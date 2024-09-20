import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import optimize, stats
from scipy.special import logsumexp, softmax
from extract_data import make_dir, log
from cme_toolbox import CMEModel  # may be unnecessary
import multiprocessing
import os
import itertools
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity

# lbfgsb has a deprecation warning for .tostring(), probably in FORTRAN interface
import warnings
from plot_aesthetics import aesthetics

from tqdm import tqdm

# from tqdm.contrib.concurrent import process_map  # or thread_map


# warnings.filterwarnings("ignore", category=DeprecationWarning) #let's do more gargeted stuff...


class InferenceParameters:
    """Stores parameters and distributes the multi-grid point inference procedure.

    Attributes
    ----------
    gradient_params: dict
        settings for gradient descent.
        "max_iterations" defines the maximum number of gradient descent iterations.
        "init_pattern" defines whether the first try starts at the method of moments estimate.
        "num_restarts" defines how many attempts should be made.
    phys_lb: float np.ndarray
        log10 lower bounds on biological parameters.
    phys_ub: float np.ndarray
        log10 upper bounds on biological parameters.
    grad_bnd: scipy.optimize.Bounds
        log10 lower and upper bounds on biological parameters.
    use_lengths: bool
        if True, the nascent Poisson model technical variation parameter is a
        coefficient multiplied by gene length.
        if False, the parameter is the genome-wide nascent sampling rate.
    samp_lb: np.ndarray
        log10 lower bounds on technical variation parameters.
    samp_ub: np.ndarray
        log10 upper bounds on technical variation parameters.
    gridsize: list of ints or int np.ndarray
        grid size for evaluating the technical variation parameters.
    model: monod.cme_toolbox.CMEModel
        CME model used for inference.
    k: int
        number of components in mixture model, default 10
    epochs: int
        number of epochs to run EM procedure for, default 100
    n_phys_pars: int
        number of biological model parameters.
    n_samp_pars: int
        number of technical variation model parameters. Set to 2 for consistency.
    dataset_string: str
        dataset-specific directory location.
    inference_string: str
        run-specific directory location within dataset_string.
    sampl_vals: list of lists of floats
        list of grid points.
    X: np.ndarray
        grid point values representing unspliced RNA sampling parameters.
    Y: np.ndarray
        grid point values representing spliced RNA sampling parameters.
    n_grid_pts: int
        total number of grid points to evaluate.

    """

    def __init__(
        self,
        dataset_string,
        model,
        use_lengths=True,
        gradient_params={
            "max_iterations": 10,
            "init_pattern": "moments",
            "num_restarts": 1,
        },
        run_meta="",
        phys_lb=None,
        phys_ub=None,
        samp_lb=None,
        samp_ub=None,
        gridsize=None,
        poisson_average_log_length=5,
        k=10,
        epochs=100
    ):
        """Initialize the InferenceParameters instance.

        Parameters
        ----------
        phys_lb: list of floats or float np.ndarray
            log10 lower bounds on biological parameters.
        phys_ub: list of floats or float np.ndarray
            log10 upper bounds on biological parameters.
        samp_lb: list of floats or float np.ndarray
            log10 lower bounds on technical variation parameters.
        samp_ub: list of floats or float np.ndarray
            log10 upper bounds on technical variation parameters.
        gridsize: list of ints or int np.ndarray
            grid size for evaluating the technical variation parameters.
        dataset_string: str
            dataset-specific directory location.
        model: monod.cme_toolbox.CMEModel
            CME model used for inference.
        use_lengths: bool, optional
            if True, the nascent Poisson model technical variation parameter is a
            coefficient multiplied by gene length.
            if False, the parameter is the genome-wide nascent sampling rate.
        gradient_params: dict, optional
            settings for gradient descent.
            "max_iterations" defines the maximum number of gradient descent iterations.
            "init_pattern" defines whether the first try starts at the method of moments estimate.
            "num_restarts" defines how many attempts should be made.
        run_meta: str, optional
            any additional metadata to append to the run directory name.
        """
        # Set biophysical parameter values to defaults.
        if phys_lb is None:
            phys_lb = model.bio_bounds['phys_lb']
        if phys_ub is None:
            phys_ub = model.bio_bounds['phys_ub']

        # Set technical sequencing parameter values to defaults.
        if samp_lb is None:
            samp_lb = model.seq_bounds['samp_lb']
        if samp_ub is None:
            samp_ub = model.seq_bounds['samp_ub']
        if gridsize is None:
            gridsize = model.seq_bounds['gridsize']
        
        self.gradient_params = gradient_params
        self.phys_lb = np.array(phys_lb)
        self.phys_ub = np.array(phys_ub)
        self.grad_bnd = scipy.optimize.Bounds(phys_lb, phys_ub)

        self.use_lengths = use_lengths
        self.poisson_average_log_length = poisson_average_log_length

        if model.seq_model == "None":
            log.info(
                "Sequencing model set to None. All sampling parameters set to null."
            )
            samp_lb = [1, 1]
            samp_ub = [1, 1]
            gridsize = [1, 1]

        self.samp_lb = np.array(samp_lb)
        self.samp_ub = np.array(samp_ub)
        self.gridsize = gridsize

        self.construct_grid()
        self.model = model

        self.k = k
        self.epochs = epochs

        self.n_phys_pars = model.get_num_params()
        self.n_samp_pars = len(self.samp_ub)  # this will always be 2 for now

        if len(run_meta) > 0:
            run_meta = "_" + run_meta

        self.dataset_string = dataset_string
        
        inference_string = f"{dataset_string}/{model.bio_model}_{model.seq_model}_"
        for i in range(len(gridsize)):
            inference_string += f"{gridsize[i]:.0f}x"
        inference_string = inference_string[:-1]
        inference_string += f"{run_meta}"
        
        make_dir(inference_string)
        self.inference_string = inference_string
        inference_parameter_string = inference_string + "/parameters.pr"
        self.store_inference_parameters(inference_parameter_string)

    def construct_grid(self):
        """Creates a grid of points over the two-dimensional technical variation parameter domain.

        Sets
        ----
        sampl_vals: list of lists of floats
            list of grid points.
        grid_values_sampl: list of np.ndarrays
            grid point values representing sampling parameters for each modality.
        n_grid_pts: int
            total number of grid points to evaluate.
        """
        linspaces = [np.linspace(self.samp_lb[i], self.samp_ub[i], self.gridsize[i]) for i in range(len(self.gridsize))]
        grid_values_sampl = np.meshgrid(*linspaces, indexing="ij")
        
        grid_values_sampl = [i.flatten() for i in grid_values_sampl]
        self.grid_values_sampl = grid_values_sampl
        
        self.sampl_vals = list(zip(*grid_values_sampl))
        self.n_grid_points = len(grid_values_sampl[0])    


    def store_inference_parameters(self, inference_parameter_string):
        """This helper method attempts to save the InferenceParameters object.

        Parameters
        ----------
        inference_parameter_string: str
            file location.
        """
        try:
            with open(inference_parameter_string, "wb") as ipfs:
                pickle.dump(self, ipfs)
            log.info(
                "Global inference parameters stored to {}.".format(
                    inference_parameter_string
                )
            )
        except:
            log.error(
                "Global inference parameters could not be stored to {}.".format(
                    inference_parameter_string
                )
            )

    def fit_all_grid_points(self, search_data, num_cores=1):
        """Fits the search data for all genes over all grid points.

        Parameters
        ----------
        num_cores: int
            number of cores to use for parallelization over grid points.
        search_data: monod.extract_data.SearchData
            SearchData object with the data to fit.

        Returns
        -------
        full_result_string: str
            disk location of the SearchResults object.

        """

        t1 = time.time()
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # ***** Assuming no parallelized grid search for now ***** 
        # if num_cores > 1: 
        #     log.info("Starting parallelized grid scan.") 
        #     parallelize(
        #         function=self.par_fun,
        #         iterable=zip(
        #             range(self.n_grid_points),
        #             [[search_data, self.model]] * self.n_grid_points,
        #             [self.k] * self.n_grid_points,
        #             [self.epochs] * self.n_grid_points,
        #         ),
        #         num_cores=num_cores,
        #         num_entries=self.n_grid_points,
        #         completion_message="Parallelized grid scan complete.",
        #         termination_message="The scan has been manually terminated.",
        #         error_message="The scan has been terminated due to computation issues. Please check MoM estimates.",
        #     )
        # else:
        if self.n_grid_points > 1:
            raise ValueError("Multiple grid points not implemented yet for meK-Means")
        log.info("Starting non-parallelized grid scan.")
        [
            self.par_fun(x)
            for x in zip(
                range(self.n_grid_points),
                [[search_data, self.model]] * self.n_grid_points,
                [self.k] * self.n_grid_points,
                [self.epochs] * self.n_grid_points,
                [num_cores] * self.n_grid_points,
            )
        ]
        log.info("Non-parallelized grid scan complete.")

        #Loop through assignments, and save each k results
        warnings.resetwarnings()
        full_result_strings = []
        full_results = []
        for i in range(self.k):
            results = SearchResults(self, search_data, i)
            results.aggregate_grid_points(clean=False)
            if results.save == True:
                full_result_string = results.store_on_disk()
                full_result_strings += [full_result_string]
                full_results += [results]

        t2 = time.time()
        log.info("Runtime: {:.1f} seconds.".format(t2 - t1))
        return full_results

 

    def par_fun(self, inputs):
        """Helper method for the grid point parallelization procedure.

        Parameters
        ----------
        inputs: tuple
            entry 0: int
                point index within [0, n_grid_points) to evaluate at.
            entry 1: tuple
                entry 0: monod.extract_data.SearchData
                    SearchData object with the data to fit.
                entry 1: monod.cme_toolbox.CMEModel
                    CME model used for inference.
            entry 2: int
                number of mixture components
        """
        point_index, (search_data, model), k, epochs, num_cores = inputs
        grad_inference = GradientInference(self, model, search_data, point_index, k, epochs)
        grad_inference.fit_all_genes(model, search_data, num_cores)


class GradientInference:
    """Runs the grid point-specific inference procedures.

    Attributes
    ----------
    grid_point: list of floats
        genome-wide technical variation parameter values at the current grid point.
    point_index: int
        the index of the current point, within [0, n_grid_points).
    k: int
        number of components in mixture model
    epochs: int
        number of epochs for EM
    regressor: np.ndarray
        gene-specific technical variation parameter values at the current grid point.
        these values will be different for each gene if use_lengths=True in the
        InferenceParameters constructor.
    grad_bnd: scipy.optimize.Bounds
        log10 lower and upper bounds on biological parameters.
    gradient_params: dict
        settings for gradient descent.
        "max_iterations" defines the maximum number of gradient descent iterations.
        "init_pattern" defines whether the first try starts at the method of moments estimate.
        "num_restarts" defines how many attempts should be made.
    phys_lb: float np.ndarray
        log10 lower bounds on biological parameters.
    phys_ub: float np.ndarray
        log10 upper bounds on biological parameters.
    n_phys_pars: int
        number of biological model parameters.
    n_samp_pars: int
        number of technical variation model parameters.
    inference_string: str
        run-specific directory location.
    param_MoM: np.ndarray
        method of moments estimates for all genes under the current technical variation parameters.
    """

    def __init__(self, global_parameters, model, search_data, point_index, k, epochs):
        """Initialize a GradientInference object.

        Parameters
        ----------
        global_parameters: InferenceParameters
            information about the global parameter inference procedure.
        model: monod.cme_toolbox.CMEModel
            CME model used for inference.
        search_data: monod.extract_data.SearchData
            SearchData object with the data to fit.
        point_index: int
            the index of the current point, within [0, n_grid_points).
        k: int
            number of components in mixture model
        epochs: int
            number of epochs for EM

        Sets
        ----
        grid_point: list of floats
            genome-wide technical variation parameter values at the current grid point.
        point_index: int
            the index of the current point, within [0, n_grid_points).
        regressor: np.ndarray
            gene-specific technical variation parameter values at the current grid point.
            these values will be different for each gene if use_lengths=True in the
            InferenceParameters constructor.
        grad_bnd: scipy.optimize.Bounds
            log10 lower and upper bounds on biological parameters.
        gradient_params: dict
            settings for gradient descent.
            "max_iterations" defines the maximum number of gradient descent iterations.
            "init_pattern" defines whether the first try starts at the method of moments estimate.
            "num_restarts" defines how many attempts should be made.
        phys_lb: float np.ndarray
            log10 lower bounds on biological parameters.
        phys_ub: float np.ndarray
            log10 upper bounds on biological parameters.
        n_phys_pars: int
            number of biological model parameters.
        n_samp_pars: int
            number of technical variation model parameters.
        weights: list of floats
            p(z=k) values
        theta: dict 
            dict of (param_estimates,kld,obj_fun,d_time) for each k component 
        inference_string: str
            run-specific directory location.
        param_MoM: np.ndarray
            method of moments estimates for all genes under the current technical variation parameters.

        """
        regressor = np.array(
            [global_parameters.sampl_vals[point_index]] * search_data.n_genes
        )
        if global_parameters.use_lengths:
            if model.seq_model == "Bernoulli":
                raise ValueError(
                    "The Bernoulli model does not yet have a physical length-based model."
                )
            elif model.seq_model == "None":
                raise ValueError(
                    "The model without technical noise has no length effects."
                )
            elif model.seq_model == "Poisson":
                regressor[:, 0] += search_data.gene_log_lengths
            else:
                raise ValueError(
                    "Please select a technical noise model from {Poisson}, {Bernoulli}, {None}."
                )
        self.grid_point = global_parameters.sampl_vals[point_index]
        self.point_index = point_index
        self.k = k
        self.epochs = epochs
        self.regressor = regressor
        self.grad_bnd = global_parameters.grad_bnd
        self.gradient_params = global_parameters.gradient_params
        self.phys_lb = global_parameters.phys_lb
        self.phys_ub = global_parameters.phys_ub
        self.n_phys_pars = global_parameters.n_phys_pars
        self.n_samp_pars = global_parameters.n_samp_pars

        #Init weights
        self.weights = np.ones(self.k)/self.k
        self.theta = {}

        self.inference_string = global_parameters.inference_string
        if self.gradient_params["init_pattern"] == "moments":
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            self.param_MoM = np.asarray(
                [
                    model.get_MoM(
                        search_data.moments[i],
                        global_parameters.phys_lb,
                        global_parameters.phys_ub,
                        regressor[i],
                    )
                    for i in range(search_data.n_genes)
                ]
            )
            warnings.resetwarnings()

    def _get_parameters(self,search_data):
        """Get inferred parameter results for storage.

        Parameters
        ----------
        search_data: monod.extract_data.SearchData
            SearchData object with the data to fit.

        Returns
        ----------
        params: np.ndarray
            n_genes x n_phys_pars x k parameters
        kl: np.ndarray
            n_genes x k klds
        obj: np.ndarray
            k, kld sums
        t: np.ndarray
            k, d_times
        self.weights:
            k, mixture weights 

        """
        theta = self.theta.copy()
        params = np.zeros((search_data.n_genes,self.n_phys_pars,self.k))
        kl = np.zeros((search_data.n_genes,self.k))
        obj = np.zeros(self.k)
        t = np.zeros(self.k)

        for k in list(theta.keys()):
            param_estimates, klds, obj_func, d_time = theta[k]
            params[:,:,k] = param_estimates
            kl[:,k] = klds
            obj[k] = obj_func
            t[k] = d_time

        return params, kl, obj, t, self.weights.copy()  
    
    def _initialize_Q(self,search_data):
        """Initialize posterior values p(z=k|x).

        Parameters
        ----------
        search_data: monod.extract_data.SearchData
            SearchData object with the data to fit.

        Returns
        ----------
        Q: np.ndarray
            obs x k mixture components for p(z=k|x)

        """
        n = search_data.n_cells

        #Init Q with U+S K-Means clusters for now
        # Assume first two layers are spliced and unspliced (true for all models so far).
        # S = 
        # U = search_data.layers[0,:,:]
        S_total = search_data.layers[0,:,:] + search_data.layers[1,:,:]
        tots = np.sum(S_total,axis=1)
        divids = (1e4/tots)[:,None]
        S_total = S_total*divids
        S_total = np.log1p(S_total)
        S_total[np.isnan(S_total)] = 0

        # KMeans input: (n_samples, n_features)
        kmeans = KMeans(n_clusters=self.k, random_state=0).fit(S_total)
        labs = kmeans.labels_

        #Bias Q towards initial cluster assignments
        Q=np.random.uniform(0,1,size=(n, self.k))
        for ind in range(self.k):
            inds = labs==ind
            Q[inds,ind] = 0.9
        
        Q *= self.weights[None,:]
        Q=Q/Q.sum(axis=(-1),keepdims=True)
        return Q

    def _part_search_data(self,search_data,Q,EPS=1e-6,padding=None): 
        """Returns search_data counts after hard assignment to mixture comp.

        Parameters
        ----------
        search_data: monod.extract_data.SearchData
            SearchData object with the data to fit.
        Q: posterior values
            Posterior probs of k mixture components

        Returns
        ----------
        SearchData dict
            dict, with k (keys) and SearchData objects (values)
        """

        datas = []
        inds = []

        #Select k with max post for each obs
        max_ks = np.argmax(Q, axis=1)  
        # options = range(self.k)
        # #Can select based on probability, maybe for initial epoch
        # max_ks = np.array([np.random.choice(options,1,list(Q[p,:])) for p in range(Q.shape[0])]).squeeze()
        
        layer_names = search_data.layer_names
        gene_names = search_data.gene_names
        n_genes = len(gene_names)
        n_layers = len(layer_names)

        # Define default padding if None
        if padding is None:
            padding = [10] * n_layers  # One padding value per layer
        
        # Ensure padding is a column vector
        padding = np.asarray(padding)[:, None]
        
        for k in np.unique(max_ks):
            #Select which obs in k 
            obs_inds = max_ks == k
            layers = search_data.layers[:,obs_inds,:]
            n_cells = np.sum(obs_inds)

            new_layers = np.transpose(layers, axes=(0, 2, 1))
            # Compute maximum expression value across cells for each gene and each layer
            max_values = np.amax(new_layers, axis=2)  # Shape: (n_genes, n_layers)
        
            # Add padding to the maximum values
            M = (max_values + padding).astype(int)

            hist_type = search_data.hist_type

            hist = make_histogram(layers, layer_names, hist_type, M)
            moments = get_moment_dicts(layers, layer_names)
            

            #Remake SearchData object
            attr_names = [
                "M",
                "hist",
                "moments",
                "n_genes",
                "gene_names",
                "n_cells",
                "layers",
                "hist_type",
                "layer_names",
            ]

            attr_values = [M,
                hist,
                moments,
                n_genes,
                gene_names,
                n_cells,
                layers,
                hist_type,
                layer_names]

            try:
                gene_log_lengths = search_data.gene_log_lengths
                attr_names += ['gene_log_lengths']
                attr_values += [gene_log_lengths]
                
            except AttributeError:
                pass
            
            sub_data = SearchData(attr_names, *attr_values)                


            inds += [k]
            datas += [sub_data]

        return dict(zip(inds,datas))
    
    def _m_step(self,model,k_dict,Q,EPS=1e-6,num_cores=1):
        """Update values for the k component weights and parameters theta_k.

        Parameters
        ----------
        model: monod.cme_toolbox.CMEModel
            CME model used for inference.
        k_dict: SearchData dict
            dict, with k (keys) and SearchData objects (values)
        Q: np.ndarray
            obs x k mixture components for p(z=k|x)

        """
        #Update weights
        self.weights = EPS+np.sum(Q,axis=0)
        self.weights /= self.weights.sum()

        #Get optimal parameters
        if num_cores > 1: # ****** PARALLELIZE *****
            ks = len(list(k_dict.keys()))

            log.info("Starting parallelized MLE param fits for EM.")  #[k_dict] * ks
            all_outs = parallelize(
                function=self._m_par_fun,
                iterable=zip(
                    [model] * ks,
                    list(k_dict.keys()),
                    [k_dict[k] for k in k_dict.keys()],
                ),
                num_cores=num_cores,
                num_entries=ks,
                completion_message="Parallelized MLE fits complete.",
                termination_message="The M step has been manually terminated.",
                error_message="The M step has been terminated due to computation issues. Please check MoM estimates.",
            )

            out_keys, out_params = zip(*all_outs)
            
            for o in range(len(out_keys)):
                self.theta[out_keys[o]] = out_params[o]  #Update only relevant/assigned ks
        else:
            all_outs = [self.iterate_over_genes(model, k_dict[key]) for key in list(k_dict.keys())] 
            for o in range(len(list(k_dict.keys()))):
                self.theta[list(k_dict.keys())[o]] = all_outs[o]

        return
    
    def _m_par_fun(self, inputs):
        """Helper method for the M step parallelization procedure.

        Parameters
        ----------
        inputs: tuple
            entry 0: monod.cme_toolbox.CMEModel
                CME model used for inference.
            entry 1: list
                list of k mixture components
            entry 2: list of dicts
                list of dicts with SearchData obj for each k component
        """
        model, key, k_dict = inputs #k_dict[key]
        return key, self.iterate_over_genes(model, k_dict)

    
    def _e_step(self,model,search_data,EPS=1e-15): 
        """Update posterior p(z=k|x).

        Parameters
        ----------
        model: monod.cme_toolbox.CMEModel
            CME model used for inference.
        search_data: monod.extract_data.SearchData
            SearchData object with the data to fit.

        Returns
        ----------  
        Q: np.ndarray
            obs x k mixture components for p(z=k|x)

        """
        #Get params for each k
        if search_data.hist_type == "grid":
            raise ValueError("Mixture model not yet implemented for grid hist type")
        elif search_data.hist_type == "unique":
            n_cells = search_data.n_cells
            logL = np.zeros((n_cells,self.k))

            for k in list(self.theta.keys()):
                
                params, klds, obj_fun, d_time = self.theta[k]
                logL_k = np.zeros(n_cells)

                for gene_index in range(search_data.n_genes):
                    S = search_data.layers[1][:,gene_index].astype(int)
                    U = search_data.layers[0][:,gene_index].astype(int)
                    x = np.array([U,S])

                    proposal = model.eval_model_pss(params[gene_index], search_data.M[:, gene_index], self.regressor[gene_index])
                    proposal[proposal < EPS] = EPS

                    proposal = proposal[tuple(x)]
                    logL_k += np.log(proposal) #logL for each obs, per gene

                logL[:,k] = logL_k

            logL += np.log(self.weights)[None,:]
            Q = softmax(logL, axis=1) #Posterior
            lower_bound = np.mean(logsumexp(a=logL, axis=1))
            q_func = np.sum(Q*logL) #Full EM Q-function (Q(theta|theta_t))


        return Q, lower_bound, q_func 
    
    def _fit(self,model,search_data,EPS=1e-15,num_cores=1): 
        """Update posterior p(z=k|x).

        Parameters
        ----------
        model: monod.cme_toolbox.CMEModel
            CME model used for inference.
        search_data: monod.extract_data.SearchData
            SearchData object with the data to fit.

        Returns
        ----------  
        Q: np.ndarray
            obs x k mixture components for p(z=k|x)
        lower_bound: float
            log-likelihood lower bound
        all_qs: list
            1 x epochs, list of Q(theta|theta_t) values at each epoch
        all_klds: list
            1 x epochs, list of cellxk KLD matrices at each epoch

        """

        #E-step, partition, m_step
        all_qs = []
        all_klds = []

        if self.epochs < 1:
            raise ValueError("No. of epochs must be an int > 0")
        else:
            for i in range(self.epochs):
                log.info("EM Epoch "+str(i+1)+'/'+str(self.epochs)+': ')

                Q, lower_bound, q_func = self._e_step(model,search_data) 
                k_dict = self._part_search_data(search_data,Q)

                self._m_step(model,k_dict,Q,num_cores=num_cores)
                kl = np.zeros((search_data.n_genes,self.k))
                for k in range(self.k):
                    params, klds, obj_fun, d_time = self.theta[k]
                    kl[:,k] = klds
                   
                all_klds += [kl]
                print('mstep self.weights: ', self.weights)
                print('Q Function: ', q_func) 
                print()
                all_qs += [q_func]

            final_k_dict = self._part_search_data(search_data,Q)
            
            return Q, lower_bound, all_qs, all_klds


    def optimize_gene(self, gene_index, model, search_data):
        """Fit the data for a single gene using KL divergence gradient descent.

        If init_pattern = moments, the first search's starting point is set to the
            method of moments parameter estimate for the current grid point.
        If num_restarts>1, the optimum is only updated if new KLD is lower
            than 0.99*previous lowest KLD.

        Parameters
        ----------
        gene_index: int
            index of the current gene, as defined by the gene_names attribute of search_data.
        model: monod.cme_toolbox.CMEModel
            CME model used for inference.
        search_data: monod.extract_data.SearchData
            SearchData object with the data to fit.

        Returns
        -------
        x: np.ndarray
            optimal biological parameter values.
        err: float
            Kullback-Leibler divergence of the model at x, relative to data.
        """
        x0 = (
            np.random.rand(self.gradient_params["num_restarts"], self.n_phys_pars)
            * (self.phys_ub - self.phys_lb)
            + self.phys_lb
        )
        if (
            self.gradient_params["init_pattern"] == "moments"
        ):  # this can be extended to other initialization patterns, like latin squares
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            param_MoM = np.asarray(
                [
                    model.get_MoM(
                        search_data.moments[i],
                        self.phys_lb,
                        self.phys_ub,
                        self.regressor[i],
                    )
                    for i in range(search_data.n_genes)
                ]
            )
            warnings.resetwarnings()
            x0[0] = param_MoM[gene_index] #self.param_MoM[gene_index] 
        x = x0[0]
        err = np.inf
        ERR_THRESH = 0.99

        # print(hist_type)
        hist_type = get_hist_type(search_data)

        for restart in range(self.gradient_params["num_restarts"]):
            res_arr = scipy.optimize.minimize(
                lambda x: model.eval_model_kld(
                    p=x,  
                    limits=search_data.M[:, gene_index],
                    samp=self.regressor[gene_index],
                    data=search_data.hist[gene_index],
                    hist_type=hist_type,
                ),
                x0=x0[restart],
                bounds=self.grad_bnd,
                options={
                    "maxiter": self.gradient_params["max_iterations"],
                    "disp": False,
                },
            )
            if (
                res_arr.fun < err * ERR_THRESH
            ):  # do not replace old best estimate if there is little marginal benefit
                x = res_arr.x
                err = res_arr.fun
        if not (np.isfinite(x).all()):
            log.warning("Gene index: " + str(gene_index))
            raise ValueError("Search failed. Please check input data.")
        return x, err

    def iterate_over_genes(self, model, search_data):
        """Run gradient descent for every gene at the current grid point.

        Parameters
        ----------
        model: monod.cme_toolbox.CMEModel
            CME model used for inference.
        search_data: monod.extract_data.SearchData
            SearchData object with the data to fit.

        Returns
        -------
        param_estimates: np.ndarray
            optimal biological parameter values for each gene, an n_genes x n_phys_pars array.
        klds: np.ndarray
            Kullback-Leibler divergence of the model for each gene at param_estimates.
        obj_func: float
            sum of klds; total error at the current grid point.
        d_time: float
            runtime in seconds.
        """
        t1 = time.time()

        param_estimates, klds = zip(
            *[
                self.optimize_gene(gene_index, model, search_data)
                for gene_index in range(search_data.n_genes)
            ]
        )

        klds = np.asarray(klds)
        param_estimates = np.asarray(param_estimates)
        obj_func = klds.sum()

        t2 = time.time()
        d_time = t2 - t1

        return param_estimates, klds, obj_func, d_time

    def fit_all_genes(self, model, search_data, num_cores=1):
        """Wraps iterate_over_genes and EM procedure, and stores the results on disk.

        Parameters
        ----------
        model: monod.cme_toolbox.CMEModel
            CME model used for inference.
        search_data: monod.extract_data.SearchData
            SearchData object with the data to fit.

        """
        
        #Init posterior Q
        Q = self._initialize_Q(search_data)
        #Partition search_data based on Q
        k_dict = self._part_search_data(search_data,Q)

        log.info("M Step Initial Run: ")
        self._m_step(model,k_dict,Q,num_cores=num_cores)

        #Do EM procedure over epochs
        Q, lower_bound, all_qs, all_klds = self._fit(model,search_data,num_cores=num_cores) 

        #m_step
        #search_out = self.iterate_over_genes(model, search_data)

        #Save theta (params,klds,obj_fun,d_time,weights), aic , assignments
        search_out = self._get_parameters(search_data)
        assigns = np.argmax(Q, axis=1)

        num_comp = len(np.unique(assigns))
        aic = lower_bound - (self.n_phys_pars * search_data.n_genes * num_comp + num_comp - 1)/search_data.n_cells


        results = GridPointResults(  #****** Update how stored ****** 
            *search_out,
            aic,
            assigns,
            all_qs,
            all_klds,
            self.regressor,
            self.grid_point,
            self.point_index,
            self.inference_string,
        )
        results.store_grid_point_results()

########################
## Helper functions
########################

def make_histogram(layers, layer_names, hist_type, M):
    """
    Generate histograms based on the provided layers and layer names.

    Parameters
    ----------
    layers : list of np.ndarray
        List of arrays, each corresponding to a different layer (modality) of data.
    layer_names : list of str
        List of names corresponding to each layer in the same order as `layers`.
    hist_type : str
        Type of histogram to generate ("grid", "unique", or "none").
    M : list of np.ndarray
        List of arrays specifying the bin structure for histogram calculation.

    Returns
    -------
    hist : list
        A list where each entry corresponds to a histogram for a specific gene.
    """


    hist = []
    n_cells = layers[0].shape[0]  # Assuming all layers have the same number of cells
    n_genes = layers[0].shape[1]  # Assuming all layers have the same number of genes

    for gene_index in range(n_genes):
        
        if hist_type == "grid":
            bins = [np.arange(x[gene_index] + 1) - 0.5 for x in M]
            stacked_data = np.vstack([x[:, gene_index] for x in layers]).T
            H, edges = np.histogramdd(
                stacked_data,
                bins=bins,
                density=True
            )
            xedges = edges[0]  # Assuming only one dimension for each bin
            yedges = edges[1] 
        elif hist_type == "unique":
            unique, unique_counts = np.unique(
                np.vstack([x[:, gene_index] for x in layers]).T, axis=0, return_counts=True
            )
            frequencies = unique_counts / n_cells
            unique = unique.astype(int)
            H = (unique, frequencies)
        elif hist_type == "none":
            H = [x[:, gene_index] for x in layers]

        hist.append(H)

    return hist

def get_moment_dicts(layers, layer_names, cov_matrix_key='layer_covariances'):
    """
    Compute and add mean and variance for each gene within each layer, and add covariances
    between layers for each gene, returning a list of dictionaries where each dictionary 
    corresponds to a gene with calculated moments and covariances.

    Parameters
    ----------
    adata: anndata.AnnData
        AnnData object with layers containing gene expression data.
    layer_names: list of strings.
    cov_matrix_key: str, optional
        Key under which the covariance matrix will be stored in `adata.uns`.

    Returns
    -------
    gene_moments: list of dict
        A list where each entry is a dictionary representing a gene, with keys as the column names 
        (e.g., mean, variance, covariance) and values as the corresponding values for that gene.
    """
    n_layers = len(layer_names)
    n_genes = np.shape(layers[0])[1]
    # print('n_genes', n_genes)

    gene_moments = []

    # Compute mean, variance, and covariances for each gene
    for gene_index in range(n_genes):
        gene_dict = {}

        for i in range(n_layers):
            # These have already been ordered.
            modality_name = layer_names[i]
            layer = layers[i]
            
            mean_col = f"MOM_{modality_name}_mean"
            var_col = f"MOM_{modality_name}_var"
            
            # Calculate mean and variance for each layer
            gene_dict[mean_col] = layer[:, gene_index].mean()
            gene_dict[var_col] = layer[:, gene_index].var()

        # Compute covariances between each pair of layers
        for i in range(n_layers):
            for j in range(i + 1, n_layers):
                layer_i = layers[i]
                layer_j = layers[j]
                mod_i, mod_j = layer_names[i], layer_names[j]
                layer_layer_string = f"MOM_cov_{mod_i}_{mod_j}"

                covar = np.cov(
                    [layer_i[:, gene_index].flatten(), 
                     layer_j[:, gene_index].flatten()]
                )[0, 1]
                
                gene_dict[layer_layer_string] = covar
        
        gene_moments.append(gene_dict)

    return gene_moments



def get_hist_type(search_data):
    """A helper function for backwards compatibility.

    If the histogram type is not specified in the SearchData object, assume it is the legacy
    type "grid".

    Parameters
    ----------
    search_data: monod.extract_data.SearchData
        SearchData object with the data to fit.

    Returns
    -------
    hist_type: str
        flavor of histogram used to generate search_data, either "unique" or "grid".
    """

    if hasattr(search_data, "hist_type") and search_data.hist_type == "unique":
        hist_type = "unique"
    else:
        hist_type = "grid"
    return hist_type


########################
## Helper classes
########################
class GridPointResults:
    """Temporarily stores the fit parameters for a single grid point.

    Attributes
    ----------
    param_estimates: np.ndarray
        optimal biological parameter values for each gene, an n_genes x n_phys_pars x k array.
    klds: np.ndarray
        Kullback-Leibler divergence of the model for each gene at param_estimates, n_genes x k.
    obj_func: float
        sum of klds; total error at the current grid point, (k,).
    d_time: float
        runtime in seconds, (k,).
    weights: np.ndarray
        weights for mixture components (k,)
    aic: float
        final AIC statistic for model fit
    assigns: np.ndarray
        final k components assignments for each cell (n_cells,)
    all_qs: float list
        all Q function values for each EM epoch, list of gene x k
    all_klds: np.array list
        all kld values for each EM epoch, list of gene x k
    regressor: np.ndarray
        gene-specific technical variation parameter values at the current grid point.
        these values will be different for each gene if use_lengths=True in the
        InferenceParameters constructor.
    grid_point: list of floats
        genome-wide technical variation parameter values at the current grid point.
    point_index: int
        the index of the current point, within [0, n_grid_points).
    inference_string: str
        run-specific directory location.

    """

    def __init__(
        self,
        param_estimates,
        klds,
        obj_func,
        d_time,
        weights,
        aic,
        assigns,
        all_qs,
        all_klds,
        regressor,
        grid_point,
        point_index,
        inference_string,
    ):
        """Creates a GridPointResults object and sets all of its attributes."""
        self.param_estimates = param_estimates
        self.klds = klds
        self.obj_func = obj_func
        self.d_time = d_time

        self.regressor = regressor
        self.grid_point = grid_point
        self.point_index = point_index
        self.inference_string = inference_string

        self.weights = weights
        self.aic = aic
        self.assigns = assigns
        self.all_qs = all_qs
        self.all_klds = all_klds

    

    def store_grid_point_results(self):
        """This helper method attempts to store the grid point results to disk as a grid_point_X.gp object."""
        try:
            grid_point_result_string = (
                self.inference_string + "/grid_point_" + str(self.point_index) + ".gp"
            )
            with open(grid_point_result_string, "wb") as gpfs:
                pickle.dump(self, gpfs)
            log.debug(
                "Grid point {:.0f} results stored to {}.".format(
                    self.point_index, grid_point_result_string
                )
            )
        except:
            log.error(
                "Grid point {:.0f} results could not be stored to {}.".format(
                    self.point_index, grid_point_result_string
                )
            )


class SearchResults:
    """Stores and analyzes the results of a single inference run.

    The first thirteen attributes relate to data loaded from the search.
    The others relate to data processing after the search is completed.

    Attributes
    ----------
    sp: InferenceParameters
        search parameters used to generate the run.
    inference_string: str
        run-specific directory location.
    model: monod.cme_toolbox.CMEModel
        CME model used for inference.
    n_genes: int
        number of analyzed genes.
    n_cells: int
        number of cells in the dataset.
    gene_log_lengths: float np.ndarray
        log lengths of analyzed genes.
    gene_names: str np.ndarray
        list of analyzed genes.
    param_estimates: float np.ndarray
        optimal biological parameter values for each gene, an n_grid_pts x n_genes x n_phys_pars array.
    klds: float np.ndarray
        Kullback-Leibler divergence of the model for each gene at param_estimates, an n_grid_pts x n_genes array.
    obj_func: float np.ndarray
        sum of klds at each grid points; total error at the current grid point, a length-n_grid_pts array.
    d_time: float np.ndarray
        runtime in seconds for each grid point, a length-n_grid_pts array.
    regressor: float np.ndarray
        gene-specific technical variation parameter values at each grid point.
        an n_grid_pts x n_genes array.
        these values will be different for each gene if use_lengths=True in the
        InferenceParameters constructor.
    weights: float 
        weight of mixture component for assigned cluster (k)
    aic: float
        negative AIC score for the final model fit (over all cells)
    assigns: int
        assigned cluster k
    all_qs: float np.ndarray
        array of Q function values at each epoch (over all cells)
    all_kld: float np.ndarray
        array of KLD values at each epoch (over cells in cluster k)
    filt: boolean np.array
        boolean filter array for cells in assigned cluster k (i.e. k=assigns)
    save: boolean
        boolean to save SearchResults object to disk
    analysis_figure_string: str
        directory for analysis figures.

    samp_optimum: list of floats
        estimated value of the technical noise parameters.
    samp_optimum_ind: int
        index of the sampling parameter optimum grid point.
    phys_optimum: float np.ndarray
        gene-specific physical parameter values at the sampling parameter optimum.
    regressor_optimum: float np.ndarray
        gene-specific technical variation parameter values at the sampling parameter optimum.
    csq: np.ndarray
        chi-squared statistics for all genes, computed at the grid point indexed by rejection_index.
    pval: np.ndarray
        p-values calculated by the chi-squared test at the grid point indexed by rejection_index.
    rejected_genes: bool np.ndarray
        a boolean filter that reports the genes rejected by the goodness-of-fit procedure,
        whose parameters cannot be safely interpeted.
    rejection_index: int
        the grid point at which the goodness-of-fit procedure was performed to generate
        the rejected_genes attribute.
    sigma: float np.ndarray
        the standard error of the parameter maximum likelihood estimate at the at the grid
        point indexed by sigma_index.
        a n_genes x n_phys_pars array.
    sigma_index: int
        the grid point at which the Fisher information procedure was performed to generate
        the sigma attribute.
    batch_analysis_string: str
        location of the directory for batch-wide analyses.
    """

    ####################################
    #   Construction and I/O methods   #
    ####################################
    def __init__(self, inference_parameters, search_data, assign): 
        """Creates a SearchResults object.

        Parameters
        ----------
        inference_parameters: InferenceParameters
            search parameters used to generate the run.
        search_data: monod.extract_data.SearchData
            SearchData object with the fit data.
        """
        # pull in info from search parameters.
        self.sp = inference_parameters

        self.inference_string = inference_parameters.inference_string
        self.model = inference_parameters.model

        # pull in small amount of non-cell-specific info from search data
        self.n_genes = search_data.n_genes
        self.n_cells = search_data.n_cells 
    
        self.gene_log_lengths = search_data.gene_log_lengths
        self.gene_names = search_data.gene_names

        self.param_estimates = []
        self.klds = []
        self.obj_func = []
        self.d_time = []
        self.regressor = []
        self.assigns = assign #Which state or mixture component k

        self.save = False #Save output if assign corresponds to a k with cells

        #Add mixture properties, pass in assignment and use to filter gp results
        self.weights = []
        self.aic = []
        
        self.all_qs = []
        self.all_klds = []
        self.filt = []

    def aggregate_grid_points(self,clean=True):
        """This helper method concatenates all of the grid point results.

        The method runs append_grid_point for all grid points, then removes the original grid point files.
        """
        for point_index in range(self.sp.n_grid_points):
            self.append_grid_point(point_index)
        
        self.clean_up(clean)

    def append_grid_point(self, point_index): 
        """This helper method updates the result attributes from a GridPointResult object stored on disk.

        Parameters
        ----------
        point_index: int
            index of the grid point results to load from disk.
        """
        grid_point_result_string = (
            self.inference_string + "/grid_point_" + str(point_index) + ".gp"
        )
        with open(grid_point_result_string, "rb") as ipfs:
            grid_point_results = pickle.load(ipfs)
            #Subset results based on assignments
            if self.assigns in np.unique(grid_point_results.assigns):
                self.save = True 

                self.param_estimates += [grid_point_results.param_estimates[:,:,self.assigns]]
                self.klds += [grid_point_results.klds[:,self.assigns]]
                self.obj_func += [grid_point_results.obj_func[self.assigns]]
                self.d_time += [grid_point_results.d_time[self.assigns]]
                self.regressor += [grid_point_results.regressor]

                self.aic += [grid_point_results.aic]
                self.weights += [grid_point_results.weights[self.assigns]]
                self.all_qs += [grid_point_results.all_qs]
                self.all_klds += [[i[:,self.assigns] for i in grid_point_results.all_klds]]

                self.filt = grid_point_results.assigns == self.assigns

                self.n_cells = np.sum(self.filt)
           

    def clean_up(self,clean=True):
        """This helper method removes temporary files and finalizes the SearchResults object.

        The GridPointResult objects are erased from disk, the attributes are converted to
        np.ndarrays, and a directory for analysis figures is created.
        """
        if clean:
            for point_index in range(self.sp.n_grid_points):
                os.remove(self.inference_string + "/grid_point_" + str(point_index) + ".gp")
            log.info("All grid point data cleaned from disk.")

        self.param_estimates = np.asarray(self.param_estimates)
        self.klds = np.asarray(self.klds)
        self.obj_func = np.asarray(self.obj_func)
        self.d_time = np.asarray(self.d_time)
        self.regressor = np.asarray(self.regressor)

        self.aic = np.asarray(self.aic)
        self.weights = np.asarray(self.weights)
        self.all_qs = np.asarray(self.all_qs)
        self.all_klds = np.asarray(self.all_klds)

        if self.save:
            analysis_figure_string = self.inference_string + "/analysis_figures_"+str(self.assigns)
            self.analysis_figure_string = analysis_figure_string
            make_dir(analysis_figure_string)

    def store_on_disk(self):
        """This helper method attempts to store the SearchResults object to disk.

        Returns
        -------
        full_result_string: str
            file location.
        """
        
        try:
            full_result_string = self.inference_string + "/grid_scan_results_"+str(self.assigns)+".res" #Add assign, which k component
            with open(full_result_string, "wb") as srfs:
                pickle.dump(self, srfs)
            log.debug("Grid scan results stored to {}.".format(full_result_string))
        except:
            log.error(
                "Grid scan results could not be stored to {}.".format(
                    full_result_string
                )
            )
        self.full_result_string = full_result_string
        return full_result_string

    def update_on_disk(self):
        """This helper method attempts to store a modified SearchResults object to disk.

        This is separate from store_on_disk() to avoid overwriting data.

        Returns
        -------
        upd_result_string: str
            file location.
        """
        try:
            upd_result_string = self.inference_string + "/grid_scan_results_"+str(self.assigns)+"_upd.res"
            with open(upd_result_string, "wb") as srfs:
                pickle.dump(self, srfs)
            log.debug("Updated results stored to {}.".format(upd_result_string))
        except:
            log.error(
                "Updated results could not be stored to {}.".format(upd_result_string)
            )
        self.upd_result_string = upd_result_string
        return upd_result_string

    ####################################
    #         Analysis methods         #
    ####################################

    def _subset_search_data(self,search_data,EPS=1e-6,padding=None):
        """Returns search_data counts after in specified cluster.

        Parameters
        ----------
        search_data: monod.extract_data.SearchData
            SearchData object with the data to fit.
        assign: int
            Which of the k clusters to select cells from

        Returns
        ----------
        SearchData object
            SearchData object for cells in assigned k (self.assigns)
        """
        layers = search_data.layers[:,self.filt,:]
        n_cells = self.n_cells

        layer_names = search_data.layer_names
        gene_names = search_data.gene_names
        n_genes = len(gene_names)

        new_layers = np.transpose(layers, axes=(0, 2, 1))
        # new_layers = layers
        
        # Compute maximum expression value across cells for each gene and each layer
        max_values = np.amax(new_layers, axis=2)  # Shape: (n_genes, n_layers)
    
        # Define default padding if None
        if padding is None:
            padding = [10] * max_values.shape[1]  # One padding value per layer
        
        # Ensure padding is a column vector
        padding = np.asarray(padding)[:, None]
    
        # Add padding to the maximum values
        M = (max_values + padding.T).astype(int)

        hist_type = search_data.hist_type

        hist = make_histogram(layers, layer_names, hist_type, M)
        moments = get_moment_dicts(layers, layer_names)

        #Remake SearchData object
        attr_names = [
            "M",
            "hist",
            "moments",
            "gene_log_lengths",
            "n_genes",
            "gene_names",
            "n_cells",
            "layers",
            "hist_type",
            "layer_names"
        ]

        
        sub_data = SearchData(
            attr_names,
            M,
            hist,
            moments,
            search_data.gene_log_lengths,
            n_genes,
            gene_names,
            n_cells,
            layers,
            hist_type,
            layer_names
        )


        # S = layers[1,:,:]
        # U = layers[0,:,:]
        # l = [U,S]
        # if padding is None:
        #     padding = np.asarray([10] * len(l))

        # M = np.amax(l, axis=2) + padding[:, None]

        # hist = []
        # moments = []
        # for gene_index in range(n_genes):
        #     if search_data.hist_type == "grid":
        #         H, xedges, yedges = np.histogramdd(
        #             *[x[gene_index] for x in l],
        #             bins=[np.arange(x[gene_index] + 1) - 0.5 for x in M],
        #             density=True
        #         )
        #     elif search_data.hist_type == "unique":
        #         unique, unique_counts = np.unique(
        #             np.vstack([x[gene_index] for x in l]).T, axis=0, return_counts=True
        #         )
        #         frequencies = unique_counts / n_cells
        #         unique = unique.astype(int)
        #         H = (unique, frequencies)

        #     hist.append(H)

        #     moments.append(
        #         {
        #             "S_mean": S[gene_index].mean(),
        #             "U_mean": U[gene_index].mean(),
        #             "S_var": S[gene_index].var(),
        #             "U_var": U[gene_index].var(),
        #         }
        #     )
        

        # #Remake SearchData object
        # attr_names = [
        #     "M",
        #     "hist",
        #     "moments",
        #     "gene_log_lengths",
        #     "n_genes",
        #     "gene_names",
        #     "n_cells",
        #     "layers",
        #     "hist_type",
        # ]

        
        # sub_data = SearchData(
        #     attr_names,
        #     M,
        #     hist,
        #     moments,
        #     search_data.gene_log_lengths,
        #     n_genes,
        #     gene_names,
        #     n_cells,
        #     layers,
        #     search_data.hist_type,
        # )

        
        return sub_data

    def find_sampling_optimum(self, gene_filter=None, discard_rejected=False):
        """Identify and set the technical parameter optimum by minimizing the total KLD.

        Parameters
        ----------
        gene_filter: None or np.ndarray, optional
            If None, use all genes.
            If a boolean or integer filter, use the filtered gene subset.
        discard_rejected: bool, optional
            whether to omit genes in the rejected_genes attribute during the calculation.

        Returns
        -------
        samp_optimum: list of floats
            estimated value of the technical noise parameters.
        """
        if gene_filter is None:
            total_divergence = self.obj_func
        else:
            gene_filter = self.get_bool_filt(gene_filter, discard_rejected)
            total_divergence = self.klds[:, gene_filter].sum(1)
        samp_optimum_ind = np.argmin(total_divergence)
        self.set_sampling_optimum(samp_optimum_ind)
        return self.samp_optimum

    def set_sampling_optimum(self, samp_optimum_ind):
        """Set the technical parameter optimum to a specific grid point index.

        Define the value, then update the samp_optimum_ind, samp_optimum,
        phys_optimum, and regressor_optimum attributes that depend on this value.

        Parameters
        ----------
        samp_optimum_ind: int
            index of the grid point.

        Returns
        -------
        samp_optimum: list of floats
            value of the technical noise parameters at the grid point.
        """
        self.samp_optimum_ind = samp_optimum_ind
        self.samp_optimum = self.sp.sampl_vals[samp_optimum_ind]
        self.phys_optimum = self.param_estimates[samp_optimum_ind]
        self.regressor_optimum = self.regressor[samp_optimum_ind]
        return self.samp_optimum

    def plot_landscape(
        self,
        ax,
        plot_optimum=True,
        gene_filter=None,
        discard_rejected=False,
        logscale=True,
        colorbar=False,
        hideticks=False,
        savefig=False,
    ):
        """Plot the 2D Kullback-Leibler divergence (KLD) landscape over the evaluated grid points.

        The landscape is computed by adding together gene-specific values at each grid points,
        potentially only over a subset of the genes.

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            axes to plot into.
        plot_optimum: bool, optional
            whether to plot a dot at the estimated optimum.
        gene_filter: bool, optional
            If None, plot landscape for all genes.
            If a boolean or integer filter, plot lanscape determined by the filtered gene subset.
        discard_rejected: bool, optional
             whether to omit genes in the rejected_genes attribute during the calculation.
        logscale: bool, optional
            whether to show log10 of the total KLD instead of the raw value.
        colorbar: bool, optional
            whether to display a colobrar.
        hideticks: bool, optional
            whether to hide the ticks and coordinates around the plot.
        savefig: bool, optional
            whether to save the figure to disk.
        """

        # if gene_filter is None:
        #     total_divergence = self.obj_func
        # else:
        #     gene_filter = self.get_bool_filt(gene_filter, discard_rejected)
        gene_filter = self.get_bool_filt(gene_filter, discard_rejected)
        total_divergence = self.klds[:, gene_filter].sum(1)

        if logscale:
            total_divergence = np.log10(total_divergence)

        dx = (np.asarray(self.sp.samp_ub) - np.asarray(self.sp.samp_lb)) / (
            np.asarray(self.sp.gridsize) - 1
        )
        dx[dx < 1e-10] = 0.1
        extent = [
            self.sp.samp_lb[0] - dx[0] / 2,
            self.sp.samp_ub[0] + dx[0] / 2,
            self.sp.samp_lb[1] - dx[1] / 2,
            self.sp.samp_ub[1] + dx[1] / 2,
        ]
        lnd = ax.imshow(
            np.flipud(np.reshape(total_divergence, self.sp.gridsize).T), extent=extent
        )

        if plot_optimum:
            ax.scatter(self.samp_optimum[0], self.samp_optimum[1], c="crimson", s=50)
        if colorbar:
            plt.colorbar(lnd, ax=ax)
        if hideticks:
            ax.set_xticks([])
            ax.set_yticks([])
        if savefig:
            fig_string = self.analysis_figure_string + "/landscape.png"
            plt.savefig(fig_string, dpi=450)
            log.info("Figure stored to {}.".format(fig_string))

    def get_bool_filt(self, gene_filter, discard_rejected):
        """This helper method constructs gene filters.

        The method intersects an arbitrary (potentially None, boolean, or integer) gene selection with the
        rejected_genes attribute (a boolean filter) to produce an overall set of genes to be analyzed or visualized.

        Parameters
        ----------
        gene_filter: None, bool np.ndarray, or int np.ndarray
            if an array, select the genes in the array,
            if None, consider all genes.
        discard_rejected: bool
            if True, omit genes that are rejected by the goodness-of-fit procedure.

        Returns
        -------
        gene_filter: bool np.ndarray
            genes that are retained by the filter.
        """

        if gene_filter is None:
            gene_filter = np.ones(self.n_genes, dtype=bool)
        else:
            if gene_filter.dtype is not bool:
                gf_temp = np.zeros(self.n_genes, dtype=bool)
                gf_temp[gene_filter] = True
                gene_filter = gf_temp

        if discard_rejected:
            if hasattr(self, "rejection_index"):
                if self.rejection_index != self.samp_optimum_ind:
                    raise ValueError("Sampling parameter value is inconsistent.")
                gene_filter = np.logical_and(~self.rejected_genes, gene_filter)
            else:
                log.info("No rejection statistics have been computed.")
        return gene_filter

    def plot_param_marg(
        self,
        gene_filter=None,
        discard_rejected=True,
        nbin=15,
        fitlaw=scipy.stats.norminvgauss,
        axis_search_bounds=True,
        figsize=None,
    ):
        """Plot and fit the biological parameter distributions at the sampling parameter optimum.

        Parameters
        ----------
        gene_filter: None or np.ndarray, optional
            If None, plot all genes.
            If a boolean or integer filter, plot the filtered gene subset.
        discard_rejected: bool, optional
             whether to omit genes in the rejected_genes attribute.
        nbin: int, optional
            number of bins used to construct the histogram.
        fitlaw: scipy.stats.rv_continuous, optional
            statistical law used to fit the parameter distributions.
        axis_search_bounds: bool, optional
            whether to place the x-limits of the plots at the parameter search bounds.
        figsize: tuple or None, optional
            figure dimensions.
        """
        num_params = self.sp.n_phys_pars
        figsize = figsize or (4 * num_params, 4)
        fig1, ax1 = plt.subplots(nrows=1, ncols=num_params, figsize=figsize)

        # identify genes to plot, extract their data
        gene_filter = self.get_bool_filt(gene_filter, discard_rejected)
        param_data = self.phys_optimum[gene_filter, :]

        for i in range(num_params):
            ax1[i].hist(
                param_data[:, i],
                nbin,
                density=True,
                color=aesthetics["hist_face_color"],
            )
            if fitlaw is not None:
                fitparams = fitlaw.fit(param_data[:, i])

                xmin, xmax = ax1[i].get_xlim()
                x = np.linspace(xmin, xmax, 100)
                p = fitlaw.pdf(x, *fitparams)
                ax1[i].plot(
                    x,
                    p,
                    "--",
                    linewidth=aesthetics["hist_fit_lw"],
                    color=aesthetics["hist_fit_color"],
                )

            if axis_search_bounds:
                ax1[i].set_xlim([self.sp.phys_lb[i], self.sp.phys_ub[i]])
            ax1[i].set_title(self.model.get_log_name_str()[i])
            ax1[i].set_xlabel(r"$log_{10}$ value")
        fig1.tight_layout()
        fig_string = self.analysis_figure_string + "/parameter_marginals.png"
        plt.savefig(fig_string, dpi=450)
        log.info("Figure stored to {}.".format(fig_string))

    def plot_KL(self, ax, gene_filter=None, discard_rejected=True, nbin=15):
        """Plot the distribution of KL divergences at the sampling parameter optimum.

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            axes to plot into.
        gene_filter: None or np.ndarray, optional
            If None, plot all genes.
            If a boolean or integer filter, plot the filtered gene subset.
        discard_rejected: bool, optional
             whether to omit genes in the rejected_genes attribute.
        nbin: int, optional
            number of bins used to construct the histogram.
        """
        gene_filter = self.get_bool_filt(gene_filter, discard_rejected)
        kld_data = self.klds[self.samp_optimum_ind][gene_filter]

        ax.hist(kld_data, nbin, color=aesthetics["hist_face_color"])
        ax.set_xlabel("KL divergence")
        ax.set_ylabel("# genes")
        fig_string = self.analysis_figure_string + "/kldiv.png"
        plt.savefig(fig_string, dpi=450)
        log.info("Figure stored to {}.".format(fig_string))

    def chisquare_testing(
        self,
        search_data,
        viz=False,
        EPS=1e-15,
        threshold=0.05,
        bonferroni=True,
        reject_at_bounds=True,
        bound_thr=0.01,
        grouping_thr=5,
        use_hellinger=True,
        hellinger_thr=0.05,
    ):
        """Perform goodness-of-fit testing at the current sampling parameter optimum to identify poor fits.

        This method performs two rounds of goodness-of-fit testing.
        First, it applies a chi-squared test to the distributions induced by biological parameter values
        at the sampling parameter optimum.
        Optionally, it also rejects genes that are too close to the search parameter bounds,
        as they typically exhibit poor gradient descent performance or do not have enough counts to
        reliably estimate parameters.
        This is typically sufficient to reject genes with proposed distributions that are grossly
        dissimilar to the raw data, whether due to inference failure or model misspecification.

        We typically expect about 5-15% of the genes to be rejected.

        Parameters
        ----------
        search_data: monod.extract_data.SearchData
            SearchData object with the fit data.
        viz: bool, optional
            whether to visualize the histogram of the chi-square statistic.
        EPS: float, optional
            probability rounding parameter: anything below this value is rounded to EPS.
        threshold: float, optional
            chi-square rejection criterion; everything below this critical p-value
            is rejected as unlikely to have been generated by the model.
        bonferroni: float, optional
            whether to apply the Bonferroni correction to the p-value threshold.
        reject_at_bounds: bool, optional
            whether to also discard genes that are near the search bounds.
        bound_thr: float, optional
            how close the parameters must be close to the bounds (in units of the allowed
            parameter range) to trigger rejection.
        grouping_thr: float or int, optional
            minimum bin size for chi-squared test.
        use_hellinger: bool, optional
            whether to use the Hellinger distance as an additional "effect size" estimate for goodness-of-fit.
        hellinger_thr: float, optional
            which threshold to use for the Hellinger distance rejection.

        Returns
        -------
        csq: np.ndarray
            chi-squared statistics for all genes, computed at the grid point indexed by rejection_index.
        pval: np.ndarray
            p-values calculated by the chi-squared test at the grid point indexed by rejection_index.
        hellinger: np.ndarray
            hellinger distances for all genes, computed at the grid point indexed by rejection_index.

        Sets
        ----
        csq: np.ndarray
            chi-squared statistics for all genes, computed at the grid point indexed by rejection_index.
        pval: np.ndarray
            p-values calculated by the chi-squared test at the grid point indexed by rejection_index.
        hellinger: np.ndarray
            hellinger distances for all genes, computed at the grid point indexed by rejection_index.
        rejected_genes: bool np.ndarray
            a boolean filter that reports the genes rejected by the goodness-of-fit procedure,
            whose parameters cannot be safely interpeted.
        rejection_index: int
            the grid point at which the goodness-of-fit procedure was performed to generate
            the rejected_genes attribute.

        """
        t1 = time.time()
        #search_data =  self._subset_search_data(search_data) #Already subset
        hist_type = get_hist_type(search_data)

        csqarr = []
        hellinger = []
        for gene_index in range(self.n_genes):
            lm = search_data.M[:, gene_index]
            expect_freq = (
                self.model.eval_model_pss(
                    self.phys_optimum[gene_index],
                    lm,
                    self.regressor_optimum[gene_index],
                )
                * search_data.n_cells
            )
            # expected_freq[expected_freq < EPS] = EPS
            # expected_freq /= expected_freq.sum()
            # PROPOSAL = search_data.n_cells * expected_freq

            if hist_type == "grid":
                raise ValueError("Not implemented in current version.")
            elif hist_type == "unique":
                counts = np.concatenate(
                    (search_data.n_cells * search_data.hist[gene_index][1], [0])
                )
                expect_freq = expect_freq[
                    search_data.hist[gene_index][0][:, 0],
                    search_data.hist[gene_index][0][:, 1],
                ]
                expect_freq = np.concatenate(
                    (expect_freq, [search_data.n_cells - expect_freq.sum()])
                )

            hellinger_ = (
                1
                / np.sqrt(2)
                * (
                    (
                        np.sqrt(expect_freq / search_data.n_cells)
                        - np.sqrt(counts / search_data.n_cells)
                    )
                    ** 2
                ).sum()
            )

            bins = []
            bin_ind = 0
            run_bin_obs = 0
            run_bin_exp = 0
            bin_obs = []
            bin_exp = []
            for i in range(len(counts)):
                bins.append(bin_ind)
                run_bin_obs += counts[i]
                run_bin_exp += expect_freq[i]
                if min(run_bin_obs, run_bin_exp) < 5:  # and i
                    pass
                else:
                    bin_ind += 1
                    bin_obs.append(run_bin_obs)
                    bin_exp.append(run_bin_exp)
                    run_bin_obs = 0
                    run_bin_exp = 0
            bins = np.asarray(bins)
            observed = np.asarray(bin_obs)
            proposed = np.asarray(bin_exp)
            observed[-1] += run_bin_obs
            proposed[-1] += run_bin_exp
            bins[bins == len(observed)] = len(observed) - 1

            for b_ in range(len(bin_obs)):
                assert np.isclose(observed[b_], counts[bins == b_].sum())
                assert np.isclose(proposed[b_], expect_freq[bins == b_].sum())
            assert np.isclose(observed.sum(), search_data.n_cells)
            assert np.isclose(proposed.sum(), search_data.n_cells)
            assert np.isclose(search_data.n_cells, counts.sum())
            assert np.isclose(search_data.n_cells, expect_freq.sum())

            csqarr += [
                scipy.stats.mstats.chisquare(
                    observed,
                    proposed,  # chisq_data, chisq_prop,
                    self.model.get_num_params(),
                )
            ]

            hellinger.append(hellinger_)

        csq, pval = zip(*csqarr)
        csq = np.asarray(csq)
        pval = np.asarray(pval)
        hellinger = np.asarray(hellinger)

        if bonferroni:
            threshold /= self.n_genes
        self.rejected_genes = pval < threshold
        if use_hellinger:
            rej_hellinger = hellinger > hellinger_thr
            self.rejected_genes = (self.rejected_genes) & rej_hellinger

        if reject_at_bounds:
            bound_range = self.sp.phys_ub - self.sp.phys_lb
            lb = self.sp.phys_lb + bound_range * bound_thr
            ub = self.sp.phys_ub - bound_range * bound_thr

            rej_at_bounds = ((self.phys_optimum < lb) | (self.phys_optimum > ub)).any(1)
            self.rejected_genes = (self.rejected_genes) | rej_at_bounds

        self.pval = pval
        self.csq = csq
        self.hellinger = hellinger
        self.rejection_index = self.samp_optimum_ind  # mostly for debug.

        if viz:
            plt.hist(csq)
            plt.xlabel("Chi-square statistic")
            plt.ylabel("# genes")
            fig_string = self.analysis_figure_string + "/chisquare.png"
            plt.savefig(fig_string, dpi=450)
            log.info("Figure stored to {}.".format(fig_string))

        t2 = time.time()
        log.info(
            "Chi-square computation complete. Rejected {:.0f} genes out of {:.0f}. Runtime: {:.1f} seconds.".format(
                np.sum(self.rejected_genes), self.n_genes, t2 - t1
            )
        )
        return (csq, pval, hellinger)

    def par_fun_hess(self, inputs):
        """Helper method for the Hessian parallelization procedure.

        Parameters
        ----------
        inputs: tuple
            entry 0: int
                gene index within [0, n_genes) to evaluate at.
            entry 1: monod.extract_data.SearchData
                SearchData object with the fit data.

        Returns
        -------
        hess: float np.ndarray
            Hessian of the current gene's KLD at the sampling parameter optimum and the
            corresponding biological parameters, evaluated with respect to the
            biological parameters.
        """
        import numdifftools  # this will fail if numdifftools has not been evaluted.

        gene_index, search_data = inputs
        search_data = self._subset_search_data(search_data)
        hist_type = get_hist_type(search_data)
        # if hasattr(search_data, "hist_type") and search_data.hist_type == "unique":
        #     hist_type = "unique"
        # else:
        #     hist_type = "grid"
        Hfun = numdifftools.Hessian(
            lambda x: self.model.eval_model_kld(
                p=x,
                limits=search_data.M[:, gene_index],
                samp=self.regressor_optimum[gene_index],
                data=search_data.hist[gene_index],
                hist_type=hist_type,
            )
        )
        hess = Hfun(self.phys_optimum[gene_index])
        return hess

    def compute_sigma(self, search_data, num_cores=1):
        """Estimate uncertainty in biological parameter values.

        This method iterates over genes, computes the Fisher information matrix by inverting
        the Hessian with respect to biological parameters, and reports the local estimate
        of standard errors of the parameters.

        This method is fairly computationally demanding, so parallelization is recommended.

        If inversion fails, the gene-specific standard errors are replaced with their mean over
        all genes that did not fail.

        Parameters
        ----------
        search_data: monod.extract_data.SearchData
            SearchData object with the fit data.
        num_cores: int, optional
            number of cores to use for parallelization over genes.

        Sets
        ----
        sigma: float np.ndarray
            the standard error of the parameter maximum likelihood estimate at the sampling
            parameter optimum.
            a n_genes x n_phys_pars array.
        sigma_index: int
            the grid point at which the Fisher information procedure was performed to generate
            the sigma attribute.

        """
        log.info("Computing local Hessian.")
        t1 = time.time()
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        if num_cores > 1:
            log.info("Starting parallelized Hessian computation.")
            hess = parallelize(
                function=self.par_fun_hess,
                iterable=zip(range(self.n_genes), [search_data] * self.n_genes),
                num_cores=num_cores,
                num_entries=self.n_genes,
                completion_message="Parallelized Hessian computation complete.",
                termination_message="The Hessian computation has been manually terminated.",
                error_message="The Hessian computation has been terminated due to numerical issues.",
                use_tqdm=False,
            )
        else:
            log.info("Starting non-parallelized Hessian computation.")
            hess = [
                self.par_fun_hess(x)
                for x in zip(range(self.n_genes), [search_data] * self.n_genes)
            ]
            log.info("Non-parallelized Hessian computation complete.")
        warnings.resetwarnings()
        hess = np.asarray(hess)

        fail = np.zeros(self.n_genes, dtype=bool)
        sigma = np.zeros((self.n_genes, self.sp.n_phys_pars))

        for gene_index in range(self.n_genes):
            try:

                hess_inv = np.linalg.inv(hess[gene_index, :, :])
                sigma[gene_index, :] = np.sqrt(np.diag(hess_inv)) / np.sqrt(
                    self.n_cells
                )
            except:
                fail[gene_index] = True
                log.info(
                    "Gene {:.0f} ran into singularity; replaced with mean. (Search converged to local minimum?) ".format(
                        gene_index
                    )
                )
                # errorbars[i,:] = np.mean(errorbars[:i,:])
            if np.any(~np.isfinite(sigma[gene_index, :])):
                fail[gene_index] = True
                log.info(
                    "Gene {:.0f} gives negative stdev; replaced with mean. (Search converged to local minimum?)".format(
                        gene_index
                    )
                )
                # errorbars[i,:] = np.mean(errorbars[:i,:])
        sigma[fail, :] = sigma[~fail, :].mean(0)
        self.sigma = sigma
        self.sigma_index = self.samp_optimum_ind  # mostly for debug

        t2 = time.time()
        log.info(
            "Standard error of the MLE computation complete. Runtime: {:.1f} seconds.".format(
                t2 - t1
            )
        )

    def resample_opt_viz(
        self, resamp_vec=(5, 10, 20, 40, 60), Ntries=4, figsize=(10, 10)
    ):
        """Test the sensitivity of the technical noise parameter landscape to the number of genes analyzed.

        Parameters
        ----------
        resamp_vec: list or tuple of ints, optional
            number of genes to select for downsampling (without replacement).
        Ntries: int, optional
            number of times to resample.
        figsize: tuple of floats, optional
            figure dimensions.
        """
        Nsamp = len(resamp_vec)

        fig1, ax1 = plt.subplots(nrows=Nsamp, ncols=Ntries, figsize=figsize)
        for samp_num in range(Nsamp):
            for i_ in range(Ntries):
                axloc = (samp_num, i_)
                gene_filter = np.random.choice(
                    self.n_genes, resamp_vec[samp_num], replace=False
                )
                subsampled_samp_optimum = self.find_sampling_optimum(gene_filter)
                self.plot_landscape(ax1[axloc], gene_filter=gene_filter, hideticks=True)

                if i_ == 0:
                    ax1[axloc].set_ylabel("n_genes = " + str(resamp_vec[samp_num]))

        fig_string = self.analysis_figure_string + "/subsampling.png"
        plt.savefig(fig_string, dpi=450)
        log.info("Figure stored to {}.".format(fig_string))
        self.find_sampling_optimum()  # reset sampling optimum here

    def resample_opt_mc_viz(
        self, resamp_vec=(5, 10, 20, 40, 60), Ntries=1000, figsize=(16, 4)
    ):
        """Test the sensitivity of technical noise parameter optima under gene downsampling.

        The optima for each downsampled set are visualized on the parameter landscape
        generated from the entire gene set.

        Parameters
        ----------
        resamp_vec: list or tuple of ints, optional
            number of genes to select for downsampling (without replacement).
        Ntries: int, optional
            number of times to resample.
        figsize: tuple of floats, optional
            figure dimensions.
        """
        Nsamp = len(resamp_vec)

        fig1, ax1 = plt.subplots(nrows=1, ncols=Nsamp, figsize=figsize)
        for samp_num in range(Nsamp):
            axloc = samp_num
            subsampled_samp_optimum_array = []
            for i__ in range(Ntries):
                gene_filter = np.random.choice(
                    self.n_genes, resamp_vec[samp_num], replace=False
                )
                subsampled_samp_optimum = self.find_sampling_optimum(gene_filter)
                subsampled_samp_optimum_array.append(subsampled_samp_optimum)
            subsampled_samp_optimum_array = np.asarray(subsampled_samp_optimum_array)

            self.plot_landscape(ax1[axloc], hideticks=True)
            jit = np.random.normal(scale=0.1, size=subsampled_samp_optimum_array.shape)
            subsampled_samp_optimum_array = subsampled_samp_optimum_array + jit
            ax1[axloc].scatter(
                subsampled_samp_optimum_array[:, 0],
                subsampled_samp_optimum_array[:, 1],
                c="r",
                s=3,
                alpha=0.3,
            )
            ax1[axloc].set_title("n_genes = " + str(resamp_vec[samp_num]))

        fig_string = self.analysis_figure_string + "/subsampling_stability.png"
        plt.savefig(fig_string, dpi=450)
        log.info("Figure stored to {}.".format(fig_string))

        self.find_sampling_optimum()  # reset sampling optimum.

    def chisq_best_param_correction(
        self,
        search_data,
        threshold=0.05,
        Ntries=10,
        viz=True,
        szfig=(2, 5),
        figsize=(10, 4),
        overwrite=True,
    ):
        """Test the sensitivity of technical noise parameter optima after gene rejection.

        This method demonstrates the sensitivity of the sampling parameter landscape
        and optimum to the specific genes retained after chi-squared testing.
        It performs fixed-point iteration to illustrate whether the optimum converges.

        This can be used with viz=True to inspect the qualitative behavior of the optimum,
        or with viz=False as a sanity check to make sure the optimum is not strongly skewed
        by a small number of very poorly fit genes.

        The procedure does not typically move the optimum.

        Parameters
        ----------
        search_data: monod.extract_data.SearchData
            SearchData object with the fit data.
        threshold: float, optional
            chi-square rejection criterion; everything below this critical p-value
            is rejected as unlikely to have been generated by the model.
        Ntries: int, optional
            number of steps of chi-squared testing to perform.
        viz: bool, optional
            whether to visualize the results.
        szfig: tuple of ints, optional
            dimensions of the figure subplot grid.
        figsize: tuple of floats, optional
            figure dimensions.
        overwrite: bool, optional
            whether to retain the optimum obtained at the end of the procedure.
        """
        search_data = self._subset_search_data(search_data)

        if viz:
            fig1, ax1 = plt.subplots(nrows=szfig[0], ncols=szfig[1], figsize=figsize)
        log.info(
            "Original optimum: {:.2f}, {:.2f}.".format(
                self.samp_optimum[0], self.samp_optimum[1]
            )
        )
        for i_ in range(Ntries):
            self.chisquare_testing(search_data, threshold=threshold)
            # gene_filter = ~self.rejected_genes
            well_fit_samp_optimum = self.find_sampling_optimum(discard_rejected=True)
            log.info(
                "New optimum: {:.2f}, {:.2f}.".format(
                    self.samp_optimum[0], self.samp_optimum[1]
                )
            )

            if viz:
                axloc = (
                    np.unravel_index(i_, szfig)
                    if (szfig[0] > 1 and szfig[1] > 1)
                    else i_
                )
                self.plot_landscape(ax1[axloc], discard_rejected=True, hideticks=True)
        if viz:
            fig_string = self.analysis_figure_string + "/chisquare_stability.png"
            plt.savefig(fig_string, dpi=450)
            log.info("Figure stored to {}.".format(fig_string))

        if overwrite:
            self.chisquare_testing(search_data, threshold=threshold)
            log.info(
                "Optimum retained at {:.2f}, {:.2f}.".format(
                    self.samp_optimum[0], self.samp_optimum[1]
                )
            )
        else:
            self.find_sampling_optimum()
            self.chisquare_testing(search_data, threshold=threshold)
            log.info(
                "Optimum restored to {:.2f}, {:.2f}.".format(
                    self.samp_optimum[0], self.samp_optimum[1]
                )
            )

    def plot_param_L_dep(
        self,
        gene_filter_=None,
        plot_errorbars=False,
        figsize=None,
        c=2.576,
        axis_search_bounds=True,
        plot_fit=False,
        distinguish_rej=True,
    ):
        """Plot and fit the biological parameter dependence on length at the sampling parameter optimum.

        This dependence is expected to be weak; burst size may show a very slight negative trend.

        Parameters
        ----------
        gene_filter_: None or np.ndarray, optional
            If None, plot all genes.
            If a boolean or integer filter, plot the filtered gene subset.
        plot_errorbars: bool, optional
            whether to use inferred standard error of maximum likelihood estimates to plot error bars.
        figsize: None or tuple of floats, optional
            figure dimensions.
        c: float
            if plotting the errorbars, the number of standard deviations to display.
            this can be used for Gaussian approximations to the MLE confidence intervals.
        axis_search_bounds: bool, optional
            whether to place the x-limits of the plots at the parameter search bounds.
        plot_fit: bool, optional
            whether to plot a linear fit to the data points.
        distinguish_rej: bool, optional
             whether to distinguish the genes in the rejected_genes attribute by color.
        """
        num_params = self.sp.n_phys_pars
        figsize = figsize or (4 * num_params, 4)

        fig1, ax1 = plt.subplots(nrows=1, ncols=num_params, figsize=figsize)

        gene_filter = self.get_bool_filt(gene_filter_, discard_rejected=False)
        gene_filter_rej = np.zeros(self.n_genes, dtype=bool)

        if distinguish_rej:  # default
            filt_rej = self.get_bool_filt(gene_filter_, discard_rejected=True)
            gene_filter_rej = np.logical_and(
                gene_filter, np.logical_not(filt_rej)
            )  # subset for rejected genes
            gene_filter = np.logical_and(gene_filter, filt_rej)
            acc_point_aesth = (
                "accepted_gene_color",
                "accepted_gene_alpha",
                "accepted_gene_ms",
            )
            rej_point_aesth = (
                "rejected_gene_color",
                "rejected_gene_alpha",
                "rejected_gene_ms",
            )
        else:  # don't distinguish
            acc_point_aesth = (
                "generic_gene_color",
                "generic_gene_alpha",
                "generic_gene_ms",
            )
            log.info("Falling back on generic marker properties.")

        for i in range(num_params):
            if plot_errorbars:

                lfun = lambda x, a, b: a * x + b
                if plot_fit:
                    popt, pcov = scipy.optimize.curve_fit(
                        lfun,
                        self.gene_log_lengths[gene_filter],
                        self.phys_optimum[gene_filter, i],
                        sigma=self.sigma[gene_filter, i],
                        absolute_sigma=True,
                    )
                    xl = np.array(
                        [min(self.gene_log_lengths), max(self.gene_log_lengths)]
                    )

                    min_param = (
                        popt[0] - np.sqrt(pcov[0, 0]) * c,
                        popt[1] - np.sqrt(pcov[1, 1]) * c,
                    )
                    max_param = (
                        popt[0] + np.sqrt(pcov[0, 0]) * c,
                        popt[1] + np.sqrt(pcov[1, 1]) * c,
                    )
                    ax1[i].fill_between(
                        xl,
                        lfun(xl, min_param[0], min_param[1]),
                        lfun(xl, max_param[0], max_param[1]),
                        facecolor=aesthetics["length_fit_face_color"],
                        alpha=aesthetics["length_fit_face_alpha"],
                    )
                    ax1[i].plot(
                        xl,
                        lfun(xl, popt[0], popt[1]),
                        c=aesthetics["length_fit_line_color"],
                        linewidth=aesthetics["length_fit_lw"],
                    )
                ax1[i].errorbar(
                    self.gene_log_lengths[gene_filter],
                    self.phys_optimum[gene_filter, i],
                    self.sigma[gene_filter, i] * c,
                    c=aesthetics["errorbar_gene_color"],
                    alpha=aesthetics["errorbar_gene_alpha"],
                    linestyle="None",
                    linewidth=aesthetics["errorbar_lw"],
                )

            ax1[i].scatter(
                self.gene_log_lengths[gene_filter],
                self.phys_optimum[gene_filter, i],
                c=aesthetics[acc_point_aesth[0]],
                alpha=aesthetics[acc_point_aesth[1]],
                s=aesthetics[acc_point_aesth[2]],
            )
            if np.any(gene_filter_rej):
                ax1[i].scatter(
                    self.gene_log_lengths[gene_filter_rej],
                    self.phys_optimum[gene_filter_rej, i],
                    c=aesthetics[rej_point_aesth[0]],
                    alpha=aesthetics[rej_point_aesth[1]],
                    s=aesthetics[rej_point_aesth[2]],
                )

            ax1[i].set_xlabel(r"$\log_{10}$ L")
            ax1[i].set_ylabel(self.model.get_log_name_str()[i])
            if axis_search_bounds:
                ax1[i].set_ylim([self.sp.phys_lb[i], self.sp.phys_ub[i]])
        fig1.tight_layout()
        fig_string = self.analysis_figure_string + "/length_dependence.png"
        plt.savefig(fig_string, dpi=450)
        log.info("Figure stored to {}.".format(fig_string))

    def plot_gene_distributions(
        self,
        search_data,
        sz=(5, 5),
        figsize=(10, 10),
        marg="joint",
        logscale=None,
        title=True,
        genes_to_plot=None,
        savefig=True,
    ):
        """Plot the gene count distributions and their fits at the sampling parameter optimum.

        Parameters
        ----------
        search_data: monod.extract_data.SearchData
            SearchData object with the fit data.
        szfig: tuple of ints, optional
            dimensions of the figure subplot grid.
        figsize: tuple of floats, optional
            figure dimensions.
        marg: str, optional
            if 'nascent': plot unspliced RNA marginal.
            if 'mature': plot spliced RNA marginal.
            if 'joint': plot the bivariate distribution.
        logscale: None or bool, optional
            whether to plot probabilities or log-probabilities.
            by default, True for 'joint', False for marginals.
        title: bool, optional
            whether to report the gene name in each subplot title.
        genes_to_plot: bool or int np.ndarray or None, optional
            if array, which genes to plot.
            if None, plot by internal order.
        savefig: bool, optional
            whether to save the figure to disk.
        """

        if logscale is None:
            if marg == "joint":
                logscale = True
            else:
                logscale = False

        search_data = self._subset_search_data(search_data)

        (nrows, ncols) = sz
        fig1, ax1 = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        nax = np.prod(sz)
        if genes_to_plot is None:
            genes_to_plot = np.arange(nax)

        genes_to_plot = np.asarray(genes_to_plot)
        if genes_to_plot.dtype == bool:
            gtp_temp = np.arange(self.n_genes)
            genes_to_plot = gtp_temp[genes_to_plot]

        number_of_genes_to_plot = len(genes_to_plot)
        if number_of_genes_to_plot > self.n_genes:
            number_of_genes_to_plot = self.n_genes
            genes_to_plot = genes_to_plot[: self.n_genes]
        if number_of_genes_to_plot > nax:  # This should no longer break...
            number_of_genes_to_plot = nax
            genes_to_plot = genes_to_plot[:nax]

        j_ = 0
        for i_ in genes_to_plot:
            lm = np.copy(search_data.M[:, i_])
            if marg == "mature":
                lm[0] = 1
            if marg == "nascent":
                lm[1] = 1
            axloc = np.unravel_index(j_, sz) if (sz[0] > 1 and sz[1] > 1) else j_

            samp = self.regressor_optimum[i_]
            Pa = np.squeeze(self.model.eval_model_pss(self.phys_optimum[i_], lm, samp))

            if marg == "joint":
                if logscale:
                    Pa[Pa < 1e-10] = 1e-10
                    Pa = np.log10(Pa)

                ax1[axloc].imshow(Pa.T, aspect="auto", cmap="summer")
                ax1[axloc].invert_yaxis()

                jitter_magn = 0.1
                jitter = np.random.randn(2, self.n_cells) * jitter_magn
                ax1[axloc].scatter(
                    *search_data.layers[:2, :,i_] + jitter, c="k", s=1, alpha=0.1
                )

                ax1[axloc].set_xlim([-0.5, search_data.M[0, i_] - 1.5])
                ax1[axloc].set_ylim([-0.5, search_data.M[1, i_] - 1.5])
            else:
                plot_hist_and_fit(ax1[axloc], search_data, i_, Pa, marg)
                if logscale:
                    ax1[axloc].set_yscale("log")
            if title:
                titlestr = self.gene_names[i_]
                if hasattr(self, "rejected_genes") and self.rejected_genes[i_]:
                    titlestr += " (rej.)"
                ax1[axloc].set_title(titlestr, fontdict={"fontsize": 9})
            ax1[axloc].set_xticks([])
            ax1[axloc].set_yticks([])
            j_ += 1
        fig1.tight_layout(pad=0.02)

        if savefig:
            fig_string = (
                self.analysis_figure_string + "/gene_distributions_{}.png".format(marg)
            )
            plt.savefig(fig_string, dpi=450)
            log.info("Figure stored to {}.".format(fig_string))

    # the next two functions are useful for model selection, but are not currently in use.
    def get_logL(self, search_data, EPS=1e-20, offs=0):
        """
        This method calculates the log-likelihood for all genes at the sampling parameter optimum.

        Parameters
        ----------
        search_data: a SearchData instance.
        EPS: probability rounding parameter -- anything below this is rounded to EPS.

        Output:
        logL: a vector of size n_genes containing model log-likelihoods.
        """
        search_data = self._subset_search_data(search_data)
        hist_type = get_hist_type(search_data)
        logL = np.zeros(self.n_genes)
        for gene_index in range(self.n_genes):
            logL[gene_index] = self.model.eval_model_logL(
                p=self.phys_optimum[gene_index],
                limits=search_data.M[:, gene_index] + offs,
                samp=self.regressor_optimum[gene_index],
                data=search_data.hist[gene_index],
                n_cells = search_data.n_cells,
                hist_type=hist_type,
                EPS=EPS,
            )
            # Pss = self.model.eval_model_pss(self.phys_optimum[gene_index],lm,samp)
            # if np.any(Pss<EPS):
            #     Pss[Pss<EPS] = EPS
            # expected_log_lik = np.log(Pss)
            # eval_model_kld(self, p, limits, samp, data, hist_type="unique", EPS=EPS)
            # logL[gene_index] = expected_log_lik[search_data.U[gene_index].astype(int),search_data.S[gene_index].astype(int)].sum()
        return logL

    def get_noise_decomp(self):
        """
        #obsolete.
        This method reports the fractions of normalized variance attributable to intrinsic,
        extrinsic, and technical noise under the instantiated model, using the eval_model_noise
        method of the CMEModel class.

        Output:
        f: array with size n_genes x 3 x 2.
            dim 0: gene
            dim 1: variance fraction (intrinsic, extrinsic, technical)
            dim 2: species (unspliced, spliced)
        The null technical noise model has dim 0 of size 2, as it has no technical noise component.
        """
        f = []
        for gene_index in range(self.n_genes):
            f.append(
                [
                    self.model.eval_model_noise(
                        self.phys_optimum[gene_index],
                        samp=self.regressor_optimum[gene_index],
                    )
                ]
            )
        return np.asarray(f).squeeze()


def parallelize(
    function,
    iterable,
    num_cores,
    num_entries,
    completion_message="Parallelization complete.",
    termination_message="Parallelization manually terminated.",
    error_message="Parallelization terminated due to error.",
    use_tqdm=True,
):
    """Helper function to safely parallelize computations.

    Inputs a single-parameter function and an iterable, requests a number of cores, \
    and gracefully shuts down if needed.

    Parameters
    ----------
    function: function
        a one-parameter function that can be mapped using entries of the iterable.
    iterable: iterable
        an iterable to be passed into the function.        
    num_cores: int
        number of cores to use for parallelization.
    num_entries: int
        length of the iterable, used for tqdm.
    completion_message: str, optional
        string to print to log upon completion.
    termination_message: str, optional
        string to print to log if the process is manually terminated.
    error_message: str, optional
        string to print to log if the process fails due to a ValueError.
    use_tqdm: bool, optional
        whether to visualize progress using tqdm.

    Returns
    -------
    x: iterable
        result of applying function to iterable.

    """
    try:
        pool = multiprocessing.Pool(processes=num_cores)
        if use_tqdm:
            x = list(tqdm(pool.imap(function, iterable), total=num_entries))  # hacky
        else:
            x = pool.map(function, iterable)
        pool.close()
        pool.join()
        log.info(completion_message)
    except KeyboardInterrupt:
        log.warning(termination_message)
        pool.terminate()
        pool.join()
    except ValueError:
        log.warning(error_message)
        pool.terminate()
        pool.join()
    return x


def plot_hist_and_fit(
    ax1,
    sd,
    i_,
    Pa,
    marg="nascent",
    facecolor=aesthetics["hist_face_color"],
    fitcolor=aesthetics["hist_fit_color"],
    facealpha=aesthetics["hist_face_alpha"],
    linestyle=aesthetics["linestyle"],
):
    """Plots marginal gene count distributions and their fits at the sampling parameter optimum.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        axes to plot into.
    sd: monod.extract_data.SearchData
        SearchData object with the fit data.
    i_: int
        gene index to plot.
    Pa: float np.ndarray
        univariate probability mass function, typically computed through CMEModel.
    marg: str, optional
        which marginal to plot, typically 'nascent' or 'mature'.
    facecolor: str or tuple, optional
        histogram face color in a matplotlib-compatible format.
    fitcolor: str or tuple, optional
        model fit line color in a matplotlib-compatible format.
    facealpha: float, optional
        histogram face alpha.
    linestyle: str, optional
        model fit line style in a matplotlib-compatible format.
    """

    if marg == "nascent":
        lind = 0
    elif marg == "mature":
        lind = 1
    elif marg == "ambiguous":
        lind = 2
    ax1.hist(
        sd.layers[lind, i_],
        bins=np.arange(sd.M[lind, i_]) - 0.5,
        density=True,
        color=facecolor,
        alpha=facealpha,
    )
    ax1.plot(np.arange(sd.M[lind, i_]), Pa, color=fitcolor, linestyle=linestyle)
    ax1.set_xlim([-0.5, sd.layers[lind, i_].max() + 2.5])


# Use class to make inference faster.
class SearchData:
    """Container for data for for inference, visualization, and testing.

    Attributes
    ----------
    attr_names: tuple of str

    layers: int np.ndarray
        raw data from the layers of interest, size n_species x n_genes x n_cells.
    M: int np.ndarray
        grid size for PMF evaluation, size n_species x n_genes.
    hist: tuple or np.ndarray
        histogram of raw data, used to evaluate divergences.
        if tuple, generated by np.unique.
        if np.ndarray, generated by np.histogramdd.
    moments: list of dict
        length-n_genes list containing moments for each gene.
        moments include 'mod2_mean', 'mod1_mean', 'mod2_var', 'mod1_var', and are used to define MoM estimates.
        Also covariances in form 'mod1_mod2_covar'
    gene_log_lengths: float np.ndarray
        log lengths of analyzed genes.
    n_genes: int
        number of genes to analyze.
    gene_names: str np.ndarray
        list of genes to analyze.
    n_cells: int
        number of cells in the dataset.
    hist_type: str
        metadata defining the type of histogram.
    """

    def __init__(self, attr_names, *input_data):
        """Creates a SearchData object from raw data.

        Parameters
        ----------
        attr_names: tuple
            list of attributes to store, provided in extract_data.
        *input_data
            attributes to store, as enumerated in the class definition.
        """
        for j in range(len(input_data)):
            setattr(self, attr_names[j], input_data[j])
