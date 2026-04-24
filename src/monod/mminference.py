import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import optimize, stats
from scipy.special import logsumexp, softmax
from extract_data import make_dir, log, _build_sampling_grid
from cme_toolbox import CMEModel  # may be unnecessary
from inference import (
    GradientInference as _BaseGradientInference,
    InferenceParameters as _BaseInferenceParameters,
    GridPointResults as _BaseGridPointResults,
    SearchResults as _BaseSearchResults,
)
import multiprocessing
import os
import itertools
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity

# lbfgsb has a deprecation warning for .tostring(), probably in FORTRAN interface
import warnings
from plot_aesthetics import aesthetics

try:
    import monod_core as _mc
    _HAS_RUST = True
except ImportError:
    _mc = None
    _HAS_RUST = False

_RUST_MODELS_2D = {"Constitutive", "Bursty", "CIR", "Extrinsic", "Delay", "DelayedSplicing"}

# True when monod_core was compiled with --features kmeans (linfa KMeans available).
_HAS_KMEANS = _HAS_RUST and hasattr(_mc, "initialize_q_kmeans")

from tqdm import tqdm

# from tqdm.contrib.concurrent import process_map  # or thread_map


# warnings.filterwarnings("ignore", category=DeprecationWarning) #let's do more gargeted stuff...


class MEKMeansParameters(_BaseInferenceParameters):
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
            "use_rust_lbfgsb": _HAS_RUST,
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
            samp_lb = [0, 0]
            samp_ub = [0, 0]
            gridsize = [1, 1]

        self.samp_lb = np.array(samp_lb)
        self.samp_ub = np.array(samp_ub)
        self.gridsize = gridsize

        self.grid_values_sampl, self.sampl_vals, self.n_grid_points = _build_sampling_grid(
            self.samp_lb, self.samp_ub, self.gridsize
        )
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



    def run_mek_means(self, search_data, num_cores=1):
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
        gp_results = {}
        for x in zip(
            range(self.n_grid_points),
            [[search_data, self.model]] * self.n_grid_points,
            [self.k] * self.n_grid_points,
            [self.epochs] * self.n_grid_points,
            [num_cores] * self.n_grid_points,
        ):
            r = self.par_fun(x)
            if r is not None:
                gp_results[r.point_index] = r
        log.info("Non-parallelized grid scan complete.")

        warnings.resetwarnings()
        full_results = []
        for i in range(self.k):
            results = SearchResults(self, search_data, i)
            results.aggregate_grid_points(gp_results)
            if results.save:
                results.store_on_disk()
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
        return grad_inference.fit_all_genes(model, search_data, num_cores)


class GradientInference(_BaseGradientInference):
    """Runs the grid point-specific inference procedures.

    Extends inference.GradientInference with MEK-Means EM methods.

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
        MEKMeansParameters constructor.
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
        global_parameters: MEKMeansParameters
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
            MEKMeansParameters constructor.
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
        # mminference.InferenceParameters.use_lengths is a bool; the base class
        # expects None or a string ("unspliced"/"spliced"/"both"). Normalise here.
        if isinstance(global_parameters.use_lengths, bool):
            global_parameters.use_lengths = "unspliced" if global_parameters.use_lengths else None
        # Delegate regressor setup, param_MoM, restart bounds, and warm_start
        # to the base class, then add MEK-Means-specific attributes.
        super().__init__(global_parameters, model, search_data, point_index)
        self.k = k
        self.epochs = epochs
        self.weights = np.ones(self.k) / self.k
        self.theta = {}

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

        # Rust fast-path: log1p-normalised KMeans + biased Q initialisation.
        if _HAS_KMEANS and isinstance(search_data, _mc.SearchData):
            layers_3d = np.ascontiguousarray(search_data.layers, dtype=np.int64)
            Q, _ = _mc.initialize_q_kmeans(layers_3d, self.k, seed=0)
            Q = Q * self.weights[None, :]
            Q = Q / Q.sum(axis=-1, keepdims=True)
            return Q

        # Python fallback: U+S KMeans via sklearn.
        S_total = search_data.layers[0,:,:] + search_data.layers[1,:,:]
        tots = np.sum(S_total,axis=1)
        divids = (1e4/tots)[:,None]
        S_total = S_total*divids
        S_total = np.log1p(S_total)
        S_total[np.isnan(S_total)] = 0

        kmeans = KMeans(n_clusters=self.k, random_state=0).fit(S_total)
        labs = kmeans.labels_

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

        max_ks = np.argmax(Q, axis=1).astype(int)

        # Rust fast-path: partition + histogram rebuild entirely in Rust (rayon).
        if _HAS_RUST and isinstance(search_data, _mc.SearchData):
            pad = padding[0] if padding is not None else 10
            subsets = _mc.partition_searchdata_2d(
                search_data, max_ks.tolist(), self.k, int(pad)
            )
            return {k: sd for k, sd in enumerate(subsets) if sd is not None}

        # Python fallback.
        datas = []
        inds = []

        layer_names = search_data.layer_names
        gene_names = search_data.gene_names
        n_genes = len(gene_names)
        n_layers = len(layer_names)

        if padding is None:
            padding = [10] * n_layers
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

        n_cells = search_data.n_cells
        n_genes = search_data.n_genes
        ks_present = sorted(self.theta.keys())

        # Rust CPU fast-path: all PSS + accumulate + softmax in one Rust call (rayon)
        if (
            _HAS_RUST
            and ks_present
            and model.bio_model in _RUST_MODELS_2D
            and model.seq_model in ("None", "Poisson")
            and model.amb_model == "None"
            and model.quad_method == "fixed_quad"
        ):
            params_per_k = [self.theta[k][0].tolist() for k in ks_present]
            limits_list = [[int(v) for v in search_data.M[:, g]] for g in range(n_genes)]
            u_obs = [search_data.layers[0][:, g].astype(int).tolist() for g in range(n_genes)]
            s_obs = [search_data.layers[1][:, g].astype(int).tolist() for g in range(n_genes)]
            samp_list = None
            if model.seq_model == "Poisson":
                samp_list = [
                    self.regressor[g].tolist() if self.regressor[g] is not None else None
                    for g in range(n_genes)
                ]
            weights_subset = self.weights[ks_present].tolist()

            Q_sub, lower_bound, q_func = _mc.e_step_2d(
                bio_model=model.bio_model,
                params_per_k=params_per_k,
                limits_list=limits_list,
                u_obs=u_obs,
                s_obs=s_obs,
                weights=weights_subset,
                fixed_quad_t=float(model.fixed_quad_T),
                quad_order=int(model.quad_order),
                samp_list=samp_list,
                eps=EPS,
            )
            Q = np.zeros((n_cells, self.k))
            for i, k in enumerate(ks_present):
                Q[:, k] = np.array(Q_sub)[:, i]
            return Q, lower_bound, q_func

        # Python fallback
        logL = np.zeros((n_cells, self.k))
        for k in ks_present:
            params, klds, obj_fun, d_time = self.theta[k]
            logL_k = np.zeros(n_cells)
            for gene_index in range(n_genes):
                S = search_data.layers[1][:,gene_index].astype(int)
                U = search_data.layers[0][:,gene_index].astype(int)
                x = np.array([U,S])
                proposal = model.eval_model_pss(params[gene_index], search_data.M[:, gene_index], self.regressor[gene_index])
                proposal[proposal < EPS] = EPS
                proposal = proposal[tuple(x)]
                logL_k += np.log(proposal)
            logL[:,k] = logL_k

        logL += np.log(self.weights)[None,:]
        Q = softmax(logL, axis=1)
        lower_bound = np.mean(logsumexp(a=logL, axis=1))
        q_func = np.sum(Q*logL)
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
                all_qs += [q_func]

            final_k_dict = self._part_search_data(search_data,Q)
            
            return Q, lower_bound, all_qs, all_klds

    # optimize_gene and iterate_over_genes are inherited from inference.GradientInference.

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


        return GridPointResults(
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
        unique, unique_counts = np.unique(
            np.vstack([x[:, gene_index] for x in layers]).T, axis=0, return_counts=True
        )
        hist.append((unique.astype(int), unique_counts / n_cells))

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



########################
## Helper classes
########################
class GridPointResults(_BaseGridPointResults):
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
        MEKMeansParameters constructor.
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
        super().__init__(param_estimates, klds, obj_func, d_time, regressor, grid_point, point_index, inference_string)
        self.weights = weights
        self.aic = aic
        self.assigns = assigns
        self.all_qs = all_qs
        self.all_klds = all_klds


class SearchResults(_BaseSearchResults):
    """Stores and analyzes the results of a single inference run.

    The first thirteen attributes relate to data loaded from the search.
    The others relate to data processing after the search is completed.

    Attributes
    ----------
    sp: MEKMeansParameters
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
        MEKMeansParameters constructor.
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
        assign: int
            Which mixture component k this object tracks.
        """
        super().__init__(inference_parameters, search_data)
        self.assigns = assign
        self.save = False
        self.weights = []
        self.aic = []
        self.all_qs = []
        self.all_klds = []
        self.filt = []

    def _append_from_object(self, gpr):
        """Update result attributes from an in-memory GridPointResults object.

        Parameters
        ----------
        gpr: GridPointResults
        """
        if self.assigns in np.unique(gpr.assigns):
            self.save = True
            self.param_estimates += [gpr.param_estimates[:, :, self.assigns]]
            self.klds += [gpr.klds[:, self.assigns]]
            self.obj_func += [gpr.obj_func[self.assigns]]
            self.d_time += [gpr.d_time[self.assigns]]
            self.regressor += [gpr.regressor]
            self.aic += [gpr.aic]
            self.weights += [gpr.weights[self.assigns]]
            self.all_qs += [gpr.all_qs]
            self.all_klds += [[i[:, self.assigns] for i in gpr.all_klds]]
            self.filt = gpr.assigns == self.assigns
            self.n_cells = np.sum(self.filt)

    def clean_up(self, remove_files=False):
        """Finalize the SearchResults object.

        Optionally removes .gp checkpoint files, then converts list attributes
        to np.ndarrays and creates the analysis figure directory.

        Parameters
        ----------
        remove_files: bool, optional
            If True, delete .gp files from disk. Default False.
        """
        if remove_files:
            for point_index in range(self.sp.n_grid_points):
                gp_path = self.inference_string + "/grid_point_" + str(point_index) + ".gp"
                if os.path.exists(gp_path):
                    os.remove(gp_path)
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

    @property
    def _result_filename(self):
        return self.inference_string + "/grid_scan_results_" + str(self.assigns) + ".res"

    @property
    def _upd_result_filename(self):
        return self.inference_string + "/grid_scan_results_" + str(self.assigns) + "_upd.res"

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

    # Inherited from _BaseSearchResults (identical or superset behavior):
    # aggregate_grid_points, find_sampling_optimum, set_sampling_optimum, plot_landscape,
    # get_bool_filt, plot_param_marg, plot_KL, chisquare_testing, compute_sigma,
    # resample_opt_viz, resample_opt_mc_viz, plot_param_L_dep, get_noise_decomp.

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
        Hfun = numdifftools.Hessian(
            lambda x: self.model.eval_model_kld(
                p=x,
                limits=search_data.M[:, gene_index],
                samp=self.regressor_optimum[gene_index],
                data=search_data.hist[gene_index],
            )
        )
        hess = Hfun(self.phys_optimum[gene_index])
        return hess

    # compute_sigma, resample_opt_viz, resample_opt_mc_viz are inherited from _BaseSearchResults.
    # compute_sigma calls self.par_fun_hess which is overridden above to apply _subset_search_data.

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

    # plot_param_L_dep is inherited from _BaseSearchResults.

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
        logL = np.zeros(self.n_genes)
        for gene_index in range(self.n_genes):
            logL[gene_index] = self.model.eval_model_logL(
                p=self.phys_optimum[gene_index],
                limits=search_data.M[:, gene_index] + offs,
                samp=self.regressor_optimum[gene_index],
                data=search_data.hist[gene_index],
                n_cells=search_data.n_cells,
                EPS=EPS,
            )
            # Pss = self.model.eval_model_pss(self.phys_optimum[gene_index],lm,samp)
            # if np.any(Pss<EPS):
            #     Pss[Pss<EPS] = EPS
            # expected_log_lik = np.log(Pss)
            # eval_model_kld(self, p, limits, samp, data, hist_type="unique", EPS=EPS)
            # logL[gene_index] = expected_log_lik[search_data.U[gene_index].astype(int),search_data.S[gene_index].astype(int)].sum()
        return logL

    # get_noise_decomp is inherited from _BaseSearchResults.


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
