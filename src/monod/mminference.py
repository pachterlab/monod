import time
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import optimize, stats
from scipy.special import logsumexp, softmax
from .extract_data import make_dir, log, _build_sampling_grid
from .cme_toolbox import CMEModel, _MODELS_2D
from .inference import (
    SearchData,
    parallelize,
    _normalize_use_lengths,
    _validate_bounds,
    _write_h5,
    search_result_list_to_uns,
    plot_hist_and_fit,
    GradientInference as _BaseGradientInference,
    InferenceParameters as _BaseInferenceParameters,
    GridPointResults as _BaseGridPointResults,
    SearchResults as _BaseSearchResults,
    DEFAULT_GRADIENT_PARAMS,
)
import os
import itertools
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity

# lbfgsb has a deprecation warning for .tostring(), probably in FORTRAN interface
import warnings
from .plot_aesthetics import aesthetics

from ._core import _mc          # None when the compiled extension is not installed
_HAS_RUST = _mc is not None


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
    use_lengths: None | "unspliced" | "spliced" | "both"
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
        use_lengths="auto",
        gradient_params=None,
        run_meta="",
        phys_lb=None,
        phys_ub=None,
        samp_lb=None,
        samp_ub=None,
        gridsize=None,
        poisson_average_log_length=5,
        k=10,
        epochs=100,
        em_tol=1e-4,
        em_patience=2,
        random_state=0,
        init_modality="sum",
        save=False,
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
        use_lengths: "auto" | None | "unspliced" | "spliced" | "both", optional
            "auto" (default) resolves to "unspliced" for Poisson and None otherwise.
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
        k: int, optional
            number of mixture components.
        epochs: int, optional
            maximum number of EM epochs.
        em_tol: float, optional
            relative change in the per-cell log-likelihood lower bound below which an
            epoch counts as stalled; EM stops early after ``em_patience`` consecutive
            stalled epochs. Set to 0 to always run all ``epochs``.
        em_patience: int, optional
            number of consecutive stalled epochs required to stop early.
        random_state: int or None, optional
            seed for the posterior initialisation (k-means and the uniform
            perturbation, on both the Rust and Python paths). The same seed and
            data give the same clustering; None draws a fresh seed each run.
        init_modality: str, optional
            count matrix the initial k-means runs on, after per-cell scaling to
            1e4 and log1p. "sum" (default) uses the sum of the first two
            modalities (U + S); a modality name such as "spliced" uses that layer
            alone, which is the setting described in the meK-Means paper.
        save: bool, optional
            create the run directory and write parameters.h5 at construction.
            Default False: nothing is written to disk unless you ask for it
            (see ``run_mek_means(save=True)``).

        Notes
        -----
        Early stopping (``em_tol`` > 0) is an addition to the published
        algorithm, which runs a fixed number of epochs and reports the Q
        function and KLD per epoch. Pass ``em_tol=0`` to reproduce that
        behaviour exactly.
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
        
        # meK-Means defaults: one start per gene and the Rust optimizer when available.
        self.gradient_params = {
            **DEFAULT_GRADIENT_PARAMS, "num_restarts": 1, "use_rust_lbfgsb": _HAS_RUST,
            **(gradient_params or {}),
        }
        self.use_lengths, self.use_lengths_explicit = _normalize_use_lengths(use_lengths, model)
        if model.seq_model == "None":
            samp_lb, samp_ub, gridsize = [0, 0], [0, 0], [1, 1]
        _validate_bounds(model, phys_lb, phys_ub, samp_lb, samp_ub, gridsize)
        self.phys_lb = np.array(phys_lb)
        self.phys_ub = np.array(phys_ub)
        self.grad_bnd = scipy.optimize.Bounds(phys_lb, phys_ub)

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
        self.em_tol = float(em_tol)
        self.em_patience = int(em_patience)
        self.random_state = random_state
        if init_modality != "sum" and init_modality not in model.model_modalities:
            raise ValueError(
                f"init_modality must be 'sum' or one of {list(model.model_modalities)}; got {init_modality!r}"
            )
        self.init_modality = init_modality

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
        
        inference_string = os.path.abspath(inference_string)
        self.inference_string = inference_string
        if save:
            make_dir(inference_string)
            self.store_inference_parameters(inference_string + "/parameters.h5")

    # ------------------------------------------------------------------
    # Structured (pickle-free) serialisation
    # ------------------------------------------------------------------
    _SERIAL_KIND = "MEKMeansParameters"

    def to_dict(self):
        d = super().to_dict()
        d.update({
            "k": int(self.k),
            "epochs": int(self.epochs),
            "em_tol": float(self.em_tol),
            "em_patience": int(self.em_patience),
            "init_modality": str(self.init_modality),
        })
        if self.random_state is not None:
            d["random_state"] = int(self.random_state)
        return d

    @classmethod
    def _from_dict_impl(cls, d):
        kw = cls._ctor_kwargs_from_dict(d)
        return cls(
            str(d["dataset_string"]),
            k=int(d["k"]), epochs=int(d["epochs"]),
            em_tol=float(d.get("em_tol", 1e-4)), em_patience=int(d.get("em_patience", 2)),
            random_state=(int(d["random_state"]) if "random_state" in d else None),
            init_modality=str(d.get("init_modality", "sum")),
            save=False, **kw,
        )



    def run_mek_means(self, search_data, num_cores=1, save=False):
        """Fits the search data for all genes over all grid points.

        Parameters
        ----------
        num_cores: int
            number of cores to use for parallelization over grid points.
        search_data: monod.extract_data.SearchData
            SearchData object with the data to fit.

        save: bool, optional
            write all per-cluster results to a single
            ``<inference_string>/grid_scan_results.h5`` (default False).

        Returns
        -------
        full_results: list of SearchResults
            one entry per cluster that received cells, ordered by cluster id.

        """

        t1 = time.time()
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
        full_results = []
        for i in range(self.k):
            results = SearchResults(self, search_data, i)
            results.aggregate_grid_points(gp_results)
            if results.save:
                full_results += [results]

        if save:
            make_dir(self.inference_string)
            path = self.inference_string + "/grid_scan_results.h5"
            _write_h5(path, search_result_list_to_uns(full_results))
            log.info("meK-Means results (%d clusters) stored to %s.", len(full_results), path)

        t2 = time.time()
        log.info("Runtime: {:.1f} seconds.".format(t2 - t1))
        return full_results

    def fit_all_grid_points(self, search_data, num_cores=1, checkpoint=False, save=False):
        """Run meK-Means EM and return a list of SearchResults (one per cluster).

        Overrides the base class to delegate to run_mek_means, which handles
        the MEK-Means-specific par_fun signature and returns a per-cluster list.
        """
        return self.run_mek_means(search_data, num_cores=num_cores, save=save)

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


def _partition_search_data(search_data, labels, k, padding=None):
    """Split ``search_data`` by the integer cell ``labels`` (0..k-1) into one
    SearchData per label with its own grid limits and histograms; labels with
    no cells are left out.  Rust when the data is a Rust SearchData, else the
    Python fallback."""
    max_ks = np.asarray(labels).astype(int)

    # Rust fast-path: partition + histogram rebuild entirely in Rust (rayon).
    if _HAS_RUST and isinstance(search_data, _mc.SearchData):
        pad = padding[0] if padding is not None else 10
        subsets = _mc.partition_searchdata_2d(
            search_data, max_ks.tolist(), k, int(pad)
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
        # Delegate regressor setup, param_MoM, restart bounds, and warm_start
        # to the base class, then add MEK-Means-specific attributes.
        super().__init__(global_parameters, model, search_data, point_index)
        self.k = k
        self.epochs = epochs
        self.em_tol = float(getattr(global_parameters, "em_tol", 1e-4))
        self.em_patience = int(getattr(global_parameters, "em_patience", 2))
        self.random_state = getattr(global_parameters, "random_state", 0)
        self.init_modality = getattr(global_parameters, "init_modality", "sum")
        self.init_modalities = list(model.model_modalities)
        self.n_epochs_run = 0
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

        # One seed drives k-means and the uniform perturbation on both paths,
        # so a rerun with the same data and random_state reproduces the
        # initial partition.  (The 0.9-plus-uniform scheme itself is a
        # deliberate perturbation of the k-means labels.)
        seed = getattr(self, "random_state", 0)
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])
        seed = int(seed) & 0xFFFFFFFF
        self._init_seed_used = seed

        # Which counts k-means sees: U + S (default) or a single modality
        # ("spliced" reproduces the paper's description).  The Rust initializer
        # always sums its first two layers, so a single modality is passed as
        # [layer, zeros]; after per-cell scaling to 1e4 the result is identical.
        init_modality = getattr(self, "init_modality", "sum")
        layers_arr = np.asarray(search_data.layers)
        if init_modality == "sum":
            init_layers = layers_arr[:2]
        else:
            modalities = getattr(self, "init_modalities", None)
            if not modalities or init_modality not in modalities:
                raise ValueError(f"init_modality={init_modality!r} is not one of the modalities {modalities}")
            lind = modalities.index(init_modality)
            init_layers = np.stack([layers_arr[lind], np.zeros_like(layers_arr[lind])])

        # Rust fast-path: log1p-normalised KMeans + biased Q initialisation.
        if _HAS_KMEANS and isinstance(search_data, _mc.SearchData):
            layers_3d = np.ascontiguousarray(init_layers, dtype=np.int64)
            Q, _ = _mc.initialize_q_kmeans(layers_3d, self.k, seed=seed)
            Q = Q * self.weights[None, :]
            Q = Q / Q.sum(axis=-1, keepdims=True)
            return Q

        # Python fallback: KMeans via sklearn on the chosen counts.
        S_total = init_layers[0].astype(float) + init_layers[1].astype(float)
        tots = np.sum(S_total,axis=1)
        divids = (1e4/tots)[:,None]
        S_total = S_total*divids
        S_total = np.log1p(S_total)
        S_total[np.isnan(S_total)] = 0

        kmeans = KMeans(n_clusters=self.k, random_state=seed).fit(S_total)
        labs = kmeans.labels_

        rng = np.random.default_rng(seed)
        Q = rng.uniform(0, 1, size=(n, self.k))
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

        return _partition_search_data(search_data, np.argmax(Q, axis=1), self.k, padding)
    
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

        # A cluster that received no cells this epoch has no M-step update:
        # drop its stale parameters so the E-step treats it as unfitted
        # (zero posterior) instead of scoring cells against old theta.
        for k_stale in [k for k in self.theta if k not in k_dict]:
            del self.theta[k_stale]

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
        if not ks_present:
            raise ValueError("E-step called before any cluster has fitted parameters.")
        missing = [k for k in range(self.k) if k not in self.theta]
        if missing:
            log.warning(
                "E-step: clusters %s have no fitted parameters (no cells were assigned to them); "
                "their posterior is set to 0 for this epoch.", missing,
            )

        # Rust CPU fast-path: all PSS + accumulate + softmax in one Rust call (rayon)
        if (
            _HAS_RUST
            and ks_present
            and model.bio_model in _MODELS_2D
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

        # Python fallback.  Columns of clusters without a fitted theta stay at
        # -inf (zero posterior), matching the Rust E-step; a 0 there would
        # dominate the real (large negative) log-likelihoods and pull every
        # cell into the empty cluster.
        logL = np.full((n_cells, self.k), -np.inf)
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
        q_func = float(np.sum(np.where(Q > 0, Q * logL, 0.0)))   # 0 * -inf -> 0
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
            one Q(theta|theta_t) value per epoch actually run
        all_klds: list
            one n_genes x k KLD matrix per epoch actually run (NaN for clusters
            that had no cells in that epoch)

        Notes
        -----
        EM stops early when the lower bound changes by less than ``em_tol``
        (relative) for ``em_patience`` consecutive epochs.  After the loop a
        final E-step is run so that the returned Q and lower bound are
        evaluated under the final parameters.  ``self.n_epochs_run`` records
        the number of M-steps performed inside the loop.

        """

        #E-step, partition, m_step
        all_qs = []
        all_klds = []

        if self.epochs < 1:
            raise ValueError("No. of epochs must be an int > 0")

        # Tolerate objects built without __init__ (e.g. notebooks that assemble a
        # GradientInference by hand): fall back to the constructor defaults.
        em_tol = float(getattr(self, "em_tol", 1e-4))
        em_patience = int(getattr(self, "em_patience", 2))
        prev_lb = None
        stalled = 0
        n_run = 0
        for i in range(self.epochs):
            log.info("EM Epoch "+str(i+1)+'/'+str(self.epochs)+': ')

            Q, lower_bound, q_func = self._e_step(model,search_data)
            k_dict = self._part_search_data(search_data,Q)

            self._m_step(model,k_dict,Q,num_cores=num_cores)
            kl = np.full((search_data.n_genes,self.k), np.nan)
            for k in range(self.k):
                if k in self.theta:
                    params, klds, obj_fun, d_time = self.theta[k]
                    kl[:,k] = klds

            all_klds += [kl]
            all_qs += [q_func]
            n_run = i + 1

            # Convergence: stop once the lower bound has stopped moving for
            # em_patience consecutive epochs (em_tol=0 disables early stopping).
            if prev_lb is not None and em_tol > 0 and np.isfinite(lower_bound) and np.isfinite(prev_lb):
                if abs(lower_bound - prev_lb) <= em_tol * max(1.0, abs(prev_lb)):
                    stalled += 1
                else:
                    stalled = 0
                if stalled >= em_patience:
                    log.info(
                        "EM stopped after %d/%d epochs: lower bound changed by less than em_tol=%g "
                        "for %d consecutive epochs.", n_run, self.epochs, em_tol, em_patience,
                    )
                    break
            prev_lb = lower_bound

        # One more E-step so the reported posteriors, assignments and lower
        # bound correspond to the final theta rather than the one before it.
        Q, lower_bound, q_func = self._e_step(model, search_data)
        self.n_epochs_run = n_run
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
        aic = mek_means_aic(Q, lower_bound, self.n_phys_pars, search_data.n_genes, search_data.n_cells)


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

def mek_means_aic(Q, lower_bound, n_phys_pars, n_genes, n_cells):
    """Per-cell AIC of a fitted meK-Means mixture, as defined in the paper.

        AIC = -log p(X, Z; theta) / N + K_param / N,
        K_param = n_phys_pars * G * K_final + (K_final - 1),

    where Z is the hard (argmax) assignment and K_final the number of clusters
    that received at least one cell.  Lower is better.

    The joint log-likelihood at the hard assignment is recovered from the
    E-step outputs without a second pass: for each cell the log-likelihood of
    its assigned cluster equals the log-marginal (``logsumexp`` over clusters)
    plus the log of its maximum posterior, and ``lower_bound`` is the mean
    log-marginal over cells.

    Parameters
    ----------
    Q: np.ndarray, (n_cells, k)
        posterior p(z=k|x) from the final E-step (rows sum to 1).
    lower_bound: float
        mean over cells of logsumexp_k(log p(x|theta_k) + log pi_k).
    n_phys_pars, n_genes, n_cells: int
    """
    Q = np.asarray(Q, dtype=float)
    assigns = np.argmax(Q, axis=1)
    k_final = len(np.unique(assigns))
    log_joint_per_cell = lower_bound + np.mean(np.log(np.max(Q, axis=1)))
    k_param = n_phys_pars * n_genes * k_final + (k_final - 1)
    return float(-log_joint_per_cell + k_param / n_cells)


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
def _deprecated_aic_property():
    """``.aic`` used to be ``lower_bound - penalty`` (higher is better). It is now
    the paper's per-cell AIC (lower is better) under the name ``aic_per_cell``.
    Reading ``.aic`` still works but warns, so scripts that pick K with
    ``argmax`` are told to switch to ``argmin``."""
    def getter(self):
        warnings.warn(
            "'.aic' was renamed to '.aic_per_cell' and its convention changed: it is now the "
            "meK-Means paper's AIC = -log p(X,Z;theta)/N + K_param/N, where LOWER is better "
            "(the old value was its negative). Use .aic_per_cell and pick K with argmin.",
            DeprecationWarning, stacklevel=2,
        )
        return self.aic_per_cell
    def setter(self, value):
        self.aic_per_cell = value
    return property(getter, setter)


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
    aic_per_cell: float
        per-cell AIC of the fitted mixture, as defined in the meK-Means paper
        (lower is better); see ``mek_means_aic``. ``aic`` is a deprecated alias.
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
        self.aic_per_cell = aic
        self.assigns = assigns
        self.all_qs = all_qs
        self.all_klds = all_klds

    _SERIAL_KIND = "MEKMeansGridPointResults"
    _SERIAL_ATTRS = _BaseGridPointResults._SERIAL_ATTRS + ("weights", "aic_per_cell", "assigns", "all_qs", "all_klds")

    aic = _deprecated_aic_property()


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
    aic_per_cell: float
        per-cell AIC of the fitted mixture, as defined in the meK-Means paper
        (lower is better); see ``mek_means_aic``. ``aic`` is a deprecated alias.
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
        self.aic_per_cell = []
        self.all_qs = []
        self.all_klds = []
        self.filt = []

    _SERIAL_KIND = "MEKMeansSearchResults"
    _SERIAL_ATTRS = _BaseSearchResults._SERIAL_ATTRS + ("assigns", "save", "weights", "aic_per_cell", "all_qs", "all_klds", "filt")

    aic = _deprecated_aic_property()

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
            self.aic_per_cell += [gpr.aic_per_cell]
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
        super().clean_up(remove_files)
        self.aic_per_cell = np.asarray(self.aic_per_cell)
        self.weights = np.asarray(self.weights)
        self.all_qs = np.asarray(self.all_qs)
        self.all_klds = np.asarray(self.all_klds)

        if self.save:
            # Created lazily by _savefig the first time a figure is actually saved.
            self.analysis_figure_string = self.inference_string + "/analysis_figures_" + str(self.assigns)

    @property
    def _result_filename(self):
        return self.inference_string + "/grid_scan_results_" + str(self.assigns) + ".h5"

    ####################################
    #         Analysis methods         #
    ####################################

    # Inherited from _BaseSearchResults (identical or superset behavior):
    # aggregate_grid_points, find_sampling_optimum, set_sampling_optimum, plot_landscape,
    # get_bool_filt, plot_param_marg, plot_KL, chisquare_testing, compute_sigma,
    # resample_opt_viz, resample_opt_mc_viz, plot_param_L_dep, get_noise_decomp.

    def _search_data_for(self, search_data):
        """The search data restricted to this cluster's cells (label 0 = in the cluster)."""
        labels = np.where(self.filt, 0, 1)
        return _partition_search_data(search_data, labels, 2)[0]

    # par_fun_hess, compute_sigma, chisq_best_param_correction, plot_gene_distributions,
    # plot_param_L_dep, resample_opt_viz and get_logL are inherited; each calls
    # _search_data_for first, which is where the cluster subset is applied.

    # get_noise_decomp is inherited from _BaseSearchResults.


# Use class to make inference faster.
