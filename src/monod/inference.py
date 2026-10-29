import itertools
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import optimize, stats
from ._core import make_dir, log, _build_sampling_grid, to_dense
from .extract_data import extract_data, make_histogram, grid_limits, _uns_unpack, model_from_uns
from .cme_toolbox import CMEModel, _HAS_RUST, _MODELS_2D, _format_direction, _PROTEIN_ODE, _CUSTOM_ODE
from .reaction_network import ReactionNetworkWarning, IdentifiabilityWarning
from ._core import _mc          # None when the compiled extension is not installed


#: Default settings for the per-gene fits.  Keys:
#:   max_iterations   maximum L-BFGS-B iterations per start
#:   init_pattern     "moments" starts the first attempt at the method-of-moments estimate
#:   num_restarts     number of starts per gene; later starts are drawn uniformly in the restart box
#:   num_gene_cores   gene-level parallelism per grid point (-1 = all cores)
#:   n_jac_jobs       joblib parallelism over finite-difference perturbations (Python path)
#:   use_rust_lbfgsb  "auto" (default): run the whole fit in the Rust L-BFGS-B kernel
#:                    whenever the extension is installed and the model has a kernel;
#:                    False uses the scipy optimizer
#: Optional keys: presearch / presearch_maxiter
#: (derivative-free pre-search, on by default for ProteinBursty), check_identifiability
#: ("auto": numerical rank check of the model at the first grid point for Custom
#: networks; True for every model; False to skip).
DEFAULT_GRADIENT_PARAMS = {
    "max_iterations": 10,
    "init_pattern": "moments",
    "num_restarts": 3,
    "num_gene_cores": 1,
    "n_jac_jobs": 1,
    "use_rust_lbfgsb": "auto",
}

import multiprocessing
import os

# lbfgsb has a deprecation warning for .tostring(), probably in FORTRAN interface
import warnings
from .plot_aesthetics import aesthetics

from tqdm import tqdm
# from tqdm.contrib.concurrent import process_map  # or thread_map
# warnings.filterwarnings("ignore", category=DeprecationWarning) #let's do more gargeted stuff...

def perform_inference(h5ad_filepath,
    model,
    transcriptome_filepath=None,
    n_genes=100,
    seed=2813308004,
    viz=True,
    modality_name_dict=None,
    dataset_string=None,
    filt_param=None,
    genes_to_fit=None,
    gradient_params=None,
    use_lengths="auto",
    run_meta="",
    phys_lb=None,
    phys_ub=None,
    samp_lb=None,
    samp_ub=None,
    gridsize=None,
    hist_type='unique',
    exclude_sigma=True,
    poisson_average_log_length=5,
    mek_means_params=None,
    num_cores=1, AIC_EPS=1e-20, AIC_offs=0,
    save=False, checkpoint=False):
    '''
    Load and filter data from h5ad file.
    Run inference procedure for the desired model, save parameters, uncertainty from Hessian and AIC values automatically.
    '''
    
    if gradient_params is None:
        gradient_params = dict(DEFAULT_GRADIENT_PARAMS)
    if genes_to_fit is None:
        genes_to_fit = []
    if mek_means_params:
        k, epochs = mek_means_params
    
    if not dataset_string:
        try:
            dataset_string = ''.join(h5ad_filepath.split('/')[-1].split('.')[:-1])
        except AttributeError:
            # For anndata, use the name of the anndata object.
            dataset_string = model.bio_model + '_' + model.seq_model
            log.info("No dataset name given (dataset_string=None). Saving as {}".format(dataset_string))

    if save:
        make_dir(dataset_string)
    
    monod_adata = extract_data(h5ad_filepath,
    model,
    dataset_name=dataset_string,
    transcriptome_filepath=transcriptome_filepath,
    n_genes=n_genes,
    seed=seed,
    viz=viz,
    filt_param=filt_param,
    modality_name_dict=modality_name_dict,
    genes_to_fit=genes_to_fit, hist_type=hist_type, mek_means_params=mek_means_params)
    log.info('Data extracted')
    log.debug('hist type is %s',hist_type)
    
    search_data = searchdata_from_adata(monod_adata)
    log.info('Search data created.')

    if not transcriptome_filepath:
        if 'log_lengths' in monod_adata.var.columns:
            log.info('Gene lengths found in adata.var, using for inference.')
        else:
            use_lengths = None
            log.info('Lengths have not been given so are not being used')

    if not mek_means_params:
        inference_parameters = InferenceParameters(
            dataset_string,
            model,
            use_lengths=use_lengths,
            gradient_params=gradient_params,
            run_meta=run_meta,
            phys_lb=phys_lb,
            phys_ub=phys_ub,
            samp_lb=samp_lb,
            samp_ub=samp_ub,
            gridsize=gridsize,
            poisson_average_log_length=poisson_average_log_length)

    else:
        from . import mminference
        # Stage 1: standard inference to find optimal technical parameters.
        # Technical noise is a batch-level property shared across all clusters,
        # so eta is estimated once from the full population before clustering.
        standard_ip = InferenceParameters(
            dataset_string,
            model,
            use_lengths=use_lengths,
            gradient_params=gradient_params,
            run_meta=run_meta,
            phys_lb=phys_lb,
            phys_ub=phys_ub,
            samp_lb=samp_lb,
            samp_ub=samp_ub,
            gridsize=gridsize,
            poisson_average_log_length=poisson_average_log_length)

        if standard_ip.n_grid_points > 1:
            log.info('Running standard inference to find optimal technical parameters for meK-Means.')
            standard_result = standard_ip.fit_all_grid_points(
                search_data, num_cores=num_cores,
                checkpoint=checkpoint, save=False)
            standard_result.find_sampling_optimum(discard_rejected=False)
            optimal_eta = list(standard_result.samp_optimum)
            log.info('Optimal technical parameters for meK-Means: {}'.format(optimal_eta))
            fixed_samp_lb = optimal_eta
            fixed_samp_ub = optimal_eta
            fixed_gridsize = [1] * len(optimal_eta)
        else:
            fixed_samp_lb = samp_lb
            fixed_samp_ub = samp_ub
            fixed_gridsize = gridsize

        # Stage 2: meK-Means at the fixed optimal technical parameters.
        inference_parameters = mminference.MEKMeansParameters(
            dataset_string,
            model,
            use_lengths=use_lengths,
            gradient_params=gradient_params,
            run_meta=run_meta,
            phys_lb=phys_lb,
            phys_ub=phys_ub,
            samp_lb=fixed_samp_lb,
            samp_ub=fixed_samp_ub,
            gridsize=fixed_gridsize,
            poisson_average_log_length=poisson_average_log_length,
            k=k, epochs=epochs)
    
    log.info('Global inference parameters set.')

    # One or several SearchResults (one per meK-Means cluster); the var
    # columns of a cluster result carry the prefix c<j>_.  save=False here:
    # with save=True the whole run is written once, as monod_adata.h5ad.
    fitted = inference_parameters.fit_all_grid_points(search_data, num_cores=num_cores, checkpoint=checkpoint, save=False)
    log.info('Grid points fit.')
    results = list(fitted) if mek_means_params else [fitted]
    prefixes = [f"c{j}_" for j in range(len(results))] if mek_means_params else [""]
    for sr in results:
        sr.find_sampling_optimum(discard_rejected=False)
    log.info('Optimal parameters found.')
    names = [p.replace(r"\log_{10} ", "") for p in results[0].model.param_str]

    # Load the AnnData object into memory if it is in backed mode
    if monod_adata.isbacked:
        monod_adata = monod_adata.to_memory()

    for sr, pre in zip(results, prefixes):
        for i, name in enumerate(names):
            monod_adata.var[pre + name] = sr.phys_optimum[:, i]
    log.info('Optimal parameters saved to anndata under .var attributes "%s<parameter>" (log-base 10).',
             'ci_' if mek_means_params else '')

    if mek_means_params:
        # AIC values from meK-Means are the same for all clusters.
        monod_adata.var['AIC'] = get_AIC_mek_means(results, search_data, AIC_EPS=AIC_EPS, AIC_offs=AIC_offs)
    else:
        AIC_per_gene(results[0], monod_adata, search_data)
    log.info('AIC values calculated and saved under .var attribute: AIC.')

    if not exclude_sigma:
        for sr, pre in zip(results, prefixes):
            # compute_sigma restricts the search data to the cluster's cells itself.
            sr.compute_sigma(search_data, num_cores=num_cores)
            for i, name in enumerate(names):
                monod_adata.var[pre + 'error_' + name] = sr.sigma[:, i]
            monod_adata.var[pre + 'error_unreliable'] = np.asarray(sr.sigma_fail, dtype=bool)
        log.info('Uncertainties per gene saved under .var attributes "%serror_<parameter>" (log-base 10).',
                 'ci_' if mek_means_params else '')

    if mek_means_params:
        monod_adata.obs['cluster'] = None
        for sr in results:
            monod_adata.obs.loc[sr.filt, 'cluster'] = sr.assigns
        log.info('Cluster labels added to anndata .obs')
        monod_adata.uns['search_result_list'] = search_result_list_to_uns(results)
    else:
        monod_adata.uns['search_result'] = results[0].to_dict()

    # The SearchData is deliberately not stored: it is rebuilt in memory from
    # the AnnData (layers + var['M_<layer>']) via searchdata_from_adata(), which is fast
    # and avoids serialising the (Rust-backed) histogram container.
    log.info('Search data is not stored; rebuild it with searchdata_from_adata(monod_adata).')

    if save:
        # Everything (data, model, parameters, results, cluster labels) lives in
        # the AnnData, so one h5ad is the whole run on disk.
        inference_string = results[0].inference_string
        make_dir(inference_string)
        adata_file_path = inference_string + '/monod_adata.h5ad'
        monod_adata.write_h5ad(adata_file_path)
        log.info('Run stored to %s (load results with analysis.load_search_results).', adata_file_path)

    return monod_adata

def searchdata_from_adata(adata):

    n_genes = adata.n_vars

    # NB the order of the layers here will be enforced to be the same as the order of the model
    # modalities defined in cme_toolbox.
    modality_name_dict = adata.uns['modality_name_dict']
    model = model_from_uns(adata.uns['model'])

    ordered_modalities = model.model_modalities
    ordered_layer_names = [modality_name_dict[modality] for modality in ordered_modalities]

    M = grid_limits(adata)
    n_cells = adata.n_obs
    gene_names = list(adata.var.index)

    gene_log_lengths = None
    try:
        gene_log_lengths = np.asarray(adata.var['log_lengths'], dtype=np.float64)
    except KeyError:
        pass

    gene_log_lengths_spliced = None
    spliced_lengths_col = adata.uns.get('spliced_lengths_col', None)
    if spliced_lengths_col is not None:
        try:
            gene_log_lengths_spliced = np.asarray(adata.var[spliced_lengths_col], dtype=np.float64)
        except KeyError:
            pass

    k = adata.uns.get('k', None)
    epochs = adata.uns.get('epochs', None)

    if _mc is not None:
        # Build the pure-Rust SearchData container.
        def _to_c_int64(arr):
            a = to_dense(arr)
            return np.ascontiguousarray(a, dtype=np.int64)
        rust_layers = [_to_c_int64(adata.layers[ln]) for ln in ordered_layer_names]
        coords_list, freqs_list = _mc.make_state_dist(rust_layers)
        limits_arr  = np.ascontiguousarray(M, dtype=np.int64)
        return _mc.searchdata_from_arrays(
            rust_layers,
            ordered_layer_names,
            limits_arr,
            coords_list,
            freqs_list,
            gene_names,
            n_cells,
            "unique",
            gene_log_lengths,
            gene_log_lengths_spliced,
            k,
            epochs,
        )

    # Python fallback (Rust unavailable).
    hist = make_histogram(adata, "unique", M)
    layers  = np.array([adata.layers[layer_name] for layer_name in ordered_layer_names])
    moments = get_gene_moments(adata)

    attr_names = [
        "M", "hist", "moments", "n_genes", "gene_names",
        "n_cells", "layers", "hist_type", "layer_names"
    ]
    attr_values = [M, hist, moments, n_genes, gene_names, n_cells, layers, "unique", ordered_layer_names]

    if gene_log_lengths is not None:
        attr_names  += ['gene_log_lengths']
        attr_values += [gene_log_lengths]

    if k is not None:
        attr_names  += ['k', 'epochs']
        attr_values += [k, epochs]

    return SearchData(attr_names, *attr_values)


def get_gene_moments(adata):
    
    # Specify the columns you want to extract
    columns_of_interest = [i for i in adata.var.columns if 'MOM' in i]
    
    # Ensure the columns exist in adata.var
    for col in columns_of_interest:
        if col not in adata.var.columns:
            raise ValueError(f"Column '{col}' does not exist in adata.var")
    
    # Extract the selected columns
    selected_data = adata.var[columns_of_interest]
    
    # Convert the DataFrame to a list of dictionaries
    list_of_dicts = selected_data.to_dict(orient='records')
    
    return list_of_dicts


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

            

def aic_values(search_result, search_data, EPS=1e-20, offs=0):
    """Per-gene AIC of one fit at its sampling optimum, ``2 k - 2 logL`` with
    ``k`` the number of biological parameters: the convention of
    ``adata.var['AIC']``, the Akaike weights and the meK-Means score.  (The
    model-comparison functions in :mod:`monod.analysis` report the deviance
    form ``2 k + 2 n_cells KLD`` instead, which differs by the entropy of the
    empirical distribution, a per-gene constant that cancels in differences.)"""
    logL = np.asarray(search_result.get_logL(search_data, EPS=EPS, offs=offs))
    return 2 * search_result.sp.n_phys_pars - 2 * logL


def AIC_per_gene(search_result, monod_adata, search_data=None):
    """Per-gene AIC (:func:`aic_values`), stored in ``monod_adata.var['AIC']`` and returned.

    ``search_data`` is rebuilt from the AnnData when not given."""
    if search_data is None:
        search_data = searchdata_from_adata(monod_adata)
    AIC = aic_values(search_result, search_data)
    monod_adata.var['AIC'] = AIC
    return AIC



# ---------------------------------------------------------------------------
# Structured (pickle-free) serialisation.
#
# Every persistent object (CMEModel, InferenceParameters, GridPointResults,
# SearchResults and their meK-Means subclasses) implements to_dict()/from_dict()
# producing nested dicts of numpy arrays, scalars and strings.  Those dicts go
# straight into adata.uns (so the AnnData writes to h5ad) and into standalone
# HDF5 files via anndata's write_elem.  No pickle is involved, so loading does
# not execute code and does not depend on the Python class layout at write time.
# ---------------------------------------------------------------------------

def _to_plain(v):
    """Coerce an attribute value into something h5ad/HDF5 can store."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return np.zeros(0)
        a = np.asarray(v)
        if a.dtype == object:
            raise TypeError("cannot serialise a ragged or object-typed attribute")
        return a
    return v


def _attr_dict(obj, names):
    d = {}
    for n in names:
        v = getattr(obj, n, None)
        if v is None:
            continue
        d[n] = _to_plain(v)
    return d


def _write_h5(path, d):
    """Write a structured dict to an HDF5 file (anndata element encoding)."""
    import h5py
    from anndata.io import write_elem
    with h5py.File(path, "w") as f:
        write_elem(f, "monod", d)


def _read_h5(path):
    import h5py
    from anndata.io import read_elem
    with h5py.File(path, "r") as f:
        return read_elem(f["monod"])


def search_result_from_uns(v):
    """Return the SearchResults stored in ``adata.uns['search_result']``."""
    if hasattr(v, "keys"):
        return SearchResults.from_dict(v)
    return _uns_unpack(v)  # legacy pickle


def search_result_list_from_uns(v):
    """Return the per-cluster SearchResults list stored in ``adata.uns['search_result_list']``."""
    if hasattr(v, "keys"):
        n = int(v["n_clusters"])
        return [SearchResults.from_dict(v[f"cluster_{i}"]) for i in range(n)]
    return _uns_unpack(v)  # legacy pickle


def search_result_list_to_uns(srs):
    d = {"format_version": 1, "kind": "SearchResultsList", "n_clusters": len(srs)}
    for i, sr in enumerate(srs):
        d[f"cluster_{i}"] = sr.to_dict()
    return d


_USE_LENGTHS_MODES = (None, "unspliced", "spliced", "both")


def _validate_bounds(model, phys_lb, phys_ub, samp_lb, samp_ub, gridsize):
    """Raise ValueError for malformed parameter bounds / sampling grids.

    Checks lengths, finiteness and lb <= ub for the biological bounds, and
    that samp_lb, samp_ub and gridsize agree in length with integer grid sizes
    >= 1.  Without this, an inverted bound is silently accepted by
    scipy.optimize.Bounds and by the Rust L-BFGS-B wrapper, which then reports
    the projected starting point as a converged optimum.
    """
    phys_lb = np.asarray(phys_lb, dtype=float)
    phys_ub = np.asarray(phys_ub, dtype=float)
    n_par = model.get_num_params()
    if phys_lb.ndim != 1 or phys_ub.ndim != 1 or phys_lb.shape != phys_ub.shape:
        raise ValueError(
            f"phys_lb and phys_ub must be 1-D with equal length; got shapes {phys_lb.shape} and {phys_ub.shape}"
        )
    if phys_lb.shape[0] < n_par:
        raise ValueError(
            f"{model.bio_model} has {n_par} parameters but phys_lb/phys_ub have {phys_lb.shape[0]} entries"
        )
    if not (np.all(np.isfinite(phys_lb)) and np.all(np.isfinite(phys_ub))):
        raise ValueError("phys_lb and phys_ub must be finite log10 values")
    bad = np.where(phys_lb > phys_ub)[0]
    if bad.size:
        i = int(bad[0])
        raise ValueError(f"phys_lb[{i}]={phys_lb[i]} exceeds phys_ub[{i}]={phys_ub[i]}")
    samp_lb = np.asarray(samp_lb, dtype=float)
    samp_ub = np.asarray(samp_ub, dtype=float)
    gridsize = list(gridsize)
    if not (len(samp_lb) == len(samp_ub) == len(gridsize)):
        raise ValueError(
            f"samp_lb ({len(samp_lb)}), samp_ub ({len(samp_ub)}) and gridsize ({len(gridsize)}) must have the same length"
        )
    if any((int(g) != g) or g < 1 for g in gridsize):
        raise ValueError(f"gridsize entries must be integers >= 1, got {gridsize}")
    bad = np.where(samp_lb > samp_ub)[0]
    if bad.size:
        i = int(bad[0])
        raise ValueError(f"samp_lb[{i}]={samp_lb[i]} exceeds samp_ub[{i}]={samp_ub[i]}")


def _warn_optimizer_status(status, label):
    """Log a summary of how the Rust L-BFGS-B runs ended.

    Status codes: 0 converged (tolerance met), 1 hit ``max_iterations``,
    2 stopped because the line search could find no further decrease,
    3 solver error.  Code 2 is the normal end point for an objective with a
    finite-difference gradient: once the remaining improvement is below the
    gradient's noise floor no step along it is accepted, exactly as scipy's
    ``ABNORMAL_TERMINATION_IN_LNSRCH``.  The returned point is the best one
    seen, so it is reported at INFO level; only codes 1 and 3 are warnings.
    """
    if status is None:
        return
    st = np.asarray(status)
    n_max, n_ls, n_err = int((st == 1).sum()), int((st == 2).sum()), int((st == 3).sum())
    if n_max or n_err:
        log.warning(
            "%s: %d/%d genes did not converge (%d hit max_iterations, %d solver errors). "
            "Consider raising gradient_params['max_iterations'] or checking the bounds.",
            label, n_max + n_err, st.size, n_max, n_err,
        )
    if n_ls:
        log.info(
            "%s: %d/%d genes stopped at the precision of the finite-difference gradient "
            "(line search found no further decrease).",
            label, n_ls, st.size,
        )


def _normalize_use_lengths(use_lengths, model):
    """Resolve the ``use_lengths`` option against the model's sequencing model.

    Returns a tuple ``(mode, explicit)`` where ``mode`` is one of None,
    "unspliced", "spliced", "both" and ``explicit`` is False when the caller
    left the default ("auto"), so downstream code may silently fall back to
    no length correction when gene lengths are unavailable.

    "auto" (the default) means "unspliced" for the Poisson sequencing model
    and None otherwise, since only the Poisson model has a length effect.
    Legacy bools are accepted: True -> "unspliced", False -> None.
    """
    if use_lengths == "auto":
        mode = "unspliced" if model.seq_model == "Poisson" else None
        return mode, False
    if use_lengths is True:
        use_lengths = "unspliced"
    elif use_lengths is False:
        use_lengths = None
    if use_lengths not in _USE_LENGTHS_MODES:
        raise ValueError(
            "use_lengths must be 'auto', None, 'unspliced', 'spliced', or 'both'; "
            f"got {use_lengths!r}"
        )
    if use_lengths is not None and model.seq_model != "Poisson":
        raise ValueError(
            f"use_lengths={use_lengths!r} requires seq_model='Poisson'; the "
            f"{model.seq_model!r} sequencing model has no gene-length effect. "
            "Pass use_lengths=None (or leave the default)."
        )
    return use_lengths, True


class InferenceParameters:
    """Stores parameters and distributes the multi-grid point inference procedure.

    Attributes
    ----------
    gradient_params: dict
        settings for the per-gene fits; see ``DEFAULT_GRADIENT_PARAMS`` for the keys.
    phys_lb: float np.ndarray
        log10 lower bounds on biological parameters.
    phys_ub: float np.ndarray
        log10 upper bounds on biological parameters.
    grad_bnd: scipy.optimize.Bounds
        log10 lower and upper bounds on biological parameters.
    use_lengths: None | "unspliced" | "spliced" | "both"
        Controls which Poisson sampling channels receive a per-gene log-length offset.
        None: no length correction (genome-wide sampling rate for all channels).
        "unspliced": add log10(gene_length) to the unspliced sampling parameter.
        "spliced": add log10(gene_length) to the spliced sampling parameter.
        "both": add log10(gene_length) to both sampling parameters.
    samp_lb: np.ndarray
        log10 lower bounds on technical variation parameters.
    samp_ub: np.ndarray
        log10 upper bounds on technical variation parameters.
    gridsize: list of ints or int np.ndarray
        grid size for evaluating the technical variation parameters.
    model: monod.cme_toolbox.CMEModel
        CME model used for inference.
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
            Which Poisson sampling channels receive a per-gene log-length offset.
            "auto" (default): "unspliced" for the Poisson sequencing model, None
            otherwise; falls back to None if the search data carries no lengths.
            None: no correction. "unspliced": unspliced channel only.
            "spliced": spliced channel only. "both": both channels.
            Any non-None value with a non-Poisson sequencing model raises.
            Legacy bool values are accepted: True → "unspliced", False → None.
        gradient_params: dict, optional
            settings for the per-gene fits; missing keys take ``DEFAULT_GRADIENT_PARAMS``.
            "max_iterations" defines the maximum number of gradient descent iterations.
            "init_pattern" defines whether the first try starts at the method of moments estimate.
            "num_restarts" defines how many attempts should be made.
            "presearch" selects a derivative-free search run from the first starting point before
                gradient descent: "auto" (default; Nelder-Mead for ProteinBursty, nothing otherwise),
                "nelder-mead", or None. "presearch_maxiter" caps its iterations (default 300).
            "check_identifiability" runs CMEModel.check_identifiability once, at the first grid
                point and the first gene's starting point, and warns about flat directions:
                "auto" (default; Custom networks only), True (every model) or False.
            "use_rust_lbfgsb" selects the optimizer: "auto" (default) runs the whole fit in the
                Rust L-BFGS-B kernel whenever the extension is installed and the model has a
                kernel, about ten times faster than the scipy loop with the same results;
                False uses scipy.
            "num_gene_cores" controls gene-level parallelism per grid point (-1 = all cores)
                when the scipy optimizer is used (a thread pool with one gene per task).
            "n_jac_jobs" controls joblib parallelism over FD perturbations (-1 = all cores); only
            effective when num_gene_cores=1 to avoid nested parallelism.
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
        
        self.gradient_params = {**DEFAULT_GRADIENT_PARAMS, **(gradient_params or {})}
        self.use_lengths, self.use_lengths_explicit = _normalize_use_lengths(use_lengths, model)
        if model.seq_model == "None":
            log.info("Sequencing model set to None. All sampling parameters set to null.")
            samp_lb, samp_ub, gridsize = model.null_sampling_grid()
        _validate_bounds(model, phys_lb, phys_ub, samp_lb, samp_ub, gridsize)
        self.phys_lb = np.array(phys_lb)
        self.phys_ub = np.array(phys_ub)
        self.grad_bnd = scipy.optimize.Bounds(phys_lb, phys_ub)

        self.poisson_average_log_length = poisson_average_log_length

        self.samp_lb = np.array(samp_lb)
        self.samp_ub = np.array(samp_ub)
        self.gridsize = gridsize

        self.construct_grid()
        self.model = model

        if model.bio_model == "Custom":
            # The constructor already warned about the other structural issues;
            # the time-unit degeneracy only matters once the network is fitted.
            for issue in model.network.structural_issues():
                if issue["code"] == "timescale":
                    warnings.warn(
                        f"ReactionNetwork({model.network.network_str!r}): {issue['message']}",
                        ReactionNetworkWarning, stacklevel=2,
                    )

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

        # Absolute so results, figures and checkpoints land in the same place
        # regardless of later chdir calls, and so stored paths are unambiguous.
        inference_string = os.path.abspath(inference_string)
        if save:
            make_dir(inference_string)
            inference_parameter_string = inference_string + "/parameters.h5"
            self.store_inference_parameters(inference_parameter_string)
        self.inference_string = inference_string

    # ------------------------------------------------------------------
    # Structured (pickle-free) serialisation
    # ------------------------------------------------------------------
    _SERIAL_KIND = "InferenceParameters"

    def to_dict(self):
        gp = {}
        for k, v in self.gradient_params.items():
            if v is None:
                continue
            gp[k] = v.item() if isinstance(v, np.generic) else v
        d = {
            "format_version": 1,
            "kind": self._SERIAL_KIND,
            "dataset_string": self.dataset_string,
            "inference_string": self.inference_string,
            "model": self.model.to_dict(),
            "use_lengths": self.use_lengths or "",
            "use_lengths_explicit": bool(getattr(self, "use_lengths_explicit", True)),
            "gradient_params": gp,
            "phys_lb": np.asarray(self.phys_lb, dtype=float),
            "phys_ub": np.asarray(self.phys_ub, dtype=float),
            "samp_lb": np.asarray(self.samp_lb, dtype=float),
            "samp_ub": np.asarray(self.samp_ub, dtype=float),
            "gridsize": np.asarray(self.gridsize, dtype=int),
            "poisson_average_log_length": float(self.poisson_average_log_length),
        }
        return d

    @staticmethod
    def _ctor_kwargs_from_dict(d):
        from .cme_toolbox import CMEModel
        ul = d.get("use_lengths", "")
        return dict(
            model=CMEModel.from_dict(d["model"]),
            use_lengths=(None if ul == "" else str(ul)),
            gradient_params={k: (v.item() if isinstance(v, np.generic) else v) for k, v in dict(d.get("gradient_params", {})).items()},
            phys_lb=np.asarray(d["phys_lb"], dtype=float),
            phys_ub=np.asarray(d["phys_ub"], dtype=float),
            samp_lb=np.asarray(d["samp_lb"], dtype=float),
            samp_ub=np.asarray(d["samp_ub"], dtype=float),
            gridsize=[int(g) for g in np.asarray(d["gridsize"]).ravel()],
            poisson_average_log_length=float(d.get("poisson_average_log_length", 5)),
        )

    @classmethod
    def from_dict(cls, d):
        kind = str(d.get("kind", "InferenceParameters"))
        if kind == "MEKMeansParameters":
            from .mminference import MEKMeansParameters
            obj = MEKMeansParameters._from_dict_impl(d)
        else:
            kw = cls._ctor_kwargs_from_dict(d)
            obj = InferenceParameters(str(d["dataset_string"]), save=False, **kw)
        obj.inference_string = str(d["inference_string"])
        obj.use_lengths_explicit = bool(d.get("use_lengths_explicit", True))
        return obj

    def construct_grid(self):
        """Creates a grid of points over the two-dimensional technical variation parameter domain.

        Sets
        ----
        sampl_vals: list of lists of floats
            list of grid points.
        grid_values_sampl: list of np.ndarrays
            grid point values representing sampling parameters for each modality.
        n_grid_points: int
            total number of grid points to evaluate.
        """
        self.grid_values_sampl, self.sampl_vals, self.n_grid_points = _build_sampling_grid(
            self.samp_lb, self.samp_ub, self.gridsize
        )

    def store_inference_parameters(self, inference_parameter_string):
        """This helper method attempts to save the InferenceParameters object.

        Parameters
        ----------
        inference_parameter_string: str
            file location.
        """
        try:
            _write_h5(inference_parameter_string, self.to_dict())
            log.info(
                "Global inference parameters stored to {}.".format(
                    inference_parameter_string
                )
            )
        except Exception:
            log.error(
                "Global inference parameters could not be stored to {}.".format(
                    inference_parameter_string
                ),
                exc_info=True,
            )

    def fit_all_grid_points(self, search_data, num_cores=1, checkpoint=False, save=False):
        """Fits the search data for all genes over all grid points.

        Parameters
        ----------
        search_data: SearchData
            SearchData object with the data to fit.
        num_cores: int
            Gene-level parallelism at each grid point (a thread pool with
            scipy L-BFGS-B per gene; the Rust kernels release the GIL).  -1
            uses all cores.  Overrides ``gradient_params["num_gene_cores"]``
            for this call.
        checkpoint: bool, optional
            Write each grid point result to disk (``grid_point_<i>.h5``) as it
            finishes. Default False.
        save: bool, optional
            Write the final SearchResults to disk. Default False.

        Returns
        -------
        results: SearchResults

        Every grid point starts from the moments estimate (or the random box).
        Grid points are not warm-started from their neighbours: the biological
        optimum shifts systematically with the sampling parameters (burst size
        trades off against capture), so a neighbour's optimum is a worse start
        than the moments estimate and a poor solution at one point used to
        cascade across the whole grid.
        """
        t1 = time.time()
        self._checkpoint = checkpoint
        if checkpoint:
            make_dir(self.inference_string)

        _prev_gene_cores = self.gradient_params.get("num_gene_cores", 1)
        if num_cores != 1:
            self.gradient_params["num_gene_cores"] = num_cores

        gp_results: dict = {}
        log.info("Starting grid scan (%d points).", self.n_grid_points)
        for idx in tqdm(range(self.n_grid_points), desc="Grid scan"):
            r = self.par_fun(idx, search_data, self.model)
            if r is not None:
                gp_results[r.point_index] = r
        log.info("Grid scan complete.")

        self.gradient_params["num_gene_cores"] = _prev_gene_cores

        results = SearchResults(self, search_data)
        results.aggregate_grid_points(gp_results if gp_results else None)
        if save:
            make_dir(self.inference_string)
            results.store_on_disk()

        t2 = time.time()
        log.info("Runtime: {:.1f} seconds.".format(t2 - t1))

        return results

    def par_fun(self, point_index, search_data, model):
        """Fit every gene at one grid point.

        Parameters
        ----------
        point_index: int
            index within [0, n_grid_points) to evaluate at.
        search_data: SearchData
        model: CMEModel

        Returns
        -------
        GridPointResults or None
        """
        grad_inference = GradientInference(self, model, search_data, point_index)
        return grad_inference.fit_all_genes(model, search_data, checkpoint=getattr(self, '_checkpoint', False))


class GradientInference:
    """Runs the grid point-specific inference procedures.

    Attributes
    ----------
    grid_point: list of floats
        genome-wide technical variation parameter values at the current grid point.
    point_index: int
        the index of the current point, within [0, n_grid_points).
    regressor: np.ndarray
        gene-specific technical variation parameter values at the current grid point.
        these values will be different for each gene if use_lengths is not None in the
        InferenceParameters constructor.
    grad_bnd: scipy.optimize.Bounds
        log10 lower and upper bounds on biological parameters.
    gradient_params: dict
        settings for the per-gene fits; see ``DEFAULT_GRADIENT_PARAMS`` for the keys.
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

    def __init__(self, global_parameters, model, search_data, point_index):
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

        Sets
        ----
        grid_point: list of floats
            genome-wide technical variation parameter values at the current grid point.
        point_index: int
            the index of the current point, within [0, n_grid_points).
        regressor: np.ndarray
            gene-specific technical variation parameter values at the current grid point.
            these values will be different for each gene if use_lengths is not None in the
            InferenceParameters constructor.
        grad_bnd: scipy.optimize.Bounds
            log10 lower and upper bounds on biological parameters.
        gradient_params: dict
            settings for the per-gene fits; missing keys take ``DEFAULT_GRADIENT_PARAMS``.
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
 
        regressor = np.array(
            [global_parameters.sampl_vals[point_index]] * search_data.n_genes
        )

        use_lengths_mode = global_parameters.use_lengths
        if use_lengths_mode is not None and getattr(search_data, "gene_log_lengths", None) is None:
            if getattr(global_parameters, "use_lengths_explicit", True):
                raise ValueError(
                    f"use_lengths={use_lengths_mode!r} was requested but the search data has "
                    "no gene lengths. Provide a transcriptome_filepath / adata.var['log_lengths'], "
                    "or pass use_lengths=None."
                )
            log.info("No gene lengths in search data; use_lengths='auto' resolved to None.")
            use_lengths_mode = None
        if use_lengths_mode is not None:
            if model.seq_model == "Bernoulli":
                raise ValueError(
                    "The Bernoulli model does not yet have a physical length-based model."
                )
            elif model.seq_model == "None":
                raise ValueError(
                    "The model without technical noise has no length effects."
                )
            elif model.seq_model == "Poisson":
                if use_lengths_mode in ("unspliced", "both"):
                    regressor[:, 0] += search_data.gene_log_lengths
                if use_lengths_mode in ("spliced", "both"):
                    spl_lengths = getattr(search_data, 'gene_log_lengths_spliced', None)
                    if spl_lengths is None:
                        spl_lengths = search_data.gene_log_lengths
                    regressor[:, 1] += spl_lengths
            else:
                raise ValueError(
                    "Please select a technical noise model from {Poisson}, {Bernoulli}, {None}."
                )
        else:
            # No lengths given: the protein model's unspliced Poisson channel gets an
            # average log length (the two-species models use the raw sampling rate).
            if model.seq_model == "Poisson" and model.bio_model == "ProteinBursty" and model.fit_unspliced:
                regressor[:, 0] += global_parameters.poisson_average_log_length
            
        self.grid_point = global_parameters.sampl_vals[point_index]
        self.point_index = point_index
        self.regressor = regressor
        self.use_lengths = global_parameters.use_lengths
        self.grad_bnd = global_parameters.grad_bnd
        self.gradient_params = global_parameters.gradient_params
        self.phys_lb = global_parameters.phys_lb
        self.phys_ub = global_parameters.phys_ub
        self.n_phys_pars = global_parameters.n_phys_pars
        self.n_samp_pars = global_parameters.n_samp_pars

        self.inference_string = global_parameters.inference_string

        # Pre-allocate restart initialisation bounds (constant across all genes/restarts).
        self._restart_lb = -2.0 * np.ones(self.n_phys_pars)
        self._restart_lb[0] = 0.0
        self._restart_ub = 2.0 * np.ones(self.n_phys_pars)
        self._restart_range = self._restart_ub - self._restart_lb

        if self.gradient_params["init_pattern"] == "moments":
            _sd_is_rust_sd = _mc is not None and isinstance(search_data, _mc.SearchData)
            if model.bio_model == "Custom":
                # No moment estimator for an arbitrary network: every gene starts
                # from the middle of the parameter box (see CMEModel.get_MoM).
                mid = 0.5 * (np.asarray(global_parameters.phys_lb, float) + np.asarray(global_parameters.phys_ub, float))
                self.param_MoM = np.tile(mid, (search_data.n_genes, 1))
            elif model.bio_model == "Chromatin":
                # chain from the configuration marginal, rates from conditional means: needs the histogram
                self.param_MoM = np.asarray([
                    model.get_MoM(None, global_parameters.phys_lb, global_parameters.phys_ub, regressor[i],
                                  hist=search_data.hist[i], limits=np.asarray(search_data.M[:, i]))
                    for i in range(search_data.n_genes)
                ])
            elif _sd_is_rust_sd:
                # Zero-Python-loop path: Rust computes MoM for all genes at once.
                samp_list_mom = None
                if model.seq_model in ("Poisson", "Bernoulli"):
                    samp_list_mom = [
                        (regressor[gi].tolist() if regressor[gi] is not None else None)
                        for gi in range(search_data.n_genes)
                    ]
                self.param_MoM = np.asarray(
                    search_data.mom_x0_all(
                        model.bio_model,
                        model.seq_model,
                        model.amb_model,
                        global_parameters.phys_lb.tolist(),
                        global_parameters.phys_ub.tolist(),
                        samp_list_mom,
                    )
                )
            else:
                _saved_filters = warnings.filters[:]
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
                warnings.filters[:] = _saved_filters   # restore the caller's filters

        if point_index == 0:
            global_parameters.identifiability = self._check_identifiability(model, search_data)

    def _check_identifiability(self, model, search_data):
        """Numerical rank check of the model, once per inference.

        Runs :meth:`CMEModel.check_identifiability` at the first gene's
        starting point (the moments estimate when there is one, otherwise the
        middle of the parameter box) on that gene's grid, and warns about any
        flat direction. Controlled by ``gradient_params["check_identifiability"]``:
        ``"auto"`` runs it for Custom networks only, ``True`` for every model
        and ``False`` never. Returns the result dict or None when skipped.
        """
        mode = self.gradient_params.get("check_identifiability", "auto")
        if mode == "auto":
            mode = model.bio_model in ("Custom", "Chromatin")
        if not mode:
            return None
        x0 = None
        if hasattr(self, "param_MoM") and np.all(np.isfinite(self.param_MoM[0])):
            x0 = np.asarray(self.param_MoM[0], dtype=float)
        if x0 is None:
            x0 = 0.5 * (np.asarray(self.phys_lb, float) + np.asarray(self.phys_ub, float))
        x0 = np.clip(x0, self.phys_lb, self.phys_ub)
        try:
            result = model.check_identifiability(
                x0, np.asarray(search_data.M[:, 0]), samp=self.regressor[0], warn=False
            )
        except Exception as exc:  # the fit itself will surface a real evaluation problem
            log.info("Identifiability check skipped: %s", exc)
            return None
        log.info(
            "Identifiability check at %s: relative singular values %s",
            np.array2string(x0, precision=2),
            np.array2string(result["singular_values"], precision=2),
        )
        for rel_sv, direction in result["flat"]:
            warnings.warn(
                f"{model.bio_model} model: the steady state is insensitive to the parameter direction "
                f"{_format_direction(direction)} (relative singular value {rel_sv:.1e} at the first "
                "gene's starting point); this combination of parameters cannot be inferred from the "
                "data. Set gradient_params['check_identifiability']=False to silence this check.",
                IdentifiabilityWarning, stacklevel=3,
            )
        return result

    def optimize_gene(self, gene_index, model, search_data, x0):
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
        x0: array-like
            starting points, n_restarts x n_params (see :meth:`_starting_points`).

        Returns
        -------
        x: np.ndarray
            optimal biological parameter values.
        err: float
            Kullback-Leibler divergence of the model at x, relative to data.
        """
        x0 = np.asarray(x0)
        err = np.inf
        ERR_THRESH = 0.99
        if log.isEnabledFor(logging.DEBUG):
            log.debug('Optimizing gene %d with initial value %s', gene_index, np.array2string(10**x0))
        
        for restart in range(self.gradient_params["num_restarts"]):
            n_jac_jobs = self.gradient_params.get("n_jac_jobs", 1)

            def _obj_and_grad(x):
                return model.eval_model_kld_and_grad(
                    p=x,
                    limits=search_data.M[:, gene_index],
                    samp=self.regressor[gene_index],
                    data=search_data.hist[gene_index],
                    n_jobs=n_jac_jobs,
                )

            # Adaptive integrators search at their coarse tolerance (a no-op for
            # the other models) and report the optimum at the fine one.
            with model.search_tolerance():
                res_arr = scipy.optimize.minimize(
                    _obj_and_grad,
                    x0=x0[restart],
                    method='L-BFGS-B',
                    jac=True,
                    bounds=self.grad_bnd,
                    options={
                        "maxiter": self.gradient_params["max_iterations"],
                        "ftol": 2.22e-9,
                        "gtol": 1e-5,
                    },
                )
            fun = res_arr.fun
            if model.has_search_tolerance():
                fun = model.eval_model_kld(res_arr.x, search_data.M[:, gene_index], self.regressor[gene_index],
                                           search_data.hist[gene_index])
            if (
                fun < err * ERR_THRESH
            ):  # do not replace old best estimate if there is little marginal benefit
                x = res_arr.x
                err = fun
        if not (np.isfinite(x).all()):
            log.warning("Gene index: " + str(gene_index))
            raise ValueError("Search failed. Please check input data.")
        if log.isEnabledFor(logging.DEBUG):
            log.debug('Optimized parameters for gene %d is %s', gene_index, np.array2string(10**x))
        return x, err

    def _presearch_x0(self, model, search_data, x0_all, n_gene_cores=1):
        """Refine each gene's first starting point with a derivative-free search.

        The ProteinBursty likelihood has a long, nearly flat valley along the
        k_p/gamma_p direction. L-BFGS-B with a finite-difference gradient stops
        early on that plateau (the gradient falls below its own noise), usually
        after drifting toward small rates where every evaluation is slowest.
        A short Nelder-Mead run from the moments estimate crosses the plateau in
        a few hundred evaluations; the gradient fit then only has to polish.

        Controlled by gradient_params["presearch"] ("auto", "nelder-mead" or
        None) and gradient_params["presearch_maxiter"]. Only the first starting
        point of each gene (the moments or warm start) is refined; random
        restarts are left untouched.

        Parameters
        ----------
        model: monod.cme_toolbox.CMEModel
        search_data: SearchData
        x0_all: list of lists
            per-gene starting points, n_genes x n_restarts x n_phys_pars.
        n_gene_cores: int
            thread count for the per-gene search; 1 runs serially, <=0 uses all cores.

        Returns
        -------
        x0_all: list of lists
            the starting points with the first one per gene replaced.
        """
        mode = self.gradient_params.get("presearch", "auto")
        if mode == "auto":
            mode = "nelder-mead" if model.bio_model == "ProteinBursty" else None
        if not mode:
            return x0_all
        if mode != "nelder-mead":
            raise ValueError(f"gradient_params['presearch'] must be 'auto', 'nelder-mead' or None, got {mode!r}")
        maxiter = int(self.gradient_params.get("presearch_maxiter", 300))
        bounds = list(zip(self.phys_lb, self.phys_ub))

        def _one(gi):
            limits = search_data.M[:, gi]
            samp = self.regressor[gi]
            data = search_data.hist[gi]
            x0 = np.asarray(x0_all[gi][0], dtype=float)

            def kld(x):
                return model.eval_model_kld(np.asarray(x, dtype=float), limits, samp, data)

            res = scipy.optimize.minimize(
                kld, x0, method="Nelder-Mead", bounds=bounds,
                options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-5},
            )
            if np.isfinite(res.fun) and res.fun <= kld(x0):
                return np.clip(res.x, self.phys_lb, self.phys_ub).tolist()
            return x0.tolist()

        n_genes = len(x0_all)
        if n_gene_cores == 1 or n_genes == 1:
            refined = [_one(gi) for gi in range(n_genes)]
        else:
            from concurrent.futures import ThreadPoolExecutor
            n_workers = os.cpu_count() if n_gene_cores <= 0 else n_gene_cores
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                refined = list(pool.map(_one, range(n_genes)))
        out = [list(map(list, x0)) for x0 in x0_all]
        for gi in range(n_genes):
            out[gi][0] = refined[gi]
        if log.isEnabledFor(logging.DEBUG):
            log.debug("Nelder-Mead pre-search refined %d starting points", n_genes)
        return out

    def _rust_optimizer_family(self, model, search_data):
        """Kernel family of the Rust optimizer that applies, or None.

        Requires the compiled extension and a Rust ``SearchData``.
        ``gradient_params["use_rust_lbfgsb"]`` is ``"auto"`` by
        default, which uses the Rust kernel whenever those hold; ``False``
        selects the scipy optimizer. The family itself comes from
        ``CMEModel._kernel_family``.
        """
        flag = self.gradient_params.get("use_rust_lbfgsb", "auto")
        if (
            _mc is None
            or flag is False
            or not isinstance(search_data, _mc.SearchData)
        ):
            return None
        return model._kernel_family()

    def _optimize_genes_rust(self, family, model, search_data, x0_all, n_gene_cores):
        """Fit every gene with the Rust L-BFGS-B kernel of ``family``.

        Returns (param_estimates, klds) as arrays.  All families share the
        solver settings; they differ only in how the model is described.
        """
        n_genes = search_data.n_genes
        common = dict(
            x0_list=x0_all,
            lb=self.phys_lb.tolist(),
            ub=self.phys_ub.tolist(),
            fd_eps=1e-6,
            maxiter=self.gradient_params["max_iterations"],
            ftol=2.22e-9,
            gtol=1e-5,
            eps=1e-15,
            m_lbfgs=10,
            num_threads=None if n_gene_cores <= 1 else n_gene_cores,
            return_status=True,
        )
        quad = dict(fixed_quad_t=float(model.fixed_quad_T), quad_order=int(model.quad_order))
        if family == "2d":
            _ul = self.use_lengths if model.seq_model == "Poisson" else None
            use_unsp = _ul in ("unspliced", "both")
            use_spl = _ul in ("spliced", "both")
            if model.seq_model in ("Poisson", "Bernoulli"):
                base_samp = list(self.grid_point) if (use_unsp or use_spl) else self.regressor[0].tolist()
            else:
                base_samp = None
            name = "optimize_genes_2d_sd"
            out = _mc.optimize_genes_2d_sd(
                search_data, bio_model=model.bio_model, seq_model=model.seq_model,
                base_samp=base_samp, use_lengths_unspliced=use_unsp, use_lengths_spliced=use_spl,
                **quad, **common,
            )
        elif family == "amb":
            samp_list = None
            if model.seq_model == "Poisson":
                samp_list = [self.regressor[gi].tolist() for gi in range(n_genes)]
            name = "optimize_genes_2d_amb_sd"
            out = _mc.optimize_genes_2d_amb_sd(
                search_data, bio_model=model.bio_model, amb_model=model.amb_model,
                samp_list=samp_list, **quad, **common,
            )
        elif family == "protein":
            name = "optimize_genes_protein_bursty_sd"
            out = _mc.optimize_genes_protein_bursty_sd(
                search_data, fit_unspliced=bool(model.fit_unspliced),
                protein_limit=float(model.protein_limit),
                ode=_PROTEIN_ODE,
                **common,
            )
        elif family == "custom":
            net = model.network
            topo = net.rust_topology()
            name = "optimize_genes_custom_sd"
            out = _mc.optimize_genes_custom_sd(
                search_data, n_species=len(net.species),
                rxn_kinds=topo["rxn_kinds"], rxn_rate_idxs=topo["rxn_rate_idxs"],
                rxn_extra1=topo["rxn_extra1"], rxn_extra2=topo["rxn_extra2"],
                prod_sp=topo["prod_sp"], prod_st=topo["prod_st"], prod_off=topo["prod_off"],
                has_norm_rate=(net._norm_rate is not None), ode=_CUSTOM_ODE,
                **common,
            )
        else:
            raise ValueError(f"unknown Rust optimizer family {family!r}")
        params_arr, klds_arr, status_arr = out
        _warn_optimizer_status(status_arr, name)
        return np.asarray(params_arr), np.asarray(klds_arr)

    def _starting_points(self, n_genes):
        """Starting points for every gene: ``num_restarts`` draws from the
        restart box per gene, the first replaced by the moments estimate when
        ``init_pattern == "moments"``.  Built once per grid point so the scipy
        and Rust paths share identical restarts.  Returns nested lists
        ``[gene][restart][parameter]``."""
        n_restarts = self.gradient_params["num_restarts"]
        x0_all = []
        for gi in range(n_genes):
            x0 = np.random.rand(n_restarts, self.n_phys_pars) * self._restart_range + self._restart_lb
            if self.gradient_params["init_pattern"] == "moments":
                x0[0] = self.param_MoM[gi]
            x0_all.append(x0.tolist())
        return x0_all

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

        n_gene_cores = self.gradient_params.get("num_gene_cores", 1)

        # Rust L-BFGS-B: the whole fit runs in Rust with rayon parallelism over
        # genes; the default whenever it applies (gradient_params["use_rust_lbfgsb"]).
        rust_family = self._rust_optimizer_family(model, search_data)

        x0_all_shared = self._starting_points(search_data.n_genes)
        x0_all_shared = self._presearch_x0(model, search_data, x0_all_shared, n_gene_cores)

        if rust_family is not None:
            param_estimates, klds = self._optimize_genes_rust(
                rust_family, model, search_data, x0_all_shared, n_gene_cores
            )
        elif n_gene_cores != 1:
            # ThreadPoolExecutor + scipy L-BFGS-B per gene; the Rust PSS
            # fast-path fires automatically inside each thread.
            from concurrent.futures import ThreadPoolExecutor, as_completed
            n_workers = os.cpu_count() if n_gene_cores < 0 else n_gene_cores
            results_list = [None] * search_data.n_genes
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(
                        self.optimize_gene,
                        gene_index=i, model=model, search_data=search_data,
                        x0=x0_all_shared[i],
                    ): i
                    for i in range(search_data.n_genes)
                }
                for f in as_completed(futures):
                    results_list[futures[f]] = f.result()
            param_estimates, klds = zip(*results_list)
        else:
            param_estimates, klds = zip(
                *[
                    self.optimize_gene(gene_index=gi, model=model, search_data=search_data,
                                       x0=x0_all_shared[gi])
                    for gi in range(search_data.n_genes)
                ]
            )

        klds = np.asarray(klds)
        param_estimates = np.asarray(param_estimates)
        obj_func = klds.sum()

        t2 = time.time()
        d_time = t2 - t1

        return param_estimates, klds, obj_func, d_time

    def fit_all_genes(self, model, search_data, checkpoint=False):
        """Wraps iterate_over_genes and optionally stores the results on disk.

        Parameters
        ----------
        model: monod.cme_toolbox.CMEModel
            CME model used for inference.
        search_data: monod.extract_data.SearchData
            SearchData object with the data to fit.
        checkpoint: bool, optional
            If True, write result to disk as a crash-recovery checkpoint. Default False.

        Returns
        -------
        results: GridPointResults
        """

        search_out = self.iterate_over_genes(model=model, search_data=search_data)
        results = GridPointResults(
            *search_out,
            self.regressor,
            self.grid_point,
            self.point_index,
            self.inference_string,
        )
        if checkpoint:
            results.store_grid_point_results()
        return results

########################
## Helper functions
########################


########################
## Helper classes
########################
class GridPointResults:
    """Temporarily stores the fit parameters for a single grid point.

    Attributes
    ----------
    param_estimates: np.ndarray
        optimal biological parameter values for each gene, an n_genes x n_phys_pars array.
    klds: np.ndarray
        Kullback-Leibler divergence of the model for each gene at param_estimates.
    obj_func: float
        sum of klds; total error at the current grid point.
    d_time: float
        runtime in seconds.
    regressor: np.ndarray
        gene-specific technical variation parameter values at the current grid point.
        these values will be different for each gene if use_lengths is not None in the
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

    _SERIAL_KIND = "GridPointResults"
    _SERIAL_ATTRS = ("param_estimates", "klds", "obj_func", "d_time", "regressor",
                     "grid_point", "point_index", "inference_string")

    def to_dict(self):
        d = {"format_version": 1, "kind": self._SERIAL_KIND}
        d.update(_attr_dict(self, self._SERIAL_ATTRS))
        return d

    @classmethod
    def from_dict(cls, d):
        kind = str(d.get("kind", "GridPointResults"))
        if kind == "MEKMeansGridPointResults":
            from .mminference import GridPointResults as target
        else:
            target = GridPointResults
        obj = target.__new__(target)
        for n in target._SERIAL_ATTRS:
            if n in d:
                v = d[n]
                setattr(obj, n, v.item() if isinstance(v, np.generic) else v)
        return obj

    def store_grid_point_results(self):
        """Store the grid point results to disk as grid_point_X.h5 (structured HDF5, no pickle)."""
        try:
            grid_point_result_string = (
                self.inference_string + "/grid_point_" + str(self.point_index) + ".h5"
            )
            _write_h5(grid_point_result_string, self.to_dict())
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
        these values will be different for each gene if use_lengths is not None in the
        InferenceParameters constructor.
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
    def __init__(self, inference_parameters, search_data):
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
        try:
            self.gene_log_lengths = search_data.gene_log_lengths
        # Do not define an attribute for gene_lengths if non is given.
        except AttributeError:
            pass
            
        self.gene_names = search_data.gene_names

        self.param_estimates = []
        self.klds = []
        self.obj_func = []
        self.d_time = []
        self.regressor = []

    def aggregate_grid_points(self, grid_point_results=None):
        """This helper method concatenates all of the grid point results.

        Parameters
        ----------
        grid_point_results: dict or None, optional
            If provided, a dict mapping point_index -> GridPointResults for in-memory
            aggregation. If None or incomplete, falls back to reading .gp files from disk.
        """
        for point_index in range(self.sp.n_grid_points):
            if grid_point_results is not None and point_index in grid_point_results:
                self._append_from_object(grid_point_results[point_index])
            else:
                self.append_grid_point(point_index)
        self.clean_up(remove_files=(grid_point_results is None))

    def _append_from_object(self, gp):
        """This helper method updates the result attributes from an in-memory GridPointResults object.

        Parameters
        ----------
        gp: GridPointResults
            in-memory grid point results.
        """
        self.param_estimates += [gp.param_estimates]
        self.klds += [gp.klds]
        self.obj_func += [gp.obj_func]
        self.d_time += [gp.d_time]
        self.regressor += [gp.regressor]

    def append_grid_point(self, point_index):
        """This helper method updates the result attributes from a GridPointResult object stored on disk.

        Parameters
        ----------
        point_index: int
            index of the grid point results to load from disk.
        """
        grid_point_result_string = (
            self.inference_string + "/grid_point_" + str(point_index) + ".h5"
        )
        gp = GridPointResults.from_dict(_read_h5(grid_point_result_string))
        self._append_from_object(gp)

    def clean_up(self, remove_files=False):
        """This helper method removes temporary files and finalizes the SearchResults object.

        The GridPointResult objects are optionally erased from disk, the attributes are converted to
        np.ndarrays, and a directory for analysis figures is created (only if inference_string exists).

        Parameters
        ----------
        remove_files: bool, optional
            If True, delete .gp checkpoint files from disk. Default False.
        """
        if remove_files:
            for point_index in range(self.sp.n_grid_points):
                gp_path = self.inference_string + "/grid_point_" + str(point_index) + ".h5"
                if os.path.exists(gp_path):
                    os.remove(gp_path)
            log.info("All grid point data cleaned from disk.")
        self.param_estimates = np.asarray(self.param_estimates)
        self.klds = np.asarray(self.klds)
        self.obj_func = np.asarray(self.obj_func)
        self.d_time = np.asarray(self.d_time)
        self.regressor = np.asarray(self.regressor)

        # Created lazily by _savefig the first time a figure is actually saved.
        self.analysis_figure_string = self.inference_string + "/analysis_figures"

    @property
    def _result_filename(self):
        return self.inference_string + "/grid_scan_results.h5"

    # ------------------------------------------------------------------
    # Structured (pickle-free) serialisation
    # ------------------------------------------------------------------
    _SERIAL_KIND = "SearchResults"
    _SERIAL_ATTRS = (
        "inference_string", "n_genes", "n_cells", "gene_names", "gene_log_lengths",
        "param_estimates", "klds", "obj_func", "d_time", "regressor",
        "samp_optimum_ind", "samp_optimum", "phys_optimum", "regressor_optimum",
        "rejected_genes", "pval", "csq", "hellinger", "rejection_index", "sigma",
        "sigma_fail", "sigma_flat_direction", "sigma_index",
        "analysis_figure_string", "full_result_string",
    )

    def to_dict(self):
        """Nested dict of arrays/scalars describing this result (h5ad/HDF5 friendly)."""
        d = {"format_version": 1, "kind": self._SERIAL_KIND, "sp": self.sp.to_dict()}
        d.update(_attr_dict(self, self._SERIAL_ATTRS))
        return d

    @classmethod
    def from_dict(cls, d):
        """Rebuild a SearchResults (or meK-Means subclass) from :meth:`to_dict` output."""
        kind = str(d.get("kind", "SearchResults"))
        if kind == "MEKMeansSearchResults":
            from .mminference import SearchResults as target
        else:
            target = SearchResults
        obj = target.__new__(target)
        obj.sp = InferenceParameters.from_dict(d["sp"])
        obj.model = obj.sp.model
        for n in ("param_estimates", "klds", "obj_func", "d_time", "regressor"):
            setattr(obj, n, np.zeros(0))
        for n in target._SERIAL_ATTRS:
            if n in d:
                v = d[n]
                setattr(obj, n, v.item() if isinstance(v, np.generic) else v)
        if "aic" in d and "aic_per_cell" not in d and "aic_per_cell" in target._SERIAL_ATTRS:
            obj.aic_per_cell = d["aic"]   # dicts written before the rename
        if hasattr(obj, "gene_names"):
            obj.gene_names = np.asarray(obj.gene_names).astype(str)
        return obj

    def store_on_disk(self):
        """This helper method attempts to store the SearchResults object to disk.

        Returns
        -------
        full_result_string: str
            file location.
        """
        try:
            full_result_string = self._result_filename
            _write_h5(full_result_string, self.to_dict())
            log.info("Grid scan results stored to {}.".format(full_result_string))
        except Exception:
            log.error(
                "Grid scan results could not be stored to {}.".format(
                    full_result_string
                ),
                exc_info=True,
            )
        self.full_result_string = full_result_string
        return full_result_string

    ####################################
    #         Analysis methods         #
    ####################################

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

    def _savefig(self, filename, dpi=450):
        """Save the current figure into the analysis figure directory if it exists.

        The directory is created only when the search was run with save=True or
        checkpoint=True; otherwise the figure is left on screen and a message is
        logged instead of raising FileNotFoundError.
        """
        figdir = getattr(self, "analysis_figure_string", None)
        run_dir = os.path.dirname(figdir) if figdir else None
        if run_dir and os.path.isdir(run_dir):
            os.makedirs(figdir, exist_ok=True)
            fig_string = figdir + "/" + filename
            plt.savefig(fig_string, dpi=dpi)
            log.info("Figure stored to {}.".format(fig_string))
        else:
            log.info("Run directory %s does not exist (run with save=True to create it); %s not saved.", run_dir, filename)

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

        # Cell width per axis; a one-point axis (gridsize 1) gets a nominal width
        # instead of the 0/0 that would make the extent NaN.
        span = np.asarray(self.sp.samp_ub, dtype=float) - np.asarray(self.sp.samp_lb, dtype=float)
        steps = np.asarray(self.sp.gridsize, dtype=float) - 1
        with np.errstate(divide="ignore", invalid="ignore"):
            dx = np.where(steps > 0, span / np.maximum(steps, 1), np.nan)
        dx[~np.isfinite(dx) | (dx < 1e-10)] = 0.1
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
            self._savefig("landscape.png")

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
        self._savefig("parameter_marginals.png")

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
        self._savefig("kldiv.png")

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

        _sd_is_rust = _mc is not None and isinstance(search_data, _mc.SearchData)
        if (
            _sd_is_rust
            and self.model.bio_model in _MODELS_2D
            and self.model.amb_model == "None"
            and self.model.quad_method == "fixed_quad"
            and self.model.seq_model in ("None", "Poisson", "Bernoulli")
        ):
            samp_list = None
            if self.model.seq_model in ("Poisson", "Bernoulli"):
                samp_list = [
                    (self.regressor_optimum[gi].tolist()
                     if self.regressor_optimum[gi] is not None else None)
                    for gi in range(self.n_genes)
                ]
            csq, pval, hellinger = _mc.chisquare_testing_2d_sd(
                search_data,
                self.model.bio_model,
                [row.tolist() for row in self.phys_optimum],
                samp_list,
                float(self.model.fixed_quad_T),
                int(self.model.quad_order),
                int(self.model.get_num_params()),
                float(grouping_thr),
                float(EPS),
                self.model.seq_model,
            )
            backend = "Rust"
        else:
            backend = "Python"
            csqarr = []
            hellinger = []
            for gene_index in range(self.n_genes):
                expect_freq = self.gene_pss(search_data, gene_index) * search_data.n_cells
                # expected_freq[expected_freq < EPS] = EPS
                # expected_freq /= expected_freq.sum()
                # PROPOSAL = search_data.n_cells * expected_freq

                counts = np.concatenate(
                    (search_data.n_cells * search_data.hist[gene_index][1], [0])
                )
                number_of_modalities = len(search_data.hist[gene_index][0][0])
                if number_of_modalities == 2:
                    expect_freq = expect_freq[
                        search_data.hist[gene_index][0][:, 0],
                        search_data.hist[gene_index][0][:, 1],
                    ]
                elif number_of_modalities == 3:
                    expect_freq = expect_freq[
                        search_data.hist[gene_index][0][:, 0],
                        search_data.hist[gene_index][0][:, 1],
                        search_data.hist[gene_index][0][:, 2]
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
                # Same rule as the Rust path (group_bins): accumulate microstates until
                # both observed and expected counts reach grouping_thr, then close a
                # bin; the tail is merged into the last bin, or forms the only bin if
                # none ever closed (tiny datasets).
                for i in range(len(counts)):
                    bins.append(bin_ind)
                    run_bin_obs += counts[i]
                    run_bin_exp += expect_freq[i]
                    if min(run_bin_obs, run_bin_exp) >= grouping_thr:
                        bin_ind += 1
                        bin_obs.append(run_bin_obs)
                        bin_exp.append(run_bin_exp)
                        run_bin_obs = 0
                        run_bin_exp = 0
                bins = np.asarray(bins)
                if not bin_obs:
                    observed = np.asarray([run_bin_obs], dtype=float)
                    proposed = np.asarray([run_bin_exp], dtype=float)
                else:
                    observed = np.asarray(bin_obs, dtype=float)
                    proposed = np.asarray(bin_exp, dtype=float)
                    observed[-1] += run_bin_obs
                    proposed[-1] += run_bin_exp
                bins[bins >= len(observed)] = len(observed) - 1

                for b_ in range(len(bin_obs)):
                    assert np.isclose(observed[b_], counts[bins == b_].sum())
                    assert np.isclose(proposed[b_], expect_freq[bins == b_].sum())
                assert np.isclose(observed.sum(), search_data.n_cells)
                assert np.isclose(proposed.sum(), search_data.n_cells)
                assert np.isclose(search_data.n_cells, counts.sum())
                assert np.isclose(search_data.n_cells, expect_freq.sum())


                # Ensure the sums of f_obs and f_exp match
                f_obs_sum = observed.sum()
                f_exp_sum = proposed.sum()
            
                # Check if normalization is needed
                if not np.isclose(f_obs_sum, f_exp_sum, rtol=1e-9):
                    # Print a warning/info message
                    warnings.warn(
                        f"The sums of observed and expected frequencies differ by {f_obs_sum - f_exp_sum:.2e}. "
                        "Normalizing expected frequencies to match observed frequencies.",
                        UserWarning,
                    )
                    # Normalize f_exp
                    proposed = proposed * (f_obs_sum / f_exp_sum)


                # chi-squared takes only the number of biological parameters?
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

        # the decision, shared by both backends
        if bonferroni:
            threshold /= self.n_genes
        log.info('P-value threshold: %s (Bonferroni: %s), Hellinger threshold: %s', threshold, bonferroni, hellinger_thr)
        self.rejected_genes = pval < threshold
        if use_hellinger:
            self.rejected_genes = self.rejected_genes & (hellinger > hellinger_thr)
        if reject_at_bounds:
            bound_range = self.sp.phys_ub - self.sp.phys_lb
            lb = self.sp.phys_lb + bound_range * bound_thr
            ub = self.sp.phys_ub - bound_range * bound_thr
            self.rejected_genes = self.rejected_genes | ((self.phys_optimum < lb) | (self.phys_optimum > ub)).any(1)
        self.pval = pval
        self.csq = csq
        self.hellinger = hellinger
        self.rejection_index = self.samp_optimum_ind

        if viz:
            plt.hist(csq)
            plt.xlabel("Chi-square statistic")
            plt.ylabel("# genes")
            self._savefig("chisquare.png")

        t2 = time.time()
        log.info(
            "Chi-square computation complete (%s). Rejected %.0f genes out of %.0f. Runtime: %.1f seconds.",
            backend, np.sum(self.rejected_genes), self.n_genes, t2 - t1,
        )
        return (csq, pval, hellinger)

    def _search_data_for(self, search_data):
        """The search data these results describe.

        Subclasses whose fit covers a subset of the cells (meK-Means clusters)
        override this; every method that takes ``search_data`` calls it first.
        """
        return search_data

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

        Two methods, chosen by ``compute_sigma``: ``"gradient"`` (default) takes
        central differences of the gradient returned by
        ``CMEModel.eval_model_kld_and_grad`` and symmetrises, 2 n gradient
        evaluations per gene; ``"numdifftools"`` uses Richardson extrapolation
        on the KLD itself, about 150 evaluations per gene, with every probe kept
        within a fifth of a decade of the optimum.
        """
        gene_index, search_data = inputs
        search_data = self._search_data_for(search_data)
        method = getattr(self, "_hess_method", "gradient")
        limits = search_data.M[:, gene_index]
        samp = self.regressor_optimum[gene_index]
        data = search_data.hist[gene_index]
        x0 = np.asarray(self.phys_optimum[gene_index], dtype=float)

        # The protein and custom-network integrators adapt their steps; every
        # stencil point replays the optimum's schedule so the integrator's
        # tolerance does not enter the differences divided by the small step.
        with self.model.pinned_schedule(x0, limits, samp):
            if method == "gradient":
                h = float(getattr(self, "_hess_step", 1e-2))
                n = x0.size
                J = np.empty((n, n))
                for j in range(n):
                    e = np.zeros(n); e[j] = h
                    _, g_plus = self.model.eval_model_kld_and_grad(x0 + e, limits, samp, data)
                    _, g_minus = self.model.eval_model_kld_and_grad(x0 - e, limits, samp, data)
                    J[:, j] = (np.asarray(g_plus) - np.asarray(g_minus)) / (2 * h)
                return 0.5 * (J + J.T)

            import numdifftools  # this will fail if numdifftools has not been evaluted.
            # numdifftools' default step sequence starts about two decades from the
            # optimum, where custom-network evaluations can take minutes. Richardson
            # extrapolation over 0.1 / 2**k (k = 0..7) decades keeps every probe
            # within a fifth of a decade and gives the same Hessian for smooth KLDs.
            Hfun = numdifftools.Hessian(
                lambda x: self.model.eval_model_kld(p=x, limits=limits, samp=samp, data=data),
                step=numdifftools.MaxStepGenerator(base_step=0.1, num_steps=8, step_ratio=2.0),
            )
            return Hfun(x0)

    def compute_sigma(self, search_data, num_cores=1, max_relative_sigma=0.5,
                      method="gradient", step=1e-2):
        """Estimate uncertainty in biological parameter values.

        This method iterates over genes, computes the Fisher information matrix by inverting
        the Hessian with respect to biological parameters, and reports the local estimate
        of standard errors of the parameters.

        This method is fairly computationally demanding, so parallelization is recommended.

        A gene is flagged in ``sigma_fail`` when its Hessian is not finite, is not positive
        definite (the fit is not at a minimum, or the likelihood is exactly flat in some
        direction), or gives a standard error wider than ``max_relative_sigma`` times the
        width of the parameter box (the data do not constrain that parameter). The standard
        errors of flagged genes are replaced with the mean over the other genes so that
        downstream steps keep working, and an :class:`IdentifiabilityWarning` reports how
        many genes were flagged and along which direction their likelihood is flattest.

        Parameters
        ----------
        search_data: monod.extract_data.SearchData
            SearchData object with the fit data.
        num_cores: int, optional
            number of cores to use for parallelization over genes.
        max_relative_sigma: float, optional
            flag a gene when any standard error exceeds this fraction of the parameter's
            bound width.
        method: str, optional
            ``"gradient"`` (default): Hessian by central differences of the model's
            gradient, 2 n gradient evaluations per gene. ``"numdifftools"``: Richardson
            extrapolation on the KLD, about 150 evaluations per gene; the previous default.
        step: float, optional
            log10 step for the gradient-difference method.

        Sets
        ----
        sigma: float np.ndarray
            the standard error of the parameter maximum likelihood estimate at the sampling
            parameter optimum.
            a n_genes x n_phys_pars array.
        sigma_fail: bool np.ndarray
            per-gene flag; True where ``sigma`` is a fallback rather than an estimate.
        sigma_flat_direction: float np.ndarray
            the eigenvector of the smallest Hessian eigenvalue per gene (n_genes x
            n_phys_pars, unit length): the direction the data constrain least. NaN where
            the Hessian was not finite.
        sigma_index: int
            the grid point at which the Fisher information procedure was performed to generate
            the sigma attribute.

        """
        if method not in ("gradient", "numdifftools"):
            raise ValueError(f"method must be 'gradient' or 'numdifftools', got {method!r}")
        self._hess_method = method
        self._hess_step = float(step)
        log.info("Computing local Hessian (%s).", method)
        t1 = time.time()
        if num_cores > 1:
            log.info("Starting parallelized Hessian computation.")
            hess = parallelize(
                function=self.par_fun_hess,
                iterable=zip(range(self.n_genes), itertools.repeat(search_data, self.n_genes)),
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
                for x in zip(range(self.n_genes), itertools.repeat(search_data, self.n_genes))
            ]
            log.info("Non-parallelized Hessian computation complete.")
        hess = np.asarray(hess, dtype=float)

        n_par = self.sp.n_phys_pars
        fail = np.zeros(self.n_genes, dtype=bool)
        reason = [""] * self.n_genes
        sigma = np.zeros((self.n_genes, n_par))
        flat_dir = np.full((self.n_genes, n_par), np.nan)
        width = np.asarray(self.sp.phys_ub, float) - np.asarray(self.sp.phys_lb, float)

        for gene_index in range(self.n_genes):
            H = hess[gene_index]
            if not np.all(np.isfinite(H)):
                fail[gene_index] = True
                reason[gene_index] = "non-finite Hessian"
                continue
            evals, evecs = np.linalg.eigh(0.5 * (H + H.T))
            flat_dir[gene_index] = evecs[:, 0]
            if evals[0] <= 0:
                fail[gene_index] = True
                reason[gene_index] = "Hessian not positive definite"
                continue
            hess_inv = np.linalg.inv(H)
            sig = np.sqrt(np.maximum(np.diag(hess_inv), 0.0)) / np.sqrt(self.n_cells)
            sigma[gene_index] = sig
            if np.any(~np.isfinite(sig)):
                fail[gene_index] = True
                reason[gene_index] = "non-finite standard error"
            elif np.any(sig > max_relative_sigma * width):
                fail[gene_index] = True
                reason[gene_index] = "standard error wider than the parameter box"

        if np.any(fail):
            for gene_index in np.flatnonzero(fail):
                log.info("Gene %d: %s; standard errors replaced with the mean over the other genes.",
                         gene_index, reason[gene_index])
            if np.all(fail):
                sigma[:] = np.nan
            else:
                sigma[fail, :] = sigma[~fail, :].mean(0)
            self._warn_sigma_fail(fail, flat_dir, reason)

        self.sigma = sigma
        self.sigma_fail = fail
        self.sigma_flat_direction = flat_dir
        self.sigma_index = self.samp_optimum_ind  # mostly for debug

        t2 = time.time()
        log.info(
            "Standard error of the MLE computation complete. Runtime: {:.1f} seconds.".format(
                t2 - t1
            )
        )

    def _warn_sigma_fail(self, fail, flat_dir, reason):
        """One IdentifiabilityWarning summarising the genes flagged by compute_sigma."""
        from collections import Counter
        names = self.model.param_names()
        n_fail = int(np.sum(fail))
        counts = Counter(reason[i] for i in np.flatnonzero(fail))
        why = "; ".join(f"{c} {r}" for r, c in counts.most_common())
        msg = (f"{n_fail} of {self.n_genes} genes have unreliable standard errors ({why}). "
               "Their entries in sigma are the mean over the other genes; see sigma_fail and "
               "sigma_flat_direction.")
        dirs = flat_dir[fail]
        dirs = dirs[np.all(np.isfinite(dirs), axis=1)]
        if len(dirs):
            # sign-invariant average direction: top eigenvector of the scatter matrix
            w, v = np.linalg.eigh(dirs.T @ dirs)
            msg += (f" Their likelihood is flattest along {_format_direction(dict(zip(names, v[:, -1])))}"
                    f"{' (this is the protein k_p/gamma_p valley)' if self.model.bio_model == 'ProteinBursty' else ''}.")
        warnings.warn(msg, IdentifiabilityWarning, stacklevel=3)

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
        try: 
            gene_log_lengths = self.gene_log_lengths
        except AttributeError:
            log.info('No gene lengths given, length-dependence cannot be plotted. Function exiting')
            return 
            
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
        self._savefig("length_dependence.png")

    def gene_pss(self, search_data, i_, marg="joint"):
        """The fitted PMF of gene ``i_`` at the sampling optimum over the data's
        grid ``search_data.M[:, i_]``: the joint PMF for ``marg="joint"``, the
        marginal of one modality (``'unspliced'``, ``'spliced'``, ...) otherwise."""
        limits = np.array(search_data.M[:, i_])
        if marg != "joint":
            lind = _resolve_marg(marg, self.model.model_modalities)
            keep = limits[lind]
            limits[:] = 1
            limits[lind] = keep
        samp = None if self.model.seq_model == "None" else self.regressor_optimum[i_]
        return np.squeeze(self.model.eval_model_pss(self.phys_optimum[i_], limits, samp))

    def plot_gene_distributions(
        self,
        search_data,
        sz=(5, 5),
        figsize=(10, 10),
        marg="joint",
        logscale=None,
        title=True,
        show_axes=False,
        genes_to_plot=None,
        savefig=True,
    ):
        """Plot the gene count distributions and their fits at the sampling parameter optimum.

        Parameters
        ----------
        search_data: monod.extract_data.SearchData
            SearchData object with the fit data.
        sz: tuple of ints, optional
            dimensions of the figure subplot grid.
        figsize: tuple of floats, optional
            figure dimensions.
        marg: str, optional
            if 'unspliced', 'spliced', 'protein' etc., plot the marginal for this modality.
            if 'joint': plot the bivariate distribution of the two modalities (for the
            protein model, of spliced RNA and protein).
        logscale: None or bool, optional
            whether to plot probabilities or log-probabilities.
            by default, True for 'joint', False for marginals.
        title: bool, optional
            whether to report the gene name in each subplot title.
        show_axes: bool, optional
            whether to keep the ticks of the outer panels and label the figure axes.
        genes_to_plot: bool or int np.ndarray or None, optional
            if array, which genes to plot.
            if None, plot by internal order.
        savefig: bool, optional
            whether to save the figure to disk.
        """
        search_data = self._search_data_for(search_data)
        joint = marg == "joint"
        if logscale is None:
            logscale = joint
        modalities = list(self.model.model_modalities)
        if joint and len(modalities) not in (2, 3):
            raise ValueError("Joint distribution plot only implemented for 2 or 3 modalities")
        # the two axes of a joint panel: both modalities, or spliced and protein for the protein model
        ax_x, ax_y = (0, 1) if len(modalities) == 2 else (1, 2)

        fig1, ax1 = plt.subplots(nrows=sz[0], ncols=sz[1], figsize=figsize)
        axes = np.atleast_1d(ax1).ravel()
        genes_to_plot = _genes_to_plot(genes_to_plot, self.n_genes, np.prod(sz))

        for j_, (ax, i_) in enumerate(zip(axes, genes_to_plot)):
            Pa = self.gene_pss(search_data, i_, marg)
            if joint:
                if len(modalities) == 3:
                    Pa = Pa.sum(axis=0)
                if logscale:
                    Pa = np.log10(np.maximum(Pa, 1e-10))
                ax.imshow(Pa.T, aspect="auto", cmap="summer")
                ax.invert_yaxis()
                jitter = np.random.randn(2, search_data.n_cells) * 0.1
                ax.scatter(*search_data.layers[ax_x:ax_y + 1, :, i_] + jitter, c="k", s=1, alpha=0.1)
                ax.set_xlim([-0.5, search_data.M[ax_x, i_] - 1.5])
                ax.set_ylim([-0.5, search_data.M[ax_y, i_] - 1.5])
            else:
                plot_hist_and_fit(ax, search_data, i_, Pa, marg, modalities=modalities)
                if logscale:
                    ax.set_yscale("log")
            if title:
                titlestr = self.gene_names[i_]
                if hasattr(self, "rejected_genes") and self.rejected_genes[i_]:
                    titlestr += " (rej.)"
                ax.set_title(titlestr, fontdict={"fontsize": 9})
            row, col = divmod(j_, sz[1])
            if not (show_axes and col == 0):
                ax.set_yticks([])
            if not (show_axes and row == sz[0] - 1):
                ax.set_xticks([])
        if show_axes:
            fig1.supxlabel(modalities[ax_x])
            fig1.supylabel(modalities[ax_y])
        fig1.tight_layout(pad=0.02, rect=[0.05, 0.05, 0.98, 0.98])

        if savefig:
            self._savefig("gene_distributions_{}.png".format(marg))
        return fig1

    # the next two functions are useful for model selection, but are not currently in use.
    def get_logL(self, search_data, n_cells=None, EPS=1e-20, offs=0):
        """
        This method calculates the log-likelihood for all genes at the sampling parameter optimum.

        Parameters
        ----------
        search_data: a SearchData instance.
        n_cells: int, optional. Defaults to search_data.n_cells.
        EPS: probability rounding parameter -- anything below this is rounded to EPS.

        Output:
        logL: a vector of size n_genes containing model log-likelihoods.
        """
        search_data = self._search_data_for(search_data)
        if n_cells is None:
            n_cells = search_data.n_cells
        model = self.model
        if (
            _HAS_RUST
            and offs == 0
            and isinstance(search_data, _mc.SearchData)
            and model._kernel_family() == "2d"
        ):
            phys_optimum = [p.tolist() for p in self.phys_optimum]
            if model.seq_model == "None":
                samp_list = None
            else:
                samp_list = [
                    r.tolist() if r is not None else None
                    for r in self.regressor_optimum
                ]
            return np.array(_mc.eval_logl_2d_sd(
                search_data,
                model.bio_model,
                phys_optimum,
                float(n_cells),
                samp_list,
                float(model.fixed_quad_T),
                int(model.quad_order),
                float(EPS),
                model.seq_model,
            ))

        logL = np.zeros(self.n_genes)
        for gene_index in range(self.n_genes):
            logL[gene_index] = model.eval_model_logL(
                p=self.phys_optimum[gene_index],
                limits=search_data.M[:, gene_index] + offs,
                samp=self.regressor_optimum[gene_index],
                data=search_data.hist[gene_index],
                n_cells=n_cells,
                EPS=EPS,
            )
        return logL

def reject_genes(adata, viz=False, EPS=1e-15, threshold=0.05, bonferroni=True,
                 reject_at_bounds=True, bound_thr=0.01, grouping_thr=5,
                 use_hellinger=True, hellinger_thr=0.05, mek_means=None,
                 save_csq=True, save_pval=True, save_hellinger=True):
    """Chi-square goodness-of-fit testing on a fitted AnnData; flags poorly fit genes.

    Works on the AnnData returned by :func:`perform_inference`.  The search data
    is rebuilt from the layers and the stored result(s) are updated in place, so
    downstream calls with ``discard_rejected=True`` (e.g. ``analysis.DE_parameters``)
    see the rejection flags.

    Adds to ``adata.var``: ``rejected_genes`` plus ``csq``, ``pval``, ``hellinger``
    (prefixed ``c<k>_`` per cluster for a meK-Means run).

    Parameters
    ----------
    adata: AnnData from perform_inference.
    mek_means: tuple or None
        pass the same ``mek_means_params`` used for the fit (anything truthy
        selects the per-cluster path).
    Other arguments are forwarded to ``SearchResults.chisquare_testing``.

    Returns
    -------
    adata (modified in place).
    """
    kw = dict(viz=viz, EPS=EPS, threshold=threshold, bonferroni=bonferroni,
              reject_at_bounds=reject_at_bounds, bound_thr=bound_thr,
              grouping_thr=grouping_thr, use_hellinger=use_hellinger,
              hellinger_thr=hellinger_thr)
    search_data = searchdata_from_adata(adata)

    if not mek_means:
        if 'search_result' not in adata.uns:
            raise KeyError("adata.uns has no 'search_result'; for a meK-Means run pass mek_means=<params>.")
        sr = search_result_from_uns(adata.uns['search_result'])
        csq, pval, hellinger = sr.chisquare_testing(search_data, **kw)
        adata.var['rejected_genes'] = np.asarray(sr.rejected_genes, dtype=bool)
        if save_csq:
            adata.var['csq'] = np.asarray(csq)
        if save_pval:
            adata.var['pval'] = np.asarray(pval)
        if save_hellinger:
            adata.var['hellinger'] = np.asarray(hellinger)
        adata.uns['search_result'] = sr.to_dict()
        log.info("Rejected %d/%d genes; flags stored in adata.var['rejected_genes'].",
                 int(np.sum(sr.rejected_genes)), sr.n_genes)
    else:
        if 'search_result_list' not in adata.uns:
            raise KeyError("adata.uns has no 'search_result_list'; this AnnData is not from a meK-Means run.")
        srs = search_result_list_from_uns(adata.uns['search_result_list'])
        for sr in srs:
            k = int(sr.assigns)
            csq, pval, hellinger = sr.chisquare_testing(search_data, **kw)   # subsets to the cluster itself
            adata.var[f'c{k}_rejected_genes'] = np.asarray(sr.rejected_genes, dtype=bool)
            if save_csq:
                adata.var[f'c{k}_csq'] = np.asarray(csq)
            if save_pval:
                adata.var[f'c{k}_pval'] = np.asarray(pval)
            if save_hellinger:
                adata.var[f'c{k}_hellinger'] = np.asarray(hellinger)
        adata.uns['search_result_list'] = search_result_list_to_uns(srs)
        log.info("Rejection flags stored per cluster in adata.var['c<k>_rejected_genes'].")
    return adata


def get_AIC_mek_means(search_result_list, search_data, AIC_EPS=1e-20, AIC_offs=0):
    """Per-gene AIC of a meK-Means fit (lower is better).

    Each gene carries one parameter set per surviving cluster, so the penalty
    is 2 * n_phys_pars * K_final, where K_final = len(search_result_list) is
    the number of clusters that received cells.  The log-likelihood is the sum
    over clusters of the gene's log-likelihood on that cluster's cells.  (The
    paper's whole-model AIC additionally counts K_final - 1 mixing weights,
    which cannot be attributed to individual genes; see
    ``mminference.mek_means_aic`` for the whole-model score.)
    """
    return sum(aic_values(sr, search_data, EPS=AIC_EPS, offs=AIC_offs) for sr in search_result_list)


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
    """Helper function to safely parallelize computations using joblib.

    Inputs a single-parameter function and an iterable, requests a number of cores,
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
    x: list
        result of applying function to each item in iterable.
    """
    from joblib import Parallel, delayed
    x = []
    try:
        items = list(tqdm(iterable, total=num_entries) if use_tqdm else iterable)
        # Thread backend: items may contain a SearchData, which is shared in
        # memory rather than pickled to worker processes.
        x = Parallel(n_jobs=num_cores, prefer="threads")(delayed(function)(item) for item in items)
        log.info(completion_message)
    except KeyboardInterrupt:
        log.warning(termination_message)
    except ValueError:
        log.warning(error_message)
    return x


_MARG_ALIASES = {"nascent": "unspliced", "mature": "spliced", "U": "unspliced", "S": "spliced"}


def _resolve_marg(marg, layer_names):
    """Map a marginal name (or legacy alias) onto an index into layer_names."""
    layer_names = list(layer_names)
    if marg in layer_names:
        return layer_names.index(marg)
    alias = _MARG_ALIASES.get(marg)
    if alias in layer_names:
        return layer_names.index(alias)
    raise ValueError(f"marg={marg!r} is not one of the modalities {layer_names}")


def _genes_to_plot(genes_to_plot, n_genes, n_axes):
    """Gene indices for a grid of ``n_axes`` panels: internal order when
    ``genes_to_plot`` is None, the indices of a boolean mask, or an index
    array; truncated to the panels and genes available."""
    if genes_to_plot is None:
        genes_to_plot = np.arange(n_axes)
    genes_to_plot = np.asarray(genes_to_plot)
    if genes_to_plot.dtype == bool:
        genes_to_plot = np.flatnonzero(genes_to_plot)
    return genes_to_plot[: min(int(n_genes), int(n_axes))]


def plot_hist_and_fit(
    ax1,
    search_data,
    i_,
    Pa,
    marg="unspliced",
    modalities=None,
    facecolor=aesthetics["hist_face_color"],
    fitcolor=aesthetics["hist_fit_color"],
    facealpha=aesthetics["hist_face_alpha"],
    linestyle=aesthetics["linestyle"],
):
    """Plot one gene's marginal count histogram and its fitted PMF.

    Parameters
    ----------
    ax1: matplotlib.axes.Axes
        axes to plot into.
    search_data: SearchData (Rust ``monod_core.SearchData`` or the Python fallback)
        holds ``layers`` (n_layers x n_cells x n_genes) and ``M`` (n_layers x n_genes).
    i_: int
        gene index to plot.
    Pa: float np.ndarray
        univariate PMF for this gene and modality, typically from CMEModel.eval_model_pss
        with the other modalities' limits set to 1.
    marg: str, optional
        modality to plot ('unspliced', 'spliced', 'protein', ...; 'nascent'/'mature' accepted).
    modalities: list of str, optional
        modality names in layer order; defaults to ``search_data.layer_names``.
    facecolor, fitcolor, facealpha, linestyle: matplotlib styling.
    """
    layer_names = modalities if modalities is not None else getattr(search_data, "layer_names", None)
    if layer_names is None:
        raise ValueError("plot_hist_and_fit needs modality names: pass modalities=model.model_modalities")
    lind = _resolve_marg(marg, layer_names)
    counts = np.asarray(search_data.layers[lind])[:, i_]
    M = int(search_data.M[lind, i_])
    Pa = np.asarray(Pa).ravel()
    ax1.hist(
        counts,
        bins=np.arange(M + 1) - 0.5,
        density=True,
        color=facecolor,
        alpha=facealpha,
    )
    ax1.plot(np.arange(min(M, Pa.size)), Pa[:M], color=fitcolor, linestyle=linestyle)
    ax1.set_xlim([-0.5, counts.max() + 2.5])
