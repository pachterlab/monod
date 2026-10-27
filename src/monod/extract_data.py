import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
from datetime import datetime
import pytz
import anndata
import numpy as np
import matplotlib.pyplot as plt
import time
import anndata as ad
import os
import warnings
import scipy
from scipy import sparse
import pandas as pd
code_ver_global = "029"  # bumping up version April 2024

# ---------------------------------------------------------------------------
# uns legacy reader
# ---------------------------------------------------------------------------
# Older monod versions stored pickled objects in adata.uns; everything is a
# nested dict now (to_dict/from_dict).  _uns_unpack only reads the old format.

def _uns_unpack(v):
    """Restore an object previously packed with the (legacy) pickle format.

    Returns *v* unchanged unless it is a bytes / uint8 blob.  This unpickles,
    so it executes code embedded in the bytes: only call it on ``uns`` entries
    from h5ad files you trust.  Files written by current monod versions store
    structured dicts instead and never hit this path.
    """
    if isinstance(v, (bytes, np.ndarray)) and (
        isinstance(v, bytes) or v.dtype == np.uint8
    ):
        try:
            log.warning("Loading a legacy pickled object from adata.uns; re-run extract_data/perform_inference to store it in the structured format.")
            return pickle.loads(bytes(v))
        except Exception:
            pass
    return v


def model_from_uns(v):
    """Return the CMEModel stored in ``adata.uns['model']`` (structured dict or legacy pickle)."""
    from .cme_toolbox import CMEModel
    if hasattr(v, "keys"):
        return CMEModel.from_dict(v)
    return _uns_unpack(v)

def _build_sampling_grid(samp_lb, samp_ub, gridsize):
    """Build a flat meshgrid over the sampling-parameter space.

    Returns (grid_values, sampl_vals, n_grid_points) where grid_values is a
    list of flattened per-axis arrays, sampl_vals is a list of (v0, v1, ...)
    tuples, and n_grid_points is the total number of points.
    """
    linspaces = [np.linspace(samp_lb[i], samp_ub[i], gridsize[i]) for i in range(len(gridsize))]
    grid_values = np.meshgrid(*linspaces, indexing="ij")
    grid_values = [g.flatten() for g in grid_values]
    sampl_vals = list(zip(*grid_values))
    return grid_values, sampl_vals, len(grid_values[0])

import logging
from scipy.sparse import csr_matrix
from scipy.sparse import issparse


def to_dense(arr):
    """Return ``arr`` as a dense numpy array, whether it arrived sparse or dense."""
    return arr.toarray() if issparse(arr) else np.asarray(arr)

# Library logger.  monod does not configure handlers or touch the root logger;
# messages propagate to whatever logging setup the application has.  To see
# progress messages: ``logging.getLogger("monod").setLevel(logging.INFO)`` after
# ``logging.basicConfig()``.
log = logging.getLogger("monod")

def _plain_string_index(adata):
    """Make obs/var names and string columns plain object dtype so the AnnData writes to h5ad.

    pandas may give h5ad-loaded indices and text columns the nullable
    ``string`` dtype, which anndata refuses to write unless
    ``allow_write_nullable_strings`` is set.  Converting to object dtype keeps
    the AnnData writable with default settings.
    """
    adata.obs_names = pd.Index(np.asarray(adata.obs_names, dtype=object).astype(str), dtype=object)
    adata.var_names = pd.Index(np.asarray(adata.var_names, dtype=object).astype(str), dtype=object)
    for df in (adata.obs, adata.var):
        for col in df.columns:
            if isinstance(df[col].dtype, pd.StringDtype):
                df[col] = df[col].astype(object)
    return adata


def make_dir(dir_string):
    """Create *dir_string* (and any missing parents); a no-op if it already exists."""
    try:
        os.makedirs(dir_string, exist_ok=True)
        log.debug("Directory %s ready.", dir_string)
    except OSError as e:
        log.error("Directory %s could not be created: %s", dir_string, e)
        raise

def extract_data(
    h5ad_filepath,
    model,
    dataset_name=None,
    transcriptome_filepath=None,
    n_genes=100,
    seed=2813308004,
    viz=True,
    filt_param=None,
    modality_name_dict=None,
    cf=None,
    code_ver=code_ver_global,
    exp_filter_threshold=1,
    genes_to_fit=None,
    hist_type = 'unique',
    padding=None,
    mek_means_params=None,
):
    """Extract data for selected genes from a single dataset.

    Parameters
    ----------
    dataset_filepath: str or AnnData
        if str, location of file with count data (disabled).
        if AnnData, an object in memory that contains the count data.
    transcriptome_filepath: str
        location of the transcriptome length reference.
    dataset_name: str, optional
        dataset metadata; name assigned to dataset-specific directory.
        Defaults to "<bio_model>_<seq_model>" when not given.
    dataset_string: str
        dataset-specific directory location.
    dir_string: str
        batch directory location.
    viz: bool, optional
        whether to visualize and store spliced and unspliced count statistics.
    dataset_attr_names: length-3 tuple, optional
        entry 0: layers to use (typically 'unspliced' and 'spliced').
        entry 1: variable name (typically 'gene_name')
        entry 2: observation name (typically 'barcode')
        Entries 1 and 2 are unused in the typical workflow, and kept in for compatibility with
        files not loaded into memory.
    padding: list of int or None
        the PMF computation procedure uses a grid on max Z + padding[z] for each species Z.
        this list determines the amount of padding to be applied to the grid.
        if None, the padding is 10 for each species.
    cf: bool np.ndarray, optional
        array of size n_cells, mandatory if several datasets are stored in a single file.
        reports which cells are assigned to the current dataset.
        if None, all of the cells are used.
    hist_type: str, optional
        flavor of histogram.
        if 'grid', use np.histogramdd to compute over a grid.
        if 'unique', use np.unique to get observed microstate frequencies.
        'unique' is preferred.

    Returns
    -------
    anndata: adata object.
    """
    log.info("Beginning data extraction.")
    if dataset_name is None:
        dataset_name = f"{model.bio_model}_{model.seq_model}"
    log.info("Dataset: %s", dataset_name)

    if mek_means_params:
        k, epochs = mek_means_params

    if not filt_param:
        filt_param = model.filter_bounds
    if genes_to_fit is None:
        genes_to_fit = []
        log.info('Using default gene filtering parameters')

    # If no mapping given, assume that the layers are named according to the conventional modalities for the model
    # e.g. 'spliced', 'unspliced'
    if not modality_name_dict:
        modality_name_dict = {v:v for v in model.model_modalities}
        log.info(
            "Reading layers %s. If your AnnData layers use other names, pass "
            "modality_name_dict={<modality>: <layer>, ...}.", model.model_modalities,
        )

    ordered_modalities = model.model_modalities
    ordered_layer_names = [modality_name_dict[modality] for modality in ordered_modalities]
    
    if transcriptome_filepath:
        transcriptome_dict = get_transcriptome(transcriptome_filepath)
    else:
        transcriptome_dict = None

    # Import anndata from file, or use anndata object directly.
    if type(h5ad_filepath) == str: 
        # Import h5ad.
        monod_adata = import_h5ad(h5ad_filepath)
    else:
        monod_adata = h5ad_filepath.copy()

    # Load the AnnData object into memory if it is in backed mode
    if monod_adata.isbacked:
        monod_adata = monod_adata.to_memory()
    # Store the raw data.
    monod_adata = _plain_string_index(monod_adata)
    raw_adata = monod_adata.copy()

    # Check whether the given data is in sparse or numpy format.
    layer_data = monod_adata.layers[[i for i in ordered_layer_names][0]]
    log.debug('layers')

    layer_type = None
    if issparse(layer_data):
        layer_type = 'sparse'  # For sparse matrices
    elif isinstance(layer_data, np.ndarray):
        layer_type = 'numpy_array'
    else:
        pass

    log.debug('1'+ str(monod_adata.n_vars))


    if layer_type == 'sparse':
        monod_adata = CSRDataset_to_arrays(monod_adata)
        log.debug('layers are sparse; converting to dense arrays')
    elif layer_type == 'numpy_array':
        pass
    else:
        monod_adata = CSRDataset_to_arrays(monod_adata)
        log.debug('layers are backed CSRDataset; converting to dense arrays')

    # # Save entire raw data object to monod_adata
    # mono

    # Subset to the layers required in the model.
    adata = monod_adata.to_memory()

    
    # Build a new AnnData containing only the required layers.
    # Avoid copying large data arrays — share references with adata (which
    # is itself a copy of the input, so mutations are safe).
    adata_subset = ad.AnnData(
        X=adata.X,
        obs=adata.obs,
        var=adata.var,
        uns={},   # fresh dict; uns will be populated below
    )
    adata_subset.obs_names = adata.obs_names
    adata_subset.var_names = adata.var_names
    for layer in ordered_layer_names:
        if layer in adata.layers:
            adata_subset.layers[layer] = adata.layers[layer]

    monod_adata = adata_subset.to_memory()



    monod_adata.uns['modality_name_dict'] = modality_name_dict
    monod_adata.uns['model'] = model.to_dict()

    if padding is None:
        padding = np.asarray([10] * len(ordered_layer_names))

    # identify genes that are in the length annotations. discard all the rest.
    # for models without length-based sequencing, this may not be necessary
    # though I do not see the purpose of analyzing genes that not in the
    # genome.
    # if this is ever necessary, just make a different reference list.
    
    if transcriptome_filepath:

        # Add gene attribute indicating whether each gene's length is included in the transcriptome,
        # and add an attribute called 'lengths' containing the lengths, and log_lengths.
        monod_adata = add_gene_lengths(monod_adata, transcriptome_dict, attribute_name='length_given')

        # Filter the data based on whether the length is given.
        gene_filter = monod_adata.var['length_given'] == 1
        monod_adata = monod_adata[:, gene_filter].copy()

    elif 'log_lengths' in monod_adata.var.columns:
        # log_lengths already present — use as-is (no gene filtering needed).
        log.info('Using log_lengths from adata.var for gene length bias.')

    elif 'gene_length' in monod_adata.var.columns:
        # Compute log10 lengths from raw gene_length column (same convention as transcriptome path).
        monod_adata.var['log_lengths'] = np.log10(monod_adata.var['gene_length'])
        log.info('Computed log_lengths from adata.var[gene_length] for gene length bias.')

    # Filter genes, then pick a number of random genes to make up the total desired number of genes.
    monod_adata = process_adata(monod_adata, filt_param, genes_to_fit, exp_filter_threshold, n_genes, transcriptome_dict=transcriptome_dict, seed=seed)

    # Visualize selected and filtered genes before filtering.
    if viz:
        if transcriptome_filepath:
            visualize_gene_filtering_lengths(monod_adata)
        else:
            visualize_gene_filtering(monod_adata)

    # # Save the filter for selected genes, which should be applied to the raw adata.
    monod_adata.uns['selected_genes'] = monod_adata.var['selected_genes'].to_numpy().astype(bool)

    # Filter for selected genes.
    monod_adata = monod_adata[:, monod_adata.var['selected_genes'].astype(bool)].to_memory()
    
    gene_names, n_cells = monod_adata.var.index, len(monod_adata.obs.index)

    # Extract layers — densify sparse matrices if necessary.
    layers = np.array([to_dense(monod_adata.layers[layer_name]).T
                       for layer_name in ordered_layer_names])

    # Compute maximum expression value across cells for each gene and each layer
    max_values = np.amax(layers, axis=2)  # Shape: (n_genes, n_layers)

    # Define default padding if None
    if padding is None:
        padding = [10] * max_values.shape[1]  # One padding value per layer
    
    # Ensure padding is a column vector
    padding = np.asarray(padding)[:, None]

    # Add padding to the maximum values
    M = max_values + padding
    if model.bio_model == "Chromatin":
        # the configuration axis is categorical with exactly 2**n_sites states
        idx = model.model_modalities.index('configuration')
        M[idx, :] = 2 ** model.n_sites                      # M is (n_layers, n_genes)
    monod_adata.uns['M'] = M.astype(int)
    
    # Load the AnnData object into memory if it is in backed mode
    if monod_adata.isbacked:
        monod_adata = monod_adata.to_memory()

    if hist_type != "unique":
        raise ValueError(
            f"hist_type={hist_type!r} is not implemented; only 'unique' (per-gene unique "
            "microstate counts) is supported in this version."
        )
    monod_adata.uns['hist_type'] = hist_type

    monod_adata = add_moments(monod_adata)
    log.debug('Moments were added')
    
    monod_adata.uns['model'] = model.to_dict()

    # Histograms are not stored in uns: searchdata_from_adata rebuilds them from
    # the layers (in Rust when available), which is faster than unpickling and
    # keeps the AnnData writable as a plain h5ad.
    # Save adata?
    if mek_means_params:
        monod_adata.uns['k'] = k
        monod_adata.uns['epochs'] = epochs

    # Save the raw data as an attribute.
    monod_adata = _plain_string_index(monod_adata)
    monod_adata.raw = raw_adata
    
    return monod_adata

def CSRDataset_to_arrays(adata):

    # Assuming `adata` is your existing AnnData object
    adata_dense_layers = ad.AnnData(
        X=adata.X,
        obs=adata.obs,
        var=adata.var,
        obsm=adata.obsm,
        varm=adata.varm,
        obsp=adata.obsp,
        varp=adata.varp,
        uns=adata.uns
    )
    
    # Replace sparse layers with dense NumPy arrays
    for layer in adata.layers:
        if issparse(adata.layers[layer]):
            adata_dense_layers.layers[layer] = adata.layers[layer].toarray()  # Convert to dense array
        else:
            adata_dense_layers.layers[layer] = adata.layers[layer].copy()  # Copy if already dense
    
    # Preserve obs_names and var_names
    adata_dense_layers.obs_names = adata.obs_names
    adata_dense_layers.var_names = adata.var_names

    return adata_dense_layers

def add_moments(adata, modality_name_dict=None, cov_matrix_key='layer_covariances'):
    """
    Compute and add mean and variance for each gene within each layer to `adata.var`, 
    and add covariances between layers for each gene to `adata.var`.

    Parameters
    ----------
    adata: anndata.AnnData
        AnnData object with layers containing gene expression data.
    cov_matrix_key: str, optional
        Key under which the covariance matrix will be stored in `adata.uns`.
    modality_name_dict: dictionary
        mapping of form {'modality_name':'anndata layer name'} for modality names as given in cme_toolbox,
        and anndata layer names from the data.

    Returns
    -------
    adata: anndata.AnnData
        AnnData object with moments and layer covariances added to `adata.var` and `adata.uns`.
    """
    layer_names = adata.layers.keys()
    n_layers = len(layer_names)
    n_genes = adata.n_vars

    # If no mapping specified, assume layers are named the same as CME model modalities.
    if modality_name_dict:
        layer_name_to_modality = {v:k for k,v in modality_name_dict.items()}
    else:
        layer_name_to_modality = {v:v for v in layer_names}

    
    # Create DataFrame to store moments
    moments_df = pd.DataFrame(index=adata.var.index)

    # Compute mean and variance for each gene within each layer
    for layer in layer_names:

        modality_name = layer_name_to_modality[layer]
        
        mean_col = f"MOM_{modality_name}_mean"
        var_col = f"MOM_{modality_name}_var"
        
        # # Calculate mean and variance for each layer
        # moments_df[mean_col] = adata.layers[layer].mean(axis=0).flatten()
        # moments_df[var_col] = adata.layers[layer].var(axis=0).flatten()

        # Calculate mean and variance for each layer
        adata.var[mean_col] = adata.layers[layer].mean(axis=0).flatten() # toarray
        adata.var[var_col] = adata.layers[layer].var(axis=0).flatten() #toarray
    
    index = 0
    for i in range(n_layers):
        for j in range(i + 1, n_layers):
            layer_i = [p for p in layer_names][i]
            layer_j = [p for p in layer_names][j]
            mod_i, mod_j = layer_name_to_modality[layer_i], layer_name_to_modality[layer_j]
            layer_layer_string = f"MOM_cov_{mod_i}_{mod_j}"

            gene_covars = []
            for gene_index in range(n_genes):
                covar = np.cov([adata.layers[layer_i][:, gene_index], adata.layers[layer_j][:, gene_index]])[0, 1] # toarray
                # covariances[gene_index, index] = covar
                gene_covars += [covar]

            adata.var[layer_layer_string] = np.array(gene_covars)
            
            index += 1

    # # Add layer covariances DataFrame to adata.var
    # covariances_df = pd.DataFrame(covariances, index=adata.var.index)
    
    # covariances_df.columns = [f"MOM_cov_{[p for p in layers][i]}_{[p for p in layers][j]}" for i in range(n_layers) for j in range(i + 1, n_layers)]
    
    # adata.var = adata.var.join(covariances_df)

    # # Add the covariance matrix between layers to `adata.uns` (not needed?)
    # adata.uns[cov_matrix_key] = covariances

    return adata


def _make_histogram_one_gene(gene_index, layers, hist_type, M, n_cells):
    """Compute the histogram for a single gene (parallelisable worker)."""
    gene_cols = [to_dense(x[:, gene_index]).ravel() for x in layers]
    stacked = np.column_stack(gene_cols)  # (n_cells, n_modalities) — built once

    unique, unique_counts = np.unique(stacked, axis=0, return_counts=True)
    return (unique.astype(int), unique_counts / n_cells)


def make_histogram(monod_adata, hist_type, M, n_jobs=1):
    """Build per-gene expression histograms.

    Parameters
    ----------
    monod_adata : AnnData
    hist_type : str
        Must be "unique".
    M : array-like
        Grid limits per layer per gene.
    n_jobs : int
        Number of joblib workers (1 = serial; -1 = all cores).
    """
    from joblib import Parallel, delayed as jdelayed

    # NB the order of the layers here will be enforced to be the same as the order of the model
    # modalities defined in cme_toolbox.

    modality_name_dict = monod_adata.uns['modality_name_dict']
    model = model_from_uns(monod_adata.uns['model'])

    ordered_modalities = model.model_modalities
    ordered_layer_names = [modality_name_dict[modality] for modality in ordered_modalities]

    layers = [monod_adata.layers[layer_name] for layer_name in ordered_layer_names]
    n_cells = monod_adata.n_obs
    n_genes = monod_adata.n_vars

    if n_jobs == 1:
        hist = [
            _make_histogram_one_gene(i, layers, hist_type, M, n_cells)
            for i in range(n_genes)
        ]
    else:
        hist = Parallel(n_jobs=n_jobs)(
            jdelayed(_make_histogram_one_gene)(i, layers, hist_type, M, n_cells)
            for i in range(n_genes)
        )

    return hist

## Helper functions.
# Don't need store_search_data function anymore, can just save adata object in the normal way.

########################
## Helper functions
########################


def add_gene_lengths(adata, length_dict, attribute_name='length_given', lengths_name='lengths'):
    """
    Adds a gene attribute to each gene in the AnnData object.
    The attribute is set to 1 if the gene's name is found within the keys of the dictionary, and 0 otherwise.
    Additionally, adds the values from the dictionary as a new attribute called 'lengths'.

    Parameters
    ----------
    adata: anndata.AnnData
        AnnData object containing gene expression data.
    gene_dict: dict
        Dictionary with gene names as keys and values as the new attribute to be added.
    attribute_name: str, optional
        Name of the attribute to be added to adata.var indicating presence in the dictionary.
    lengths_name: str, optional
        Name of the attribute to be added to adata.var containing the values from the dictionary.

    Returns
    -------
    adata: anndata.AnnData
        AnnData object with the new gene attributes added to adata.var.
    """
    # Initialize the attributes
    adata.var[attribute_name] = adata.var_names.isin(length_dict).astype(int)

    
    adata.var[lengths_name] = adata.var_names.map(length_dict).fillna(0).astype(float)
    with np.errstate(divide='ignore'):   # genes absent from the reference have length 0 and are dropped later
        adata.var['log_lengths'] = np.log10(adata.var[lengths_name])

    log.info('Added lengths')
    return adata

def threshold_by_expression(adata, filt_param=None):
    """
    Filters genes in an AnnData object based on expression thresholds for each layer.

    Parameters
    ----------
    adata: anndata.AnnData
        AnnData object with gene expression data in layers.
    filt_param: dict, optional
        Dictionary with filtering parameters.
        - 'min_means': Minimum mean expression for each modality.
        - 'max_maxes': Maximum expression for each modality.
        - 'min_maxes': Minimum expression for each modality.

    Returns
    -------
    adata: anndata.AnnData
        Filtered AnnData object with genes that meet the expression thresholds.
    """
    if filt_param is None:
        filt_param = {'min_means': [0.01, 0.01], 'max_maxes': [350, 350], 'min_maxes': [4, 4]}
    model = model_from_uns(adata.uns['model'])
    ordered_modalities = model.model_modalities
    modality_name_dict = adata.uns['modality_name_dict']
    ordered_layer_names = [modality_name_dict[mod] for mod in ordered_modalities]
        
    # Initialize gene filter to keep all genes
    gene_exp_filter = np.ones(adata.n_vars, dtype=bool)
    
    layers = [adata.layers[layer] for layer in ordered_layer_names] # toarray

    # # Iterate over each layer to apply filters
    for i in range(len(layers)):
        layer = layers[i]
    
        means = layer.mean(axis=0)
        maxes = layer.max(axis=0)
        total_gene_num = len(means)

        min_means_filt = (means >= filt_param['min_means'][i])
        max_maxes_filt = (maxes <= filt_param['max_maxes'][i])
        min_maxes_filt = (maxes >= filt_param['min_maxes'][i])
        
        layer_filter = min_means_filt  &  max_maxes_filt & min_maxes_filt

        log.debug('{} filt_param'.format(filt_param))
        log.debug('{} genes failed to meet minimum {} means'.format( total_gene_num - sum(min_means_filt), ordered_layer_names[i]))

        log.debug('{} genes exceeded maximum {} means'.format( total_gene_num - sum(max_maxes_filt), ordered_layer_names[i]))

        log.debug('{} genes failed to meet minimum {} maximum'.format(total_gene_num - sum(min_maxes_filt), ordered_layer_names[i]))
        
        # Combine filters across layers
        gene_exp_filter &= layer_filter

    ## Do not remove filtered genes.
    # # Update AnnData object with filtered genes
    # adata = adata[:, gene_exp_filter].to_memory() #copy()
    adata.var['gene_exp_filter'] = gene_exp_filter

    # Log the number of genes retained
    log.info("%d genes retained after expression filter.", int(np.sum(gene_exp_filter)))
    
    return adata
    

def import_h5ad(filename, cf=None):

    with warnings.catch_warnings():
        
        warnings.simplefilter("ignore", UserWarning)
        
        # Load or process your AnnData object here
        ds = ad.read_h5ad(filename, backed="r")
        # Handle duplicates if necessary
        if not ds.var_names.is_unique:
            log.info("Duplicate variable names found. Making them unique...")
            ds.var_names_make_unique()

    # if not ds.var_names.is_unique:
    #     print("Duplicate variable names found. Making them unique...")
    #     ds.var_names_make_unique()
        
    return ds

def visualize_gene_filtering_lengths(monod_adata):
    """
    Visualizes gene filtering results given lengths.
    
    Parameters
    ----------
    adata: anndata.AnnData
        AnnData object with gene expression data in layers.
    transcriptome_dict: dict
        Dictionary with transcriptome data for gene lengths.
    modality_names: list
        List of attribute names for visualization.
    """
    gene_names = monod_adata.var_names

    try:
        len_arr = monod_adata.var['log_lengths']
    except AttributeError:
        log.error('No gene lengths given')
    modality_names = model_from_uns(monod_adata.uns['model']).model_modalities
    
    # len_arr = np.array([transcriptome_dict[k] for k in gene_names])
    var_name = tuple([name[0].upper() for name in modality_names[::-1]])
    var_arr = tuple([monod_adata.layers[layer].mean(axis=0) for layer in monod_adata.layers][::-1])

    fig1, ax1 = plt.subplots(nrows=1, ncols=len(monod_adata.layers), figsize=(12, 4))
    for i, layer in enumerate(monod_adata.layers):
        ax1[i].scatter(
            len_arr[~monod_adata.var['selected_genes'].values],
            np.log10(var_arr[i][~monod_adata.var['selected_genes'].values] + 0.001),
            s=3,
            c="silver",
            alpha=0.15,
        )
        ax1[i].scatter(
            len_arr[monod_adata.var['selected_genes'].values],
            np.log10(var_arr[i][monod_adata.var['selected_genes'].values] + 0.001),
            s=3,
            c="indigo",
            alpha=0.3,
        )
        ax1[i].set_xlabel("log10 gene length")
        ax1[i].set_ylabel("log10 (mean " + var_name[i] + " + 0.001)")

def visualize_gene_filtering(monod_adata):
    """
    Visualizes gene filtering results without lengths.
    
    Parameters
    ----------
    adata: anndata.AnnData
        AnnData object with gene expression data in layers.
    """
    gene_names = monod_adata.var_names

    modality_names = model_from_uns(monod_adata.uns['model']).model_modalities
    
    # len_arr = np.array([transcriptome_dict[k] for k in gene_names])
    var_name = tuple([name[0].upper() for name in modality_names[::-1]])
    var_arr = tuple([monod_adata.layers[layer].mean(axis=0) for layer in monod_adata.layers][::-1])

    fig1 = plt.figure(figsize=(5, 5))

    # Show first two layers.        
    plt.scatter(
        np.log10(var_arr[0][~monod_adata.var['selected_genes'].values] + 0.001),
        np.log10(var_arr[1][~monod_adata.var['selected_genes'].values] + 0.001),
        s=3,
        c="silver",
        alpha=0.15,
    )
    plt.scatter(
        np.log10(var_arr[0][monod_adata.var['selected_genes'].values] + 0.001),
        np.log10(var_arr[1][monod_adata.var['selected_genes'].values] + 0.001),
        s=3,
        c="indigo",
        alpha=0.3,
    )
    plt.xlabel("log10 (mean " + var_name[0] + " + 0.001)")
    plt.ylabel("log10 (mean " + var_name[1] + " + 0.001)")

def process_adata(adata, filt_param, genes_to_fit, exp_filter_threshold, n_genes, seed=5, dir_string='./fits', transcriptome_dict=None):
    """
    Processes a single AnnData object, applies expression filters, and stores the results in AnnData.

    Parameters
    ----------
    adata: anndata.AnnData
        AnnData object to be processed.
    filt_param: dict
        Filtering parameters for expression.
    transcriptome_dict: dict
        Dictionary with transcriptome data for gene lengths.
    genes_to_fit: list
        List of genes to be specifically included in the selection.
    exp_filter_threshold: float or None
        Threshold for expression filtering.
    n_genes: int
        Number of genes to select.
    seed: int
        Random seed for reproducibility.
    dir_string: str
        Directory to save the gene list.

    Returns
    -------
    adata: anndata.AnnData
        Processed AnnData object with gene filters applied.
    """
    n_cells = adata.n_obs
    log.info(f"{n_cells} cells detected.")
    n_genes_original = adata.n_vars
    log.info(f"{n_genes_original} genes detected.")
    
    # Filter genes by expression
    adata = threshold_by_expression(adata, filt_param)
    gene_names = adata.var_names

    gene_name_reference = np.copy(gene_names)
    # Apply expression threshold filter
    gene_exp_filter = adata.var['gene_exp_filter']

    # Randomly select genes to meet the required n_genes.
    n_vars = adata.n_vars
    selected_genes_filter = np.zeros(n_vars, dtype=bool)

    enforced_genes = np.zeros(n_vars, dtype=bool)
    gene_exp_filter = np.asarray(gene_exp_filter, dtype=bool).copy()  # plain array: positional indexing below
    
    # First mark the enforced genes
    for gene in genes_to_fit:
        gene_loc = np.where(adata.var_names == gene)[0]
        if len(gene_loc) == 0:
            log.warning(f"Gene {gene} not found.")
        elif len(gene_loc) > 1:
            log.error(f"Multiple entries found for gene {gene}: this should never happen.")
        else:
            gene_exp_filter[gene_loc[0]] = False  # Exclude from sampling pool
            n_genes -= 1
            enforced_genes[gene_loc[0]] = True

    log.debug(f"Number of additional genes to select: {n_genes}")
    np.random.seed(seed)
    # Get indices where gene_exp_filter is True (available for sampling) AND enforced_genes is False
    sampling_gene_set = np.where(gene_exp_filter & ~enforced_genes)[0]
    log.info(f"Number of enforced genes: {sum(enforced_genes)}")
    log.debug(f"Size of sampling pool: {len(sampling_gene_set)}")
    log.debug(f"Sampling gene indices: {sampling_gene_set}")
    
    # Initialize the final selection filter
    selected_genes_filter = np.zeros(n_vars, dtype=bool)
    
    # Select random genes from the sampling pool
    if n_genes < len(sampling_gene_set):
        gene_select_ind = np.random.choice(sampling_gene_set, max(n_genes, 0), replace=False)
        log.info(f"{len(gene_select_ind)} random genes selected.")
    else:
        gene_select_ind = sampling_gene_set
        if n_genes > len(sampling_gene_set):
            log.warning(f"Only {len(sampling_gene_set)} genes pass the filters; cannot satisfy query of {n_genes} genes.")
        else:
            log.info(f"All {len(sampling_gene_set)} genes passing the filters selected.")
    
    # Set both randomly selected and enforced genes to True
    selected_genes_filter[gene_select_ind] = True
    selected_genes_filter[enforced_genes] = True
    adata.var['selected_genes'] = selected_genes_filter
    log.info(f"Total of {selected_genes_filter.sum()} genes selected.")
    
    return adata



def get_transcriptome(transcriptome_filepath, repeat_thr=15):
    """Imports transcriptome length/repeat statistics from a previously generated file.

    Parameters
    ----------
    transcriptome_filepath: str
        location of the transcriptome length reference.
        this is a simple space-separated file.
        The convention for each line is name - length - # of 5mers - # of 6mers -
            .... - # of 50mers - # of repeats with more than 50 A bases in a row
    repeat_thr: int
        threshold for minimum repeat length to consider.
        By default, this is 15, and so will return number of polyA stretches of
        length 15 or more in the gene.

    Returns
    -------
    len_dict: dict
        dictionary with structure {gene name : gene length}

    The repeat dictionary is not used in this version of the code.
    """
    #repeat_dict = {}
    len_dict = {}
    #thr_ind = repeat_thr - 3
    with open(transcriptome_filepath, "r") as file:
        for line in file.readlines():
            d = [i for i in line.split(" ") if i]
            #repeat_dict[d[0]] = int(d[thr_ind])
            len_dict[d[0]] = int(d[1])
    return len_dict


# same for ATAC?
def knee_plot(X, ax1=None, thr=None, viz=False):
    """
    Plot the knee plot for a gene x cell dataset.

    Parameters
    ----------
    X: np.ndarray
        gene x cell count matrix.
    ax1: matplotlib.axes.Axes, optional
        axes to plot into.
    thr: float or int, optional
        minimum molecule count cutoff.
    viz: bool, optional
        whether to visualize the knee plot.

    Returns
    -------
    cf: bool np.ndarray
        cells that meet the minimum molecule count cutoff.
    """

    umi_sum = X.sum(0)
    n_cells = len(umi_sum)
    umi_rank = np.argsort(umi_sum)
    usf = np.flip(umi_sum[umi_rank])
    if viz:
        ax1.plot(np.arange(n_cells), usf, "k")
        ax1.set_xlabel("Cell rank")
        ax1.set_ylabel("UMI count+1")
        ax1.set_yscale("log")
    if thr is not None:
        cf = umi_sum > thr
        rank_ = np.argmin(np.abs(usf - thr))
        if viz:
            ax1.plot([0, n_cells + 1], thr * np.ones(2), "r--")
            ys = ax1.get_ylim()
            ax1.plot(rank_ * np.ones(2), ys, "r--")
        return cf


# For multiple datasets, choose genes which pass the expression filter in the highest fraction of datasets 
