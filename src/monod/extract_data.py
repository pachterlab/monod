import numpy as np
import csv
import matplotlib.pyplot as plt
import pickle
import sys
from datetime import datetime
import pytz
import scanpy as sc
import anndata
import numpy as np
import matplotlib.pyplot as plt
import time
import loompy as lp  # will be necessary later, probably
import anndata as ad
import os
import collections
import warnings
import scipy
from scipy import sparse
import pandas as pd
code_ver_global = "029"  # bumping up version April 2024

import logging, sys
from scipy.sparse import csr_matrix
from scipy.sparse import issparse

logging.basicConfig(stream=sys.stdout)
log = logging.getLogger()
log.setLevel(logging.DEBUG)

import logging.config

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": True,
    }
)

def make_dir(dir_string):
    """Convenience function to create a directory.

    Parameters
    ----------
    dir_string: str
        directory location to create.
    """
    try:
        os.mkdir(dir_string)
        log.info("Directory " + dir_string + " created.")
    except OSError as error:
        log.info(
            "Directory " + dir_string + " already exists."
        )  # used to be warning, but should be ok for now



########################
## Main code
########################

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
    genes_to_fit=[],
    hist_type = 'grid',
    padding=None,
    mek_means_params=None
):
    """Extract data for selected genes from a single dataset.

    Parameters
    ----------
    dataset_filepath: str or AnnData
        if str, location of file with count data (disabled).
        if AnnData, an object in memory that contains the count data.
    transcriptome_filepath: str
        location of the transcriptome length reference.
    dataset_name: str
        dataset metadata; name assigned to dataset-specific directory.
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
    log.info("Dataset: " + dataset_name)

    if mek_means_params:
        k, epochs = mek_means_params

    if not filt_param:
        filt_param = model.filter_bounds
        log.info('Using default gene filtering parameters')

    # If no mapping given, assume that the layers are named according to the conventional modalities for the model
    # e.g. 'spliced', 'unspliced'
    if not modality_name_dict:
        modality_name_dict = {v:v for v in model.model_modalities}

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
        print('is sparse')
    elif layer_type == 'numpy_array':
        pass
    else:
        monod_adata = CSRDataset_to_arrays(monod_adata)
        print('CSRDataset')

    # # Save entire raw data object to monod_adata
    # mono

    # Subset to the layers required in the model.
    adata = monod_adata.to_memory()

    
    adata_subset = ad.AnnData(
    X=adata.X.copy(),
    obs=adata.obs.copy(),
    var=adata.var.copy(),
    obsm=adata.obsm.copy(),
    varm=adata.varm.copy(),
    obsp=adata.obsp.copy(),
    varp=adata.varp.copy(),
    uns=adata.uns.copy()
)
    # Ensure var and obs names are preserved
    adata_subset.obs_names = adata.obs_names
    adata_subset.var_names = adata.var_names


    # Copy over only the desired layers
    for layer in ordered_layer_names:
        if layer in adata.layers:
            adata_subset.layers[layer] = adata.layers[layer].copy()

    monod_adata = adata_subset.to_memory()



    monod_adata.uns['modality_name_dict'] = modality_name_dict
    monod_adata.uns['model'] = model


    
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

    # Filter genes, then pick a number of random genes to make up the total desired number of genes.
    monod_adata = process_adata(monod_adata, filt_param, genes_to_fit, exp_filter_threshold, n_genes, transcriptome_dict=transcriptome_dict, seed=seed)

    # Visualize selected and filtered genes before filtering.
    if viz:
        if transcriptome_filepath:
            visualize_gene_filtering_lengths(monod_adata)
        else:
            visualize_gene_filtering(monod_adata)

    # # Save the filter for selected genes, which should be applied to the raw adata.
    monod_adata.uns['selected_genes'] = monod_adata.var['selected_genes'].astype(bool)

    # Filter for selected genes.
    monod_adata = monod_adata[:, monod_adata.var['selected_genes'].astype(bool)].to_memory()
    
    gene_names, n_cells = monod_adata.var.index, len(monod_adata.obs.index)

    # Extract layers
    layers = np.array([monod_adata.layers[layer_name].T for layer_name in ordered_layer_names]) # toarray

    # Compute maximum expression value across cells for each gene and each layer
    max_values = np.amax(layers, axis=2)  # Shape: (n_genes, n_layers)

    # Define default padding if None
    if padding is None:
        padding = [10] * max_values.shape[1]  # One padding value per layer
    
    # Ensure padding is a column vector
    padding = np.asarray(padding)[:, None]

    # Add padding to the maximum values
    M = max_values + padding
    monod_adata.uns['M'] = M.astype(int)
    
    # Load the AnnData object into memory if it is in backed mode
    if monod_adata.isbacked:
        monod_adata = monod_adata.to_memory()

    monod_adata.uns['hist_type'] = hist_type

    monod_adata = add_moments(monod_adata)
    log.debug('Moments were added')
    
    monod_adata.uns['model'] = model

    hist = make_histogram(monod_adata, hist_type, M)
    
    monod_adata.uns['hist'] = hist
    # Save adata?
    if mek_means_params:
        monod_adata.uns['k'] = k
        monod_adata.uns['epochs'] = epochs

    # Save the raw data as an attribute.
    monod_adata.raw = raw_adata
    
    return monod_adata

def sparse_to_arrays(adata):

    # Assuming `adata` is your existing AnnData object
    adata_dense_layers = ad.AnnData(
        X=adata.X.copy(),
        obs=adata.obs.copy(),
        var=adata.var.copy(),
        obsm=adata.obsm.copy(),
        varm=adata.varm.copy(),
        obsp=adata.obsp.copy(),
        varp=adata.varp.copy(),
        uns=adata.uns.copy()
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

# TODO make work.
def plot_vs_lengths(adata):
    
    mods = layers # usually unspliced, spliced
    
    mod_names = dataset_attr_names[0] # default 'unspliced', 'spliced'
    
    # Use first letter of names as labels (make sure modalities have different first letters.
    mod_labels = [mod_name[0].upper() for mod_name in mod_names]
    
    var_name = tuple(mod_labels[::-1])
    var_arr = tuple([mod.mean(1) for mod in mods][::-1])
    
    fig1, ax1 = plt.subplots(nrows=1, ncols=len(mods), figsize=(12, 4))
    for i in range(len(mods)):
        ax1[i].scatter(
            np.log10(len_arr),
            np.log10(var_arr[i] + 0.001),
            s=3,
            c="silver",
            alpha=0.15,
        )
        ax1[i].set_xlabel("log10 gene length")
        ax1[i].set_ylabel("log10 (mean " + var_name[i] + " + 0.001)")

# TODO make work.
def plot_diagnostic(monod_adata):
    dataset_diagnostics_dir_string = dataset_string + "/diagnostic_figures"
    make_dir(dataset_diagnostics_dir_string)

    if transcriptome_filepath:
        for i in range(len(mods)):
            var_arr = tuple([mod.mean(1) for mod in mods][::-1])
            ax1[i].scatter(
                np.log10(len_arr),
                np.log10(var_arr[i] + 0.001),
                s=5,
                c="firebrick",
                alpha=0.9,
            )
        
        plt.savefig(
            dataset_diagnostics_dir_string + "/{}.png".format(dataset_name), dpi=450
        )

def make_histogram(monod_adata, hist_type, M):

    # NB the order of the layers here will be enforced to be the same as the order of the model 
    # modalities defined in cme_toolbox.

    modality_name_dict = monod_adata.uns['modality_name_dict']
    model = monod_adata.uns['model']

    ordered_modalities = model.model_modalities
    ordered_layer_names = [modality_name_dict[modality] for modality in ordered_modalities]

    hist = []
    layers = [monod_adata.layers[layer_name] for layer_name in ordered_layer_names] # toarray
    n_cells = monod_adata.n_obs
    n_genes = monod_adata.n_vars
    
    for gene_index in range(n_genes):
        
        if hist_type == "grid":
            bins = [np.arange(x[gene_index] + 1) - 0.5 for x in M]
            stacked_data = np.vstack([x[:,gene_index] for x in layers]).T
            H, edges = np.histogramdd(
                stacked_data,
                bins=bins,
                density=True
            )
            xedges = edges[0]  # Assuming only one dimension for each bin
            yedges = edges[1] 
        elif hist_type == "unique":
            
            unique, unique_counts = np.unique(
                np.vstack([x[:,gene_index] for x in layers]).T, axis=0, return_counts=True
            )
            frequencies = unique_counts / n_cells
            unique = unique.astype(int)
            H = (unique, frequencies)
            
        elif hist_type == "none":
            H = [x[:, gene_index] for x in layers]

        hist.append(H)

    return hist

## Helper functions.
# Don't need store_search_data function anymore, can just save adata object in the normal way.

def get_noise_decomp(
    adata, sizefactor="pf", lognormalize=True, pcount=0, which_variance_measure="CV2"
):
    """
    This function performs normalization and variance stabilization on the raw data, and
    reports the fractions of normalized variance retained and removed as a result of the process.
    The (unspliced and spliced) species are analyzed independently.

    Parameters
    ----------
    adata: anndata.AnnData
        AnnData object with layers containing raw data (e.g., 'unspliced', 'spliced').
    sizefactor: str, float, int, or None, optional
        What size factor to use.
        If 'pf', use proportional fitting; set the size of each cell to the mean size.
        If int or float, use this number (e.g., 1e4 for cp10k).
        If None, do not do size/depth normalization.
    lognormalize: bool, optional
        Whether to apply log(1+x) transformation.
    pcount: int or float, optional
        Pseudocount added in size normalization to ensure division by zero does not occur.
    which_variance_measure: str, optional
        Measure of variance to use ('CV2', 'var', 'Fano').

    Returns
    -------
    adata: anndata.AnnData
        AnnData object with additional columns in `adata.obs` containing the fraction of variance retained or discarded.
    """

    layers = adata.layers.keys()

    for layer in layers:
        # Copy the layer to a temporary AnnData object for normalization
        temp_adata = anndata.AnnData(X=adata.layers[layer])

        temp_adata = normalize_count_matrix(temp_adata, sizefactor=sizefactor, lognormalize=lognormalize, pcount=pcount)

        # Compute variance before and after normalization
        original_variance = var_fun(anndata.AnnData(X=adata.layers[layer]), which_variance_measure)
        normalized_variance = var_fun(temp_adata, which_variance_measure)

        # Store the results in the adata.obs
        adata.obs[f'var_retained_{layer}'] = normalized_variance / original_variance
        adata.obs[f'var_discarded_{layer}'] = 1 - adata.obs[f'var_retained_{layer}']

    return adata


def var_fun(adata, measure):
    '''
    Define a function to measure variance of gene count matrix.
    '''    
    if measure == "CV2":
        mean = adata.X.mean(axis=1).A1
        variance = adata.X.var(axis=1).A1
        return variance / (mean ** 2)
    elif measure == "var":
        return adata.X.var(axis=1).A1
    elif measure == "Fano":
        mean = adata.X.mean(axis=1).A1
        variance = adata.X.var(axis=1).A1
        return variance / mean


def normalize_count_matrix(
    adata, sizefactor="pf", lognormalize=True, pcount=1, logbase=np.e
):
    """
    This function performs normalization and variance stabilization on a raw data matrix in an AnnData object
    using Scanpy's built-in functions.

    Parameters
    ----------
    adata: anndata.AnnData
        AnnData object with a gene x cell count matrix in adata.X.
    sizefactor: str, float, int, or None, optional
        What size factor to use.
        If 'pf', use proportional fitting; set the size of each cell to the mean size.
        If int or float, use this number (e.g., 1e4 for cp10k).
        If None, do not do size/depth normalization.
    lognormalize: bool, optional
        Whether to apply log transformation.
    pcount: int or float, optional
        Pseudocount added in size normalization to ensure division by zero does not occur.
    logbase: int or float
        log base for scanpy.

    Returns
    -------
    adata: anndata.AnnData
        AnnData object with normalized and transformed count matrix in adata.X.
    """
    if sizefactor is not None:
        if sizefactor == "pf":
            sc.pp.normalize_total(adata, target_sum=adata.X.sum(0).mean(), inplace=True)
        else:
            sc.pp.normalize_total(adata, target_sum=sizefactor, inplace=True)

    if lognormalize:
        sc.pp.log1p(adata, base=logbase)
    
    return adata

########################
## Main code
########################

# From construct batch.
    
        
    if dataset_index == 0:  # this presupposes the data are well-structured
        gene_name_reference = np.copy(gene_names)
        expression_filter_array = np.zeros(
            (n_datasets, len(gene_name_reference)), dtype=bool
        )
    else:
        if not np.all(gene_name_reference == gene_names):
            raise ValueError(
                "Gene names do not match: the data may not be consistently structured."
            )

    


# TODO make work.
def preprocess_viz(monod_adata, transcriptome_filepath):

    if transcriptome_filepath:
        # Use first letters of the modalities as names in visualization.
        # mod1_name, mod2_name = attribute_names[0]
        
        mod_names = attribute_names[0][0]
        # var_name = (mod2_name[0].upper(), mod1_name[0].upper()) # e.g. ("S", "U")
        # NB reversed order here.
        var_name = tuple([name[0].upper() for name in mod_names[::-1]])
        # var_arr = (mod2.mean(1), mod1.mean(1))
        var_arr = tuple([mod.mean(1) for mod in mods][::-1])

        fig1, ax1 = plt.subplots(nrows=1, ncols=len(mods), figsize=(12, 4))
        for i in range(len(mods)):
            ax1[i].scatter(
                np.log10(len_arr)[~gene_exp_filter],
                np.log10(var_arr[i][~gene_exp_filter] + 0.001),
                s=3,
                c="silver",
                alpha=0.15,
            )
            ax1[i].scatter(
                np.log10(len_arr[gene_exp_filter]),
                np.log10(var_arr[i][gene_exp_filter] + 0.001),
                s=3,
                c="indigo",
                alpha=0.3,
            )
            ax1[i].set_xlabel("log10 gene length")
            ax1[i].set_ylabel("log10 (mean " + var_name[i] + " + 0.001)")

########################
## Helper functions
########################


def filter_by_gene(filter, *args):
    """Convenience function to filter arrays by gene.

    This function takes in a filter over genes,
    then selects the entries of inputs that match the filter.

    Parameters
    ----------
    filter: bool or int np.ndarray
        filter over the gene dimension.
    *args: variable number of np.ndarrays
        np.ndarrays with dimension 0 that matches the filter dimension.

    Returns
    -------
    tuple(out): tuple
        tuple of filtered *args.

    Examples
    --------
    >>> S_filt, U_filt = filter_by_gene(filter,S,U)
    >>> assert((S_filt.shape[0]==filter.sum()) & (U_filt.shape[0]==filter.sum()))

    """
    out = []
    for arg in args:
        out += [arg[filter].squeeze()]
    return tuple(out)

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
    adata.var['log_lengths'] = np.log10(adata.var['lengths'])

    log.info('Added lengths')
    return adata

def threshold_by_expression(adata, filt_param={'min_means': [0.01, 0.01], 
                                               'max_maxes': [350, 350], 'min_maxes': [4, 4]}):
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
    model = adata.uns['model']
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
    print(f"{np.sum(gene_exp_filter)} genes retained after expression filter.")
    
    return adata
    

def save_gene_list(dir_string, gene_list, filename):
    """Store a list of genes to disk.

    Parameters
    ----------
    dir_string: str
        batch directory location.
    gene_list: list of str
        list of genes to store.
    filename: str
        file name string.
    """
    with open(dir_string + "/" + filename + ".csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(gene_list)



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
    modality_names = monod_adata.uns['model'].model_modalities
    
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

    modality_names = monod_adata.uns['model'].model_modalities
    
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
    gene_exp_filter = gene_exp_filter.copy()  # Make a copy to avoid modifying the original
    
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
        log.warning(f"{len(sampling_gene_set)} random genes selected: cannot satisfy query of {n_genes} genes.")
    
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


def identify_annotated_genes(gene_names, feat_dict):
    """Check which gene names are unique and have annotations in a feature dictionary.

    Parameters
    ----------
    gene_names: str np.ndarray
        gene names from raw data file.
    feat_dict: dict
        annotation dictionary imported by get_transcriptome.

    Returns
    -------
    ann_filt: bool np.ndarray
        boolean filter of genes that have annotations.
    """
    n_gen_tot = len(gene_names)
    capitalize = False
    if capitalize:
        sel_ind_annot = [k for k in range(len(gene_names)) if gene_names[k].capitalize() in feat_dict]
    else:
        sel_ind_annot = [k for k in range(len(gene_names)) if gene_names[k] in feat_dict]

    NAMES = [gene_names[k] for k in range(len(sel_ind_annot))]
    COUNTS = collections.Counter(NAMES)
    sel_ind = [x for x in sel_ind_annot if COUNTS[gene_names[x]] == 1]

    log.info(
        str(len(gene_names))
        + " features observed, "
        + str(len(sel_ind_annot))
        + " match genome annotations. "
        + str(len(sel_ind))
        + " were unique."
    )

    ann_filt = np.zeros(n_gen_tot, dtype=bool)
    ann_filt[sel_ind] = True
    return ann_filt

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
def select_genes_across_datasets(adata_list, filt_param, exp_filter_threshold, genes_to_fit, n_genes, seed):
    n_datasets = len(adata_list)
    expression_filter_array = np.zeros((n_datasets, adata_list[0].n_vars), dtype=bool)
    gene_name_reference = adata_list[0].var_names

    for dataset_index, adata in enumerate(adata_list):
        log.info("Processing dataset: " + str(dataset_index))

        # Apply expression filter
        adata = threshold_by_expression(adata, filt_param)
        gene_exp_filter = adata.var['expression_filter'].values
        expression_filter_array[dataset_index, :] = gene_exp_filter

    exp_fractions = expression_filter_array.mean(axis=0)

    if exp_filter_threshold is None:
        n_genes_enforced = 0
        selected_genes_filter = np.zeros(exp_fractions.shape, dtype=bool)
        
        for gene in genes_to_fit:
            gene_loc = np.where(gene_name_reference == gene)[0]
            if len(gene_loc) == 0:
                log.warning(f"Gene {gene} not found or has multiple entries in annotations.")
            elif len(gene_loc) > 1:
                log.error(f"Multiple entries found for gene {gene}: this should never happen.")
            else:
                exp_fractions[gene_loc[0]] = 0
                n_genes_enforced += 1
                selected_genes_filter[gene_loc[0]] = True
        
        q = np.quantile(exp_fractions, 1 - (n_genes - n_genes_enforced) / len(exp_fractions))

        selected_genes_filter[exp_fractions > q] = True
        np.random.seed(seed)
        random_genes = np.where(exp_fractions == q)[0]
        
        if random_genes.size > 0:
            random_genes_sel = np.random.choice(random_genes, n_genes - selected_genes_filter.sum(), replace=False)
            selected_genes_filter[random_genes_sel] = True

        selected_genes = gene_name_reference[selected_genes_filter]
        log.info(f"Total of {selected_genes_filter.sum()} genes selected.")
        log.warning(f"Selecting {selected_genes_filter.sum()} genes required {q * 100:.1f}% dataset threshold.")
    else:
        exp_filter = exp_fractions >= exp_filter_threshold
        log.info(f"Gene set size according to a {exp_filter_threshold * 100:.1f}% dataset threshold: {exp_filter.sum()}")
        
        enforced_genes = np.zeros(exp_filter.shape, dtype=bool)
        for gene in genes_to_fit:
            gene_loc = np.where(gene_name_reference == gene)[0]
            if len(gene_loc) == 0:
                log.warning(f"Gene {gene} not found or has multiple entries in annotations.")
            elif len(gene_loc) > 1:
                log.error(f"Multiple entries found for gene {gene}: this should never happen.")
            else:
                exp_filter[gene_loc[0]] = False
                n_genes -= 1
                enforced_genes[gene_loc[0]] = True

        np.random.seed(seed)
        sampling_gene_set = np.where(exp_filter)[0]
        if n_genes < len(sampling_gene_set):
            gene_select_ind = np.random.choice(sampling_gene_set, n_genes, replace=False)
            log.info(f"{n_genes} random genes selected.")
        else:
            gene_select_ind = sampling_gene_set
            log.warning(f"{len(sampling_gene_set)} random genes selected: cannot satisfy query of {n_genes} genes.")
        
        selected_genes_filter = np.zeros(exp_filter.shape, dtype=bool)
        selected_genes_filter[gene_select_ind] = True
        selected_genes_filter[enforced_genes] = True

        selected_genes = gene_name_reference[selected_genes_filter]
        sampling_gene_set = gene_name_reference[exp_filter]
        log.info(f"Total of {selected_genes_filter.sum()} genes selected.")
    
    for adata in adata_list:
        adata.var['selected_genes'] = selected_genes_filter

    return adata_list

