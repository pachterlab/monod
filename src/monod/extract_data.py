import numpy as np
import matplotlib.pyplot as plt
import pickle
import anndata as ad
import warnings
import zlib
from pathlib import Path
import pandas as pd

from scipy.sparse import issparse

from ._core import log, to_dense

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
    genes_to_fit=None,
    hist_type='unique',
    padding=None,
    mek_means_params=None,
    ambiguous=None,
):
    """Select the genes to fit and prepare their counts for inference.

    Parameters
    ----------
    h5ad_filepath: str or anndata.AnnData
        path to an h5ad file, or an AnnData in memory.  The input is not modified.
    model: monod.cme_toolbox.CMEModel
        the model to be fit; names the modalities (layers) to read.
    dataset_name: str, optional
        dataset name for the logs; defaults to "<bio_model>_<seq_model>".
    transcriptome_filepath: str, optional
        location of the transcriptome length reference.  Genes without a length
        are discarded; without a reference, ``var['log_lengths']`` or
        ``var['gene_length']`` are used when present.
    n_genes: int, optional
        number of genes to select among those passing the expression filter.
    seed: int, optional
        seed for the random gene selection.
    viz: bool, optional
        whether to plot the spliced and unspliced count statistics of the selected genes.
    filt_param: dict, optional
        expression filter bounds (see ``threshold_by_expression``); defaults to
        ``model.filter_bounds``.
    modality_name_dict: dict, optional
        mapping from the model's modality names to the AnnData layer names;
        by default the layers are assumed to carry the modality names.
    genes_to_fit: list of str, optional
        genes to include regardless of the random selection.
    hist_type: str, optional
        flavor of histogram; only 'unique' (observed microstate frequencies) is supported.
    padding: list of int or None
        the PMF is evaluated on a grid of max count + padding[z] per species Z;
        10 for each species by default.
    mek_means_params: (k, epochs) or None
        recorded in ``uns`` for meK-Means inference.
    ambiguous: None, a modality name, 'drop' or 'model'
        what to do with an ``ambiguous`` layer (reads that could belong to
        either transcript; the layer name is ``modality_name_dict['ambiguous']``
        when given).  A modality name (``'spliced'`` or ``'unspliced'`` for the
        built-in models, a species name for a custom network) adds its counts
        to that channel, ``'drop'`` ignores it, ``'model'`` fits it as a third channel
        and needs a model with ``amb_model`` (the default when the model has
        one).  When the layer is present and no choice is made, the call
        fails rather than ignore the reads silently: some quantifications
        already fold them into ``spliced``, in which case ``'drop'`` is the
        right choice and folding again would count them twice.  The choice is
        recorded in ``uns['ambiguous']``.

    Returns
    -------
    anndata.AnnData
        the selected genes with the model's layers dense, the grid limits in
        ``var['M_<layer>']`` (one column per modality, read back with
        :func:`grid_limits`, so gene subsets of the AnnData keep them),
        ``uns['model']``, ``uns['modality_name_dict']``, ``uns['hist_type']``
        and the moments in ``var``.
    """
    if hist_type != "unique":
        raise ValueError(
            f"hist_type={hist_type!r} is not implemented; only 'unique' (per-gene unique "
            "microstate counts) is supported in this version."
        )
    log.info("Beginning data extraction.")
    if dataset_name is None:
        dataset_name = f"{model.bio_model}_{model.seq_model}"
    log.info("Dataset: %s", dataset_name)

    if not filt_param:
        filt_param = model.filter_bounds
        log.info('Using default gene filtering parameters')
    if genes_to_fit is None:
        genes_to_fit = []
    if not modality_name_dict:
        # Assume the layers are named after the model's modalities, e.g. 'spliced', 'unspliced'.
        modality_name_dict = {v: v for v in model.model_modalities}
        log.info(
            "Reading layers %s. If your AnnData layers use other names, pass "
            "modality_name_dict={<modality>: <layer>, ...}.", model.model_modalities,
        )
    layer_names = [modality_name_dict[modality] for modality in model.model_modalities]

    source = import_h5ad(h5ad_filepath) if isinstance(h5ad_filepath, str) else h5ad_filepath
    if source.isbacked:
        source = source.to_memory()
    missing = [layer for layer in layer_names if layer not in source.layers]
    if missing:
        raise KeyError(f"layers {missing} not found in the AnnData (available: {list(source.layers)}); "
                       "pass modality_name_dict={<modality>: <layer>, ...}.")

    # A new AnnData with only the model's layers, dense.  The metadata tables
    # are copied so the input is left untouched; dense layers are shared.
    monod_adata = _plain_string_index(ad.AnnData(X=source.X, obs=source.obs.copy(), var=source.var.copy()))
    for layer in layer_names:
        monod_adata.layers[layer] = to_dense(source.layers[layer])
    ambiguous = _handle_ambiguous(monod_adata, source, model, modality_name_dict, ambiguous)
    if ambiguous is not None:
        monod_adata.uns['ambiguous'] = ambiguous
    # only the model's modalities: an 'ambiguous' entry used for folding is not a channel of the fit
    monod_adata.uns['modality_name_dict'] = {m: modality_name_dict[m] for m in model.model_modalities}
    monod_adata.uns['model'] = model.to_dict()   # read by threshold_by_expression

    if transcriptome_filepath:
        # Mark the genes whose length is known, store lengths and log_lengths, and keep only those genes.
        transcriptome_dict = get_transcriptome(transcriptome_filepath)
        monod_adata = add_gene_lengths(monod_adata, transcriptome_dict, attribute_name='length_given')
        monod_adata = monod_adata[:, monod_adata.var['length_given'] == 1].copy()
    elif 'log_lengths' in monod_adata.var.columns:
        log.info('Using log_lengths from adata.var for gene length bias.')
    elif 'gene_length' in monod_adata.var.columns:
        # Same convention as the transcriptome path: log10 lengths.
        monod_adata.var['log_lengths'] = np.log10(monod_adata.var['gene_length'])
        log.info('Computed log_lengths from adata.var[gene_length] for gene length bias.')

    # Filter genes by expression, then pick random genes up to the desired number.
    monod_adata = process_adata(monod_adata, filt_param, genes_to_fit, n_genes, seed=seed)
    if viz:
        if transcriptome_filepath:
            visualize_gene_filtering_lengths(monod_adata)
        else:
            visualize_gene_filtering(monod_adata)
    monod_adata = monod_adata[:, monod_adata.var['selected_genes'].astype(bool)].copy()

    # Grid sizes: the largest count per modality and gene, plus padding.
    layers = np.array([to_dense(monod_adata.layers[layer]).T for layer in layer_names])  # (n_layers, n_genes, n_cells)
    if padding is None:
        padding = [10] * len(layer_names)
    M = layers.max(axis=2) + np.asarray(padding)[:, None]
    if model.bio_model == "Chromatin":
        # the configuration axis is categorical with exactly 2**n_sites states
        M[model.model_modalities.index('configuration'), :] = 2 ** model.n_sites
    for row, layer in zip(M.astype(int), layer_names):
        monod_adata.var[f'M_{layer}'] = row
    monod_adata.uns['hist_type'] = hist_type

    # Histograms are not stored in uns: searchdata_from_adata rebuilds them from
    # the layers (in Rust when available), which is faster than unpickling and
    # keeps the AnnData writable as a plain h5ad.
    # Moments are keyed by modality (MOM_unspliced_mean, ...), whatever the layers are called:
    # the method-of-moments starts read them by modality.
    monod_adata = add_moments(monod_adata, modality_name_dict)
    if mek_means_params:
        monod_adata.uns['k'], monod_adata.uns['epochs'] = mek_means_params
    return _plain_string_index(monod_adata)


def _handle_ambiguous(monod_adata, source, model, modality_name_dict, ambiguous):
    """Apply the ``ambiguous`` choice of :func:`extract_data` and return it
    (``'model'`` when the model carries ``amb_model``)."""
    amb_layer = modality_name_dict.get('ambiguous', 'ambiguous')
    present = amb_layer in source.layers
    modelled = getattr(model, 'amb_model', 'None') != 'None'
    if ambiguous is None and modelled:
        ambiguous = 'model'
    fold_targets = [m for m in model.model_modalities if m != 'ambiguous']
    if ambiguous is not None and ambiguous not in ('model', 'drop') and ambiguous not in fold_targets:
        raise ValueError(f"ambiguous must be one of 'model', 'drop' or a modality to fold into {fold_targets}; "
                         f"got {ambiguous!r}")
    if modelled and ambiguous != 'model':
        raise ValueError(f"the model fits the ambiguous layer (amb_model={model.amb_model!r}); "
                         f"ambiguous={ambiguous!r} conflicts with that")
    if ambiguous == 'model' and not modelled:
        raise ValueError("ambiguous='model' needs a CMEModel with amb_model='Equal' or 'Unequal'")
    if ambiguous is None and present:
        raise ValueError(
            f"the AnnData has an ambiguous layer ({amb_layer!r}); say what to do with it: "
            f"ambiguous=<modality> with one of {fold_targets} (add its counts to that channel), 'drop' (ignore it), "
            "or 'model' with a CMEModel(amb_model='Equal' | 'Unequal') (fit the probability that a count "
            "is reported as ambiguous). If the spliced layer already holds these reads, 'drop' is the choice."
        )
    if ambiguous in fold_targets:
        if not present:
            raise KeyError(f"ambiguous={ambiguous!r} but the AnnData has no {amb_layer!r} layer "
                           f"(available: {list(source.layers)})")
        target = modality_name_dict[ambiguous]
        monod_adata.layers[target] = monod_adata.layers[target] + to_dense(source.layers[amb_layer])
        log.info("Ambiguous counts (%r) added to the %s layer %r.", amb_layer, ambiguous, target)
    elif ambiguous == 'drop' and present:
        log.info("Ambiguous layer %r dropped.", amb_layer)
    return ambiguous


def unique_histograms(layers):
    """Per-gene histograms of the unique joint states.

    ``layers`` are dense or sparse cells x genes arrays, one per modality, in
    the model's modality order.  Returns one ``(coords, freqs)`` pair per gene:
    the distinct state vectors (integer array, n_states x n_modalities) and
    their frequencies.  This is the Python mirror of the Rust
    ``make_state_dist``; the inference pipeline uses the Rust one whenever the
    extension is installed."""
    layers = [to_dense(x) for x in layers]
    n_cells, n_genes = layers[0].shape
    hist = []
    for g in range(n_genes):
        unique, counts = np.unique(np.column_stack([x[:, g] for x in layers]), axis=0, return_counts=True)
        hist.append((unique.astype(int), counts / n_cells))
    return hist


def layer_moments(layers, names):
    """Per-gene means, variances and pairwise covariances of the modalities.

    ``layers`` as in :func:`unique_histograms`, ``names`` their modality names.
    Returns one dict per gene with keys ``MOM_<name>_mean``, ``MOM_<name>_var``
    (population variance) and ``MOM_cov_<a>_<b>`` (sample covariance, as
    ``np.cov``), the keys the method-of-moments starts read.  Mirror of the
    moments a Rust ``SearchData`` computes when it is built."""
    X = [np.asarray(to_dense(x), dtype=float) for x in layers]
    n_cells, n_genes = X[0].shape
    cols = {}
    for name, x in zip(names, X):
        cols[f"MOM_{name}_mean"] = x.mean(0)
        cols[f"MOM_{name}_var"] = x.var(0)
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            di, dj = X[i] - X[i].mean(0), X[j] - X[j].mean(0)
            denom = max(n_cells - 1, 1)
            cols[f"MOM_cov_{names[i]}_{names[j]}"] = (di * dj).sum(0) / denom
    return [{k: float(v[g]) for k, v in cols.items()} for g in range(n_genes)]


def add_moments(adata, modality_name_dict=None):
    """Write each gene's moments (see :func:`layer_moments`) into ``adata.var``
    as ``MOM_*`` columns, naming the modalities through ``modality_name_dict``
    (``{modality: layer}``) or by their layer names.  Returns ``adata``."""
    layer_names = list(adata.layers.keys())
    if modality_name_dict:
        layer_to_modality = {v: k for k, v in modality_name_dict.items()}
    else:
        layer_to_modality = {v: v for v in layer_names}
    names = [layer_to_modality[l] for l in layer_names]
    moments = layer_moments([adata.layers[l] for l in layer_names], names)
    for key in moments[0] if moments else []:
        adata.var[key] = np.array([m[key] for m in moments])
    return adata


def _ordered_layer_names(monod_adata):
    """The AnnData's layer names in the model's modality order."""
    modality_name_dict = monod_adata.uns['modality_name_dict']
    model = model_from_uns(monod_adata.uns['model'])
    return [modality_name_dict[m] for m in model.model_modalities]


def _ordered_layers(monod_adata):
    """The AnnData's layers in the model's modality order."""
    return [monod_adata.layers[name] for name in _ordered_layer_names(monod_adata)]


def grid_limits(monod_adata):
    """Grid limits of a monod AnnData as an int array of shape (n_layers, n_genes),
    layers in the model's modality order.

    :func:`extract_data` stores them in ``var['M_<layer>']``, one column per
    modality, so they follow the genes through any AnnData subset.  Files
    written by older versions carry them as ``uns['M']`` instead, which is read
    when the columns are absent.
    """
    cols = [f'M_{name}' for name in _ordered_layer_names(monod_adata)]
    if all(c in monod_adata.var.columns for c in cols):
        return np.asarray(monod_adata.var[cols].to_numpy().T, dtype=int)
    if 'M' in monod_adata.uns:
        return np.asarray(monod_adata.uns['M'], dtype=int)
    raise KeyError(f"no grid limits in the AnnData: expected var columns {cols} (written by extract_data)")


def select_expressed_genes(adata, min_cells=10, n_genes=None, layer='spliced'):
    """Names of the genes with a nonzero count in at least ``min_cells`` cells,
    in the AnnData's gene order; the first ``n_genes`` of them when given.

    A deterministic alternative to the seeded random draw of
    :func:`extract_data`: pass the result as ``genes_to_fit`` (with
    ``n_genes=len(genes)``).  ``layer`` names the layer to count in, or ``None``
    for ``X``.  Sparse layers are counted without densifying them.
    """
    X = adata.X if layer is None else adata.layers[layer]
    if issparse(X):
        n_cells_expr = np.asarray((X != 0).sum(axis=0)).ravel()
    else:
        n_cells_expr = np.count_nonzero(np.asarray(X), axis=0)
    names = adata.var_names[n_cells_expr >= min_cells]
    return list(names if n_genes is None else names[:n_genes])


def make_histogram(monod_adata, hist_type="unique", M=None):
    """Per-gene unique-state histograms of a monod AnnData, layers in the
    model's modality order (:func:`unique_histograms`).  ``hist_type`` must be
    ``"unique"``; ``M`` is accepted for compatibility and not needed."""
    if hist_type != "unique":
        raise ValueError(f"hist_type must be 'unique'; got {hist_type!r}")
    return unique_histograms(_ordered_layers(monod_adata))

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
    

def combine_chunks(chunk_dir, output_file):
    """Reassemble an h5ad file shipped as zlib-compressed chunks.

    The `monod_examples <https://github.com/pachterlab/monod_examples>`_
    repository stores each example AnnData as ``chunk_0000.gz``,
    ``chunk_0001.gz``, ... in one directory.  The chunks are decompressed in
    name order and concatenated into ``output_file``, whose path is returned.
    """
    chunks = sorted(Path(chunk_dir).glob('chunk_*.gz'))
    if not chunks:
        raise FileNotFoundError(f"no chunk_*.gz files in {chunk_dir}")
    with open(output_file, 'wb') as out:
        for chunk in chunks:
            out.write(zlib.decompress(chunk.read_bytes()))
    return output_file


def import_h5ad(filename):
    """Read an h5ad file in backed mode, making duplicate gene names unique."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ds = ad.read_h5ad(filename, backed="r")
        if not ds.var_names.is_unique:
            log.info("Duplicate variable names found. Making them unique...")
            ds.var_names_make_unique()
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

def process_adata(adata, filt_param, genes_to_fit, n_genes, seed=5):
    """
    Processes a single AnnData object, applies expression filters, and stores the results in AnnData.

    Parameters
    ----------
    adata: anndata.AnnData
        AnnData object to be processed.
    filt_param: dict
        Filtering parameters for expression.
    genes_to_fit: list
        List of genes to be specifically included in the selection.
    n_genes: int
        Number of genes to select.
    seed: int
        Random seed for reproducibility.

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
