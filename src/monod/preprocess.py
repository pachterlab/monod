import numpy as np
import matplotlib.pyplot as plt

import time

import loompy as lp  # will be necessary later, probably
import anndata as ad
import os
from datetime import datetime
import pytz
import collections
import csv
import warnings
import scipy
from scipy import sparse

code_ver_global = "025"  # bumping up version early Nov

########################
## Debug and error logging
########################
import logging, sys

logging.basicConfig(stream=sys.stdout)
log = logging.getLogger()
log.setLevel(logging.WARNING)

import logging.config

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": True,
    }
)

########################
## Main code
########################


def construct_batch(
    dataset_filepaths,
    transcriptome_filepath,
    dataset_names,
    batch_id=1,
    n_genes=100,
    seed=2813308004,
    viz=True,
    filt_param={
        "min_U_mean": 0.01,
        "min_S_mean": 0.01,
        "max_U_max": 400,
        "max_S_max": 400,
        "min_U_max": 3,
        "min_S_max": 3,
    },
    attribute_names=[("unspliced", "spliced"), "gene_name", "barcode"],
    meta="batch",
    datestring=datetime.now(pytz.timezone("US/Pacific")).strftime("%y%m%d"),
    creator="gg",
    code_ver=code_ver_global,
    batch_location=".",
    cf=None,
    exp_filter_threshold=1,
    genes_to_fit=[],
):
    """Pre-process data and identify genes to fit.

    This function runs basic pre-processing on the batch, creates directories, and writes a
    list of genes with sufficient data to analyze.

    Parameters
    ----------
    dataset_filepaths: str, list of str, or list of AnnData
        if str, path to a single file that contains all batch data (disabled).
        if list of str, paths to multiple files (disabled).
        if list of AnnData, a single object in memory that contains all batch data.
    transcriptome_filepath: str
        location of the transcriptome length reference.
    dataset_names: list of str
        dataset metadata; names assigned to dataset-specific directories.
    batch_id: int, optional
        batch index, occasionally useful if reproducing analysis with different genes.
    n_genes: int, optional
        how many genes to analyze.
    seed: int, optional
        random number generator seed, used to select genes.
    viz: bool, optional
        whether to visualize and store spliced and unspliced count statistics.
    filt_param: dict, optional
        parameters used to select genes with sufficient data to fit.
        'min_U_mean' and 'min_S_mean': lowest allowable count mean.
        'max_U_max' and 'max_S_max': highest allowable count maximum.
        'min_U_max' and 'min_S_max': lowest allowable count maximum.
    attribute_names: length-3 tuple, optional
        entry 0: layers to use (typically 'unspliced' and 'spliced').
        entry 1: variable name (typically 'gene_name')
        entry 2: observation name (typically 'barcode')
        Entries 1 and 2 are unused in the typical workflow, and kept in for compatibility with
        files not loaded into memory.
    meta: str, optional
        batch name.
    datestring: str, optional
        batch date.
    creator: str, optional
        batch creator.
    code_ver: str, optional
        Monod version used to fit data.
    batch_location: str, optional
        directory where the analysis should be located.
    cf: bool np.ndarray, optional
        array of size n_datasets x n_cells, mandatory if several datasets are stored in a single file.
        reports which cells are assigned to which dataset.
    exp_filter_threshold: None or float
        internal parameter for selecting genes with sufficient data for analysis.
        if None, force to analyze n_genes genes.
        if a float, analyze a subset of at most n_genes genes that pass quality thresholds in at least
        a fraction exp_filter_threshold of the datasets.
    genes_to_fit: list of str
        genes that are required to be fit by Monod, bypassing the filtering routine.

    Returns
    -------
    dir_string: str
        batch directory location.
    dataset_strings: list of str
        locations of the directories with data and fits for each dataset.
    """
    log.info("Beginning data preprocessing and filtering.")
    dir_string = (
        batch_location.rstrip("/")
        + "/"
        + ("_".join((creator, datestring, code_ver, meta, str(batch_id))))
    )
    make_dir(dir_string)
    dataset_strings = []

    if type(dataset_filepaths) is str:
        dataset_filepaths = [
            dataset_filepaths
        ]  # if we get in a single string, we are processing one file /
    n_datasets = len(dataset_filepaths)

    if type(attribute_names[-1]) is str:
        attribute_names = [attribute_names] * n_datasets

    transcriptome_dict = get_transcriptome(transcriptome_filepath)

    for dataset_index in range(n_datasets):
        dataset_filepath = dataset_filepaths[dataset_index]
        log.info("Dataset: " + dataset_names[dataset_index])
        dataset_attr_names = attribute_names[
            dataset_index
        ]  # pull out the correct attribute names
        if cf is None:
            dataset_cf = None
        else:
            dataset_cf = cf[dataset_index]
        layers, gene_names, n_cells = import_raw(
            dataset_filepath, dataset_attr_names, dataset_cf
        )
        log.info(str(n_cells) + " cells detected.")

        # identify genes that are in the length annotations. discard all the rest.
        # for models without length-based sequencing, this may not be necessary
        # though I do not see the purpose of analyzing genes that not in the
        # genome.
        # if this is ever necessary, just make a different reference list.
        annotation_filter = identify_annotated_genes(gene_names, transcriptome_dict)
        *layers, gene_names = filter_by_gene(annotation_filter, *layers, gene_names)
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

        # initialize the gene length array.
        len_arr = np.array([transcriptome_dict[k] for k in gene_names])

        S = layers[1]
        U = layers[0]
        gene_exp_filter = threshold_by_expression(S, U, filt_param)
        if viz:
            var_name = ("S", "U")
            var_arr = (S.mean(1), U.mean(1))

            fig1, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
            for i in range(2):
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

        expression_filter_array[dataset_index, :] = gene_exp_filter

        *layers, gene_names = filter_by_gene(gene_exp_filter, *layers, gene_names)

        dataset_dir_string = dir_string + "/" + dataset_names[dataset_index]
        make_dir(dataset_dir_string)
        dataset_strings.append(dataset_dir_string)

    exp_fractions = expression_filter_array.mean(0)
    if exp_filter_threshold is None:  # no-tuning
        n_genes_enforced = 0
        selected_genes_filter = np.zeros(exp_fractions.shape, dtype=bool)
        for gene in genes_to_fit:
            gene_loc = np.where(gene_name_reference == gene)[0]
            if len(gene_loc) == 0:
                log.warning(
                    "Gene {} not found or has multiple entries in annotations.".format(
                        gene
                    )
                )
            elif len(gene_loc) > 1:
                log.error(
                    "Multiple entries found for gene {}: this should never happen.".format(
                        gene
                    )
                )
            else:
                exp_fractions[gene_loc[0]] = 0
                n_genes_enforced += 1
                selected_genes_filter[gene_loc[0]] = True

        q = np.quantile(
            exp_fractions, 1 - (n_genes - n_genes_enforced) / len(exp_fractions)
        )
        selected_genes_filter[exp_fractions > q] = True
        np.random.seed(seed)
        random_genes = np.where(exp_fractions == q)[0]
        random_genes_sel = np.random.choice(
            random_genes, n_genes - selected_genes_filter.sum(), replace=False
        )
        selected_genes_filter[random_genes_sel] = True

        selected_genes = gene_name_reference[selected_genes_filter]
        log.info("Total of " + str(selected_genes_filter.sum()) + " genes selected.")
        log.warning(
            "Selecting "
            + str(selected_genes_filter.sum())
            + " genes required {:.1f}% dataset threshold.".format(q * 100)
        )
        save_gene_list(dir_string, selected_genes, "genes")

    else:
        exp_filter = exp_fractions >= exp_filter_threshold

        log.info(
            "Gene set size according to a {:.1f}% dataset threshold: ".format(
                exp_filter_threshold * 100
            )
            + str(exp_filter.sum())
        )

        enforced_genes = np.zeros(exp_filter.shape, dtype=bool)
        for gene in genes_to_fit:
            gene_loc = np.where(gene_name_reference == gene)[0]
            if len(gene_loc) == 0:
                log.warning(
                    "Gene {} not found or has multiple entries in annotations.".format(
                        gene
                    )
                )
            elif len(gene_loc) > 1:
                log.error(
                    "Multiple entries found for gene {}: this should never happen.".format(
                        gene
                    )
                )
            else:
                exp_filter[gene_loc[0]] = False
                n_genes -= 1
                enforced_genes[gene_loc[0]] = True

        np.random.seed(seed)
        sampling_gene_set = np.where(exp_filter)[0]
        if n_genes < len(sampling_gene_set):
            gene_select_ind = np.random.choice(
                sampling_gene_set, n_genes, replace=False
            )
            log.info(str(n_genes) + " random genes selected.")
        else:
            gene_select_ind = sampling_gene_set
            log.warning(
                str(len(sampling_gene_set))
                + " random genes selected: cannot satisfy query of "
                + str(n_genes)
                + " genes."
            )

        selected_genes_filter = np.zeros(exp_filter.shape, dtype=bool)
        exp_filter = exp_filter | enforced_genes
        selected_genes_filter[gene_select_ind] = True
        selected_genes_filter[enforced_genes] = True

        selected_genes = gene_name_reference[selected_genes_filter]
        sampling_gene_set = gene_name_reference[exp_filter]
        log.info("Total of " + str(selected_genes_filter.sum()) + " genes selected.")

        save_gene_list(dir_string, selected_genes, "genes")
        save_gene_list(dir_string, sampling_gene_set, "gene_set")

    if viz:
        diagnostics_dir_string = dir_string + "/diagnostic_figures"
        make_dir(diagnostics_dir_string)
        for figure_ind in plt.get_fignums():
            plt.figure(figure_ind)
            plt.savefig(
                diagnostics_dir_string
                + "/{}.png".format(dataset_names[figure_ind - 1]),
                dpi=450,
            )
    return dir_string, dataset_strings


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


def threshold_by_expression(
    S,
    U,
    filt_param={
        "min_U_mean": 0.01,
        "min_S_mean": 0.01,
        "max_U_max": 350,
        "max_S_max": 350,
        "min_U_max": 4,
        "min_S_max": 4,
    },
):
    """Convenience function for filtering genes.

    This function takes in raw spliced and unspliced counts, as well as
    threshold parameters, and outputs a boolean filter of genes that meet
    these thresholds.

    Parameters
    ----------
    S: np.ndarray
         genes x cells spliced count matrix.
    U: np.ndarray
         genes x cells unspliced count matrix.

    Returns
    -------
    gene_exp_filter: bool np.ndarray
        genes that meet the expression thresholds.
    """
    S_max = S.max(1)
    U_max = U.max(1)
    S_mean = S.mean(1)
    U_mean = U.mean(1)

    gene_exp_filter = (
        (U_mean > filt_param["min_U_mean"])
        & (S_mean > filt_param["min_S_mean"])
        & (U_max < filt_param["max_U_max"])
        & (S_max < filt_param["max_S_max"])
        & (U_max > filt_param["min_U_max"])
        & (S_max > filt_param["min_S_max"])
    )
    log.info(str(np.sum(gene_exp_filter)) + " genes retained after expression filter.")
    return gene_exp_filter


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


def import_raw(filename, attribute_names, cf):
    """Import raw count data from a dataset.

    This function attempts to import raw data from disk or memory objects.
    In the current version, only the memory object interface, which
    parses an AnnData file, is accessible.

    Parameters
    ----------
    filename: AnnData
        object containing spliced and unspliced raw RNA counts.
    attribute_names: length-3 tuple
        entry 0: layers to use (typically 'unspliced' and 'spliced').
        entry 1: variable name (typically 'gene_name')
        entry 2: observation name (typically 'barcode')
        Entries 1 and 2 are unused in the typical workflow, and kept in for compatibility with
        files not loaded into memory.
        Gene names should be accessibles as counts.var_names.
    cf: bool np.ndarray, optional
        array of size n_datasets x n_cells, mandatory if several datasets are stored in a single file.
        reports which cells are assigned to which dataset.

    Output:
    out: tuple
        entry 0: np.ndarray
            layers (typically unspliced, then unspliced)
        entry 1: str np.ndarray
            gene names
        entry 2: int
            number of cells in the dataset
    """

    if type(filename) is str:
        fn_extension = filename.split(".")[-1]
        if fn_extension == "loom":  # loom file
            return import_vlm(filename, attribute_names, cf)
        elif fn_extension == "h5ad":
            raise ValueError(
                "This functionality is unsupported in the current version of Monod."
            )
            return import_h5ad(filename, attribute_names, cf)
        else:
            raise ValueError(
                "This functionality is unsupported in the current version of Monod."
            )
            return import_mtx(filename)
    else:  # assume passing in anndata
        return process_h5ad(filename, attribute_names, cf)


def process_h5ad(file, attribute_names, cf=None):
    """
    This function attempts to import raw data from disk or memory objects.
    In the current version, only the memory object interface, which
    parses an AnnData file, is accessible.

    Parameters
    ----------
    file: AnnData
        object containing spliced and unspliced raw RNA counts.
    attribute_names: length-3 tuple
        entry 0: layers to use (typically 'unspliced' and 'spliced').
        entry 1: variable name (typically 'gene_name')
        entry 2: observation name (typically 'barcode')
        Entries 1 and 2 are unused in the typical workflow.
        Gene names should be accessibles as counts.var_names.
    cf: bool np.ndarray, optional
        array of size n_datasets x n_cells, mandatory if several datasets are stored in a single file.
        reports which cells are assigned to which dataset.

    Returns
    -------
    layers: int np.ndarray
        raw data from the layers of interest
    gene_names: str np.ndarray
        gene names
    nCells: int
        number of cells in the dataset
    """

    layer_names, gene_attr, cell_attr = attribute_names
    if cf is None:
        cf = np.ones(file.shape[0], dtype=bool)

    layers = [file[cf].layers[layer].astype(int).T for layer in layer_names]
    if scipy.sparse.issparse(layers[0]):
        layers = [x.todense() for x in layers]
    layers = np.asarray(layers)  # hmm

    gene_names = file.var_names.to_numpy()
    nCells = layers.shape[2]
    warnings.resetwarnings()
    return layers, gene_names, nCells


# the next three functions are obsolete and should not be called in the current version.
def import_h5ad(filename, attribute_names, cf=None):
    """
    Imports an anndata file with spliced and unspliced RNA counts.
    Note row/column convention is opposite loompy.
    Conventions as in import_raw.
    """
    spliced_layer, unspliced_layer, gene_attr, cell_attr = attribute_names
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    ds = ad.read_h5ad(filename, backed="r")
    if cf is None:
        cf = np.ones(ds.shape[0], dtype=bool)
    # ds = ds[cf]
    S = ds[cf].layers[spliced_layer].T
    U = ds[cf].layers[unspliced_layer].T
    if scipy.sparse.issparse(S):
        S = np.asarray(S.todense())
        U = np.asarray(U.todense())
    # gene_names = ds.var[gene_attr].to_numpy()
    gene_names = ds.var_names.to_numpy()
    nCells = S.shape[1]
    warnings.resetwarnings()
    return S, U, gene_names, nCells


def import_mtx(dir_name):
    """
    Imports mtx files with spliced and unspliced RNA counts via anndata object.
    Note row/column convention is opposite loompy.
    Conventions as in import_raw.

    mtx files typically have *.genes.txt files with gene IDs rather than names. Beware incompatibilities.
    """
    dir_name = dir_name.rstrip("/")
    ds = ad.read_mtx(dir_name + "/spliced.mtx")
    nCells = ds.shape[0]
    S = ds.X.todense().T
    U = ad.read_mtx(dir_name + "/unspliced.mtx").X.todense().T
    gene_names = np.loadtxt(dir_name + "/spliced.genes.txt", dtype=str)
    return S, U, gene_names, nCells


def import_vlm(filename, attribute_names, cf=None):
    """
    Imports mtx files with spliced and unspliced RNA counts via anndata object.
    Note that there is a new deprecation warning in the h5py package
    underlying loompy.

    Conventions as in import_raw.
    """
    layer_names, gene_attr, cell_attr = attribute_names
    # spliced_layer, unspliced_layer, gene_attr, cell_attr = attribute_names
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    with lp.connect(filename, mode="r") as ds:
        if cf is None:
            cf = np.ones(len(ds.ca[cell_attr]), dtype=bool)
        layers = [ds.layers[layer][:][:, cf] for layer in layer_names]
        gene_names = ds.ra[gene_attr]
    layers = np.asarray(layers, dtype=int)
    nCells = layers.shape[2]
    warnings.resetwarnings()
    return layers, gene_names, nCells


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
    repeat_dict = {}
    len_dict = {}
    thr_ind = repeat_thr - 3
    with open(transcriptome_filepath, "r") as file:
        for line in file.readlines():
            d = [i for i in line.split(" ") if i]
            repeat_dict[d[0]] = int(d[thr_ind])
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
    sel_ind_annot = [k for k in range(len(gene_names)) if gene_names[k] in feat_dict]

    NAMES = [gene_names[k] for k in sel_ind_annot]
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
