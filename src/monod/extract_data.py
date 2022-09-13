import numpy as np
import csv
import matplotlib.pyplot as plt
import pickle

from .preprocess import (
    code_ver_global,
    filter_by_gene,
    make_dir,
    import_raw,
    process_h5ad,
    get_transcriptome,
    identify_annotated_genes,
    log,
)


########################
## Main code
########################


def extract_data(
    dataset_filepath,
    transcriptome_filepath,
    dataset_name,
    dataset_string,
    dir_string,
    viz=True,
    dataset_attr_names=[("unspliced", "spliced"), "gene_name", "barcode"],
    padding=None,
    cf=None,
    hist_type="unique",
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
    attribute_names: length-3 tuple, optional
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
    search_data: SearchData object.

    """

    log.info("Beginning data extraction.")
    log.info("Dataset: " + dataset_name)

    transcriptome_dict = get_transcriptome(transcriptome_filepath)
    layers, gene_names, n_cells = import_raw(dataset_filepath, dataset_attr_names, cf)
    # S,U,gene_names,n_cells = import_raw(dataset_filepath,dataset_attr_names,cf)

    if padding is None:
        padding = np.asarray([10] * len(layers))

    # identify genes that are in the length annotations. discard all the rest.
    # for models without length-based sequencing, this may not be necessary
    # though I do not see the purpose of analyzing genes that not in the
    # genome.
    # if this is ever necessary, just make a different reference list.
    annotation_filter = identify_annotated_genes(gene_names, transcriptome_dict)
    *layers, gene_names = filter_by_gene(annotation_filter, *layers, gene_names)

    # initialize the gene length array.
    len_arr = np.array([transcriptome_dict[k] for k in gene_names])

    gene_result_list_file = dir_string + "/genes.csv"
    try:
        with open(gene_result_list_file, newline="") as f:
            reader = csv.reader(f)
            analysis_gene_list = list(reader)[0]
        log.info("Gene list extracted from {}.".format(gene_result_list_file))
    except:
        log.error(
            "Gene list could not be extracted from {}.".format(gene_result_list_file)
        )
        # raise an error here in the next version.

    if viz:
        S = layers[1]
        U = layers[0]
        var_name = ("S", "U")
        var_arr = (S.mean(1), U.mean(1))
        fig1, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
        for i in range(2):
            ax1[i].scatter(
                np.log10(len_arr),
                np.log10(var_arr[i] + 0.001),
                s=3,
                c="silver",
                alpha=0.15,
            )
            ax1[i].set_xlabel("log10 gene length")
            ax1[i].set_ylabel("log10 (mean " + var_name[i] + " + 0.001)")

    gene_names = list(gene_names)
    gene_filter = [gene_names.index(gene) for gene in analysis_gene_list]
    gene_names = np.asarray(gene_names)
    *layers, gene_names, len_arr = filter_by_gene(
        gene_filter, *layers, gene_names, len_arr
    )

    S = layers[1]
    U = layers[0]
    if viz:
        for i in range(2):
            var_arr = (S.mean(1), U.mean(1))
            ax1[i].scatter(
                np.log10(len_arr),
                np.log10(var_arr[i] + 0.001),
                s=5,
                c="firebrick",
                alpha=0.9,
            )
        dataset_diagnostics_dir_string = dataset_string + "/diagnostic_figures"
        make_dir(dataset_diagnostics_dir_string)
        plt.savefig(
            dataset_diagnostics_dir_string + "/{}.png".format(dataset_name), dpi=450
        )

    n_genes = len(gene_names)
    M = np.amax(layers, axis=2) + padding[:, None]

    gene_log_lengths = np.log10(len_arr)

    hist = []
    moments = []
    for gene_index in range(n_genes):
        if hist_type == "grid":
            H, xedges, yedges = np.histogramdd(
                *[x[gene_index] for x in layers],
                bins=[np.arange(x[gene_index] + 1) - 0.5 for x in M],
                density=True
            )
        elif hist_type == "unique":
            unique, unique_counts = np.unique(
                np.vstack([x[gene_index] for x in layers]).T, axis=0, return_counts=True
            )
            frequencies = unique_counts / n_cells
            unique = unique.astype(int)
            H = (unique, frequencies)

        hist.append(H)

        moments.append(
            {
                "S_mean": S[gene_index].mean(),
                "U_mean": U[gene_index].mean(),
                "S_var": S[gene_index].var(),
                "U_var": U[gene_index].var(),
            }
        )

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
    ]

    layers = np.asarray(layers)
    search_data = SearchData(
        attr_names,
        M,
        hist,
        moments,
        gene_log_lengths,
        n_genes,
        gene_names,
        n_cells,
        layers,
        hist_type,
    )

    search_data_string = dataset_string + "/raw.sd"
    store_search_data(search_data, search_data_string)
    return search_data


########################
## Helper functions
########################
def store_search_data(search_data, search_data_string):
    """Attempt to store a search data object to disk.

    Parameters
    ----------
    search_data: SearchData
        object to store.
    search_data_string: str
        desired disk location for the SearchData object.
    """
    try:
        with open(search_data_string, "wb") as sdfs:
            pickle.dump(search_data, sdfs)
        log.info("Search data stored to {}.".format(search_data_string))
    except:
        log.error("Search data could not be stored to {}.".format(search_data_string))


########################
## Helper classes
########################
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
        moments include 'S_mean', 'U_mean', 'S_var', 'U_var', and are used to define MoM estimates.
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

    def get_noise_decomp(self, sizefactor="pf", lognormalize=True, pcount=0):
        """
        This method performs normalization and variance stabilization on the raw data, and
        reports the fractions of normalized variance retained and removed as a result of the process.
        The unspliced and spliced species are analyzed independently.

        Parameters
        ----------
        sizefactor: str, float, int, or None, optional
            what size factor to use.
            If 'pf', use proportional fitting; set the size of each cell to the mean size.
            If int or float, use this number (e.g., 1e4 for cp10k).
            If None, do not do size/depth normalization.
        lognormalize: bool, optional
            whether to apply log(1+x) transformation.
        pcount: int or float, optional
            pseudocount added in size normalization to ensure division by zero does not occur.

        Returns
        f: float np.ndarray
            array with size n_genes x 2 x 2, which contains the fraction of CV2 retained or discarded.
            dim 0: gene index.
            dim 1: variance fraction (retained, discarded).
            dim 2: species (unspliced, spliced).

        """
        f = np.zeros((self.n_genes, 2, 2))  # genes -- bio vs tech -- species

        S = np.copy(self.layers[1])
        U = np.copy(self.layers[0])

        CV2_S = S.var(1) / S.mean(1) ** 2
        CV2_U = U.var(1) / U.mean(1) ** 2

        S = normalize_count_matrix(S, sizefactor, lognormalize, pcount)
        U = normalize_count_matrix(U, sizefactor, lognormalize, pcount)

        CV2_S_norm = S.var(1) / S.mean(1) ** 2
        CV2_U_norm = U.var(1) / U.mean(1) ** 2

        # compute fraction of CV2 eliminated for unspliced and spliced
        f[:, 0, 0] = CV2_U_norm / CV2_U
        f[:, 1, 0] = 1 - f[:, 0, 0]

        f[:, 0, 1] = CV2_S_norm / CV2_S
        f[:, 1, 1] = 1 - f[:, 0, 1]
        return f


def normalize_count_matrix(
    X, sizefactor="pf", lognormalize=True, pcount=0, logbase="e"
):
    """
    This helper function performs normalization and variance stabilization on a raw data matrix X.

    Parameters
    ----------
    X: np.ndarray
        gene x cell count matrix.
    sizefactor: str, float, int, or None, optional
        what size factor to use.
        If 'pf', use proportional fitting; set the size of each cell to the mean size.
        If int or float, use this number (e.g., 1e4 for cp10k).
        If None, do not do size/depth normalization.
    lognormalize: bool, optional
        whether to apply log(1+x) transformation.
    pcount: int or float, optional
        pseudocount added in size normalization to ensure division by zero does not occur.
    logbase: str or int
        If 'e', use log base e.
        If 2 or 10, use the corresponding integer base.

    Returns
    X: np.ndarray
        normalized and transformed count matrix.
    """

    if sizefactor is not None:
        if sizefactor == "pf":
            sizefactor = X.sum(0).mean()
        X = X / (X.sum(0)[None, :] + pcount) * sizefactor
    if lognormalize:
        if logbase == "e":
            X = np.log(X + 1)
        elif logbase == 2:
            X = np.log2(X + 1)
        elif logbase == 10:
            X = np.log10(X + 1)
    return X
