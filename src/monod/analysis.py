from .cme_toolbox import CMEModel
from .inference import SearchResults, plot_hist_and_fit
from .extract_data import SearchData, normalize_count_matrix
from .preprocess import make_dir, log
from .plot_aesthetics import aesthetics
import pickle
import scipy
from scipy import stats
from scipy import odr
import numpy as np
import matplotlib.pyplot as plt


########################
## Helper functions
########################


def load_search_results(full_result_string):
    """Attempt to load search results from disk.

    Parameters
    ----------
    full_result_string: str
        disk location of the SearchResults object.

    Returns
    -------
    search_results: monod.inference.SearchResults
        the SearchResults object.
    """
    try:
        with open(full_result_string, "rb") as srfs:
            search_results = pickle.load(srfs)
        log.info("Grid scan results loaded from {}.".format(full_result_string))
        if not hasattr(search_results.model, "amb_model"):  # bck
            search_results.model.amb_model = "None"

        return search_results
    except Exception as ex:
        log.error(
            "Grid scan results could not be loaded from {}.".format(full_result_string)
        )
        raise ex


def load_search_data(search_data_string):
    """Attempt to load search data from disk.

    If the SearchData object is in the old format, with separate spliced and unspliced count
    matrix attributes, they are concatenated during the import.

    Parameters
    ----------
    search_data_string: str
        disk location of the SearchData object.

    Returns
    -------
    sd: monod.extract_data.SearchData
        the SearchData object.
    """
    try:
        with open(search_data_string, "rb") as sdfs:
            sd = pickle.load(sdfs)
        log.info("Search data loaded from {}.".format(search_data_string))

        if hasattr(sd, "S"):
            sd.layers = np.asarray([sd.U, sd.S])
            sd.M = np.asarray([sd.M, sd.N])
            delattr(sd, "N")
            delattr(sd, "U")
            delattr(sd, "S")
        return sd
    except Exception as ex:
        log.error("Search data could not be loaded from {}.".format(search_data_string))
        raise ex


def make_batch_analysis_dir(sr_arr, dir_string):
    """Create a directory for batch analysis.

    Parameters
    -------
    sr_arr: list of monod.inference.SearchResults
        search result array to process.
    dir_string: str
        batch analysis directory to create and assign.

    Sets
    ----
    batch_analysis_string: str
        batch analysis directory.
    """
    batch_analysis_string = dir_string + "/analysis_figures"
    make_dir(batch_analysis_string)
    for sr in sr_arr:
        sr.batch_analysis_string = batch_analysis_string


def plot_params_for_pair(
    sr1,
    sr2,
    gene_filter_=None,
    plot_errorbars=False,
    figsize=None,
    c=2.576,
    axis_search_bounds=True,
    distinguish_rej=True,
    plot_identity=True,
    meta="12",
    xlabel="dataset 1",
    ylabel="dataset 2",
):
    """Plot the inferred physical parameters at the sampling parameter optimum for a matched pair of datasets.

    This is typically useful to show similarities between controls, or show differences between different
    tissues, for a common set of genes.

    Parameters
    -------
    sr1: monod.inference.SearchResults
        search result with parameters to plot on abscissa.
    sr2: monod.inference.SearchResults
        search result with parameters to plot on ordinate.
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
    distinguish_rej: bool, optional
         whether to distinguish the genes in either of the SearchResults objects'
         rejected_genes attributes by color.
    plot_identity: bool, optional
        whether to plot the identity line. Useful for displaying biases.
    meta: str, optional
        figure filestring suffix.
    xlabel: str, optional
        figure x-label, name of the first dataset.
    ylabel: str, optional
        figure y-label, name of the second dataset.
    """
    num_params = sr1.sp.n_phys_pars
    figsize = figsize or (4 * num_params, 4)
    fig1, ax1 = plt.subplots(nrows=1, ncols=num_params, figsize=figsize)

    gene_filter = sr1.get_bool_filt(gene_filter_, discard_rejected=False)
    # gene_filter2 = sr2.get_bool_filt(gene_filter_,discard_rejected=False)
    gene_filter_rej = np.zeros(sr1.n_genes, dtype=bool)
    # gene_filter_rej2 = np.zeros(sr2.n_genes,dtype=bool)

    if distinguish_rej:  # default
        filt_rej1 = sr1.get_bool_filt(gene_filter_, discard_rejected=True)
        filt_rej2 = sr2.get_bool_filt(gene_filter_, discard_rejected=True)
        filt_rej = np.logical_and(filt_rej1, filt_rej2)

        gene_filter_rej = np.logical_and(gene_filter, np.logical_not(filt_rej))
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

    for i in range(3):
        if plot_errorbars:
            ax1[i].errorbar(
                sr1.phys_optimum[gene_filter, i],
                sr2.phys_optimum[gene_filter, i],
                sr1.sigma[gene_filter, i] * c,
                sr2.sigma[gene_filter, i] * c,
                c=aesthetics["errorbar_gene_color"],
                alpha=aesthetics["errorbar_gene_alpha"],
                linestyle="None",
                linewidth=aesthetics["errorbar_lw"],
            )
        ax1[i].scatter(
            sr1.phys_optimum[gene_filter, i],
            sr2.phys_optimum[gene_filter, i],
            c=aesthetics[acc_point_aesth[0]],
            alpha=aesthetics[acc_point_aesth[1]],
            s=aesthetics[acc_point_aesth[2]],
        )
        if np.any(gene_filter_rej):
            ax1[i].scatter(
                sr1.phys_optimum[gene_filter_rej, i],
                sr2.phys_optimum[gene_filter_rej, i],
                c=aesthetics[rej_point_aesth[0]],
                alpha=aesthetics[rej_point_aesth[1]],
                s=aesthetics[rej_point_aesth[2]],
            )

        ax1[i].set_xlabel(xlabel)
        ax1[i].set_ylabel(ylabel)
        ax1[i].set_title(sr1.model.get_log_name_str()[i])
        if axis_search_bounds:
            ax1[i].set_xlim([sr1.sp.phys_lb[i], sr1.sp.phys_ub[i]])
            ax1[i].set_ylim([sr1.sp.phys_lb[i], sr1.sp.phys_ub[i]])
        if plot_identity:
            xl = ax1[i].get_xlim()
            ax1[i].plot(xl, xl, "r--", linewidth=2)
    fig1.tight_layout()

    fig_string = sr1.batch_analysis_string + "/pair_parameter_comparison_{}.png".format(
        meta
    )
    plt.savefig(fig_string, dpi=450)
    log.info("Figure stored to {}.".format(fig_string))


def find_most_concordant_samp(sr1, sr2):
    """Find sampling parameter optimum by comparing two matched datasets.

    This function attempts to find a sampling parameter optimum by comparing two matched (control) datasets
    and finding the point at which their parameter values are most concordant, according to l2 distance over
    all genes and parameters.

    Conceptually, if the datasets are biologically similar (e.g., the same cell type), we expect their
    biophysical parameters to match. We can come up with an ad hoc estimate for the sampling parameters
    by _assuming_ they are identical for both datasets and finding the grid point that makes the
    biological parameters most similar.

    This typically works poorly.

    Parameters
    ---------
    sr1: monod.inference.SearchResults
        first search result object.
    sr2: monod.inference.SearchResults
        second search result object.

    Returns
    -------
    samp_optimum: list of floats
        value of the technical noise parameters at the grid point.
    """
    discordance = ((sr1.param_estimates - sr2.param_estimates) ** 2).sum(2).sum(1)
    srt = np.argsort(discordance)
    samp_concordant_ind = srt[0]
    sr1.set_sampling_optimum(samp_concordant_ind)
    sr2.set_sampling_optimum(samp_concordant_ind)

    log.info(f"Optimum set to {sr1.samp_optimum[0]:.2f}, {sr1.samp_optimum[1]:.2f}.")
    return sr1.samp_optimum


def get_AIC_weights(sr_arr, sd):
    """Compute Akaike weights for several proposed models.

    This method computes the Akaike Information Criterion (AIC) weights according to the optimal
    physical and sampling parameters obtained for a single dataset (sd) under several models whose
    results are stored in sr_arr.

    For more reference, see Burnham and Anderson (2002).

    This is not yet functional, because the get_logL method of SearchResults is commented out.

    Parameters
    ---------
    sr_arr: list of monod.inference.SearchResults
        search result array to process.
    sd: monod.extract_data.SearchData
        the SearchData object.

    Returns
    -------
    w: np.ndarray
        Akaike weights corresponding to each model, an n_models x n_genes array.
    """

    n_models = len(sr_arr)
    AIC = []
    for j in range(n_models):
        AIC += [2 * sr_arr[j].sp.n_phys_pars - 2 * sr_arr[j].get_logL(sd)]
    AIC = np.asarray(AIC)
    min_AIC = AIC.min(0)
    normalization = np.exp(-(AIC - min_AIC) / 2).sum(0)
    w = np.exp(-(AIC - min_AIC) / 2) / normalization
    return w


def plot_AIC_weights(
    sr_arr,
    sd,
    models,
    ax1=None,
    meta=None,
    figsize=None,
    facecolor=aesthetics["hist_face_color"],
    facealpha=aesthetics["hist_face_alpha"],
    nbin=20,
    savefig=True,
):
    """Compute Akaike weights and plot their distributions.

    This function wraps get_AIC_weights and plots the distribution of weights for
    each model.

    Parameters
    ---------
    sr_arr: list of monod.inference.SearchResults
        search result array to process.
    sd: monod.extract_data.SearchData
        the SearchData object.
    models: list of str
        model names.
    ax1: list of matplotlib.axes.Axes or None, optional
        axes to plot into.
        if None, creates a new figure.
    meta: str or None, optional
        metadata suffix for figure file.
    figsize: None or tuple of floats, optional
        figure dimensions.
    facecolor: str or tuple, optional
        histogram face color in a matplotlib-compatible format.
    facealpha: float, optional
        histogram face alpha.
    nbin: int, optional
        number of bins used to construct the histogram.
    savefig: bool, optional
        whether to save the figure to disk.

    Returns
    -------
    w: np.ndarray
        Akaike weights corresponding to each model, an n_models x n_genes array.

    """
    w = get_AIC_weights(sr_arr, sd)

    if meta is None:
        meta = ""
    else:
        meta = "_" + meta

    n_models = w.shape[0]
    figsize = figsize or (4 * n_models, 4)

    if ax1 is None:
        fig1, ax1 = plt.subplots(nrows=1, ncols=n_models, figsize=figsize)
    else:
        fig1 = plt.gcf()

    for i in range(n_models):
        ax1[i].hist(w[i], bins=nbin, density=False, color=facecolor, alpha=facealpha)
        ax1[i].set_xlabel("AIC weight at MLE")
        ax1[i].set_ylabel("# genes")
        ax1[i].set_title(models[i])

    fig1.tight_layout()
    if savefig:
        fig_string = (sr_arr[0].batch_analysis_string) + (
            "/AIC_comparison{}.png".format(meta)
        )

        plt.savefig(fig_string, dpi=450)
        log.info("Figure stored to {}.".format(fig_string))
    return w


def compare_AIC_weights(
    w, dataset_names, batch_analysis_string, model_ind=0, figsize=(12, 12), kde_bw=0.05
):
    """Compare Akaike weights across datasets.

    This function compares the consistency of AIC weights for a single model across several datasets.
    For a given gene and model j, the function takes the weight w_j and compares its absolute difference
    between two datasets. Then, it aggregates the information over all genes and plots the
    kernel density estimates for a pair of datasets.

    If the KDE is concentrated near zero, inference on the same dataset tends to choose the same model.

    Returns
    -------
    w: np.ndarray
        Akaike weights corresponding to each dataset and model, an n_datasets x n_models x n_genes array.
    dataset_names: list of str
        dataset metadata.
    batch_analysis_string: str
        location of the directory for batch-wide analyses.
    model_ind: int
        which model to plot weights for.
    figsize: None or tuple of floats, optional
        figure dimensions.
    kde_bw: str, scalar or callable, optional
        float kernel density estimate bandwidth or method, as documented for scipy.stats.gaussian_kde.

    """
    fs = 12
    n_datasets = len(dataset_names)
    fig1, ax1 = plt.subplots(nrows=n_datasets, ncols=n_datasets, figsize=figsize)
    for i in range(n_datasets):
        for k in range(n_datasets):
            if i > k:
                xx = np.linspace(-0.2, 1.2, 2000)

                kde = stats.gaussian_kde(
                    np.abs(w[i, model_ind, :] - w[k, model_ind, :]), bw_method=kde_bw
                )
                ax1[i, k].plot(xx, kde(xx), "k")
            elif i == k:
                ax1[i, k].hist(
                    w[i, model_ind, :], 30, facecolor=aesthetics["hist_face_color"]
                )
            elif i < k:
                fig1.delaxes(ax1[i, k])
            ax1[i, k].set_yticks([])
            if k == 0:
                ax1[i, k].set_ylabel(dataset_names[i], fontsize=fs)
            if i == (n_datasets - 1):
                ax1[i, k].set_xlabel(dataset_names[k], fontsize=fs)
    fig1.tight_layout()
    fig_string = batch_analysis_string + "/AIC_comparison_grid.png"

    plt.savefig(fig_string, dpi=450)
    log.info("Figure stored to {}.".format(fig_string))


def diffexp_pars(
    sr1,
    sr2,
    modeltype="id",
    gene_filter_=None,
    discard_rejected=True,
    use_sigma=True,
    figsize=None,
    meta="12",
    viz=True,
    pval_thr=0.001,
    nit=10,
):
    """Use fixed-point iteration to find differentially expressed genes without replicates.

    This function uses the optimal physical and sampling parameters obtained for a pair of datasets
    to attempt to identify sources of differential expression under a model of transcription,
    i.e., calling patterns of DE-theta, where theta is a biophysical parameter identifiable at steady state.
    Specifically, it uses a fixed-point iteration procedure to distinguish a set of genes with
    Gaussian aleatory variation from a set with systematic, high-magnitude deviations between the datasets.

    The outputs of plot_params_for_pair typically show similar parameter values for paired datasets, with
    the following structure: many genes lie around the identity line (potentially containing a small bias),
    a few genes are quite far from the identity line. This function attempts to find the divergent genes,
    which show meaningful deviations in parameter values, using a Gaussian model for non-systematic variation
    between datasets.

    By fitting a Gaussian model, we can identify genes that diverge from identity and tentatively propose that
    they are differentially expressed. Excluding them from changes the dataset used to fit the Gaussian, so we need to
    repeat the procedure several times until convergence.

    Parameters
    ---------
    sr1: monod.inference.SearchResults
        first search result object.
    sr2: monod.inference.SearchResults
        second search result object.
    modeltype: str, optional
        If 'id', the relationship between datasets' parameters is assumed to be x + b = y; n_model_pars = 1.
        If 'lin', the relationship between datasets' parameters is assumed to be ax + b = y; n_model_pars = 2.
    gene_filter_: None or np.ndarray, optional
        If None, use all genes.
        If a boolean or integer filter, use the filtered gene subset.
    discard_rejected: bool, optional
         whether to omit genes in the rejected_genes attribute from the analysis.
    use_sigma: bool, optional
        whether to use the inferred standard error of MLEs to fit the aleatory variation model.
    figsize: None or tuple of floats, optional
        figure dimensions.
    meta: str, optional
        metadata suffix for figure file.
    viz: bool, optional
        whether to visualize the histograms of residuals.
    pval_thr: float, optional
        the p-value threshold to use for the Z-test. If the Z-value of the gene's parameter deviation
        corresponds to a p-value lower than this threshold, the gene is considered differentially expressed
        with respect to this parameter.
    nit: int, optional
        number of rounds of fixed-point iteration to perform.

    Returns
    -------
    gn: list of lists of str
        list of size n_phys_pars. each entry contains names of genes identified as DE with respect
        to each parameter.
    gf: bool np.ndarray
        an n_phys_pars x n_genes array. if True, the gene has been identified as DE with respect to
        the parameter.
    offs: float np.ndarray
        an n_phys_pars x n_model_pars array. contains (b) or (b,a) for each physical parameter.
        b should typically be fairly small (under 0.5).
    resid: float np.ndarray
        an n_phys_pars x n_genes array. contains residuals for each parameter under the final
        fit statistical model.
    """
    num_params = sr1.sp.n_phys_pars
    if viz:
        figsize = figsize or (4 * num_params, 4)
        fig1, ax1 = plt.subplots(nrows=1, ncols=num_params, figsize=figsize)

    gene_filter = sr1.get_bool_filt(gene_filter_, discard_rejected=False)

    if discard_rejected:  # default
        filt_rej1 = sr1.get_bool_filt(gene_filter_, discard_rejected=True)
        filt_rej2 = sr2.get_bool_filt(gene_filter_, discard_rejected=True)
        filt_rej = np.logical_and(filt_rej1, filt_rej2)
        gene_filter = np.logical_and(gene_filter, filt_rej)

    gn = []
    gf = []
    offs = []
    resid = []

    parnames = sr1.model.get_log_name_str()
    for i in range(num_params):

        if viz:
            ax = ax1[i]
        else:
            ax = None

        resid_arr = np.asarray([np.nan] * sr1.n_genes)

        # xl = [sr1.sp.phys_lb[i],sr1.sp.phys_ub[i]]
        if use_sigma:
            gf_, offs_, resid_, pval_ = diffexp_fpi(
                sr1.phys_optimum[gene_filter, i],
                sr2.phys_optimum[gene_filter, i],
                parnames[i],
                modeltype=modeltype,
                ax1=ax,
                s1=sr1.sigma[gene_filter, i],
                s2=sr2.sigma[gene_filter, i],
                nit=nit,
                viz=viz,
                pval_thr=pval_thr,
            )
        else:
            gf_, offs_, resid_, pval_ = diffexp_fpi(
                sr1.phys_optimum[gene_filter, i],
                sr2.phys_optimum[gene_filter, i],
                parnames[i],
                modeltype=modeltype,
                ax1=ax,
                s1=None,
                s2=None,
                nit=nit,
                viz=viz,
                pval_thr=pval_thr,
            )
        resid_arr[gene_filter] = resid_

        filtind = np.arange(sr1.n_genes)
        filtind = filtind[gene_filter]
        filtind = filtind[gf_]
        gf__ = np.zeros(sr1.n_genes, dtype=bool)
        gf__[filtind] = True
        gn_ = sr1.gene_names[gf__]
        gn.append(gn_)
        gf.append(gf__)
        offs.append(offs_)
        resid.append(resid_arr)

        if i == 0 and viz:
            ax.legend()
    if viz:
        fig1.tight_layout()
        fig_string = f"{sr1.batch_analysis_string}/parameter_residuals_{meta}.png"
        plt.savefig(fig_string, dpi=450)
        log.info("Figure stored to {}.".format(fig_string))

    return gn, np.asarray(gf), np.asarray(offs), np.asarray(resid)


def linoffset(B, x, modeltype="id"):
    """Helper function for debiasing.

    Defines a statistical model that governs the relationship between the parameter values in two
    matched parameter sets, under the assumption that all differences are aleatory rather than systematic,
    i.e., governed by random noise rather than any meaningful differential variation.

    Parameters
    ----------
    B: float np.ndarray
        statistical model parameters of size n_model_pars.
    x: float np.ndarray
        an array of optimal physical parameters of size n_genes_subset, where
        n_genes_subset is within [1,n_genes] and may omit some genes.
    modeltype: str, optional
        If 'id', the relationship between datasets' parameters is assumed to be x + b = y; n_model_pars = 1.
        If 'lin', the relationship between datasets' parameters is assumed to be ax + b = y; n_model_pars = 2.

    Returns
    -------
    y: float np.ndarray
        predicted parameter values accounting for potential bias between datasets.
    """
    if modeltype == "id":
        return x + B[0]
    elif modeltype == "lin":
        return B[1] * x + B[0]


def diffexp_fpi(
    m1,
    m2,
    parname=None,
    modeltype="id",
    ax1=None,
    s1=None,
    s2=None,
    nit=10,
    pval_thr=0.001,
    viz=True,
):
    """Use fixed-point interation on a set of generic parameter values to debias them and call outliers.

    This function uses the optimal physical and sampling parameters obtained for a pair of datasets
    to attempt to identify differentially expressed genes under a model of transcription, for a single stest statistic.
    Specifically, it uses a fixed-point iteration (FPI) procedure to distinguish a set of genes with
    Gaussian aleatory variation from a set with systematic, high-magnitude deviations between the datasets.

    This is a helper function that has no access to the raw data, only generic arrays of floats, and can
    be used for debiasing more generally.

    The fit uses orthogonal distance regression to impose symmetry between datasets.
    If s1 and s2 are provided, the fit uses these error estimates to penalize higher-uncertainty data points.

    Parameters
    ----------
    m1: float np.ndarray
        parameter estimates from the first search result object.
    m2: float np.ndarray
        parameter estimates from the second search result object.
    parname: float or None, optional
        if float, the name of the parameter being considered. Used only for visualization.
    modeltype: str, optional
        If 'id', the relationship between datasets' parameters is assumed to be x + b = y; n_model_pars = 1.
        If 'lin', the relationship between datasets' parameters is assumed to be ax + b = y; n_model_pars = 2.
    ax1: matplotlib.axes.Axes or None, optional
        axes to plot into.
    s1: float np.ndarray or None, optional
        standard errors associated with values in m1.
    s2: float np.ndarray or None, optional
        standard errors associated with values in m2.
    nit: int, optional
        number of rounds of fixed-point iteration to perform.
    pval_thr: float, optional
        the p-value threshold to use for the Z-test. If the Z-value of the gene's parameter deviation
        corresponds to a p-value lower than this threshold, the gene is considered differentially expressed
        with respect to this parameter.
    viz: bool, optional
        whether to visualize the histogram of residuals.

    Returns
    -------
    gf: bool np.ndarray
        if True, the gene has been identified as DE with respect to the current parameter.
    out.beta: float np.ndarray
        an array of size n_model_pars. contains (b) or (b,a) for each physical parameter.
        b should typically be fairly small (under 0.5).
    resid: float np.ndarray
        residuals for each gene under the final fit statistical model.
    pval: float np.ndarray
        residuals' p-values under the Z-test.
    """
    fs = 12
    gf = np.ones(len(m1), dtype=bool)
    # x = np.linspace(xl[0], xl[1], 100)
    fitlaw = scipy.stats.norm

    if modeltype == "id":
        beta0 = [0]
    elif modeltype == "lin":
        beta0 = [0, 1]

    for j in range(nit):
        if s1 is None and s2 is None:
            d = odr.Data(m1[gf], m2[gf])
        else:
            d = odr.Data(m1[gf], m2[gf], s1[gf], s2[gf])

        md = lambda B, x: linoffset(B, x, modeltype=modeltype)
        odrmodel = scipy.odr.Model(md)

        myodr = odr.ODR(d, odrmodel, beta0=beta0)
        out = myodr.run()

        if modeltype == "id":
            resid = m2 - m1 - out.beta[0]
        elif modeltype == "lin":
            resid = m2 - m1 * out.beta[1] - out.beta[0]

        fitparams = fitlaw.fit(resid[gf])
        x = np.linspace(resid.min(), resid.max(), 100)
        p = fitlaw.pdf(x, *fitparams)
        if j == 0 and viz:
            ax1.plot(
                x,
                p,
                linestyle=aesthetics["init_fit_linestyle"],
                linewidth=aesthetics["init_fit_linewidth"],
                color=aesthetics["hist_fit_color_2"],
                label="Initial fit",
            )

        z = (resid - fitparams[0]) / fitparams[1]
        pval = scipy.stats.norm.sf(np.abs(z)) * 2
        gf = np.logical_not(pval < pval_thr)
        if j == (nit - 1) and viz:
            n, bins = np.histogram(resid, 200, density=True)
            binc = np.diff(bins) / 2 + bins[:-1]
            ax1.bar(
                binc,
                n,
                width=np.diff(bins),
                color=aesthetics["hist_face_color"],
                align="center",
                label="Putative static genes",
            )
            ax1.plot(
                x,
                p,
                linestyle=aesthetics["linestyle"],
                linewidth=2,
                color=aesthetics["hist_fit_color"],
                label="FPI fit",
            )
            y_, _ = np.histogram(resid[~gf], bins=bins, density=True)
            y_ *= (~gf).mean()
            ax1.bar(
                binc,
                y_,
                width=np.diff(bins),
                color=aesthetics["diffreg_gene_color"],
                align="center",
                label="Putative DE genes",
            )
    if viz:
        ax1.set_xlabel(parname + " residual", fontsize=fs)
        ax1.set_ylabel("Density", fontsize=fs)
    gf = np.logical_not(gf)
    return gf, out.beta, resid, pval


def compare_gene_distributions(
    sr_arr,
    sd_arr,
    sz=(5, 5),
    figsize=(10, 10),
    marg="mature",
    logscale=None,
    title=True,
    genes_to_plot=None,
):
    """Plot marginal histograms for multiple datasets.

    This function is analogous to the SearchResults method plot_gene_distributions, but it
    plots the marginal histograms for multiple datasets at a time (2 supported at this time).
    The intended use case is to identify a set of DE genes of interest, pass them in as
    genes_to_plot, and inspect the difference between the distributions.

    By default, dataset 1 is plotted in red and dataset 2 is plotted in blue. The colors are
    defined in monod.plot_aesthetics.aesthetics.

    Parameters
    ----------
    sr_arr: list of monod.inference.SearchResults
        search results to visualize as lines..
    sd_arr: list of monod.extract_data.SearchData
        search data to visualize as histograms.
    sz: tuple of ints, optional
        dimensions of the figure subplot grid.
    figsize: tuple of floats, optional
        figure dimensions.
    logscale: None or bool, optional
        whether to plot probabilities or log-probabilities.
        by default, False for marginals.
    title: bool, optional
        whether to report the gene name in each subplot title.
    genes_to_plot: bool or int np.ndarray or None, optional
        if array, which genes to plot.
        if None, plot by internal order.

    """
    if logscale is None:
        if marg == "joint":
            logscale = True
        else:
            logscale = False
    (nrows, ncols) = sz

    fig1, ax1 = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    nax = np.prod(sz)
    if genes_to_plot is None:
        genes_to_plot = np.arange(nax)

    genes_to_plot = np.asarray(genes_to_plot)
    if genes_to_plot.dtype == bool:
        gtp_temp = np.arange(sr_arr[0].n_genes)
        genes_to_plot = gtp_temp[genes_to_plot]

    number_of_genes_to_plot = len(genes_to_plot)
    if number_of_genes_to_plot > sr_arr[0].n_genes:
        number_of_genes_to_plot = sr_arr[0].n_genes
        genes_to_plot = genes_to_plot[: self.n_genes]
    if number_of_genes_to_plot > nax:
        number_of_genes_to_plot = nax
        genes_to_plot = genes_to_plot[:nax]

    j_ = 0
    for i_ in genes_to_plot:
        axloc = np.unravel_index(j_, sz) if (sz[0] > 1 and sz[1] > 1) else j_
        xlim = 1
        for j in range(len(sr_arr)):
            lm = np.copy(sd_arr[j].M[:, i_])
            if marg == "mature":
                lm[0] = 1
            if marg == "nascent":
                lm[1] = 1
            xlim = np.max([max(lm), xlim])
            samp = (
                None
                if (sr_arr[j].model.seq_model == "None")
                else sr_arr[j].regressor_optimum[i_]
            )
            Pa = np.squeeze(
                sr_arr[j].model.eval_model_pss(sr_arr[j].phys_optimum[i_], lm, samp)
            )

            if marg == "joint":
                log.error("Potentially implement this later...")
                raise ValueError("Cannot compare two 2D histograms!")
            else:
                plot_hist_and_fit(
                    ax1[axloc],
                    sd_arr[j],
                    i_,
                    Pa,
                    marg,
                    facealpha=aesthetics["hist_face_alpha_" + str(j + 1)],
                    facecolor=aesthetics["hist_face_color_" + str(j + 1)],
                    fitcolor=aesthetics["hist_fit_color_" + str(j + 1)],
                )
        ax1[axloc].set_xlim([-0.5, xlim - 8])
        if logscale:
            ax1[axloc].set_yscale("log")
        if title:
            ax1[axloc].set_title(sr_arr[0].gene_names[i_], fontdict={"fontsize": 9})

        ax1[axloc].set_xticks([])
        ax1[axloc].set_yticks([])

        j_ += 1
    fig1.tight_layout(pad=0.02)


def diffexp_mean(
    sd1,
    sd2,
    sizefactor="pf",
    lognormalize=True,
    pcount=0,
    pval_thr=0.001,
    method="ttest",
    bonferroni=True,
    modeltype="lin",
    viz=True,
    knee_thr=None,
    fc_thr=1,
    ax1=None,
    viz_resid=False,
):
    """Identify DE genes through the means.

    This function attempts to identify differentially expressed genes using a simple comparison of
    the means of gene-specific count distributions, i.e., DE-theta using the normalized mean as a test statistic.

    Two methods are implemented.
    The first, 'ttest', simply applies a t-test to each gene in a normalized/transformed count matrix.
    The second, 'logmeanfpi' or 'meanlogfpi', applies the fixed-point iteration to fit a model of Gaussian
        aleatory variation and call outliers.

    Parameters
    ---------
    sd1: monod.extract_data.SearchData
        first search data object.
    sd2: monod.extract_data.SearchData
        second search data object.
    sizefactor: str or float or None, optional
        size factor to use in normalization.
        if 'pf': Proportional fitting; set the size of each cell to the mean size.
        if a float: use this value (e.g., 1e4 for cp10k).
        if None: do not perform size/depth normalization.
    lognormalize: bool, optional
        whether to apply a log-transformation to the data.
    pcount: float, optional
        pseudocount to use for size-normalization.
    pval_thr: float, optional
        the p-value threshold to use for the Z-test or t-test. If the p-value is lower than
        this threshold, the gene is considered differentially expressed with respect to the mean.
    method: str, optional
        if 'ttest', use the t-test on transformed data.
        If 'logmeanfpi', use the fixed-point iteration procedure on the distribution of log-means.
        If 'meanlogfpi', use the FPI procedure on the distribution of means of log+1 counts.
    bonferroni: bool, optional
        whether to use a multiple comparison correction for the t-test.
    modeltype: str, optional
        statistical model for FPI
        If 'id', the relationship between datasets' means is assumed to be x + b = y; n_model_pars = 1.
        If 'lin', the relationship between datasets' means is assumed to be ax + b = y; n_model_pars = 2.
    viz: bool, optional
        whether to visualize t-test or FPI volcano plots.
    knee_thr: float or None, optional
        if not None, use a subset of cells with more than knee_thr total counts.
        this threshold is compared to the total number of molecules of the genes stored in sd1 and sd2.
    fc_thr: float, optional
        log2 fold change threshold for the t-test.
    ax1: matplotlib.axes.Axes or None, optional
        axes to plot into.
    viz_resid
        whether to visualize FPI residual plots.

    Returns
    -------
    gf: bool np.ndarray
        if True, the gene has been identified as differentially expressed.
    fc: float np.ndarray
        effect sizes for all genes.

    """

    s1 = np.copy(sd1.layers[1])
    s2 = np.copy(sd2.layers[1])

    if (viz or viz_resid) and ax1 is None:
        fig1, ax1 = plt.subplots(1, 1)

    if method == "ttest":

        if knee_thr is not None:
            cf1 = sd1.knee_plot(thr=knee_thr[0])
            cf2 = sd2.knee_plot(thr=knee_thr[1])
            s1 = s1[:, cf1]
            s2 = s2[:, cf2]

        gf = np.zeros(sd1.n_genes, dtype=bool)
        if bonferroni:
            pval_thr /= sd1.n_genes
        s1 = normalize_count_matrix(
            s1, sizefactor=sizefactor, lognormalize=lognormalize, pcount=0, logbase=2
        )
        s2 = normalize_count_matrix(
            s2, sizefactor=sizefactor, lognormalize=lognormalize, pcount=0, logbase=2
        )

        p = np.zeros(sd1.n_genes)
        for i in range(sd1.n_genes):
            _, p_ = scipy.stats.ttest_ind(s1[i], s2[i], equal_var=False)
            if p_ < pval_thr:
                gf[i] = True
            p[i] = p_
        if lognormalize:
            fc = s2.mean(1) - s1.mean(1)
        else:
            fc = np.log2(s1 + 1).mean(1) - np.log2(s2 + 1).mean(1)
        gf = gf & (np.abs(fc) > fc_thr)

        if viz:
            pv = -np.log10(p)
            ax1.scatter(fc[gf], pv[gf], 5, "r")
            ax1.scatter(fc[~gf], pv[~gf], 3, "darkgray")
            ax1.set_xlabel(r"Fold change ($\log_2$)")
            ax1.set_ylabel(r"$-\log_{10} p$")
    else:
        if method == "logmeanfpi":
            m1 = np.log2(s1.mean(1))
            m2 = np.log2(s2.mean(1))
        elif method == "meanlogfpi":
            m1 = np.log2(s1 + 1).mean(1)
            m2 = np.log2(s2 + 1).mean(1)
        gf, offs_, resid_, p = diffexp_fpi(
            m1,
            m2,
            "Spliced mean",
            modeltype=modeltype,
            ax1=ax1,
            s1=None,
            s2=None,
            nit=30,
            viz=viz_resid,
            pval_thr=pval_thr,
        )
        fc = m2 - m1  # ?
        if viz:
            pv = -np.log10(p)
            ax1.scatter(fc[gf], pv[gf], 5, "r")
            ax1.scatter(fc[~gf], pv[~gf], 3, "darkgray")
            ax1.set_xlabel(r"Fold change ($\log_2$)")
            ax1.set_ylabel(r"$-\log_{10} p$")
    return gf, fc
