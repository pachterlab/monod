"""End-to-end checks for the repaired analysis helpers (audit fix 7, 2026-10-15).

Everything here runs on example_adata.h5ad with a 3-gene Bursty/None fit so the
whole module takes a few seconds.  matplotlib uses the Agg backend.
"""
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from monod import analysis
from monod.cme_toolbox import CMEModel
from monod.extract_data import extract_data
from monod.inference import (
    InferenceParameters,
    perform_inference,
    plot_hist_and_fit,
    searchdata_from_adata,
)

_HERE = os.path.dirname(__file__)
EXAMPLE = os.path.join(_HERE, "..", "example_h5ad", "example_adata.h5ad")
GP = {"max_iterations": 5, "init_pattern": "moments", "num_restarts": 1, "num_gene_cores": 1, "n_jac_jobs": 1}


@pytest.fixture(scope="module")
def fit(tmp_path_factory):
    """(model, search_data, search_result) from a quick Bursty/None fit."""
    import anndata as ad
    warnings.filterwarnings("ignore")
    cwd = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("fit"))
    try:
        model = CMEModel("Bursty", "None")
        madata = extract_data(ad.read_h5ad(EXAMPLE), model, dataset_name="x", n_genes=3, viz=False, ambiguous="drop")
        sd = searchdata_from_adata(madata)
        ip = InferenceParameters("x", model, gradient_params=GP)
        sr = ip.fit_all_grid_points(sd)
        sr.find_sampling_optimum()
        return model, sd, sr
    finally:
        os.chdir(cwd)


@pytest.fixture(scope="module")
def two_adatas(tmp_path_factory):
    """Two perform_inference outputs on the same genes (for DE_parameters)."""
    warnings.filterwarnings("ignore")
    cwd = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("pi"))
    try:
        m = CMEModel("Bursty", "None")
        a1 = perform_inference(EXAMPLE, m, n_genes=3, viz=False, gradient_params=GP, seed=1, ambiguous="drop")
        a2 = perform_inference(EXAMPLE, m, n_genes=3, viz=False, gradient_params=GP, seed=1, ambiguous="drop")
        return a1, a2
    finally:
        os.chdir(cwd)


class TestPlotHistAndFit:
    def test_marginal_plot_runs_for_each_modality(self, fit):
        model, sd, sr = fit
        for marg in ("unspliced", "spliced", "nascent", "mature"):
            fig, ax = plt.subplots()
            lm = sd.M[:, 0].copy()
            lind = 0 if marg in ("unspliced", "nascent") else 1
            lm[1 - lind] = 1
            Pa = np.squeeze(model.eval_model_pss(sr.phys_optimum[0], lm))
            plot_hist_and_fit(ax, sd, 0, Pa, marg, modalities=model.model_modalities)
            assert len(ax.patches) > 0 and len(ax.lines) == 1
            plt.close(fig)

    def test_unknown_marginal_raises(self, fit):
        model, sd, sr = fit
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="not one of the modalities"):
            plot_hist_and_fit(ax, sd, 0, np.ones(5), "protein", modalities=model.model_modalities)
        plt.close(fig)

    def test_plot_gene_distributions_marginal_and_joint(self, fit):
        model, sd, sr = fit
        sr.plot_gene_distributions(sd, sz=(1, 3), marg="unspliced", savefig=False)
        sr.plot_gene_distributions(sd, sz=(1, 1), marg="joint", savefig=False, show_axes=True)
        sr.plot_gene_distributions(sd, sz=(1, 3), marg="spliced", savefig=True)  # no figure dir: logs, no crash
        plt.close("all")


class TestLogLAndAIC:
    def test_get_logL_default_n_cells(self, fit):
        _, sd, sr = fit
        a = np.asarray(sr.get_logL(sd))
        b = np.asarray(sr.get_logL(sd, sd.n_cells))
        np.testing.assert_allclose(a, b)
        assert a.shape == (sr.n_genes,) and np.all(np.isfinite(a))

    def test_aic_weights(self, fit):
        _, sd, sr = fit
        w = analysis.get_AIC_weights([sr, sr], sd)
        assert w.shape == (2, sr.n_genes)
        np.testing.assert_allclose(w.sum(0), 1.0)
        np.testing.assert_allclose(w[0], 0.5)
        w2 = analysis.plot_AIC_weights([sr, sr], sd, ["a", "b"], savefig=True)  # no batch dir: logs
        np.testing.assert_allclose(w2, w)
        plt.close("all")


class TestPairAndDE:
    def test_plot_params_for_pair(self, fit, tmp_path):
        _, sd, sr = fit
        fig = analysis.plot_params_for_pair(sr, sr)                 # no path -> not saved
        assert len(fig.axes) == sr.sp.n_phys_pars
        out = os.path.join(str(tmp_path), "pair.png")
        analysis.plot_params_for_pair(sr, sr, plot_errorbars=True, savefig=out)  # sigma missing -> warning, no crash
        assert os.path.exists(out)
        plt.close("all")

    def test_diffexp_pars_without_sigma_falls_back(self, fit):
        _, sd, sr = fit
        gn, gf, offs, resid = analysis.diffexp_pars(sr, sr, use_sigma=True, viz=True, nit=2)
        assert gf.shape == (sr.sp.n_phys_pars, sr.n_genes)
        assert resid.shape == gf.shape
        plt.close("all")

    def test_diffexp_fpi_uses_sigmas_as_errors(self):
        rng = np.random.default_rng(0)
        m1 = rng.normal(size=200)
        m2 = m1 + 0.3 + rng.normal(scale=0.05, size=200)
        m2[:5] += 3.0                                     # five clear outliers
        s = np.full(200, 0.05)
        gf, beta, resid, pval = analysis.diffexp_fpi(m1, m2, s1=s, s2=s, nit=5, viz=False)
        assert abs(beta[0] - 0.3) < 0.05
        assert gf[:5].all() and gf[5:].sum() <= 5

    def test_compare_gene_distributions(self, fit):
        model, sd, sr = fit
        fig = analysis.compare_gene_distributions([sr, sr], [sd, sd], sz=(1, 3), marg="spliced")
        assert len(fig.axes) == 3
        with pytest.raises(ValueError, match="marginals"):
            analysis.compare_gene_distributions([sr, sr], [sd, sd], sz=(1, 3), marg="joint")
        plt.close("all")

    def test_DE_parameters_two_datasets(self, two_adatas):
        a1, a2 = two_adatas
        gn, gf, offs, resid = analysis.DE_parameters(a1, a2, viz=False, use_sigma=True, nit=2)
        # Parameter columns use the model's TeX labels with the log prefix stripped: $b$, $\beta$, $\gamma$.
        assert "FC_$b$" in a2.var.columns and "adj_FC_$b$" in a1.var.columns and "adj_$b$" in a2.var.columns
        assert gf.shape[1] == a1.n_vars
        plt.close("all")

    def test_diffexp_mean_both_methods(self, fit):
        _, sd, sr = fit
        gf, fc = analysis.diffexp_mean(sd, sd, method="ttest", viz=False)
        assert gf.shape == (sd.n_genes,) and not gf.any()          # identical data: nothing DE
        gf2, fc2 = analysis.diffexp_mean(sd, sd, method="meanlogfpi", viz=False, viz_resid=False)
        assert gf2.shape == (sd.n_genes,)
        np.testing.assert_allclose(fc2, 0.0, atol=1e-12)


def test_drivers_module_removed():
    import importlib
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("monod.drivers")


def test_diffexp_fpi_tolerates_zero_sigma():
    rng = np.random.default_rng(1)
    m1 = rng.normal(size=100)
    m2 = m1 + 0.2 + rng.normal(scale=0.05, size=100)
    s = np.full(100, 0.05); s[:10] = 0.0            # pinned-at-bound genes report sigma 0
    gf, beta, resid, pval = analysis.diffexp_fpi(m1, m2, s1=s, s2=s, nit=3, viz=False)
    assert np.isfinite(beta).all() and np.isfinite(resid).all()


def test_reject_genes_round_trip(two_adatas):
    from monod.inference import reject_genes, search_result_from_uns
    a1, a2 = two_adatas
    reject_genes(a1, reject_at_bounds=False)
    for col in ("rejected_genes", "csq", "pval", "hellinger"):
        assert col in a1.var.columns
    sr = search_result_from_uns(a1.uns["search_result"])
    assert hasattr(sr, "rejected_genes") and len(sr.rejected_genes) == a1.n_vars
    # discard_rejected now has something to act on
    reject_genes(a2, reject_at_bounds=False)
    gn, gf, offs, resid = analysis.DE_parameters(a1, a2, viz=False, use_sigma=False, discard_rejected=True, nit=2)
    assert gf.shape[1] == a1.n_vars


@pytest.mark.parametrize("fit_slope", [False, True])
@pytest.mark.parametrize("use_err", [False, True])
def test_fit_line_eiv_matches_scipy_odr(fit_slope, use_err):
    """The closed-form / York fit reproduces scipy.odr (deprecated in SciPy 1.17, removed in 1.19)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            from scipy import odr
        except ImportError:
            pytest.skip("scipy.odr no longer available")
    rng = np.random.default_rng(3)
    x = rng.normal(size=300)
    sx = rng.uniform(0.05, 0.3, size=300)
    sy = rng.uniform(0.05, 0.3, size=300)
    slope = 1.3 if fit_slope else 1.0
    y = 0.4 + slope * x + rng.normal(size=300) * sy
    x = x + rng.normal(size=300) * sx
    ours = analysis._fit_line_eiv(x, y, sx if use_err else None, sy if use_err else None, fit_slope=fit_slope)
    modeltype = "lin" if fit_slope else "id"
    data = odr.RealData(x, y, sx=sx, sy=sy) if use_err else odr.Data(x, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = odr.ODR(data, odr.Model(lambda B, t: t + B[0] if modeltype == "id" else B[1] * t + B[0]),
                      beta0=[0, 1] if fit_slope else [0]).run().beta
    np.testing.assert_allclose(ours, ref, rtol=1e-4, atol=1e-6)
    assert ours.shape == (2 if fit_slope else 1,)


def test_fit_line_eiv_degenerate_inputs():
    # one point: slope stays at 1, offset is the difference; identical inputs give an exact zero offset
    np.testing.assert_allclose(analysis._fit_line_eiv([1.0], [1.5], fit_slope=True), [0.5, 1.0])
    m = np.random.default_rng(0).normal(size=20)
    np.testing.assert_array_equal(analysis._fit_line_eiv(m, m.copy(), fit_slope=True), [0.0, 1.0])
    gf, beta, resid, pval = analysis.diffexp_fpi(m, m.copy(), "x", modeltype="lin", viz=False, nit=3)
    assert not gf.any() and np.isfinite(beta).all()


class TestGenePlotHelpers:
    def test_genes_to_plot_selector(self):
        from monod.inference import _genes_to_plot
        np.testing.assert_array_equal(_genes_to_plot(None, 10, 4), [0, 1, 2, 3])
        np.testing.assert_array_equal(_genes_to_plot(None, 2, 4), [0, 1])
        np.testing.assert_array_equal(_genes_to_plot(np.array([0, 1, 0, 1, 1], bool), 5, 2), [1, 3])
        np.testing.assert_array_equal(_genes_to_plot([4, 2], 5, 4), [4, 2])

    def test_gene_pss_grids_and_aliases(self, fit):
        _, sd, sr = fit
        joint = sr.gene_pss(sd, 0)
        assert joint.shape == tuple(int(m) for m in sd.M[:, 0])
        spliced = sr.gene_pss(sd, 0, "spliced")
        assert spliced.shape == (int(sd.M[1, 0]),)
        np.testing.assert_array_equal(sr.gene_pss(sd, 0, "mature"), spliced)
        assert abs(spliced.sum() - 1) < 0.05
        with pytest.raises(ValueError, match="modalities"):
            sr.gene_pss(sd, 0, "protein")

    def test_plot_gene_distributions_honours_a_mask(self, fit):
        _, sd, sr = fit
        mask = np.zeros(sr.n_genes, dtype=bool)
        mask[-1] = True
        fig = sr.plot_gene_distributions(sd, sz=(1, 2), marg="joint", genes_to_plot=mask, savefig=False)
        assert fig.axes[0].get_title().startswith(sr.gene_names[-1])
        plt.close("all")

    def test_diffexp_mean_fold_change_is_second_minus_first(self):
        from types import SimpleNamespace
        rng = np.random.default_rng(0)
        s = rng.poisson(5.0, size=(50, 4))                     # cells x genes
        a = SimpleNamespace(layers=np.array([s, s]), n_genes=4)
        b = SimpleNamespace(layers=np.array([s, 4 * s]), n_genes=4)
        for lognormalize in (True, False):
            gf, fc = analysis.diffexp_mean(a, b, method="ttest", viz=False, sizefactor=None,
                                           lognormalize=lognormalize)
            assert (fc > 0).all() and gf.all()
        with pytest.raises(ValueError, match="method"):
            analysis.diffexp_mean(a, b, method="median", viz=False)


class TestClusterHelpers:
    """cluster_confusion, match_clusters and plot_cluster_confusion (no fit needed)."""

    def test_cluster_confusion_counts(self):
        a = np.array(["T", "B", "T", "NK", "B", "T"])
        b = np.array([1, 0, 1, 2, 0, 2])
        table, va, vb = analysis.cluster_confusion(a, b)
        assert list(va) == ["B", "NK", "T"] and list(vb) == [0, 1, 2]
        assert table.tolist() == [[2, 0, 0], [0, 0, 1], [0, 2, 1]]
        with pytest.raises(ValueError):
            analysis.cluster_confusion(a, b[:-1])

    def test_match_clusters_undoes_a_permutation(self):
        rng = np.random.default_rng(0)
        ref = rng.integers(0, 3, 300)
        perm = np.array([2, 0, 1])
        lab = perm[ref].copy()
        lab[:5] = (lab[:5] + 1) % 3                      # a few disagreements survive the relabelling
        out = analysis.match_clusters(ref, lab)
        assert np.mean(out == ref) == pytest.approx(295 / 300)
        # unassigned cells stay -1, an extra cluster gets a fresh id
        lab2 = lab.copy(); lab2[10:20] = -1; lab2[20:30] = 7
        out2 = analysis.match_clusters(ref, lab2)
        assert np.all(out2[10:20] == -1) and np.all(out2[20:30] == 3)

    def test_plot_cluster_confusion_aligns_and_scores(self):
        truth = np.repeat(["B", "NK", "T"], [50, 20, 30])
        assigns = np.repeat([2, 0, 1], [50, 20, 30])    # perfect clustering, permuted ids
        fig, ax = plt.subplots()
        res = analysis.plot_cluster_confusion(truth, assigns, ax=ax, title="x", col_names=["c0", "c1", "c2"])
        assert res["ari"] == pytest.approx(1.0) and res["ax"] is ax
        assert list(res["cols"]) == [2, 0, 1]
        assert np.all(np.diag(res["table"]) == [50, 20, 30])
        assert [t.get_text() for t in ax.get_xticklabels()] == ["c2", "c0", "c1"]
        assert ax.get_title() == "x\nARI = 1.000"
        unaligned = analysis.plot_cluster_confusion(truth, assigns, align=False)
        assert list(unaligned["cols"]) == [0, 1, 2]
        plt.close("all")


class TestProjections:
    def test_plot_pairwise_projections(self):
        rng = np.random.default_rng(0)
        pss = rng.random((4, 5, 6)); pss /= pss.sum()
        axes = analysis.plot_pairwise_projections(pss, title="t")
        assert len(axes) == 3
        assert axes[0].get_xlabel() == "Unspliced count" and axes[2].get_ylabel() == "Protein count"
        assert axes[1].images[0].get_array().shape == (6, 4)          # U x P marginal, transposed
        np.testing.assert_allclose(axes[0].images[0].get_array().sum(), 1.0)
        coords = np.array([[0, 1, 2], [3, 4, 5]]); freqs = np.array([0.5, 0.5])
        fig, axs = plt.subplots(1, 3)
        out = analysis.plot_pairwise_projections(pss, limits=[4, 5, 6], data=(coords, freqs), axes=axs,
                                                 labels=("A", "B", "C"), colorbar=True)
        assert out is axs and len(axs[0].collections) == 1 and axs[0].get_title() == "A vs B"
        with pytest.raises(ValueError):
            analysis.plot_pairwise_projections(pss[0])
        plt.close("all")
