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
        madata = extract_data(ad.read_h5ad(EXAMPLE), model, dataset_name="x", n_genes=3, viz=False)
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
        a1 = perform_inference(EXAMPLE, m, n_genes=3, viz=False, gradient_params=GP, seed=1)
        a2 = perform_inference(EXAMPLE, m, n_genes=3, viz=False, gradient_params=GP, seed=1)
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
        fig = analysis.plot_params_for_pair(sr, sr)                 # no batch dir -> not saved
        assert len(fig.axes) == sr.sp.n_phys_pars
        analysis.make_batch_analysis_dir([sr], str(tmp_path))
        analysis.plot_params_for_pair(sr, sr, plot_errorbars=True)  # sigma missing -> warning, no crash
        assert os.path.exists(os.path.join(str(tmp_path), "analysis_figures", "pair_parameter_comparison_12.png"))
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
