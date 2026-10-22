"""The Chromatin model family in the ordinary pipeline: CMEModel, starting
points (monod.chromatin_fit), extraction, inference and nested comparison."""
import os

import numpy as np
import pytest

from monod import chromatin as ch
from monod import chromatin_fit as cf
from monod.analysis import compare_nested_models
from monod.cme_toolbox import CMEModel

TRUTH = dict(k_on=[0.5, 0.8], eps=0.6, beta=0.8, gamma=0.3)


def _model(rule="switch", seq="None", **kw):
    kw.setdefault("dropout_bounds", (0.0, 0.0, 1))
    return CMEModel("Chromatin", seq, n_sites=2, rule=rule, **kw)


def _theta(rule, rates):
    return cf.join_theta(TRUTH["k_on"], TRUTH["eps"], rates, TRUTH["beta"], TRUTH["gamma"])


def _cells(model, theta, n_cells, rng, limits=(40, 60, 4), samp=None):
    pss = model.eval_model_pss(theta, list(limits), samp)
    flat = pss.ravel() / pss.sum()
    idx = rng.choice(flat.size, size=n_cells, p=flat)
    return np.column_stack(np.unravel_index(idx, pss.shape))          # (unspliced, spliced, configuration)


def _hist(cells, n_sites=2):
    lim = [int(cells[:, 0].max() * 1.3) + 3, int(cells[:, 1].max() * 1.3) + 3, 2 ** n_sites]
    coords, counts = np.unique(cells, axis=0, return_counts=True)
    return (coords.astype(np.int64), counts / counts.sum()), lim


class TestChainMarginal:
    def test_recovers_chain_from_expected_counts(self):
        k_on, eps = np.array([0.5, 0.8, 0.3]), 0.6
        counts = 5000 * ch.ising_steady_state(k_on, 1.0, eps)
        fit = cf.fit_chain_marginal(counts)
        np.testing.assert_allclose(fit.k_on, k_on, rtol=1e-3)
        assert fit.eps == pytest.approx(eps, rel=1e-3)
        assert fit.p_drop is None and fit.n_params == 4

    def test_recovers_shared_dropout_and_accepts_a_fixed_one(self):
        k_on, eps, p = np.array([0.5, 0.8, 0.3]), 0.6, 0.2
        w = ch.ising_steady_state(k_on, 1.0, eps) @ ch.dropout_matrix(3, p)
        fit = cf.fit_chain_marginal(20000 * w, dropout="shared")
        np.testing.assert_allclose(fit.k_on, k_on, rtol=2e-2)
        np.testing.assert_allclose(fit.p_drop, p, atol=2e-2)
        fixed = cf.fit_chain_marginal(20000 * w, p_drop=p)
        np.testing.assert_allclose(fixed.k_on, k_on, rtol=1e-3)
        assert fixed.eps == pytest.approx(eps, rel=1e-3)


class TestParameterVectors:
    def test_split_join_embed(self):
        theta = cf.join_theta([0.5, 0.8], 0.6, [4.0, 2.0], 0.8, 0.3)
        k_on, eps, rates, beta, gamma = cf.split_theta(theta, "additive", 2)
        np.testing.assert_allclose(k_on, [0.5, 0.8]); assert eps == pytest.approx(0.6)
        np.testing.assert_allclose(rates, [4.0, 2.0]); assert (beta, gamma) == pytest.approx((0.8, 0.3))
        tab = cf.split_theta(cf.embed_in_table(theta, "additive", 2), "table", 2)[2]
        np.testing.assert_allclose(tab, [cf.RATE_FLOOR, 4.0, 2.0, 6.0])
        assert cf.n_params("table", 2) == 9 and cf.n_params("switch", 2) == 6

    def test_dense_sparse_round_trip(self):
        rng = np.random.default_rng(0)
        dense = rng.integers(0, 3, size=(5, 6, 4)).astype(float)
        coords, freqs = cf.sparse_histogram(dense)
        back = cf.dense_histogram((coords, freqs), dense.shape) * dense.sum()
        np.testing.assert_allclose(back, dense)


class TestCMEModel:
    def test_declares_itself_like_the_other_families(self):
        m = _model("additive")
        assert m.model_modalities == ["unspliced", "spliced", "configuration"]
        assert m.get_num_params() == 7 == len(m.param_str) == len(m.param_names())
        assert len(m.bio_bounds["phys_lb"]) == 7 == len(m.bio_bounds["phys_ub"])
        assert len(m.filter_bounds["min_means"]) == 3
        assert m.seq_bounds["gridsize"] == [1, 1, 1]                   # no dropout grid in this test model
        assert m.null_sampling_grid() == ([0, 0, 0.0], [0, 0, 0.0], [1, 1, 1])
        with pytest.raises(ValueError):
            CMEModel("Chromatin", "None", n_sites=2, rule="gate")
        again = CMEModel.from_dict(m.to_dict())
        assert again.rule == "additive" and again.n_sites == 2 and again.dropout_bounds == (0.0, 0.0, 1)

    def test_default_dropout_grid_rides_on_the_sampling_grid(self):
        m = CMEModel("Chromatin", "Poisson", n_sites=2, rule="switch")
        assert len(m.seq_bounds["samp_lb"]) == 3 and m.seq_bounds["gridsize"][2] == 3
        n = CMEModel("Chromatin", "None", n_sites=2, rule="switch")
        assert n.null_sampling_grid()[2] == [1, 1, 3]

    def test_pss_matches_the_kernel_and_has_the_modality_order(self):
        m = _model("additive")
        theta = _theta("additive", [4.0, 2.0])
        pss = m.eval_model_pss(theta, [30, 40, 4])
        assert pss.shape == (30, 40, 4) and pss.sum() == pytest.approx(1.0)
        H, A, C = ch.nascent_mature_system(TRUTH["k_on"], 1.0, TRUTH["eps"], "additive", TRUTH["beta"], TRUTH["gamma"], rule_params=[4.0, 2.0])
        ref = ch.eval_joint_pss(H, A, C, [30, 40], tol=m.chromatin_tol)
        np.testing.assert_allclose(np.moveaxis(pss, -1, 0), ref, atol=1e-12)
        with pytest.raises(ValueError):
            m.eval_model_pss(theta, [30, 40, 3])

    def test_dropout_and_poisson_sampling(self):
        m = _model("switch", seq="Poisson")
        theta = _theta("switch", [5.0])
        clean = m.eval_model_pss(theta, [30, 40, 4], [0.0, 0.0, 0.0])
        dropped = m.eval_model_pss(theta, [30, 40, 4], [0.0, 0.0, np.log10(0.3)])
        # dropout thins the open-site frequencies and leaves the RNA marginal alone
        configs = ch.configurations(2)
        np.testing.assert_allclose(dropped.sum(axis=(0, 1)) @ configs, 0.7 * (clean.sum(axis=(0, 1)) @ configs), rtol=1e-10)
        np.testing.assert_allclose(dropped.sum(2), clean.sum(2), atol=1e-12)
        # Poisson capture at log10 lambda = -0.3 halves the RNA means and leaves the configuration marginal alone
        half = m.eval_model_pss(theta, [30, 40, 4], [np.log10(0.5), np.log10(0.5), 0.0])
        u = np.arange(30)
        assert (half.sum(axis=(1, 2)) @ u) == pytest.approx(0.5 * (clean.sum(axis=(1, 2)) @ u), rel=1e-3)
        np.testing.assert_allclose(half.sum(axis=(0, 1)), clean.sum(axis=(0, 1)), atol=1e-10)

    def test_pinned_gradient_and_search_tolerance(self):
        m = _model("switch")
        theta = _theta("switch", [5.0])
        rng = np.random.default_rng(1)
        data, lim = _hist(_cells(m, theta, 800, rng))
        f, g = m.eval_model_kld_and_grad(theta, lim, [0, 0, 0], data)
        g_free = np.zeros(theta.size)
        for i in range(theta.size):
            e = np.zeros(theta.size); e[i] = 1e-3
            g_free[i] = (m.eval_model_kld(theta + e, lim, [0, 0, 0], data) - m.eval_model_kld(theta - e, lim, [0, 0, 0], data)) / 2e-3
        np.testing.assert_allclose(g, g_free, rtol=3e-2, atol=3e-2 * np.abs(g_free).max())
        assert m.has_search_tolerance()
        with m.search_tolerance():
            assert m.chromatin_tol == m.chromatin_search_tol
            coarse = m.eval_model_kld(theta, lim, [0, 0, 0], data)
        assert m.chromatin_tol == 1e-5
        assert coarse == pytest.approx(f, abs=1e-5)

    def test_moments_start_recovers_the_chain_and_lands_in_the_box(self):
        m = _model("additive")
        theta = _theta("additive", [4.0, 2.0])
        rng = np.random.default_rng(2)
        data, lim = _hist(_cells(m, theta, 1500, rng))
        lb, ub = np.array(m.bio_bounds["phys_lb"]), np.array(m.bio_bounds["phys_ub"])
        x0 = m.get_MoM(None, lb, ub, [0, 0, 0], hist=data, limits=lim)
        assert x0.shape == (7,) and np.all(x0 >= lb) and np.all(x0 <= ub)
        np.testing.assert_allclose(10 ** x0[:2], TRUTH["k_on"], rtol=0.25)
        assert 10 ** x0[2] == pytest.approx(TRUTH["eps"], rel=0.3)
        # the start is a better point than the middle of the box
        mid = 0.5 * (lb + ub)
        assert m.eval_model_kld(x0, lim, [0, 0, 0], data) < m.eval_model_kld(mid, lim, [0, 0, 0], data)
        assert np.array_equal(m.get_MoM(None, lb, ub, [0, 0, 0]), mid)      # no histogram: the box middle


@pytest.mark.filterwarnings("ignore::UserWarning")
class TestPipeline:
    @pytest.fixture(scope="class")
    def synthetic_h5ad(self, tmp_path_factory):
        import anndata as ad
        import pandas as pd
        m = _model("switch")
        rng = np.random.default_rng(4)
        truths = [_theta("switch", [5.0]), _theta("switch", [3.0])]
        n_cells = 900
        layers = {k: np.zeros((n_cells, 2), dtype=np.int64) for k in ("unspliced", "spliced", "configuration")}
        for j, th in enumerate(truths):
            cells = _cells(m, th, n_cells, rng)
            layers["unspliced"][:, j], layers["spliced"][:, j], layers["configuration"][:, j] = cells.T
        adata = ad.AnnData(X=layers["spliced"].astype(float), layers=layers,
                           var=pd.DataFrame(index=["geneA", "geneB"]), obs=pd.DataFrame(index=[f"c{i}" for i in range(n_cells)]))
        path = tmp_path_factory.mktemp("chrom") / "synthetic.h5ad"
        adata.write_h5ad(path)
        return str(path), truths, n_cells

    def test_extract_sets_the_configuration_limit(self, synthetic_h5ad):
        from monod.extract_data import extract_data
        path, truths, n_cells = synthetic_h5ad
        adata = extract_data(path, _model("switch"), n_genes=2, viz=False, dataset_name="chrom_extract")
        assert adata.n_vars == 2
        assert np.all(adata.uns["M"][2, :] == 4)                       # M is (n_layers, n_genes)
        assert adata.uns["model"]["bio_model"] == "Chromatin"

    def test_perform_inference_fits_both_genes(self, synthetic_h5ad, tmp_path):
        from monod.inference import perform_inference
        path, truths, n_cells = synthetic_h5ad
        m = _model("switch")
        gp = {"num_restarts": 1, "max_iterations": 80, "check_identifiability": False}
        adata = perform_inference(path, m, n_genes=2, viz=False, gradient_params=gp, seed=1,
                                  dataset_string=str(tmp_path / "chrom_run"))
        names = m.param_names()
        cols = [c for c in adata.var.columns if c in names or c.strip("$") in [n_.strip("$") for n_ in names]]
        assert len(cols) >= 1, list(adata.var.columns)
        # fitted gamma within a factor two of the truth for both genes
        gcol = [c for c in adata.var.columns if "gamma" in c and "FC" not in c and "sigma" not in c.lower()][0]
        gam = np.asarray(adata.var[gcol], dtype=float)
        assert gam.shape == (2,) and np.all(np.isfinite(gam))
        gam_lin = 10.0 ** gam if np.all(gam < 0) else gam                # stored in log10 (as fitted) or linear
        np.testing.assert_allclose(gam_lin, TRUTH["gamma"], rtol=0.5)

    def test_nested_comparison_keeps_the_switch_on_switch_data(self, synthetic_h5ad):
        path, truths, n_cells = synthetic_h5ad
        import scipy.optimize
        small, big = _model("switch"), _model("table")
        rng = np.random.default_rng(6)
        data, lim = _hist(_cells(small, truths[0], 1200, rng))
        # the small model's optimum, as inference would hand it over
        with small.search_tolerance():
            fit = scipy.optimize.minimize(lambda x: small.eval_model_kld_and_grad(x, lim, [0, 0, 0], data), truths[0],
                                          jac=True, method="L-BFGS-B",
                                          bounds=scipy.optimize.Bounds(small.bio_bounds["phys_lb"], small.bio_bounds["phys_ub"]),
                                          options=dict(maxiter=100, ftol=1e-7, gtol=1e-3))
        res = compare_nested_models(small, big, lambda x: cf.embed_in_table(x, "switch", 2), fit.x, lim, [0, 0, 0],
                                    data, 1200, table="always")
        stat, df, p_score = res["score"]
        assert df == 2 and res["fitted"]
        assert res["kld_big"] <= res["kld_small"] + 1e-9                 # the nesting model never ends worse
        assert res["lr"] < 12.0 and not res["prefer_big"]                 # no evidence for the free table
        assert abs(stat - res["lr"]) < 6.0                                 # score and likelihood ratio agree in size
