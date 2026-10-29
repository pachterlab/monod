"""meK-Means EM loop behaviour (audit fix 8 items 1, 3, 4; 2026-10-15).

1. Clusters without fitted parameters get zero posterior on the Python E-step
   (previously a log-likelihood of 0, which swallowed every cell).
3. Reported Q / lower bound come from an E-step run after the final M-step.
4. EM stops early once the lower bound plateaus (em_tol / em_patience).
"""
import os
import types
import unittest.mock
import warnings

import numpy as np
import pytest

import monod.mminference as mm
from monod.cme_toolbox import CMEModel

from monod._core import _mc  # None unless the compiled extension is really installed
_HAS_RUST = _mc is not None

needs_rust = pytest.mark.skipif(_mc is None, reason="monod_core not installed")
GP = {"max_iterations": 15, "init_pattern": "moments", "num_restarts": 1, "num_gene_cores": 1}


# ---------------------------------------------------------------------------
# 1. Empty-cluster E-step (Python fallback)
# ---------------------------------------------------------------------------

class TestEmptyClusterEStep:
    def _stub(self, n_cells=40, n_genes=2, k=3):
        rng = np.random.default_rng(0)
        layers = np.stack([rng.poisson(2.0, (n_cells, n_genes)), rng.poisson(3.0, (n_cells, n_genes))]).astype(np.int64)
        M = layers.max(axis=1) + 6                                   # (2, n_genes)
        sd = types.SimpleNamespace(n_cells=n_cells, n_genes=n_genes, layers=layers, M=M)
        params = np.tile(np.array([0.3, 0.0, -0.1]), (n_genes, 1))
        theta = {0: (params, np.zeros(n_genes), None, None), 1: (params + 0.1, np.zeros(n_genes), None, None)}
        self_ = types.SimpleNamespace(theta=theta, k=k, weights=np.array([0.5, 0.3, 0.2]),
                                      regressor=[None] * n_genes)
        return self_, sd

    def test_missing_cluster_gets_zero_posterior(self):
        model = CMEModel("Bursty", "None")
        self_, sd = self._stub()
        with unittest.mock.patch.object(mm, "_HAS_RUST", False):
            Q, lb, q = mm.GradientInference._e_step(self_, model, sd)
        assert Q.shape == (sd.n_cells, 3)
        np.testing.assert_allclose(Q.sum(1), 1.0)
        assert np.all(Q[:, 2] == 0.0), "unfitted cluster must receive zero posterior"
        assert np.isfinite(lb) and np.isfinite(q)

    def test_no_fitted_cluster_raises(self):
        model = CMEModel("Bursty", "None")
        self_, sd = self._stub()
        self_.theta = {}
        with pytest.raises(ValueError, match="before any cluster"):
            mm.GradientInference._e_step(self_, model, sd)


# ---------------------------------------------------------------------------
# 3 + 4. EM loop ordering and early stopping (deterministic stub)
# ---------------------------------------------------------------------------

class _LoopStub:
    """Stands in for GradientInference inside _fit; records call order."""
    def __init__(self, epochs, em_tol, em_patience, lbs):
        self.epochs, self.em_tol, self.em_patience = epochs, em_tol, em_patience
        self.k = 2
        self.theta = {0: (None, np.zeros(3), None, None), 1: (None, np.zeros(3), None, None)}
        self._lbs = list(lbs)          # lower bound returned by successive E-steps
        self.calls = []
        self.n_m = 0

    def _e_step(self, model, sd):
        self.calls.append("E")
        i = min(len(self._lbs) - 1, sum(c == "E" for c in self.calls) - 1)
        Q = np.full((5, 2), 0.5) + 0.01 * self.n_m      # encodes which theta it saw
        return Q, self._lbs[i], -1.0

    def _part_search_data(self, sd, Q):
        return {0: None, 1: None}

    def _m_step(self, model, k_dict, Q, num_cores=1):
        self.calls.append("M")
        self.n_m += 1


class TestEMLoop:
    SD = types.SimpleNamespace(n_genes=3, n_cells=5)

    def test_final_e_step_follows_last_m_step(self):
        stub = _LoopStub(epochs=3, em_tol=0.0, em_patience=2, lbs=[-10, -9, -8, -8])
        Q, lb, all_qs, all_klds = mm.GradientInference._fit(stub, None, self.SD)
        assert stub.calls == ["E", "M", "E", "M", "E", "M", "E"]
        assert stub.n_epochs_run == 3 and len(all_qs) == 3 and len(all_klds) == 3
        np.testing.assert_allclose(Q, 0.5 + 0.01 * 3)   # Q evaluated under the final theta

    def test_em_tol_zero_runs_all_epochs(self):
        stub = _LoopStub(epochs=6, em_tol=0.0, em_patience=1, lbs=[-5.0] * 10)
        mm.GradientInference._fit(stub, None, self.SD)
        assert stub.n_epochs_run == 6

    def test_early_stop_on_plateau(self):
        # lower bound: big moves for 2 epochs, then flat -> stop after patience=2 stalled epochs
        stub = _LoopStub(epochs=20, em_tol=1e-4, em_patience=2, lbs=[-10, -8, -7, -7, -7, -7, -7])
        mm.GradientInference._fit(stub, None, self.SD)
        assert stub.n_epochs_run == 5         # epochs 4 and 5 are stalled relative to 3 and 4
        assert stub.calls[-1] == "E"

    def test_patience_resets_after_a_move(self):
        stub = _LoopStub(epochs=20, em_tol=1e-4, em_patience=2, lbs=[-10, -10, -6, -6, -6, -6])
        mm.GradientInference._fit(stub, None, self.SD)
        assert stub.n_epochs_run == 5


# ---------------------------------------------------------------------------
# End-to-end on synthetic two-population data (Rust path)
# ---------------------------------------------------------------------------

def _two_population_sd(n_cells=240, n_genes=4, seed=0):
    rng = np.random.default_rng(seed)
    half = n_cells // 2
    U = np.vstack([rng.poisson(1.0, (half, n_genes)), rng.poisson(6.0, (half, n_genes))])
    S = np.vstack([rng.poisson(2.0, (half, n_genes)), rng.poisson(12.0, (half, n_genes))])
    layers = [np.ascontiguousarray(U.astype(np.int64)), np.ascontiguousarray(S.astype(np.int64))]
    coords, freqs = _mc.make_state_dist(layers)
    limits = np.ascontiguousarray(np.stack([l.max(axis=0) + 8 for l in layers]).astype(np.int64))
    sd = _mc.searchdata_from_arrays(layers, ["unspliced", "spliced"], limits, coords, freqs,
                                    [f"g{i}" for i in range(n_genes)], n_cells, "unique",
                                    None, None, 2, 40)
    truth = np.r_[np.zeros(half, int), np.ones(half, int)]
    return sd, truth


@needs_rust
class TestMEKMeansEndToEnd:
    def test_converges_early_and_recovers_populations(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        warnings.filterwarnings("ignore")
        sd, truth = _two_population_sd()
        model = CMEModel("Bursty", "None")
        p = mm.MEKMeansParameters("ds", model, k=2, epochs=40, em_tol=1e-4, em_patience=2, gradient_params=GP)
        gpr = p.par_fun((0, [sd, model], 2, 40, 1))
        n_run = len(gpr.all_qs)
        assert 1 <= n_run < 40, f"expected early stopping, ran {n_run} epochs"
        assigns = np.asarray(gpr.assigns)
        agreement = max(np.mean(assigns == truth), np.mean(assigns != truth))
        assert agreement > 0.9
        assert np.isfinite(gpr.aic_per_cell)

    def test_em_tol_zero_runs_every_epoch(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        warnings.filterwarnings("ignore")
        sd, _ = _two_population_sd()
        model = CMEModel("Bursty", "None")
        p = mm.MEKMeansParameters("ds", model, k=2, epochs=3, em_tol=0.0, gradient_params=GP)
        gpr = p.par_fun((0, [sd, model], 2, 3, 1))
        assert len(gpr.all_qs) == 3


# ---------------------------------------------------------------------------
# Seeded initialisation (the 0.9 + uniform scheme itself is intentional)
# ---------------------------------------------------------------------------

class TestSeededInit:
    def _stub(self, seed, n_cells=60, n_genes=3, k=4):
        rng = np.random.default_rng(123)
        layers = np.stack([rng.poisson(3.0, (n_cells, n_genes)), rng.poisson(5.0, (n_cells, n_genes))]).astype(np.int64)
        sd = types.SimpleNamespace(n_cells=n_cells, n_genes=n_genes, layers=layers)
        self_ = types.SimpleNamespace(k=k, weights=np.ones(k) / k, random_state=seed)
        return self_, sd

    def test_python_path_is_reproducible_and_seed_dependent(self):
        with unittest.mock.patch.object(mm, "_HAS_KMEANS", False):
            a = mm.GradientInference._initialize_Q(*self._stub(0))
            b = mm.GradientInference._initialize_Q(*self._stub(0))
            c = mm.GradientInference._initialize_Q(*self._stub(1))
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, c)
        np.testing.assert_allclose(a.sum(1), 1.0)

    def test_none_seed_still_runs(self):
        with unittest.mock.patch.object(mm, "_HAS_KMEANS", False):
            self_, sd = self._stub(None)
            Q = mm.GradientInference._initialize_Q(self_, sd)
        assert Q.shape == (60, 4) and isinstance(self_._init_seed_used, int)

    @needs_rust
    def test_rust_path_uses_random_state(self, tmp_path, monkeypatch):
        if not mm._HAS_KMEANS:
            pytest.skip("monod_core built without the kmeans feature")
        monkeypatch.chdir(tmp_path)
        warnings.filterwarnings("ignore")
        sd, _ = _two_population_sd()
        model = CMEModel("Bursty", "None")
        def init(seed):
            p = mm.MEKMeansParameters("ds", model, k=2, epochs=1, random_state=seed, gradient_params=GP)
            gi = mm.GradientInference(p, model, sd, 0, 2, 1)
            return gi._initialize_Q(sd)
        np.testing.assert_array_equal(init(0), init(0))
        assert not np.array_equal(init(0), init(7))

    def test_default_random_state_is_zero(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        p = mm.MEKMeansParameters("ds", CMEModel("Bursty", "None"), k=2, epochs=1)
        assert p.random_state == 0


# ---------------------------------------------------------------------------
# Paper conformance: AIC definitions and the k-means input toggle
# ---------------------------------------------------------------------------

class TestAIC:
    def test_mek_means_aic_matches_paper_formula(self):
        rng = np.random.default_rng(0)
        n_cells, k, G, P = 50, 3, 7, 3
        logL = rng.normal(-40, 3, size=(n_cells, k))          # log p(x|theta_k) + log pi_k
        lse = np.log(np.exp(logL - logL.max(1, keepdims=True)).sum(1)) + logL.max(1)
        Q = np.exp(logL - lse[:, None])
        lower_bound = lse.mean()
        joint = logL.max(1).sum()                               # hard assignment picks the max
        k_final = len(np.unique(np.argmax(Q, 1)))
        expected = -joint / n_cells + (P * G * k_final + (k_final - 1)) / n_cells
        got = mm.mek_means_aic(Q, lower_bound, P, G, n_cells)
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_mek_means_aic_counts_only_surviving_clusters(self):
        Q = np.zeros((10, 4)); Q[:5, 0] = 1.0; Q[5:, 2] = 1.0  # clusters 1 and 3 empty
        a = mm.mek_means_aic(Q, -30.0, 3, 10, 10)
        assert a == pytest.approx(30.0 + (3 * 10 * 2 + 1) / 10)

    def test_per_gene_aic_penalty_scales_with_clusters(self):
        from monod.inference import get_AIC_mek_means
        class SR:
            n_genes = 4
            n_cells = 100
            sp = types.SimpleNamespace(n_phys_pars=3)
            def __init__(self, ll): self._ll = np.asarray(ll, float)
            def get_logL(self, sd, EPS=1e-20, offs=0): return self._ll
        srs = [SR([-10, -20, -30, -40]), SR([-1, -2, -3, -4]), SR([-5, -5, -5, -5])]
        aic = get_AIC_mek_means(srs, None)
        np.testing.assert_allclose(aic, 2 * 3 * 3 - 2 * (srs[0]._ll + srs[1]._ll + srs[2]._ll))


class TestInitModality:
    def _stub(self, init_modality, U, S, k=3, seed=0):
        layers = np.stack([U, S]).astype(np.int64)
        sd = types.SimpleNamespace(n_cells=U.shape[0], n_genes=U.shape[1], layers=layers)
        self_ = types.SimpleNamespace(k=k, weights=np.ones(k) / k, random_state=seed,
                                      init_modality=init_modality, init_modalities=["unspliced", "spliced"])
        return self_, sd

    def test_spliced_only_ignores_unspliced(self):
        rng = np.random.default_rng(5)
        S = rng.poisson(4.0, (80, 3)); U1 = rng.poisson(2.0, (80, 3)); U2 = rng.poisson(50.0, (80, 3))
        with unittest.mock.patch.object(mm, "_HAS_KMEANS", False):
            a = mm.GradientInference._initialize_Q(*self._stub("spliced", U1, S))
            b = mm.GradientInference._initialize_Q(*self._stub("spliced", U2, S))
            c = mm.GradientInference._initialize_Q(*self._stub("sum", U2, S))
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, c)

    def test_unknown_modality_rejected_at_construction(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="init_modality must be"):
            mm.MEKMeansParameters("ds", CMEModel("Bursty", "None"), k=2, epochs=1, init_modality="protein")

    @needs_rust
    def test_end_to_end_with_spliced_init(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        warnings.filterwarnings("ignore")
        sd, truth = _two_population_sd()
        model = CMEModel("Bursty", "None")
        p = mm.MEKMeansParameters("ds", model, k=2, epochs=5, init_modality="spliced", gradient_params=GP)
        gpr = p.par_fun((0, [sd, model], 2, 5, 1))
        assigns = np.asarray(gpr.assigns)
        assert max(np.mean(assigns == truth), np.mean(assigns != truth)) > 0.9
        assert np.isfinite(gpr.aic_per_cell)


# ---------------------------------------------------------------------------
# Rust vs Python backend consistency (needs monod_core built with --features kmeans
# for the initialisation test; the E-step test only needs monod_core)
# ---------------------------------------------------------------------------

def _two_composition_populations(n_cells=240, n_genes=4, seed=1):
    """Two populations that differ in gene *composition* (not total counts), so
    the per-cell normalisation to 1e4 used before k-means keeps them separable."""
    rng = np.random.default_rng(seed)
    half = n_cells // 2
    hi, lo = 12.0, 1.0
    meanU_a = np.array([hi, hi, lo, lo]); meanU_b = meanU_a[::-1]
    U = np.vstack([rng.poisson(meanU_a, (half, n_genes)), rng.poisson(meanU_b, (half, n_genes))])
    S = np.vstack([rng.poisson(2 * meanU_a, (half, n_genes)), rng.poisson(2 * meanU_b, (half, n_genes))])
    layers = np.stack([U, S]).astype(np.int64)
    truth = np.r_[np.zeros(half, int), np.ones(half, int)]
    return layers, truth


@needs_rust
class TestRustPythonBackendConsistency:
    @pytest.mark.parametrize("init_modality", ["sum", "spliced"])
    def test_kmeans_labels_agree(self, init_modality):
        """The Rust KMeans++/Lloyd and sklearn KMeans partition the same
        normalised matrix the same way (up to label permutation).  The 0.9+uniform
        perturbation is applied with independent RNGs afterwards, so Q itself is
        not expected to match; only the labels are compared."""
        if not mm._HAS_KMEANS:
            pytest.skip("monod_core built without the kmeans feature")
        from sklearn.cluster import KMeans
        layers, truth = _two_composition_populations()
        if init_modality == "sum":
            init_layers = layers[:2]
        else:
            init_layers = np.stack([layers[1], np.zeros_like(layers[1])])
        _, labels_rust = _mc.initialize_q_kmeans(np.ascontiguousarray(init_layers, dtype=np.int64), 2, seed=0)
        X = init_layers[0].astype(float) + init_layers[1].astype(float)
        X = np.log1p(X * (1e4 / X.sum(1))[:, None])
        labels_py = KMeans(n_clusters=2, random_state=0).fit(X).labels_
        lr, lp = np.asarray(labels_rust), np.asarray(labels_py)
        agree = max(np.mean(lr == lp), np.mean(lr != lp))
        assert agree > 0.95, f"Rust/Python k-means labels agree on only {agree:.0%} of cells"
        assert max(np.mean(lr == truth), np.mean(lr != truth)) > 0.9

    def test_e_step_parity(self, tmp_path, monkeypatch):
        """Given the same theta and weights, the Rust and Python E-steps return
        the same posterior, lower bound and Q function."""
        monkeypatch.chdir(tmp_path)
        warnings.filterwarnings("ignore")
        sd, _ = _two_population_sd()
        model = CMEModel("Bursty", "None")
        p = mm.MEKMeansParameters("ds", model, k=2, epochs=1, random_state=0, gradient_params=GP)
        gi = mm.GradientInference(p, model, sd, 0, 2, 1)
        Q0 = gi._initialize_Q(sd)
        gi._m_step(model, gi._part_search_data(sd, Q0), Q0)
        assert sorted(gi.theta) == [0, 1]
        Qr, lbr, qr = gi._e_step(model, sd)
        with unittest.mock.patch.object(mm, "_HAS_RUST", False):
            Qp, lbp, qp = gi._e_step(model, sd)
        np.testing.assert_allclose(Qr, Qp, rtol=1e-6, atol=1e-9)
        assert lbr == pytest.approx(lbp, rel=1e-6)
        assert qr == pytest.approx(qp, rel=1e-6)


@needs_rust
class TestReviewRegressions:
    def test_cluster_that_empties_is_dropped_from_theta(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        warnings.filterwarnings("ignore")
        sd, _ = _two_population_sd()
        model = CMEModel("Bursty", "None")
        p = mm.MEKMeansParameters("ds", model, k=2, epochs=1, random_state=0, gradient_params=GP)
        gi = mm.GradientInference(p, model, sd, 0, 2, 1)
        Q0 = gi._initialize_Q(sd)
        k_dict = gi._part_search_data(sd, Q0)
        gi._m_step(model, k_dict, Q0)
        assert sorted(gi.theta) == [0, 1]
        gi._m_step(model, {0: k_dict[0]}, Q0)       # cluster 1 received no cells this epoch
        assert sorted(gi.theta) == [0]
        Q, lb, q = gi._e_step(model, sd)
        assert np.all(Q[:, 1] == 0.0)

    def test_mek_means_plot_gene_distributions_1x1_and_unsaved(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        warnings.filterwarnings("ignore")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sd, _ = _two_population_sd()
        model = CMEModel("Bursty", "None")
        p = mm.MEKMeansParameters("ds", model, k=2, epochs=2, random_state=0, gradient_params=GP)
        results = p.run_mek_means(sd)
        sr = results[0]
        sr.find_sampling_optimum()
        sr.plot_gene_distributions(sd, sz=(1, 1), marg="joint", savefig=False)
        sr.plot_gene_distributions(sd, sz=(1, 2), marg="spliced", savefig=True)
        plt.close("all")


def test_fit_tolerates_objects_built_without_init():
    """Notebooks assemble GradientInference by hand; _fit must not require the new attributes."""
    stub = _LoopStub(epochs=3, em_tol=0.0, em_patience=2, lbs=[-3, -2, -1, -1])
    del stub.em_tol, stub.em_patience
    Q, lb, all_qs, all_klds = mm.GradientInference._fit(stub, None, types.SimpleNamespace(n_genes=3, n_cells=5))
    assert stub.n_epochs_run == 3 and stub.calls[-1] == "E"


def test_aic_attribute_is_deprecated_alias():
    gpr = mm.GridPointResults(np.zeros((1, 2, 3)), np.zeros((1, 2)), np.zeros(1), np.zeros(1),
                              np.ones(2) / 2, 1.5, np.array([0, 1]), [-1.0], [np.zeros((2, 2))],
                              np.zeros((2, 2)), np.zeros(2), 0, "x")
    assert gpr.aic_per_cell == 1.5
    with pytest.warns(DeprecationWarning, match="argmin"):
        assert gpr.aic == 1.5


# ---------------------------------------------------------------------------
# perform_inference with meK-Means: the cluster-prefixed columns end to end
# ---------------------------------------------------------------------------

def test_perform_inference_mek_means_writes_cluster_columns(tmp_path, monkeypatch):
    """The meK-Means branch of perform_inference shares the result-writing loop
    with the plain run: parameters, errors and AIC per cluster prefix, cluster
    labels in obs, and the result list in uns."""
    from monod.inference import perform_inference, search_result_list_from_uns
    example = os.path.join(os.path.dirname(__file__), "..", "example_h5ad", "example_adata.h5ad")
    monkeypatch.chdir(tmp_path)
    gp = {"max_iterations": 5, "init_pattern": "moments", "num_restarts": 1, "num_gene_cores": 1}
    adata = perform_inference(example, CMEModel("Bursty", "None"), n_genes=3, viz=False, gradient_params=gp, seed=1,
                              mek_means_params=(2, 2), exclude_sigma=False, ambiguous="drop")
    names = [p.replace(r"\log_{10} ", "") for p in CMEModel("Bursty", "None").param_str]
    for j in range(2):
        for name in names:
            assert f"c{j}_{name}" in adata.var.columns and f"c{j}_error_{name}" in adata.var.columns
        assert f"c{j}_error_unreliable" in adata.var.columns
    assert "AIC" in adata.var.columns and np.all(np.isfinite(adata.var["AIC"]))
    assert adata.obs["cluster"].notna().all() and set(adata.obs["cluster"].unique()) <= {0, 1}
    srs = search_result_list_from_uns(adata.uns["search_result_list"])
    assert len(srs) == 2 and all(sr.phys_optimum.shape == (adata.n_vars, 3) for sr in srs)   # the example file has one gene


# ---------------------------------------------------------------------------
# cell_assignments: one cluster index per cell from the result list
# ---------------------------------------------------------------------------

def test_cell_assignments_from_result_list():
    n = 10
    filt0 = np.zeros(n, bool); filt0[[0, 1, 2, 7]] = True
    filt2 = np.zeros(n, bool); filt2[[3, 4, 9]] = True
    results = [types.SimpleNamespace(filt=filt0, assigns=0), types.SimpleNamespace(filt=filt2, assigns=2)]
    out = mm.cell_assignments(results)
    assert out.tolist() == [0, 0, 0, 2, 2, -1, -1, 0, -1, 2]      # cluster 1 ended empty: -1
    np.testing.assert_array_equal(mm.cell_assignments(results, n_cells=n), out)
    with pytest.raises(ValueError):
        mm.cell_assignments([])


@needs_rust
def test_cell_assignments_matches_run_mek_means(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    warnings.filterwarnings("ignore")
    sd, truth = _two_population_sd()
    p = mm.MEKMeansParameters("ds", CMEModel("Bursty", "None"), k=2, epochs=3, gradient_params=GP, random_state=0)
    results = p.run_mek_means(sd)
    out = mm.cell_assignments(results, sd.n_cells)
    assert out.shape == (sd.n_cells,) and set(np.unique(out)) <= {0, 1}
    for sr in results:
        assert np.all(out[np.asarray(sr.filt, bool)] == int(sr.assigns))
    agreement = max(np.mean(out == truth), np.mean(out != truth))
    assert agreement > 0.9
