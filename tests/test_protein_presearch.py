"""Nelder-Mead pre-search for the ProteinBursty model.

The ProteinBursty likelihood has a flat valley along k_p/gamma_p on which the
finite-difference L-BFGS-B fit stalls. ``GradientInference._presearch_x0`` runs a
short Nelder-Mead search from the first starting point of every gene before the
gradient fit. These tests cover the dispatch rules and that the refined start is
never worse than the one it replaces; the accuracy study behind the default lives
in the docs (usage.md, "A note on the protein model").
"""
import numpy as np
import pytest

from monod.cme_toolbox import CMEModel
from monod.inference import InferenceParameters, GradientInference, _mc

pytestmark = pytest.mark.skipif(_mc is None, reason="monod_core not available")

LIMITS = np.array([[8], [6], [5]], dtype=np.int64)


@pytest.fixture(scope="module")
def protein_sd():
    rng = np.random.default_rng(7)
    layers = [np.ascontiguousarray(rng.integers(0, 4, size=(120, 1)), dtype=np.int64) for _ in range(3)]
    coords, freqs = _mc.make_state_dist(layers)
    return _mc.searchdata_from_arrays(
        layers, ["unspliced", "spliced", "protein"], LIMITS, coords, freqs, ["G0"], 120, "unique"
    )


def _grad_inference(model, sd, **gp):
    params = {"max_iterations": 5, "init_pattern": "moments", "num_restarts": 2, "num_gene_cores": 1}
    params.update(gp)
    ip = InferenceParameters("presearch_test", model, gradient_params=params)
    return GradientInference(ip, model, sd, 0)


def _record_minimize(monkeypatch):
    import scipy.optimize
    calls = []
    real = scipy.optimize.minimize

    def spy(fun, x0, *args, **kwargs):
        calls.append(kwargs.get("method"))
        return real(fun, x0, *args, **kwargs)

    monkeypatch.setattr(scipy.optimize, "minimize", spy)
    return calls


class TestDispatch:
    def test_default_runs_nelder_mead_for_protein_only(self, protein_sd, monkeypatch):
        model = CMEModel("ProteinBursty", "None", protein_limit=float("inf"))
        gi = _grad_inference(model, protein_sd, presearch_maxiter=5)
        calls = _record_minimize(monkeypatch)
        x0 = [[[0.0] * 5, [0.5] * 5]]
        out = gi._presearch_x0(model, protein_sd, x0)
        assert calls == ["Nelder-Mead"]
        assert out[0][1] == [0.5] * 5, "random restarts must be left untouched"
        assert out[0][0] != [0.0] * 5

        # a two-species model gets no pre-search by default
        bursty = CMEModel("Bursty", "None")
        gi_b = _grad_inference(bursty, protein_sd)
        calls.clear()
        x0b = [[[0.0] * 3]]
        assert gi_b._presearch_x0(bursty, protein_sd, x0b) == x0b
        assert calls == []

    def test_presearch_can_be_disabled_or_forced(self, protein_sd, monkeypatch):
        model = CMEModel("ProteinBursty", "None", protein_limit=float("inf"))
        calls = _record_minimize(monkeypatch)
        x0 = [[[0.0] * 5]]
        gi = _grad_inference(model, protein_sd, presearch=None)
        assert gi._presearch_x0(model, protein_sd, x0) == x0 and calls == []

        gi = _grad_inference(model, protein_sd, presearch="nelder-mead", presearch_maxiter=3)
        gi._presearch_x0(model, protein_sd, x0)
        assert calls == ["Nelder-Mead"]

        gi = _grad_inference(model, protein_sd, presearch="bogus")
        with pytest.raises(ValueError, match="presearch"):
            gi._presearch_x0(model, protein_sd, x0)


class TestResult:
    def test_refined_start_is_within_bounds_and_no_worse(self, protein_sd):
        model = CMEModel("ProteinBursty", "None", protein_limit=float("inf"))
        gi = _grad_inference(model, protein_sd, presearch_maxiter=40)
        x0 = np.zeros(5)
        out = np.asarray(gi._presearch_x0(model, protein_sd, [[x0.tolist()]])[0][0])
        assert np.all(out >= gi.phys_lb) and np.all(out <= gi.phys_ub)
        data = protein_sd.hist[0]
        limits = protein_sd.M[:, 0]
        kld0 = model.eval_model_kld(x0, limits, None, data)
        kld1 = model.eval_model_kld(out, limits, None, data)
        assert kld1 < kld0

    def test_threaded_matches_serial(self, protein_sd):
        model = CMEModel("ProteinBursty", "None", protein_limit=float("inf"))
        gi = _grad_inference(model, protein_sd, presearch_maxiter=10)
        x0 = [[[0.0] * 5], [[0.2, -0.1, 0.1, 0.0, 0.3]]]
        # two "genes" sharing the same histogram: patch M/hist access through a tiny shim
        class SD:
            n_genes = 2
            M = np.repeat(protein_sd.M, 2, axis=1)
            hist = [protein_sd.hist[0], protein_sd.hist[0]]
        gi.regressor = [None, None]
        serial = gi._presearch_x0(model, SD, x0, n_gene_cores=1)
        threaded = gi._presearch_x0(model, SD, x0, n_gene_cores=2)
        assert serial == threaded

    def test_fit_uses_presearch_end_to_end(self, protein_sd, monkeypatch):
        """iterate_over_genes calls the pre-search on both the scipy and Rust paths."""
        model = CMEModel("ProteinBursty", "None", protein_limit=float("inf"))
        for use_rust in (False, True):
            gi = _grad_inference(model, protein_sd, num_restarts=1, max_iterations=3,
                                 presearch_maxiter=3, use_rust_lbfgsb=use_rust)
            seen = []
            real = gi._presearch_x0

            def spy(model_, sd_, x0_all, n_gene_cores=1):
                seen.append(len(x0_all))
                return real(model_, sd_, x0_all, n_gene_cores)

            monkeypatch.setattr(gi, "_presearch_x0", spy)
            params, klds, obj, _ = gi.iterate_over_genes(model, protein_sd)
            assert seen == [1], f"use_rust_lbfgsb={use_rust}"
            assert np.isfinite(klds).all() and params.shape == (1, 5)


class TestNullSampRouting:
    """Inference passes a zero sampling vector (not None) for seq_model="None";
    the noise-free Rust fast paths must treat it as no transform."""

    def test_protein_pss_and_grad_route_to_rust_with_zero_samp(self, protein_sd, monkeypatch):
        import monod.cme_toolbox as ct
        model = CMEModel("ProteinBursty", "None", protein_limit=float("inf"))
        data = protein_sd.hist[0]
        limits = [int(v) for v in protein_sd.M[:, 0]]
        x = np.array([0.3, -0.2, 0.1, 0.4, -0.5])
        hits = []
        real_pss = ct._mc.eval_model_pss_protein_bursty
        real_grad = ct._mc.eval_kld_grad_protein_bursty
        monkeypatch.setattr(ct._mc, "eval_model_pss_protein_bursty",
                            lambda *a, **k: (hits.append("pss"), real_pss(*a, **k))[1])
        monkeypatch.setattr(ct._mc, "eval_kld_grad_protein_bursty",
                            lambda *a, **k: (hits.append("grad"), real_grad(*a, **k))[1])
        pss_none = model.eval_model_pss(x, limits, None)
        pss_zero = model.eval_model_pss(x, limits, np.zeros(2))
        k_none, g_none = model.eval_model_kld_and_grad(x, limits, None, data)
        k_zero, g_zero = model.eval_model_kld_and_grad(x, limits, np.zeros(2), data)
        assert hits == ["pss", "pss", "grad", "grad"]
        np.testing.assert_array_equal(pss_none, pss_zero)
        assert k_none == k_zero and np.array_equal(g_none, g_zero)
