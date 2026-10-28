"""ProteinBursty steady state: exponential integrator with error-controlled steps.

Checks (1) the Python and Rust integrators agree, (2) both agree with a stiff
reference solution of the characteristic equations, (3) the step count stays
small when the rates are spread over several decades, and (4) replaying a
recorded step sequence reproduces the result exactly.
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

import monod.cme_toolbox as ct
from monod._core import _mc
from monod.cme_toolbox import CMEModel

LIMITS = [12, 10, 12]

# log10 (b, beta, gamma, k_p, gamma_p)
REGIMES = {
    "balanced": [0.7, 0.0, 0.0, 0.5, 0.0],
    "ptprc_like": [1.11, 0.96, 1.32, 1.36, 1.85],
    "spread_1e3": [1.0, 2.0, -1.0, 1.5, -1.5],
    "fast_protein": [0.5, 0.5, 0.5, 3.0, 3.0],
}


def _mesh(limits):
    L0, L1, L2 = limits
    g0 = np.exp(-2j * np.pi * np.arange(L0) / L0) - 1
    g1 = np.exp(-2j * np.pi * np.arange(L1) / L1) - 1
    g2 = np.exp(-2j * np.pi * np.arange(L2 // 2 + 1) / L2) - 1
    G = np.meshgrid(g0, g1, g2, indexing="ij")
    return np.vstack([x.ravel() for x in G])


def _reference_phi(g, p, rtol=1e-10):
    """Stiff reference: BDF on (v, w, phi) per grid point."""
    b, beta, gamma, k_p, gp = p
    T = np.log(1e13) / min(beta, gamma, gp) + max(0.0, np.log(2 * k_p / gamma)) / gp
    out = np.empty(g.shape[1], dtype=complex)
    for i in range(g.shape[1]):
        g2 = g[2, i]

        def rhs(t, y):
            v, w, _ = y
            u = v - 1
            return [beta * (w - v), (-gamma + k_p * g2 * np.exp(-gp * t)) * w + gamma, b * u / (1 - b * u)]

        y0 = np.array([g[0, i] + 1, g[1, i] + 1, 0], dtype=complex)
        out[i] = solve_ivp(rhs, (0, T), y0, method="BDF", rtol=rtol, atol=1e-13).y[2, -1]
    return out


@pytest.fixture(scope="module")
def model():
    return CMEModel("ProteinBursty", "None", protein_limit=float("inf"))


class TestAccuracy:
    @pytest.mark.parametrize("regime", list(REGIMES))
    def test_python_matches_stiff_reference(self, model, regime):
        p = 10.0 ** np.asarray(REGIMES[regime])
        rng = np.random.default_rng(0)
        g = _mesh(LIMITS)
        idx = rng.choice(g.shape[1], size=6, replace=False)
        idx[0] = g.shape[1] // 2            # include a point far from g = 0
        g = g[:, idx]
        model._fixed_schedule = None
        phi = model.protein_pgf(g, p)
        ref = _reference_phi(g, p)
        # compare on the generating-function scale, where |exp(phi)| <= 1
        assert np.max(np.abs(np.exp(phi) - np.exp(ref))) < 5e-5

    @pytest.mark.parametrize("regime", list(REGIMES))
    def test_step_count_is_small_in_every_regime(self, model, regime):
        p = 10.0 ** np.asarray(REGIMES[regime])
        model._fixed_schedule = None
        model.protein_pgf(_mesh(LIMITS), p)
        n = len(model._last_schedule)
        assert 20 < n < 600, f"{regime}: {n} steps"


class TestReplay:
    def test_replaying_the_schedule_reproduces_phi(self, model):
        p = 10.0 ** np.asarray(REGIMES["balanced"])
        g = _mesh(LIMITS)
        model._fixed_schedule = None
        phi = model.protein_pgf(g, p)
        sched = model._last_schedule.copy()
        model._fixed_schedule = sched
        try:
            phi2 = model.protein_pgf(g, p)
        finally:
            model._fixed_schedule = None
        np.testing.assert_array_equal(phi, phi2)
        np.testing.assert_array_equal(model._last_schedule, sched)


@pytest.mark.skipif(_mc is None, reason="monod_core not available")
class TestRustParity:
    @pytest.mark.parametrize("regime", list(REGIMES))
    @pytest.mark.parametrize("fit_unspliced", [True, False])
    def test_pss_matches_python(self, regime, fit_unspliced, monkeypatch):
        m = CMEModel("ProteinBursty", "None", fit_unspliced=fit_unspliced, protein_limit=float("inf"))
        p = np.asarray(REGIMES[regime])
        rust = m.eval_model_pss(p, LIMITS)
        monkeypatch.setattr(ct, "_HAS_RUST", False)
        py = m.eval_model_pss(p, LIMITS)
        assert rust.shape == py.shape
        np.testing.assert_allclose(rust, py, rtol=1e-9, atol=1e-13)

    def test_rust_gradient_matches_python_gradient(self, monkeypatch):
        m = CMEModel("ProteinBursty", "None", protein_limit=float("inf"))
        rng = np.random.default_rng(3)
        counts = np.column_stack([rng.integers(0, hi, 200) for hi in (5, 4, 4)])
        coords, freqs = np.unique(counts, axis=0, return_counts=True)
        data = (coords.astype(np.int64), freqs / freqs.sum())
        x = np.array([0.3, -0.2, 0.1, 0.4, -0.5])
        k_r, g_r = m.eval_model_kld_and_grad(x, [7, 6, 6], None, data)
        monkeypatch.setattr(ct, "_HAS_RUST", False)
        k_p, g_p = m.eval_model_kld_and_grad(x, [7, 6, 6], None, data)
        assert abs(k_r - k_p) < 1e-10
        np.testing.assert_allclose(g_r, g_p, rtol=1e-6, atol=1e-9)
