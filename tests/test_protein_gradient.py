"""ProteinBursty finite-difference gradients with a frozen integration schedule.

The ProteinBursty PGF is integrated with error-controlled steps whose sequence
is chosen from the parameters and the grid, so it changes discretely as the
parameters move.  Forward finite differences across such a change are noise.
Both the Rust gradient and the Python fallback therefore replay the base
point's step sequence at every stencil point; these tests check that the
pinning happens and that both gradients agree with a pinned central-difference
reference.
"""
import numpy as np
import pytest

import monod.cme_toolbox as ct
from monod._core import _mc
from monod.cme_toolbox import CMEModel

_HAS_RUST = _mc is not None

LIMITS = [6, 5, 5]
X = np.array([0.3, -0.2, 0.1, 0.4, -0.5])


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(3)
    counts = np.column_stack([rng.integers(0, hi, 150) for hi in (4, 3, 3)])
    coords, freqs = np.unique(counts, axis=0, return_counts=True)
    return coords.astype(np.int64), freqs / freqs.sum()


@pytest.fixture(scope="module")
def model():
    return CMEModel("ProteinBursty", "None", fit_unspliced=True, protein_limit=float("inf"))


def _python_reference(model, data, monkeypatch_rust_off, h=1e-4):
    """Central differences through the Python integrator with a pinned schedule."""
    coords, freqs = data

    def kld(p):
        return model.eval_model_kld(np.asarray(p, float), LIMITS, None, (coords, freqs))

    model._fixed_protein_schedule = None
    kld(X)
    sched = model._last_protein_schedule
    model._fixed_protein_schedule = sched
    try:
        ref = np.array([(kld(X + h * e) - kld(X - h * e)) / (2 * h) for e in np.eye(5)])
    finally:
        model._fixed_protein_schedule = None
    return ref, sched


def _rel_err(g, ref):
    return np.max(np.abs(g - ref) / np.maximum(np.abs(ref), 1e-8))


class TestPythonPinnedGradient:
    def test_pinned_schedule_is_used_and_restored(self, model, data, monkeypatch):
        monkeypatch.setattr(ct, "_HAS_RUST", False)
        coords, freqs = data
        model._fixed_protein_schedule = None
        model.eval_model_kld(X, LIMITS, None, (coords, freqs))
        sched = np.array(model._last_protein_schedule)
        assert sched.ndim == 1 and len(sched) > 10 and np.all(sched > 0)
        # replaying a truncated schedule must use exactly that sequence
        model._fixed_protein_schedule = sched[:-7]
        model.eval_model_kld(X, LIMITS, None, (coords, freqs))
        model._fixed_protein_schedule = None
        np.testing.assert_array_equal(model._last_protein_schedule, sched[:-7])
        # the gradient path pins and then clears the schedule
        model.eval_model_kld_and_grad(X, LIMITS, None, (coords, freqs))
        assert model._fixed_protein_schedule is None
    def test_python_fd_gradient_matches_reference(self, model, data, monkeypatch):
        monkeypatch.setattr(ct, "_HAS_RUST", False)
        ref, _ = _python_reference(model, data, monkeypatch)
        coords, freqs = data
        _, grad = model.eval_model_kld_and_grad(X, LIMITS, None, (coords, freqs), eps=1e-6)
        # forward FD with a float32 integrator: a few 1e-3 relative is expected
        assert _rel_err(grad, ref) < 2e-2, (grad, ref)


@pytest.mark.skipif(not _HAS_RUST, reason="monod_core not available")
class TestRustPinnedGradient:
    def test_rust_gradient_matches_reference(self, model, data, monkeypatch):
        coords, freqs = data
        k_rs, g_rs = _mc.eval_kld_grad_protein_bursty(
            X.tolist(), LIMITS, coords.tolist(), freqs.tolist(),
            True, float("inf"), 0.1, 10.0, 1e-6, 1e-15,
        )
        monkeypatch.setattr(ct, "_HAS_RUST", False)
        ref, _ = _python_reference(model, data, monkeypatch)
        k_py = model.eval_model_kld(X, LIMITS, None, (coords, freqs))
        assert abs(k_rs - k_py) < 1e-8
        assert _rel_err(np.array(g_rs), ref) < 2e-3, (g_rs, ref)

    def test_python_api_routes_to_rust(self, model, data):
        coords, freqs = data
        k1, g1 = model.eval_model_kld_and_grad(X, LIMITS, None, (coords, freqs), eps=1e-6)
        k2, g2 = _mc.eval_kld_grad_protein_bursty(
            X.tolist(), LIMITS, coords.tolist(), freqs.tolist(),
            True, float("inf"), 0.1, 10.0, 1e-6, 1e-15,
        )
        assert abs(k1 - k2) < 1e-12
        np.testing.assert_allclose(g1, np.array(g2), rtol=1e-9, atol=1e-12)

    def test_rust_gradient_validates_inputs(self, data):
        coords, freqs = data
        with pytest.raises(ValueError, match="outside grid"):
            _mc.eval_kld_grad_protein_bursty(
                X.tolist(), [2, 2, 2], coords.tolist(), freqs.tolist(),
                True, float("inf"), 0.1, 10.0,
            )
        with pytest.raises(ValueError, match="5 log10"):
            _mc.eval_kld_grad_protein_bursty(
                [0.0] * 4, LIMITS, coords.tolist(), freqs.tolist(),
                True, float("inf"), 0.1, 10.0,
            )
