"""GPU evaluation with Poisson and Bernoulli sequencing noise.

The dispatch test runs everywhere (it only checks that the GPU helper receives
the sampling parameters when a GPU is present); the numerical parity test
needs a CUDA device with float64 support and is skipped otherwise.
"""
import numpy as np
import pytest

import monod.cme_toolbox as ct
from monod.cme_toolbox import CMEModel

MODELS = {
    "Constitutive": [0.3, -0.2],
    "Extrinsic": [0.5, 0.2, -0.3],
    "Delay": [0.6, 0.1, -0.4],
    "DelayedSplicing": [0.6, -0.3, 0.2],
    "Bursty": [0.7, 0.1, -0.2],
    "CIR": [0.5, 0.0, -0.3],
}
LIMITS = [24, 18]
SAMP = np.array([-0.3, -0.6])


@pytest.mark.skipif(not ct._HAS_GPU, reason="no CUDA device with float64 support")
@pytest.mark.parametrize("seq_model", ["Poisson", "Bernoulli"])
@pytest.mark.parametrize("bio_model", sorted(MODELS))
def test_gpu_matches_python_with_noise(bio_model, seq_model, monkeypatch):
    model = CMEModel(bio_model, seq_model)
    p = np.array(MODELS[bio_model])
    monkeypatch.setattr(ct, "_HAS_GPU", False)
    monkeypatch.setattr(ct, "_HAS_RUST", False)
    ref = np.asarray(model.eval_model_pss(p, LIMITS, SAMP))
    monkeypatch.setattr(ct, "_HAS_GPU", True)
    gpu = np.asarray(model.eval_model_pss(p, LIMITS, SAMP))
    np.testing.assert_allclose(gpu, ref, rtol=1e-6, atol=1e-12 * ref.max())


def test_dispatch_passes_noise_to_gpu(monkeypatch):
    """With a (fake) GPU present, Poisson/Bernoulli calls must reach the GPU helper
    with samp and seq_model, and seq_model='None' calls must not pass them."""
    seen = []

    def fake_gpu(bio_model, p_log, limits, T, order, device, samp=None, seq_model="None"):
        seen.append((bio_model, seq_model, None if samp is None else list(samp)))
        return np.full((limits[0], limits[1]), 1.0 / (limits[0] * limits[1]))

    monkeypatch.setattr(ct, "_HAS_GPU", True)
    monkeypatch.setattr(ct, "_GPU_DEVICE", "fake")
    monkeypatch.setattr(ct, "_eval_model_pss_gpu", fake_gpu)
    CMEModel("Bursty", "Poisson").eval_model_pss(np.array(MODELS["Bursty"]), LIMITS, SAMP)
    CMEModel("Bursty", "Bernoulli").eval_model_pss(np.array(MODELS["Bursty"]), LIMITS, SAMP)
    CMEModel("Bursty", "None").eval_model_pss(np.array(MODELS["Bursty"]), LIMITS)
    assert seen == [
        ("Bursty", "Poisson", SAMP.tolist()),
        ("Bursty", "Bernoulli", SAMP.tolist()),
        ("Bursty", "None", None),
    ]


def test_gpu_helper_rejects_missing_samp():
    if not ct._HAS_GPU:
        pytest.skip("needs a GPU to construct the mesh")
    with pytest.raises(ValueError, match="requires samp"):
        ct._eval_model_pss_gpu("Bursty", MODELS["Bursty"], LIMITS, 10.0, 60, ct._GPU_DEVICE, seq_model="Poisson")
