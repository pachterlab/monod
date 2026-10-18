"""Batched GPU evaluation of the two-species steady state over many genes."""
import numpy as np
import pytest

import monod.cme_toolbox as ct
from monod.cme_toolbox import CMEModel

MODELS = ["Constitutive", "Bursty", "CIR", "Extrinsic", "Delay", "DelayedSplicing"]
P = {
    "Constitutive": [[0.2, -0.3], [-0.1, 0.4], [0.5, 0.5]],
    "Bursty": [[0.7, 0.1, -0.2], [0.3, -0.4, 0.2], [0.5, 0.3, 0.3]],       # third gene: beta == gamma branch
    "CIR": [[0.7, 0.1, -0.2], [0.3, -0.4, 0.2], [0.5, 0.3, 0.3]],
    "Extrinsic": [[0.3, 0.1, -0.2], [0.6, -0.3, 0.2], [0.1, 0.2, 0.2]],
    "Delay": [[0.7, 0.1, 0.3], [0.3, -0.4, 0.5], [0.5, 0.3, 0.1]],
    "DelayedSplicing": [[0.7, 0.3, -0.2], [0.3, 0.5, 0.2], [0.5, 0.1, 0.3]],
}
LIMITS = [[14, 10], [9, 12], [14, 10]]            # two genes share a grid, one differs
SAMP = [np.array([-0.3, 0.1]), np.array([0.2, -0.2]), np.array([0.0, 0.0])]


@pytest.mark.skipif(not ct._HAS_GPU, reason="no CUDA device with float64 support")
@pytest.mark.parametrize("seq_model", ["None", "Poisson", "Bernoulli"])
@pytest.mark.parametrize("bio_model", MODELS)
def test_gpu_batch_matches_per_gene_python(bio_model, seq_model, monkeypatch):
    model = CMEModel(bio_model, seq_model)
    params = [np.array(p) for p in P[bio_model]]
    samp = None if seq_model == "None" else SAMP
    batch = model.eval_model_pss_batch(params, LIMITS, samp)
    monkeypatch.setattr(ct, "_HAS_GPU", False)
    monkeypatch.setattr(ct, "_HAS_RUST", False)
    for i in range(3):
        ref = model.eval_model_pss(params[i], LIMITS[i], None if samp is None else samp[i])
        assert batch[i].shape == ref.shape
        np.testing.assert_allclose(batch[i], ref, rtol=1e-6, atol=1e-14)


@pytest.mark.skipif(not ct._HAS_GPU, reason="no CUDA device with float64 support")
def test_gpu_batch_chunking_is_exact(monkeypatch):
    model = CMEModel("Bursty", "None")
    params = [np.array(p) for p in P["Bursty"]]
    full = model.eval_model_pss_batch(params, LIMITS)
    monkeypatch.setattr(ct, "_GPU_BATCH_BYTES", 2048)      # forces many grid chunks
    chunked = model.eval_model_pss_batch(params, LIMITS)
    for a, b in zip(full, chunked):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-16)


def test_dispatch_prefers_gpu_batch(monkeypatch):
    seen = []

    def fake(bio_model, params_list, limits_list, fixed_quad_T, quad_order, device, samp_list=None, seq_model="None"):
        seen.append((bio_model, seq_model, len(params_list), samp_list))
        return [np.zeros((int(l[0]), int(l[1]))) for l in limits_list]

    monkeypatch.setattr(ct, "_eval_model_pss_gpu_batch", fake)
    monkeypatch.setattr(ct, "_HAS_GPU", True)
    monkeypatch.setattr(ct, "_GPU_DEVICE", "fake")
    params = [np.array(p) for p in P["Bursty"]]
    CMEModel("Bursty", "None").eval_model_pss_batch(params, LIMITS)
    CMEModel("Bursty", "Poisson").eval_model_pss_batch(params, LIMITS, SAMP)
    assert [s[:3] for s in seen] == [("Bursty", "None", 3), ("Bursty", "Poisson", 3)]
    assert seen[0][3] is None and seen[1][3] is SAMP
