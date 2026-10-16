"""Parity test: Python vs Rust for amb_model=Equal and amb_model=Unequal."""
import numpy as np
import pytest
import monod.cme_toolbox as ct

orig_has_rust = ct._HAS_RUST

@pytest.fixture(autouse=True)
def restore_rust():
    yield
    ct._HAS_RUST = orig_has_rust

@pytest.mark.parametrize("bio_model,p", [
    ("Constitutive",    np.array([-0.301, -0.523, -0.523])),
    ("Bursty",          np.array([-0.301, -0.523, -0.699, -0.523])),
    ("Delay",           np.array([-0.301, -0.523,  0.0,   -0.523])),
    ("DelayedSplicing", np.array([-0.301, -0.523,  0.0,   -0.523])),
    ("CIR",             np.array([-0.301, -0.523, -0.699, -0.523])),
    ("Extrinsic",       np.array([-0.301, -0.523, -0.699, -0.523])),
])
def test_equal_parity(bio_model, p):
    m = ct.CMEModel(bio_model, "None", "Equal")
    limits = [30, 30, 30]
    ct._HAS_RUST = False
    pss_py = m.eval_model_pss(p, limits)
    ct._HAS_RUST = orig_has_rust
    pss_rust = m.eval_model_pss(p, limits)
    assert pss_py.shape == pss_rust.shape
    np.testing.assert_allclose(pss_rust, pss_py, rtol=1e-5, atol=1e-10,
                               err_msg=f"{bio_model} Equal: Python vs Rust mismatch")

@pytest.mark.parametrize("bio_model,p", [
    ("Constitutive",    np.array([-0.301, -0.523, -0.523, -0.699])),
    ("Bursty",          np.array([-0.301, -0.523, -0.699, -0.523, -0.699])),
])
def test_unequal_parity(bio_model, p):
    m = ct.CMEModel(bio_model, "None", "Unequal")
    limits = [30, 30, 30]
    ct._HAS_RUST = False
    pss_py = m.eval_model_pss(p, limits)
    ct._HAS_RUST = orig_has_rust
    pss_rust = m.eval_model_pss(p, limits)
    assert pss_py.shape == pss_rust.shape
    np.testing.assert_allclose(pss_rust, pss_py, rtol=1e-5, atol=1e-10,
                               err_msg=f"{bio_model} Unequal: Python vs Rust mismatch")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
