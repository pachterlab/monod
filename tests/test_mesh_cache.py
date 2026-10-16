"""Mesh caches stay bounded and never change results (audit fix 6, 2026-10-15)."""
import unittest.mock

import numpy as np
import pytest

import monod.cme_toolbox as ct
from monod.cme_toolbox import CMEModel

from monod._core import _mc  # None unless the compiled extension is really installed
_HAS_RUST = _mc is not None

needs_rust = pytest.mark.skipif(_mc is None, reason="monod_core not installed")
P = [0.3, 0.0, -0.1]


@needs_rust
class TestRustMeshCache:
    def test_bounded_and_consistent(self):
        stats0 = _mc.mesh_cache_stats()
        cap0 = stats0["cap_entries"][0]
        try:
            _mc.mesh_cache_clear()
            _mc.mesh_cache_set_cap(20_000)          # ~12 Poisson meshes at 40x40
            ref = {}
            for i in range(120):                    # 120 distinct samp values
                samp = [-6.0 + i * 1e-3, -6.0]
                ref[i] = _mc.eval_model_pss_2d("Bursty", P, [40, 40], 10.0, 60, samp_log=samp)
                st = _mc.mesh_cache_stats()
                assert st["poisson"][1] <= 20_000, st
                assert st["mesh_2d"][1] <= 20_000, st
            # Results are identical whether served from cache, after a clear, or rebuilt.
            _mc.mesh_cache_clear()
            for i in (0, 57, 119):
                samp = [-6.0 + i * 1e-3, -6.0]
                again = _mc.eval_model_pss_2d("Bursty", P, [40, 40], 10.0, 60, samp_log=samp)
                np.testing.assert_array_equal(again, ref[i])
        finally:
            _mc.mesh_cache_set_cap(cap0)
            _mc.mesh_cache_clear()

    def test_bernoulli_cache_bounded(self):
        cap0 = _mc.mesh_cache_stats()["cap_entries"][0]
        try:
            _mc.mesh_cache_clear()
            _mc.mesh_cache_set_cap(5_000)
            for i in range(50):
                _mc.eval_model_pss_2d("Bursty", P, [30, 30], 10.0, 60,
                                      samp_log=[-1.0 - i * 1e-3, -0.5], seq_model="Bernoulli")
            assert _mc.mesh_cache_stats()["bernoulli"][1] <= 5_000
        finally:
            _mc.mesh_cache_set_cap(cap0)
            _mc.mesh_cache_clear()


class TestPythonMeshCache:
    def test_bounded_and_consistent(self, monkeypatch):
        m = CMEModel("Bursty", "None")
        p = np.array(P)
        ct.clear_mesh_cache()
        monkeypatch.setattr(ct, "_MESH_CACHE_CAP_ENTRIES", 6_000)   # a 30x30 mesh is 2*30*16 = 960 entries
        ref = {}
        with unittest.mock.patch.object(ct, "_HAS_RUST", False), unittest.mock.patch.object(ct, "_HAS_GPU", False):
            for L in range(20, 60):                                   # 40 distinct grids
                ref[L] = m.eval_model_pss(p, [L, L])
                assert ct._MESH_CACHE_ENTRIES[0] <= 6_000
                assert sum(v.size for v in ct._MESH_CACHE.values()) == ct._MESH_CACHE_ENTRIES[0]
            ct.clear_mesh_cache()
            for L in (20, 41, 59):
                np.testing.assert_array_equal(m.eval_model_pss(p, [L, L]), ref[L])
