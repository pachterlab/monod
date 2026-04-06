"""
Tests verifying that monod_core.compute_moments matches the Python
get_moment_dicts reference implementation (mminference.py) exactly.

Covers:
  - 1-layer (single modality): mean and population variance (ddof=0)
  - 2-layer (unspliced/spliced): mean, variance, and sample covariance (ddof=1)
  - 3-layer: all pairwise covariances
  - Single cell (n_cells=1): variance=0, covariance=NaN handled the same way
  - Single gene
  - Many genes (stress/parallelism)
  - Constant column: variance should be exactly 0
  - Round-trip against example_adata.h5ad to guard against real-data drift
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "monod"))

try:
    import monod_core as _mc
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

pytestmark = pytest.mark.skipif(not _HAS_RUST, reason="monod_core not available")

# ---------------------------------------------------------------------------
# Reference implementation (mirrors mminference.get_moment_dicts exactly)
# ---------------------------------------------------------------------------

def _ref_moments(layers, layer_names):
    """Pure-Python reference matching mminference.get_moment_dicts."""
    n_layers = len(layer_names)
    n_genes = layers[0].shape[1]
    result = []
    for g in range(n_genes):
        d = {}
        means = []
        for i in range(n_layers):
            col = layers[i][:, g].astype(float)
            mean = col.mean()
            var = col.var()           # ddof=0
            d[f"MOM_{layer_names[i]}_mean"] = mean
            d[f"MOM_{layer_names[i]}_var"] = var
            means.append(mean)
        for i in range(n_layers):
            for j in range(i + 1, n_layers):
                cov = np.cov(
                    [layers[i][:, g].astype(float), layers[j][:, g].astype(float)]
                )[0, 1]               # ddof=1
                d[f"MOM_cov_{layer_names[i]}_{layer_names[j]}"] = cov
        result.append(d)
    return result


def _to_int64_c(arr):
    return np.ascontiguousarray(arr, dtype=np.int64)


def _compare(rust_list, ref_list, tol=1e-10):
    assert len(rust_list) == len(ref_list), "gene count mismatch"
    for g, (rust_d, ref_d) in enumerate(zip(rust_list, ref_list)):
        assert set(rust_d.keys()) == set(ref_d.keys()), (
            f"gene {g}: key mismatch rust={set(rust_d.keys())} ref={set(ref_d.keys())}"
        )
        for k in ref_d:
            ref_v = ref_d[k]
            rust_v = rust_d[k]
            if np.isnan(ref_v):
                assert np.isnan(rust_v), f"gene {g} key {k}: expected NaN, got {rust_v}"
            else:
                assert abs(rust_v - ref_v) <= tol + tol * abs(ref_v), (
                    f"gene {g} key {k}: rust={rust_v:.15g} ref={ref_v:.15g} "
                    f"diff={abs(rust_v - ref_v):.3e}"
                )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _make_layers(n_cells, n_genes, n_layers, high=50):
    return [
        _to_int64_c(RNG.integers(0, high, size=(n_cells, n_genes)))
        for _ in range(n_layers)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComputeMoments:

    def test_one_layer(self):
        layers = _make_layers(200, 10, 1)
        names = ["unspliced"]
        rust = _mc.compute_moments(layers, names)
        ref = _ref_moments(layers, names)
        _compare(rust, ref)

    def test_two_layers(self):
        layers = _make_layers(500, 20, 2)
        names = ["unspliced", "spliced"]
        rust = _mc.compute_moments(layers, names)
        ref = _ref_moments(layers, names)
        _compare(rust, ref)

    def test_three_layers(self):
        layers = _make_layers(300, 15, 3)
        names = ["unspliced", "spliced", "protein"]
        rust = _mc.compute_moments(layers, names)
        ref = _ref_moments(layers, names)
        _compare(rust, ref)

    def test_single_gene(self):
        layers = _make_layers(400, 1, 2)
        names = ["unspliced", "spliced"]
        rust = _mc.compute_moments(layers, names)
        ref = _ref_moments(layers, names)
        _compare(rust, ref)

    def test_many_genes_parallel(self):
        """Exercise rayon parallelism across many genes."""
        layers = _make_layers(1000, 200, 2)
        names = ["unspliced", "spliced"]
        rust = _mc.compute_moments(layers, names)
        ref = _ref_moments(layers, names)
        _compare(rust, ref)

    def test_constant_column_variance_zero(self):
        """A gene with constant expression across all cells has variance=0."""
        n_cells, n_genes = 100, 3
        layer = _to_int64_c(np.ones((n_cells, n_genes), dtype=np.int64) * 5)
        layer[0, :] = layer[0, :]   # no mutation needed
        layers = [layer]
        names = ["spliced"]
        rust = _mc.compute_moments(layers, names)
        ref = _ref_moments(layers, names)
        _compare(rust, ref)
        for d in rust:
            assert d["MOM_spliced_var"] == pytest.approx(0.0, abs=1e-12)

    def test_single_cell_variance_zero_cov_nan(self):
        """With n_cells=1: population variance=0; np.cov returns NaN (0/0)."""
        layers = [_to_int64_c(np.array([[3, 7]])),
                  _to_int64_c(np.array([[1, 2]]))]
        names = ["unspliced", "spliced"]
        rust = _mc.compute_moments(layers, names)
        ref = _ref_moments(layers, names)
        _compare(rust, ref)

    def test_empty_layers_returns_empty(self):
        rust = _mc.compute_moments([], [])
        assert rust == []

    def test_key_names_match_exactly(self):
        """Key strings must use the supplied layer names verbatim."""
        layers = _make_layers(50, 2, 2)
        names = ["mod1", "mod2"]
        rust = _mc.compute_moments(layers, names)
        expected_keys = {
            "MOM_mod1_mean", "MOM_mod1_var",
            "MOM_mod2_mean", "MOM_mod2_var",
            "MOM_cov_mod1_mod2",
        }
        assert set(rust[0].keys()) == expected_keys

    def test_mismatched_lengths_raises(self):
        layers = _make_layers(50, 5, 2)
        with pytest.raises(Exception):
            _mc.compute_moments(layers, ["only_one_name"])

    def test_mean_formula(self):
        """Spot-check mean against numpy directly."""
        n_cells, n_genes = 100, 5
        layer = _to_int64_c(RNG.integers(0, 30, size=(n_cells, n_genes)))
        rust = _mc.compute_moments([layer], ["x"])
        for g in range(n_genes):
            expected = layer[:, g].astype(float).mean()
            assert abs(rust[g]["MOM_x_mean"] - expected) < 1e-12

    def test_variance_is_population_not_sample(self):
        """Variance must use ddof=0, not ddof=1."""
        n_cells, n_genes = 50, 3
        layer = _to_int64_c(RNG.integers(0, 20, size=(n_cells, n_genes)))
        rust = _mc.compute_moments([layer], ["x"])
        for g in range(n_genes):
            col = layer[:, g].astype(float)
            assert abs(rust[g]["MOM_x_var"] - col.var(ddof=0)) < 1e-12
            # Must differ from ddof=1 when n>1
            assert abs(rust[g]["MOM_x_var"] - col.var(ddof=1)) > 1e-14

    def test_covariance_is_sample_not_population(self):
        """Covariance must use ddof=1 (np.cov default)."""
        n_cells, n_genes = 50, 3
        l0 = _to_int64_c(RNG.integers(0, 20, size=(n_cells, n_genes)))
        l1 = _to_int64_c(RNG.integers(0, 20, size=(n_cells, n_genes)))
        rust = _mc.compute_moments([l0, l1], ["a", "b"])
        for g in range(n_genes):
            expected = np.cov([l0[:, g].astype(float), l1[:, g].astype(float)])[0, 1]
            assert abs(rust[g]["MOM_cov_a_b"] - expected) < 1e-10


class TestComputeMomentsRealData:
    """Round-trip test against example_adata.h5ad."""

    EXAMPLE_PATH = os.path.join(
        os.path.dirname(__file__), "..", "example_h5ad", "example_adata.h5ad"
    )

    @pytest.fixture(scope="class")
    def adata(self):
        ad = pytest.importorskip("anndata")
        if not os.path.exists(self.EXAMPLE_PATH):
            pytest.skip("example_adata.h5ad not found")
        return ad.read_h5ad(self.EXAMPLE_PATH)

    def test_real_data_two_layers(self, adata):
        from scipy.sparse import issparse
        layer_names = ["unspliced", "spliced"]
        layers = []
        for ln in layer_names:
            arr = adata.layers[ln]
            if issparse(arr):
                arr = arr.toarray()
            layers.append(_to_int64_c(arr))
        rust = _mc.compute_moments(layers, layer_names)
        ref = _ref_moments(layers, layer_names)
        _compare(rust, ref, tol=1e-9)
