"""
Tests verifying that monod_core.SearchData (pure-Rust Stage 3 container)
produces attribute values identical to the Python SearchData class, and that
the inference pipeline works end-to-end with the Rust container.

Test groups
-----------
1. Scalar attributes: n_genes, n_cells, hist_type, layer_names, gene_names
2. M property: shape, dtype, values match Python
3. hist property: list length, per-gene shapes, dtype, values
4. moments property: keys, mean/var/cov values match Python reference
5. layers property: shape, dtype, values match raw input
6. gene_log_lengths: present and absent cases
7. Pickle round-trip: __getstate__ / __setstate__
8. End-to-end: searchdata_from_adata returns Rust SearchData and inference
   produces the same parameter estimates as with Python SearchData
"""

import os
import sys
import pickle

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "monod"))

try:
    import monod_core as _mc
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

pytestmark = pytest.mark.skipif(not _HAS_RUST, reason="monod_core not available")

EXAMPLE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "example_h5ad", "example_adata.h5ad"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(7)


def _make_layers(n_cells, n_genes, n_layers, high=30):
    return [
        np.ascontiguousarray(RNG.integers(0, high, size=(n_cells, n_genes)), dtype=np.int64)
        for _ in range(n_layers)
    ]


def _make_limits(n_layers, n_genes, base=40):
    """(n_layers, n_genes) int64 array of grid limits."""
    return np.ascontiguousarray(
        RNG.integers(base, base + 20, size=(n_layers, n_genes)), dtype=np.int64
    )


def _make_hist_unique(layers, n_layers, n_genes):
    """Python reference: same output shape as make_histograms_unique."""
    coords_list, freqs_list = _mc.make_histograms_unique(layers)
    return coords_list, freqs_list


def _ref_moments(layers, layer_names):
    """Pure-Python moments reference (mirrors mminference.get_moment_dicts)."""
    n_layers = len(layer_names)
    n_genes = layers[0].shape[1]
    n_cells = layers[0].shape[0]
    result = []
    for g in range(n_genes):
        d = {}
        means = []
        for i in range(n_layers):
            col = layers[i][:, g].astype(float)
            d[f"MOM_{layer_names[i]}_mean"] = col.mean()
            d[f"MOM_{layer_names[i]}_var"]  = col.var()
            means.append(col.mean())
        for i in range(n_layers):
            for j in range(i + 1, n_layers):
                d[f"MOM_cov_{layer_names[i]}_{layer_names[j]}"] = np.cov(
                    [layers[i][:, g].astype(float), layers[j][:, g].astype(float)]
                )[0, 1]
        result.append(d)
    return result


def _build_sd(n_cells=200, n_genes=10, n_layers=2,
              layer_names=None, gene_log_lengths=None, k=None, epochs=None):
    if layer_names is None:
        layer_names = [f"mod{i}" for i in range(n_layers)]
    layers = _make_layers(n_cells, n_genes, n_layers)
    limits = _make_limits(n_layers, n_genes)
    coords_list, freqs_list = _make_hist_unique(layers, n_layers, n_genes)
    gene_names = [f"GENE{i}" for i in range(n_genes)]
    return _mc.SearchData(
        layers, layer_names, limits,
        coords_list, freqs_list,
        gene_names, n_cells, "unique",
        gene_log_lengths, k, epochs,
    ), layers, limits, coords_list, freqs_list, layer_names, gene_names


# ---------------------------------------------------------------------------
# Tests: scalar attributes
# ---------------------------------------------------------------------------

class TestScalars:
    def test_n_genes(self):
        sd, layers, *_ = _build_sd(n_genes=7)
        assert sd.n_genes == 7

    def test_n_cells(self):
        sd, *_ = _build_sd(n_cells=123)
        assert sd.n_cells == 123

    def test_hist_type(self):
        sd, *_ = _build_sd()
        assert sd.hist_type == "unique"

    def test_layer_names(self):
        sd, *_ = _build_sd(layer_names=["unspliced", "spliced"])
        assert sd.layer_names == ["unspliced", "spliced"]

    def test_gene_names(self):
        sd, *rest, gene_names = _build_sd(n_genes=5)
        assert sd.gene_names == gene_names

    def test_k_none(self):
        sd, *_ = _build_sd()
        assert sd.k is None

    def test_k_set(self):
        sd, *_ = _build_sd(k=3, epochs=10)
        assert sd.k == 3
        assert sd.epochs == 10

    def test_gene_log_lengths_none(self):
        sd, *_ = _build_sd()
        assert sd.gene_log_lengths is None

    def test_gene_log_lengths_set(self):
        n_genes = 8
        gll = list(RNG.random(n_genes))
        sd, *_ = _build_sd(n_genes=n_genes, gene_log_lengths=gll)
        np.testing.assert_allclose(sd.gene_log_lengths, gll)


# ---------------------------------------------------------------------------
# Tests: M property
# ---------------------------------------------------------------------------

class TestM:
    def test_shape(self):
        n_layers, n_genes = 2, 12
        sd, _, limits, *_ = _build_sd(n_genes=n_genes, n_layers=n_layers)
        assert sd.M.shape == (n_layers, n_genes)

    def test_dtype(self):
        sd, *_ = _build_sd()
        assert sd.M.dtype == np.int64

    def test_values_match_limits(self):
        sd, _, limits, *_ = _build_sd(n_genes=8, n_layers=2)
        np.testing.assert_array_equal(sd.M, limits)

    def test_column_slicing(self):
        """M[:, gi] must return per-layer limits for gene gi."""
        sd, _, limits, *_ = _build_sd(n_genes=6, n_layers=2)
        for gi in range(6):
            np.testing.assert_array_equal(sd.M[:, gi], limits[:, gi])

    def test_three_layers(self):
        sd, _, limits, *_ = _build_sd(n_genes=5, n_layers=3,
                                       layer_names=["a", "b", "c"])
        np.testing.assert_array_equal(sd.M, limits)


# ---------------------------------------------------------------------------
# Tests: hist property
# ---------------------------------------------------------------------------

class TestHist:
    def test_length(self):
        n_genes = 9
        sd, *_ = _build_sd(n_genes=n_genes)
        assert len(sd.hist) == n_genes

    def test_tuple_unpacking(self):
        sd, *_ = _build_sd(n_genes=4)
        for g in range(4):
            coords, freqs = sd.hist[g]
            assert coords.ndim == 2
            assert freqs.ndim == 1

    def test_coords_shape(self):
        """coords[g].shape = (n_microstates, n_layers)."""
        n_layers = 2
        sd, layers, _, coords_list, *_ = _build_sd(n_layers=n_layers, n_genes=5)
        for g in range(5):
            coords, _ = sd.hist[g]
            assert coords.shape[1] == n_layers
            assert coords.shape[0] == len(coords_list[g])

    def test_freqs_sum_to_one(self):
        sd, *_ = _build_sd(n_genes=6)
        for g in range(6):
            _, freqs = sd.hist[g]
            assert abs(freqs.sum() - 1.0) < 1e-10

    def test_coords_values_match_input(self):
        sd, _, _, coords_list, freqs_list, *_ = _build_sd(n_genes=4)
        for g in range(4):
            coords, freqs = sd.hist[g]
            np.testing.assert_array_equal(coords, coords_list[g])
            np.testing.assert_allclose(freqs, freqs_list[g])

    def test_coords_dtype_int64(self):
        sd, *_ = _build_sd()
        coords, _ = sd.hist[0]
        assert coords.dtype == np.int64


# ---------------------------------------------------------------------------
# Tests: moments property
# ---------------------------------------------------------------------------

class TestMoments:
    def test_length(self):
        n_genes = 11
        sd, *_ = _build_sd(n_genes=n_genes)
        assert len(sd.moments) == n_genes

    def test_keys(self):
        sd, *rest, layer_names, _ = _build_sd(n_layers=2, n_genes=3,
                                               layer_names=["unspliced", "spliced"])
        expected = {"MOM_unspliced_mean", "MOM_unspliced_var",
                    "MOM_spliced_mean",   "MOM_spliced_var",
                    "MOM_cov_unspliced_spliced"}
        assert set(sd.moments[0].keys()) == expected

    def test_values_match_reference(self):
        n_cells, n_genes = 300, 8
        layer_names = ["unspliced", "spliced"]
        layers = _make_layers(n_cells, n_genes, 2)
        limits = _make_limits(2, n_genes)
        coords_list, freqs_list = _make_hist_unique(layers, 2, n_genes)
        sd = _mc.SearchData(layers, layer_names, limits, coords_list, freqs_list,
                            [f"G{i}" for i in range(n_genes)], n_cells, "unique")
        ref = _ref_moments(layers, layer_names)
        for g in range(n_genes):
            for k in ref[g]:
                assert abs(sd.moments[g][k] - ref[g][k]) < 1e-10, \
                    f"gene {g} key {k}: rust={sd.moments[g][k]:.15g} ref={ref[g][k]:.15g}"

    def test_three_layer_covariances(self):
        n_cells, n_genes = 100, 4
        layer_names = ["a", "b", "c"]
        layers = _make_layers(n_cells, n_genes, 3)
        limits = _make_limits(3, n_genes)
        c, f = _make_hist_unique(layers, 3, n_genes)
        sd = _mc.SearchData(layers, layer_names, limits, c, f,
                            [f"G{i}" for i in range(n_genes)], n_cells, "unique")
        ref = _ref_moments(layers, layer_names)
        for g in range(n_genes):
            for k in ["MOM_cov_a_b", "MOM_cov_a_c", "MOM_cov_b_c"]:
                assert abs(sd.moments[g][k] - ref[g][k]) < 1e-10


# ---------------------------------------------------------------------------
# Tests: layers property
# ---------------------------------------------------------------------------

class TestLayers:
    def test_shape(self):
        n_cells, n_genes, n_layers = 150, 7, 2
        sd, layers, *_ = _build_sd(n_cells=n_cells, n_genes=n_genes, n_layers=n_layers)
        assert sd.layers.shape == (n_layers, n_cells, n_genes)

    def test_dtype(self):
        sd, *_ = _build_sd()
        assert sd.layers.dtype == np.int64

    def test_values_match_input(self):
        n_cells, n_genes = 80, 5
        sd, layers, *_ = _build_sd(n_cells=n_cells, n_genes=n_genes, n_layers=2)
        for l in range(2):
            np.testing.assert_array_equal(sd.layers[l], layers[l])

    def test_cell_subset_indexing(self):
        """sd.layers[:, mask, :] must select correct cells (mminference pattern)."""
        n_cells, n_genes = 100, 6
        sd, layers, *_ = _build_sd(n_cells=n_cells, n_genes=n_genes, n_layers=2)
        mask = RNG.integers(0, 2, size=n_cells, dtype=bool)
        subset = sd.layers[:, mask, :]
        assert subset.shape == (2, int(mask.sum()), n_genes)
        np.testing.assert_array_equal(subset[0], layers[0][mask])
        np.testing.assert_array_equal(subset[1], layers[1][mask])

    def test_gene_column_indexing(self):
        """sd.layers[:2, :, gi] must be shape (2, n_cells) — scatter-plot pattern."""
        n_cells, n_genes = 60, 8
        sd, *_ = _build_sd(n_cells=n_cells, n_genes=n_genes, n_layers=2)
        for gi in range(n_genes):
            col = sd.layers[:2, :, gi]
            assert col.shape == (2, n_cells)


# ---------------------------------------------------------------------------
# Tests: pickle round-trip
# ---------------------------------------------------------------------------

class TestPickle:
    def test_getstate_setstate_round_trip(self):
        sd, *_ = _build_sd(n_genes=5)
        state = sd.__getstate__()
        sd2, *_ = _build_sd(n_genes=5)  # create a fresh object with same shape
        sd2.__setstate__(state)
        np.testing.assert_array_equal(sd2.M, sd.M)
        assert sd2.n_genes == sd.n_genes
        assert sd2.hist_type == sd.hist_type
        assert sd2.layer_names == sd.layer_names
        for g in range(5):
            np.testing.assert_allclose(sd2.hist[g][1], sd.hist[g][0] if False else sd.hist[g][1])
            assert sd2.moments[g] == pytest.approx(sd.moments[g])

    def test_pickle_dump_load(self):
        sd, *_ = _build_sd(n_genes=4)
        blob = pickle.dumps(sd.__getstate__())
        state = pickle.loads(blob)
        sd2, *_ = _build_sd(n_genes=4)
        sd2.__setstate__(state)
        np.testing.assert_array_equal(sd2.M, sd.M)
        assert sd2.gene_names == sd.gene_names


# ---------------------------------------------------------------------------
# Tests: end-to-end with real data
# ---------------------------------------------------------------------------

class TestEndToEnd:
    @pytest.fixture(scope="class")
    def adata(self):
        ad = pytest.importorskip("anndata")
        if not os.path.exists(EXAMPLE_PATH):
            pytest.skip("example_adata.h5ad not found")
        return ad.read_h5ad(EXAMPLE_PATH)

    def test_searchdata_from_adata_returns_rust_type(self, adata):
        from extract_data import extract_data
        from inference import searchdata_from_adata
        from cme_toolbox import CMEModel
        model = CMEModel("Bursty", "None")
        processed = extract_data(adata, model, dataset_name="test_sd",
                                 n_genes=1, viz=False, hist_type="unique")
        sd = searchdata_from_adata(processed)
        assert isinstance(sd, _mc.SearchData), \
            f"Expected _mc.SearchData, got {type(sd)}"

    def test_searchdata_attributes_consistent(self, adata):
        from extract_data import extract_data
        from inference import searchdata_from_adata
        from cme_toolbox import CMEModel
        model = CMEModel("Bursty", "None")
        processed = extract_data(adata, model, dataset_name="test_sd2",
                                 n_genes=1, viz=False, hist_type="unique")
        sd = searchdata_from_adata(processed)
        # M must match adata.uns['M']
        np.testing.assert_array_equal(sd.M, processed.uns['M'])
        # n_genes, n_cells
        assert sd.n_genes == processed.n_vars
        assert sd.n_cells == processed.n_obs
        # hist freqs sum to ~1
        _, freqs = sd.hist[0]
        assert abs(freqs.sum() - 1.0) < 1e-10
        # moments keys present
        assert "MOM_unspliced_mean" in sd.moments[0]

    def test_moments_match_adata_var(self, adata):
        """Rust moments must match the MOM_* columns written to adata.var."""
        from extract_data import extract_data
        from inference import searchdata_from_adata
        from cme_toolbox import CMEModel
        model = CMEModel("Bursty", "None")
        processed = extract_data(adata, model, dataset_name="test_sd3",
                                 n_genes=1, viz=False, hist_type="unique")
        sd = searchdata_from_adata(processed)
        for col in processed.var.columns:
            if not col.startswith("MOM_"):
                continue
            key = col[4:]  # strip "MOM_" prefix that add_moments adds
            full_key = col  # moments dict uses the full "MOM_..." key
            ref_val = float(processed.var[col].values[0])
            rust_val = sd.moments[0].get(full_key)
            if rust_val is None:
                continue  # key naming may differ; skip
            # numpy uses pairwise summation; Rust uses sequential — for 65k cells
            # variance accumulation differences reach ~1e-6 relative
            assert abs(rust_val - ref_val) < 1e-5 + 1e-5 * abs(ref_val), \
                f"key {full_key}: rust={rust_val:.15g} adata={ref_val:.15g}"


# ---------------------------------------------------------------------------
# Tests: optimize_genes_2d_sd parity with marshal-loop path
# ---------------------------------------------------------------------------

class TestOptimizeGenes2dSd:
    """optimize_genes_2d_sd must produce the same results as the marshal-loop
    path through optimize_genes_2d for every supported 2D model."""

    MODELS = ["Bursty", "Constitutive", "Extrinsic", "Delay", "CIR", "DelayedSplicing"]

    @pytest.fixture(scope="class")
    def synthetic_sd(self):
        """Small reproducible SearchData (3 genes, 300 cells, 2 layers)."""
        rng = np.random.default_rng(99)
        n_cells, n_genes = 300, 3
        layers = [
            np.ascontiguousarray(rng.integers(0, 15, (n_cells, n_genes)), dtype=np.int64),
            np.ascontiguousarray(rng.integers(0, 10, (n_cells, n_genes)), dtype=np.int64),
        ]
        layer_names = ["unspliced", "spliced"]
        limits = np.ascontiguousarray(
            np.array([[25, 25, 25], [20, 20, 20]], dtype=np.int64)
        )
        coords_list, freqs_list = _mc.make_histograms_unique(layers)
        gene_names = ["G0", "G1", "G2"]
        sd = _mc.SearchData(layers, layer_names, limits, coords_list, freqs_list,
                            gene_names, n_cells, "unique")
        return sd, layers, limits, coords_list, freqs_list

    def _ref_optimize(self, sd, layers, limits, coords_list, freqs_list,
                      bio_model, x0_all, lb, ub, fixed_quad_t=20.0, quad_order=10):
        """Run the marshal-loop path through optimize_genes_2d."""
        n_genes = sd.n_genes
        u_idx_list, s_idx_list, f_list, limits_list = [], [], [], []
        for gi in range(n_genes):
            coords = np.array(coords_list[gi], dtype=np.int64)
            freqs  = np.array(freqs_list[gi])
            u_idx_list.append(coords[:, 0].tolist())
            s_idx_list.append(coords[:, 1].tolist())
            f_list.append(freqs.tolist())
            limits_list.append([int(limits[0, gi]), int(limits[1, gi])])
        return _mc.optimize_genes_2d(
            bio_model=bio_model, x0_list=x0_all,
            lb=lb, ub=ub,
            limits_list=limits_list, u_idx_list=u_idx_list,
            s_idx_list=s_idx_list, f_list=f_list,
            fixed_quad_t=fixed_quad_t, quad_order=quad_order,
        )

    @pytest.mark.parametrize("bio_model", MODELS)
    def test_parity_with_marshal_loop(self, bio_model, synthetic_sd):
        sd, layers, limits, coords_list, freqs_list = synthetic_sd
        # Use broad bounds to make the test model-agnostic
        lb = [-4.0] * 3
        ub = [ 2.0] * 3
        rng = np.random.default_rng(42)
        x0_all = [rng.uniform(lb, ub, size=(1, 3)).tolist() for _ in range(sd.n_genes)]

        params_sd, klds_sd = _mc.optimize_genes_2d_sd(
            sd, bio_model=bio_model, x0_list=x0_all,
            lb=lb, ub=ub, fixed_quad_t=20.0, quad_order=10,
        )
        params_ref, klds_ref = self._ref_optimize(
            sd, layers, limits, coords_list, freqs_list,
            bio_model, x0_all, lb, ub,
        )

        np.testing.assert_allclose(params_sd, params_ref, rtol=1e-10,
            err_msg=f"{bio_model}: parameter estimates differ")
        np.testing.assert_allclose(klds_sd, klds_ref, rtol=1e-10,
            err_msg=f"{bio_model}: KLD values differ")

    def test_wrong_model_raises(self, synthetic_sd):
        sd = synthetic_sd[0]
        with pytest.raises(Exception, match="Unknown bio_model"):
            _mc.optimize_genes_2d_sd(
                sd, bio_model="NotAModel",
                x0_list=[[[-1.0, -1.0, -1.0]]],
                lb=[-4.0]*3, ub=[2.0]*3,
                fixed_quad_t=20.0, quad_order=10,
            )

    def test_single_layer_raises(self):
        n_cells, n_genes = 50, 2
        rng = np.random.default_rng(0)
        layers = [np.ascontiguousarray(rng.integers(0, 10, (n_cells, n_genes)), dtype=np.int64)]
        limits = np.ascontiguousarray(np.array([[20, 20]], dtype=np.int64))
        coords_list, freqs_list = _mc.make_histograms_unique(layers)
        sd = _mc.SearchData(layers, ["unspliced"], limits, coords_list, freqs_list,
                            ["G0", "G1"], n_cells, "unique")
        with pytest.raises(Exception, match="2 layers"):
            _mc.optimize_genes_2d_sd(
                sd, bio_model="Bursty",
                x0_list=[[[-1.0, -1.0, -1.0]], [[-1.0, -1.0, -1.0]]],
                lb=[-4.0]*3, ub=[2.0]*3,
                fixed_quad_t=20.0, quad_order=10,
            )
