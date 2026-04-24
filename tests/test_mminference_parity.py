"""
Parity tests for Rust implementations in monod_core that mirror mminference.py logic:

1. partition_searchdata_2d  — must produce the same n_cells, M, histograms,
   and moments as the Python _part_search_data fallback.
2. initialize_q_kmeans (kmeans feature only) — structural correctness:
   shape, value range, row sums after weight normalization.
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
    _mc = None
    _HAS_RUST = False

_HAS_KMEANS = _HAS_RUST and hasattr(_mc, "initialize_q_kmeans")

pytestmark = pytest.mark.skipif(not _HAS_RUST, reason="monod_core not available")

# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

N_CELLS  = 80
N_GENES  = 6
N_LAYERS = 2
K        = 3
PADDING  = 10
LAYER_NAMES = ["unspliced", "spliced"]
GENE_NAMES  = [f"gene_{i}" for i in range(N_GENES)]
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helpers: pure-Python reference for histogram and moments
# ---------------------------------------------------------------------------

def _ref_histogram(layers_sub):
    """
    Replicate make_histogram (hist_type='unique') for a single cell subset.

    Parameters
    ----------
    layers_sub : (n_layers, n_cells_sub, n_genes) int64 array

    Returns
    -------
    list of (coords, freqs) per gene, where
      coords : (n_unique, n_layers) int64
      freqs  : (n_unique,) float64   (counts / n_cells_sub)
    """
    nl, nc, ng = layers_sub.shape
    hist = []
    for g in range(ng):
        cols = np.vstack([layers_sub[l, :, g] for l in range(nl)]).T  # (nc, nl)
        unique, counts = np.unique(cols, axis=0, return_counts=True)
        hist.append((unique.astype(np.int64), counts / nc))
    return hist


def _ref_moments(layers_sub, layer_names):
    """
    Replicate get_moment_dicts for a single cell subset.

    Parameters
    ----------
    layers_sub  : (n_layers, n_cells_sub, n_genes) int64 array
    layer_names : list of str

    Returns
    -------
    list of dicts, one per gene
    """
    nl, nc, ng = layers_sub.shape
    gene_moments = []
    for g in range(ng):
        d = {}
        for i, name in enumerate(layer_names):
            col = layers_sub[i, :, g].astype(float)
            d[f"MOM_{name}_mean"] = col.mean()
            d[f"MOM_{name}_var"]  = col.var()          # population: /nc
        for i in range(nl):
            for j in range(i + 1, nl):
                ci = layers_sub[i, :, g].astype(float)
                cj = layers_sub[j, :, g].astype(float)
                cov = np.cov([ci, cj])[0, 1]           # sample: /(nc-1)
                key = f"MOM_cov_{layer_names[i]}_{layer_names[j]}"
                d[key] = cov
        gene_moments.append(d)
    return gene_moments


# ---------------------------------------------------------------------------
# Fixture: Rust SearchData built from synthetic data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_rust_sd():
    """Builds a Rust SearchData from random int64 count data."""
    layers_2d = [
        np.ascontiguousarray(
            RNG.integers(0, 30, size=(N_CELLS, N_GENES)), dtype=np.int64
        )
        for _ in range(N_LAYERS)
    ]
    # Compute unique histograms via Rust (same code path as partition uses).
    coords_list, freqs_list = _mc.make_state_dist(layers_2d)

    # Limits: generous upper bound (max + 40) so we don't truncate counts.
    limits = np.ascontiguousarray(
        np.stack([layers_2d[l].max(axis=0) for l in range(N_LAYERS)], axis=0) + 40,
        dtype=np.int64,
    )  # (n_layers, n_genes)

    rust_sd = _mc.searchdata_from_arrays(
        layers_2d,
        LAYER_NAMES,
        limits,
        coords_list,
        freqs_list,
        GENE_NAMES,
        N_CELLS,
        "unique",
        None,   # gene_log_lengths
        None,   # gene_log_lengths_spliced
        None,   # k
        None,   # epochs
    )
    # Also keep raw numpy layers for reference computation.
    raw = np.stack(layers_2d, axis=0)  # (n_layers, n_cells, n_genes)
    return rust_sd, raw


@pytest.fixture(scope="module")
def balanced_assigns():
    """Cluster assignments ensuring every cluster has cells."""
    assigns = np.array([i % K for i in range(N_CELLS)], dtype=int)
    RNG.shuffle(assigns)
    return assigns


# ---------------------------------------------------------------------------
# partition_searchdata_2d tests
# ---------------------------------------------------------------------------

class TestPartitionSearchdata2D:
    """partition_searchdata_2d must agree with the Python _part_search_data logic."""

    @pytest.fixture(autouse=True)
    def _setup(self, synthetic_rust_sd, balanced_assigns):
        self.rust_sd, self.raw = synthetic_rust_sd
        self.assigns = balanced_assigns
        self.partitions = _mc.partition_searchdata_2d(
            self.rust_sd, self.assigns.tolist(), K, PADDING
        )

    def _cluster_ref(self, k):
        """Return (n_cells_k, layers_k, M_k, hist_k, moments_k) for cluster k."""
        mask = self.assigns == k
        layers_k = self.raw[:, mask, :]                     # (nl, n_cells_k, ng)
        nc_k = layers_k.shape[1]
        # M: max per (layer, gene) + padding  →  (n_layers, n_genes)
        M_k = (layers_k.max(axis=1) + PADDING).astype(int)  # (nl, ng)
        hist_k = _ref_histogram(layers_k)
        mom_k  = _ref_moments(layers_k, LAYER_NAMES)
        return nc_k, layers_k, M_k, hist_k, mom_k

    # ── n_cells ───────────────────────────────────────────────────────────────

    def test_n_cells(self):
        for k in range(K):
            expected = int((self.assigns == k).sum())
            assert self.partitions[k] is not None
            assert self.partitions[k].n_cells == expected, (
                f"cluster {k}: Rust n_cells={self.partitions[k].n_cells}, expected={expected}"
            )

    # ── layers content ────────────────────────────────────────────────────────

    def test_layers_content(self):
        for k in range(K):
            mask = self.assigns == k
            expected = self.raw[:, mask, :]         # (nl, n_cells_k, ng)
            got = np.array(self.partitions[k].layers)  # (nl, n_cells_k, ng)
            np.testing.assert_array_equal(
                got, expected,
                err_msg=f"cluster {k}: layers mismatch"
            )

    # ── M values ─────────────────────────────────────────────────────────────

    def test_M_values(self):
        for k in range(K):
            nc_k, layers_k, M_k, _, _ = self._cluster_ref(k)
            got_M = np.array(self.partitions[k].M)   # (n_layers, n_genes)
            np.testing.assert_array_equal(
                got_M, M_k,
                err_msg=f"cluster {k}: M mismatch"
            )

    # ── Histograms ────────────────────────────────────────────────────────────

    def test_hist_parity(self):
        for k in range(K):
            nc_k, layers_k, _, ref_hist, _ = self._cluster_ref(k)
            rust_hist = self.partitions[k].hist
            assert len(rust_hist) == N_GENES, f"cluster {k}: hist length mismatch"
            for g in range(N_GENES):
                ref_coords, ref_freqs = ref_hist[g]
                rust_coords = np.array(rust_hist[g][0])  # (n_unique, n_layers)
                rust_freqs  = np.array(rust_hist[g][1])  # (n_unique,)

                # Sort both by coords rows to make comparison order-independent.
                ref_order  = np.lexsort(ref_coords.T[::-1])
                rust_order = np.lexsort(rust_coords.T[::-1])

                np.testing.assert_array_equal(
                    ref_coords[ref_order], rust_coords[rust_order],
                    err_msg=f"cluster {k}, gene {g}: histogram coords mismatch"
                )
                np.testing.assert_allclose(
                    ref_freqs[ref_order], rust_freqs[rust_order],
                    rtol=1e-10,
                    err_msg=f"cluster {k}, gene {g}: histogram freqs mismatch"
                )

    # ── Moments ───────────────────────────────────────────────────────────────

    def test_moments_parity(self):
        for k in range(K):
            nc_k, layers_k, _, _, ref_mom = self._cluster_ref(k)
            rust_mom = self.partitions[k].moments
            assert len(rust_mom) == N_GENES, f"cluster {k}: moments list length mismatch"
            for g in range(N_GENES):
                ref_d  = ref_mom[g]
                rust_d = rust_mom[g]
                assert set(rust_d.keys()) == set(ref_d.keys()), (
                    f"cluster {k}, gene {g}: moment keys mismatch\n"
                    f"  Rust: {sorted(rust_d.keys())}\n"
                    f"  Ref:  {sorted(ref_d.keys())}"
                )
                for key in ref_d:
                    np.testing.assert_allclose(
                        rust_d[key], ref_d[key], rtol=1e-10,
                        err_msg=f"cluster {k}, gene {g}, key '{key}': moment mismatch"
                    )

    # ── Empty cluster returns None ─────────────────────────────────────────────

    def test_empty_cluster_is_none(self):
        """Assign all cells to clusters 0 and 1; cluster 2 must be None."""
        assigns_no2 = np.where(self.assigns == 2, 0, self.assigns)
        partitions2 = _mc.partition_searchdata_2d(
            self.rust_sd, assigns_no2.tolist(), K, PADDING
        )
        assert len(partitions2) == K
        assert partitions2[2] is None, "expected None for empty cluster 2"
        assert partitions2[0] is not None
        assert partitions2[1] is not None


# ---------------------------------------------------------------------------
# initialize_q_kmeans tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_KMEANS, reason="monod_core not compiled with --features kmeans")
class TestInitializeQKmeans:
    """initialize_q_kmeans: structural correctness of returned Q and labels."""

    @pytest.fixture(autouse=True)
    def _setup(self, synthetic_rust_sd):
        self.rust_sd, self.raw = synthetic_rust_sd
        # (n_layers, n_cells, n_genes) int64
        layers_3d = np.ascontiguousarray(self.raw, dtype=np.int64)
        self.Q, self.labels = _mc.initialize_q_kmeans(layers_3d, K, seed=0)
        self.Q      = np.array(self.Q)
        self.labels = np.array(self.labels)
        self.k = K

    def test_q_shape(self):
        assert self.Q.shape == (N_CELLS, K), f"Q shape: {self.Q.shape}"

    def test_labels_shape(self):
        assert self.labels.shape == (N_CELLS,), f"labels shape: {self.labels.shape}"

    def test_labels_values(self):
        assert set(self.labels).issubset(set(range(K))), (
            f"unexpected label values: {set(self.labels)}"
        )

    def test_q_range(self):
        assert self.Q.min() >= 0.0, "Q has negative values"
        assert self.Q.max() <= 1.0 + 1e-12, f"Q has values > 1: max={self.Q.max()}"

    def test_q_row_sums(self):
        """Returned Q is already row-normalized: rows should sum to 1."""
        row_sums = self.Q.sum(axis=-1)
        np.testing.assert_allclose(
            row_sums, 1.0, atol=1e-12,
            err_msg="Q rows should sum to 1 (returned pre-normalized)"
        )

    def test_biased_cell_has_high_posterior(self):
        """Each cell's assigned cluster should tend to have the maximum Q value.

        Q[c, label]=0.9 before normalization; for k=3 the argmax will equal
        label when 0.9 > max(k-1 uniform values) — probability ~0.81.
        We use 70% as a conservative lower bound.
        """
        best_cluster = self.Q.argmax(axis=1)
        agreement = np.mean(best_cluster == self.labels)
        assert agreement >= 0.70, (
            f"only {agreement:.1%} of cells have argmax(Q)==label (expected ≥70%)"
        )

    def test_reproducible_with_same_seed(self):
        layers_3d = np.ascontiguousarray(self.raw, dtype=np.int64)
        Q2, labs2 = _mc.initialize_q_kmeans(layers_3d, K, seed=0)
        np.testing.assert_array_equal(self.labels, np.array(labs2))
        np.testing.assert_allclose(self.Q, np.array(Q2), rtol=0)
