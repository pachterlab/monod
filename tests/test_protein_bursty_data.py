"""
Integration tests using the example h5ad datasets.

USP_test.h5ad  — ProteinBursty model, fit_unspliced=True  (unspliced + spliced + protein)
SP_test.h5ad   — ProteinBursty model, fit_unspliced=False (spliced + protein only)

Both files contain a single gene (PTPRC) with ~65 988 cells.
Fitted linear-space parameters are stored in adata.var under the column names:
  '$b$', '$\\beta$', '$\\gamma$', '$k_p$', '$\\gamma_p$'

Test groups
-----------
1. h5ad structure  — layers, shape, stored moments, stored params
2. eval_model_pss  — PMF at stored fitted params sums to 1; snapshot
3. eval_model_kld  — KLD(data || fit) is finite and ≥ 0; snapshot
4. get_MoM         — estimates from stored moments; length, bounds, snapshot
5. extract_data    — end-to-end pipeline on example_adata.h5ad;
                     moments match direct calculation; histogram shape correct
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for pipeline tests

import numpy as np
import pytest
import anndata as ad

from conftest import check_snapshot

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "monod"))

from cme_toolbox import CMEModel
from extract_data import extract_data

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "example_h5ad")
USP_PATH = os.path.join(_EXAMPLE_DIR, "USP_test.h5ad")
SP_PATH = os.path.join(_EXAMPLE_DIR, "SP_test.h5ad")
EXAMPLE_PATH = os.path.join(_EXAMPLE_DIR, "example_adata.h5ad")

# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def usp_adata():
    return ad.read_h5ad(USP_PATH)


@pytest.fixture(scope="module")
def sp_adata():
    return ad.read_h5ad(SP_PATH)


@pytest.fixture(scope="module")
def example_adata():
    return ad.read_h5ad(EXAMPLE_PATH)


@pytest.fixture(scope="module")
def model_usp():
    return CMEModel("ProteinBursty", "None", fit_unspliced=True)


@pytest.fixture(scope="module")
def model_sp():
    return CMEModel("ProteinBursty", "None", fit_unspliced=False)


# ---------------------------------------------------------------------------
# Helper: extract fitted linear-space parameters and convert to log10
# ---------------------------------------------------------------------------


def _fitted_log_params(adata):
    """Return log10 of the fitted ProteinBursty params stored in adata.var."""
    b = float(adata.var["$b$"].values[0])
    beta = float(adata.var["$\\beta$"].values[0])
    gamma = float(adata.var["$\\gamma$"].values[0])
    k_p = float(adata.var["$k_p$"].values[0])
    gamma_p = float(adata.var["$\\gamma_p$"].values[0])
    return np.log10(np.array([b, beta, gamma, k_p, gamma_p]))


def _to_dense(arr):
    """Convert sparse matrix to dense; leave dense arrays unchanged."""
    if hasattr(arr, "toarray"):
        return arr.toarray()
    return np.asarray(arr)


def _limits(adata, padding=10):
    """Return grid limits = max per layer + padding (same logic as extract_data)."""
    limits = []
    for layer in ["unspliced", "spliced", "protein"]:
        arr = _to_dense(adata.layers[layer])
        limits.append(int(arr.max()) + padding)
    return limits


def _unique_histogram(adata):
    """Build a (coords, freqs) unique histogram from unspliced/spliced/protein layers."""
    u = _to_dense(adata.layers["unspliced"]).flatten().astype(int)
    s = _to_dense(adata.layers["spliced"]).flatten().astype(int)
    p = _to_dense(adata.layers["protein"]).flatten().astype(int)
    data_matrix = np.column_stack([u, s, p])
    n_cells = data_matrix.shape[0]
    unique, counts = np.unique(data_matrix, axis=0, return_counts=True)
    freqs = counts / n_cells
    return unique.astype(int), freqs


# ===========================================================================
# 1. h5ad structure
# ===========================================================================


class TestH5adStructure:
    def test_usp_has_required_layers(self, usp_adata):
        for layer in ("unspliced", "spliced", "protein"):
            assert layer in usp_adata.layers

    def test_sp_has_required_layers(self, sp_adata):
        for layer in ("unspliced", "spliced", "protein"):
            assert layer in sp_adata.layers

    def test_usp_single_gene(self, usp_adata):
        assert usp_adata.n_vars == 1

    def test_sp_single_gene(self, sp_adata):
        assert sp_adata.n_vars == 1

    def test_usp_cell_count(self, usp_adata):
        assert usp_adata.n_obs > 0
        check_snapshot("usp_n_obs", np.array([usp_adata.n_obs]))

    def test_sp_matches_usp_shape(self, usp_adata, sp_adata):
        assert sp_adata.n_obs == usp_adata.n_obs
        assert sp_adata.n_vars == usp_adata.n_vars

    def test_usp_has_fitted_params(self, usp_adata):
        for col in ("$b$", r"$\beta$", r"$\gamma$", r"$k_p$", r"$\gamma_p$"):
            assert col in usp_adata.var.columns

    def test_sp_has_fitted_params(self, sp_adata):
        for col in ("$b$", r"$\beta$", r"$\gamma$", r"$k_p$", r"$\gamma_p$"):
            assert col in sp_adata.var.columns

    def test_usp_has_moments(self, usp_adata):
        for col in (
            "MOM_unspliced_mean",
            "MOM_unspliced_var",
            "MOM_spliced_mean",
            "MOM_spliced_var",
            "MOM_protein_mean",
            "MOM_protein_var",
            "MOM_cov_unspliced_protein",
        ):
            assert col in usp_adata.var.columns

    def test_usp_stored_moments_snapshot(self, usp_adata):
        moments = np.array([
            float(usp_adata.var["MOM_unspliced_mean"].values[0]),
            float(usp_adata.var["MOM_unspliced_var"].values[0]),
            float(usp_adata.var["MOM_spliced_mean"].values[0]),
            float(usp_adata.var["MOM_spliced_var"].values[0]),
            float(usp_adata.var["MOM_protein_mean"].values[0]),
            float(usp_adata.var["MOM_protein_var"].values[0]),
        ])
        check_snapshot("usp_stored_moments", moments)

    def test_usp_stored_params_snapshot(self, usp_adata):
        check_snapshot("usp_stored_params", _fitted_log_params(usp_adata))

    def test_sp_stored_params_snapshot(self, sp_adata):
        check_snapshot("sp_stored_params", _fitted_log_params(sp_adata))

    def test_usp_sp_params_differ(self, usp_adata, sp_adata):
        """USP and SP fits should produce different parameter estimates."""
        p_usp = _fitted_log_params(usp_adata)
        p_sp = _fitted_log_params(sp_adata)
        assert not np.allclose(p_usp, p_sp)

    def test_usp_nonnegative_counts(self, usp_adata):
        for layer in ("unspliced", "spliced", "protein"):
            assert float(usp_adata.layers[layer].min()) >= 0


# ===========================================================================
# 2. eval_model_pss at fitted parameters
# ===========================================================================


class TestEvalModelPSSRealData:
    def test_usp_pss_sums_to_one(self, usp_adata, model_usp):
        p = _fitted_log_params(usp_adata)
        limits = _limits(usp_adata)
        pss = model_usp.eval_model_pss(p, limits)
        np.testing.assert_allclose(pss.sum(), 1.0, rtol=1e-4)

    def test_usp_pss_nonnegative(self, usp_adata, model_usp):
        p = _fitted_log_params(usp_adata)
        limits = _limits(usp_adata)
        pss = model_usp.eval_model_pss(p, limits)
        assert np.all(pss >= 0)

    def test_usp_pss_snapshot(self, usp_adata, model_usp):
        p = _fitted_log_params(usp_adata)
        limits = _limits(usp_adata)
        pss = model_usp.eval_model_pss(p, limits)
        check_snapshot("pss_usp_fitted_params", pss)

    def test_sp_pss_sums_to_one(self, sp_adata, model_sp):
        """SP model sets mx[0]=1 (no unspliced fit) — PMF should still sum to 1."""
        p = _fitted_log_params(sp_adata)
        limits = _limits(sp_adata)
        pss = model_sp.eval_model_pss(p, limits)
        np.testing.assert_allclose(pss.sum(), 1.0, rtol=1e-4)

    def test_sp_pss_nonnegative(self, sp_adata, model_sp):
        p = _fitted_log_params(sp_adata)
        limits = _limits(sp_adata)
        pss = model_sp.eval_model_pss(p, limits)
        assert np.all(pss >= 0)

    def test_sp_pss_snapshot(self, sp_adata, model_sp):
        p = _fitted_log_params(sp_adata)
        limits = _limits(sp_adata)
        pss = model_sp.eval_model_pss(p, limits)
        check_snapshot("pss_sp_fitted_params", pss)

    def test_usp_pss_is_3d(self, usp_adata, model_usp):
        """With fit_unspliced=True the PSS has the full (unspliced, spliced, protein) shape."""
        p = _fitted_log_params(usp_adata)
        limits = _limits(usp_adata)
        pss = model_usp.eval_model_pss(p, limits)
        assert pss.ndim == 3

    def test_sp_pss_unspliced_dim_is_uniform(self, sp_adata, model_sp):
        """With fit_unspliced=False, mx[0] is set to 1 during RFFT so the PSS
        is constant along the unspliced (first) axis — all slices are equal."""
        p = _fitted_log_params(sp_adata)
        limits = _limits(sp_adata)
        pss = model_sp.eval_model_pss(p, limits)
        assert pss.ndim == 3
        for i in range(1, pss.shape[0]):
            np.testing.assert_allclose(pss[i], pss[0], rtol=1e-5)


# ===========================================================================
# 3. eval_model_kld with real data histogram
# ===========================================================================


class TestEvalModelKLDRealData:
    def test_usp_kld_finite(self, usp_adata, model_usp):
        p = _fitted_log_params(usp_adata)
        limits = _limits(usp_adata)
        coords, freqs = _unique_histogram(usp_adata)
        kld = model_usp.eval_model_kld(p, limits, None, (coords, freqs), "unique")
        assert np.isfinite(kld)

    def test_usp_kld_nonnegative(self, usp_adata, model_usp):
        p = _fitted_log_params(usp_adata)
        limits = _limits(usp_adata)
        coords, freqs = _unique_histogram(usp_adata)
        kld = model_usp.eval_model_kld(p, limits, None, (coords, freqs), "unique")
        assert kld >= 0.0

    def test_usp_kld_snapshot(self, usp_adata, model_usp):
        p = _fitted_log_params(usp_adata)
        limits = _limits(usp_adata)
        coords, freqs = _unique_histogram(usp_adata)
        kld = model_usp.eval_model_kld(p, limits, None, (coords, freqs), "unique")
        check_snapshot("kld_usp_fitted_params", np.array([kld]))

    def test_sp_kld_finite(self, sp_adata, model_sp):
        """SP histogram: data still has 3 columns (USP), matching the 3D PSS."""
        p = _fitted_log_params(sp_adata)
        limits = _limits(sp_adata)
        coords, freqs = _unique_histogram(sp_adata)
        kld = model_sp.eval_model_kld(
            p, limits, None, (coords, freqs), "unique"
        )
        assert np.isfinite(kld)

    def test_sp_kld_snapshot(self, sp_adata, model_sp):
        p = _fitted_log_params(sp_adata)
        limits = _limits(sp_adata)
        coords, freqs = _unique_histogram(sp_adata)
        kld = model_sp.eval_model_kld(
            p, limits, None, (coords, freqs), "unique"
        )
        check_snapshot("kld_sp_fitted_params", np.array([kld]))


# ===========================================================================
# 4. get_MoM with real moments
# ===========================================================================


class TestGetMoMRealData:
    def _moments_from_var(self, adata):
        return {
            "MOM_unspliced_mean": float(adata.var["MOM_unspliced_mean"].values[0]),
            "MOM_unspliced_var": float(adata.var["MOM_unspliced_var"].values[0]),
            "MOM_spliced_mean": float(adata.var["MOM_spliced_mean"].values[0]),
            "MOM_spliced_var": float(adata.var["MOM_spliced_var"].values[0]),
            "MOM_protein_mean": float(adata.var["MOM_protein_mean"].values[0]),
            "MOM_protein_var": float(adata.var["MOM_protein_var"].values[0]),
            "MOM_cov_unspliced_protein": float(
                adata.var["MOM_cov_unspliced_protein"].values[0]
            ),
        }

    def test_usp_mom_length(self, usp_adata, model_usp):
        moments = self._moments_from_var(usp_adata)
        lb = np.array(model_usp.bio_bounds["phys_lb"])
        ub = np.array(model_usp.bio_bounds["phys_ub"])
        samp = np.array([-6.0, -6.0, -6.0])  # dummy; ignored for seq_model="None"
        x0 = model_usp.get_MoM(moments, lb, ub, samp=samp)
        assert len(x0) == model_usp.get_num_params()

    def test_usp_mom_within_bounds(self, usp_adata, model_usp):
        moments = self._moments_from_var(usp_adata)
        lb = np.array(model_usp.bio_bounds["phys_lb"])
        ub = np.array(model_usp.bio_bounds["phys_ub"])
        samp = np.array([-6.0, -6.0, -6.0])
        x0 = model_usp.get_MoM(moments, lb, ub, samp=samp)
        assert np.all(x0 >= lb)
        assert np.all(x0 <= ub)

    def test_usp_mom_finite(self, usp_adata, model_usp):
        moments = self._moments_from_var(usp_adata)
        lb = np.array(model_usp.bio_bounds["phys_lb"])
        ub = np.array(model_usp.bio_bounds["phys_ub"])
        samp = np.array([-6.0, -6.0, -6.0])
        x0 = model_usp.get_MoM(moments, lb, ub, samp=samp)
        assert np.all(np.isfinite(x0))

    def test_usp_mom_snapshot(self, usp_adata, model_usp):
        moments = self._moments_from_var(usp_adata)
        lb = np.array(model_usp.bio_bounds["phys_lb"])
        ub = np.array(model_usp.bio_bounds["phys_ub"])
        samp = np.array([-6.0, -6.0, -6.0])
        x0 = model_usp.get_MoM(moments, lb, ub, samp=samp)
        check_snapshot("mom_usp_real_moments", x0)


# ===========================================================================
# 5. extract_data pipeline
# ===========================================================================


class TestExtractDataPipeline:
    """Run extract_data on example_adata.h5ad and verify the output structure.

    PTPRC has protein mean ≈ 0.20, which falls below the default ProteinBursty
    min_mean threshold of 1.0 for the protein layer.  The gene is therefore
    passed explicitly via genes_to_fit so it is included regardless of the filter.
    """

    @pytest.fixture(scope="class")
    def processed_usp(self, example_adata, model_usp):
        return extract_data(
            example_adata,
            model_usp,
            dataset_name="test_usp",
            n_genes=1,
            viz=False,
            genes_to_fit=["PTPRC"],
            hist_type="unique",
        )

    @pytest.fixture(scope="class")
    def processed_sp(self, example_adata, model_sp):
        return extract_data(
            example_adata,
            model_sp,
            dataset_name="test_sp",
            n_genes=1,
            viz=False,
            genes_to_fit=["PTPRC"],
            hist_type="unique",
        )

    def test_usp_output_has_one_gene(self, processed_usp):
        assert processed_usp.n_vars == 1

    def test_sp_output_has_one_gene(self, processed_sp):
        assert processed_sp.n_vars == 1

    def test_usp_output_has_uns_M(self, processed_usp):
        assert "M" in processed_usp.uns

    def test_usp_M_shape(self, processed_usp):
        M = processed_usp.uns["M"]
        # M is (n_layers, n_genes), so (3, 1) for ProteinBursty
        assert M.shape == (3, 1)

    def test_sp_M_shape(self, processed_sp):
        M = processed_sp.uns["M"]
        assert M.shape == (3, 1)

    def test_usp_M_snapshot(self, processed_usp):
        check_snapshot("extract_data_usp_M", processed_usp.uns["M"])

    def test_sp_M_snapshot(self, processed_sp):
        check_snapshot("extract_data_sp_M", processed_sp.uns["M"])

    def test_usp_has_moments_in_var(self, processed_usp):
        for col in (
            "MOM_unspliced_mean",
            "MOM_unspliced_var",
            "MOM_spliced_mean",
            "MOM_spliced_var",
            "MOM_protein_mean",
            "MOM_protein_var",
        ):
            assert col in processed_usp.var.columns

    def test_usp_moments_match_direct_calculation(self, processed_usp):
        """Moments stored in var should equal direct numpy calculations on the layers."""
        u = processed_usp.layers["unspliced"].flatten().astype(float)
        s = processed_usp.layers["spliced"].flatten().astype(float)
        p = processed_usp.layers["protein"].flatten().astype(float)

        np.testing.assert_allclose(
            float(processed_usp.var["MOM_unspliced_mean"].values[0]),
            u.mean(), rtol=1e-5,
        )
        np.testing.assert_allclose(
            float(processed_usp.var["MOM_spliced_mean"].values[0]),
            s.mean(), rtol=1e-5,
        )
        np.testing.assert_allclose(
            float(processed_usp.var["MOM_protein_mean"].values[0]),
            p.mean(), rtol=1e-5,
        )

    def test_usp_moments_snapshot(self, processed_usp):
        moments = np.array([
            float(processed_usp.var["MOM_unspliced_mean"].values[0]),
            float(processed_usp.var["MOM_unspliced_var"].values[0]),
            float(processed_usp.var["MOM_spliced_mean"].values[0]),
            float(processed_usp.var["MOM_spliced_var"].values[0]),
            float(processed_usp.var["MOM_protein_mean"].values[0]),
            float(processed_usp.var["MOM_protein_var"].values[0]),
        ])
        check_snapshot("extract_data_usp_moments", moments)

    def test_usp_histogram_type(self, processed_usp):
        """Histogram should be a list of (coords, freqs) tuples for unique hist_type."""
        hist = processed_usp.uns["hist"]
        assert isinstance(hist, list)
        assert len(hist) == processed_usp.n_vars
        coords, freqs = hist[0]
        assert coords.ndim == 2
        assert coords.shape[1] == 3  # unspliced, spliced, protein
        assert freqs.ndim == 1

    def test_usp_histogram_freqs_sum_to_one(self, processed_usp):
        _, freqs = processed_usp.uns["hist"][0]
        np.testing.assert_allclose(freqs.sum(), 1.0, rtol=1e-5)

    def test_sp_histogram_freqs_sum_to_one(self, processed_sp):
        _, freqs = processed_sp.uns["hist"][0]
        np.testing.assert_allclose(freqs.sum(), 1.0, rtol=1e-5)

    def test_usp_moments_match_usp_test_stored(self, processed_usp, usp_adata):
        """extract_data on example_adata should produce the same moments as USP_test."""
        np.testing.assert_allclose(
            float(processed_usp.var["MOM_unspliced_mean"].values[0]),
            float(usp_adata.var["MOM_unspliced_mean"].values[0]),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            float(processed_usp.var["MOM_spliced_mean"].values[0]),
            float(usp_adata.var["MOM_spliced_mean"].values[0]),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            float(processed_usp.var["MOM_protein_mean"].values[0]),
            float(usp_adata.var["MOM_protein_mean"].values[0]),
            rtol=1e-5,
        )
