"""
Regression tests for all 2-modality CME bio_models using example_adata.h5ad.

All tests use the PTPRC gene (~65 988 cells, unspliced max=71, spliced max=34).
Every model produces M = [81, 44] (max + 10 padding) and passes the default
expression filter, so no genes_to_fit override is needed here.

Models tested (all with seq_model="None"):
  Bursty, Constitutive, Extrinsic, Delay, CIR, DelayedSplicing

Test groups
-----------
1. extract_data pipeline — structure, M bounds, moments, histogram
2. eval_model_pss at MOM estimates — sums to 1, non-negative, snapshot
3. eval_model_kld at MOM estimates — finite, non-negative, snapshot

Note on get_MoM: a known bug causes `samp = 10**samp` to be evaluated even for
seq_model="None".  A dummy samp of np.zeros(2) is passed so the call succeeds;
for seq_model="None" the converted value (ones) is never used in the output.
"""

import os
import sys
import unittest.mock

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
import anndata as ad

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "monod"))

import cme_toolbox as _cme_toolbox_module
from conftest import check_snapshot
from cme_toolbox import CMEModel
from extract_data import extract_data, _uns_unpack

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXAMPLE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "example_h5ad", "example_adata.h5ad"
)

MODELS_2D = [
    ("Bursty", "None"),
    ("Constitutive", "None"),
    ("Extrinsic", "None"),
    ("Delay", "None"),
    ("CIR", "None"),
    ("DelayedSplicing", "None"),
]
MODEL_IDS = [f"{m}_{s}" for m, s in MODELS_2D]

# Only Bursty and CIR have a Rust fast-path for seq_model="Poisson".
RUST_POISSON_MODELS = [
    ("Bursty", "Poisson"),
    ("CIR", "Poisson"),
]
RUST_POISSON_MODEL_IDS = [f"{m}_{s}" for m, s in RUST_POISSON_MODELS]

_DUMMY_SAMP = np.zeros(2)       # safe dummy for seq_model="None" get_MoM calls
_POISSON_SAMP = np.array([-6.0, -6.0])  # representative Poisson samp for parity tests

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def raw_example_adata():
    return ad.read_h5ad(EXAMPLE_PATH)


@pytest.fixture(scope="module", params=RUST_POISSON_MODELS, ids=RUST_POISSON_MODEL_IDS)
def processed_poisson(request, raw_example_adata):
    """Run extract_data for Bursty/CIR+Poisson — the two models with a Rust Poisson path."""
    bio_model, seq_model = request.param
    model = CMEModel(bio_model, seq_model)
    adata = extract_data(
        raw_example_adata,
        model,
        dataset_name=f"test_{bio_model}_{seq_model}",
        n_genes=1,
        viz=False,
        hist_type="unique",
    )
    return {
        "bio_model": bio_model,
        "seq_model": seq_model,
        "model": model,
        "adata": adata,
    }


@pytest.fixture(scope="module", params=MODELS_2D, ids=MODEL_IDS)
def processed(request, raw_example_adata):
    """Run extract_data once per model; return a dict with model + adata."""
    bio_model, seq_model = request.param
    model = CMEModel(bio_model, seq_model)
    adata = extract_data(
        raw_example_adata,
        model,
        dataset_name=f"test_{bio_model}_{seq_model}",
        n_genes=1,
        viz=False,
        hist_type="unique",
    )
    return {
        "bio_model": bio_model,
        "seq_model": seq_model,
        "model": model,
        "adata": adata,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _moments_dict(adata):
    """Collect all MOM_* columns from adata.var into a flat dict."""
    return {
        col: float(adata.var[col].values[0])
        for col in adata.var.columns
        if col.startswith("MOM_")
    }


def _mom_params(model, adata):
    """Return log10 MOM parameter estimates for the model."""
    moments = _moments_dict(adata)
    lb = np.array(model.bio_bounds["phys_lb"])
    ub = np.array(model.bio_bounds["phys_ub"])
    return model.get_MoM(moments, lb, ub, samp=_DUMMY_SAMP)


def _limits(adata):
    """Return grid limits from adata.uns['M'] for gene 0."""
    return adata.uns["M"][:, 0].tolist()


def _histogram(adata):
    """Return the (coords, freqs) unique histogram for gene 0."""
    return _uns_unpack(adata.uns["hist"])[0]


def _snap(bio_model, tag):
    """Consistent snapshot name scoped to the bio_model."""
    return f"{bio_model.lower()}_{tag}"


# ===========================================================================
# 1. extract_data pipeline
# ===========================================================================


class TestExtractDataAllModels:
    def test_one_gene_selected(self, processed):
        assert processed["adata"].n_vars == 1

    def test_n_cells_positive(self, processed):
        assert processed["adata"].n_obs > 0

    def test_has_uns_M(self, processed):
        assert "M" in processed["adata"].uns

    def test_M_shape_2x1(self, processed):
        """2-modality models produce M of shape (2, n_genes)."""
        M = processed["adata"].uns["M"]
        assert M.shape == (2, 1)

    def test_M_snapshot(self, processed):
        check_snapshot(
            _snap(processed["bio_model"], "extract_data_M"),
            processed["adata"].uns["M"],
        )

    def test_has_moment_columns(self, processed):
        adata = processed["adata"]
        for col in (
            "MOM_unspliced_mean",
            "MOM_unspliced_var",
            "MOM_spliced_mean",
            "MOM_spliced_var",
        ):
            assert col in adata.var.columns

    def test_moments_positive_means(self, processed):
        adata = processed["adata"]
        assert float(adata.var["MOM_unspliced_mean"].values[0]) > 0
        assert float(adata.var["MOM_spliced_mean"].values[0]) > 0

    def test_moments_nonnegative_vars(self, processed):
        adata = processed["adata"]
        assert float(adata.var["MOM_unspliced_var"].values[0]) >= 0
        assert float(adata.var["MOM_spliced_var"].values[0]) >= 0

    def test_moments_match_direct_calculation(self, processed):
        """Moments in adata.var must equal numpy means/variances of the layers."""
        adata = processed["adata"]
        u = adata.layers["unspliced"].astype(float)
        s = adata.layers["spliced"].astype(float)
        if hasattr(u, "toarray"):
            u = u.toarray()
        if hasattr(s, "toarray"):
            s = s.toarray()

        np.testing.assert_allclose(
            float(adata.var["MOM_unspliced_mean"].values[0]),
            u.mean(), rtol=1e-5,
        )
        np.testing.assert_allclose(
            float(adata.var["MOM_spliced_mean"].values[0]),
            s.mean(), rtol=1e-5,
        )

    def test_moments_snapshot(self, processed):
        adata = processed["adata"]
        moments = np.array([
            float(adata.var["MOM_unspliced_mean"].values[0]),
            float(adata.var["MOM_unspliced_var"].values[0]),
            float(adata.var["MOM_spliced_mean"].values[0]),
            float(adata.var["MOM_spliced_var"].values[0]),
        ])
        check_snapshot(_snap(processed["bio_model"], "extract_data_moments"), moments)

    def test_histogram_is_list(self, processed):
        hist = _uns_unpack(processed["adata"].uns["hist"])
        assert isinstance(hist, list)
        assert len(hist) == 1

    def test_histogram_coords_shape(self, processed):
        coords, freqs = _histogram(processed["adata"])
        assert coords.ndim == 2
        assert coords.shape[1] == 2  # unspliced, spliced

    def test_histogram_freqs_sum_to_one(self, processed):
        _, freqs = _histogram(processed["adata"])
        np.testing.assert_allclose(freqs.sum(), 1.0, rtol=1e-5)

    def test_histogram_coords_nonnegative(self, processed):
        coords, _ = _histogram(processed["adata"])
        assert np.all(coords >= 0)

    def test_histogram_freqs_positive(self, processed):
        _, freqs = _histogram(processed["adata"])
        assert np.all(freqs > 0)


# ===========================================================================
# 2. eval_model_pss at MOM estimates
# ===========================================================================


class TestPSSAtMOMAllModels:
    def test_mom_params_length(self, processed):
        model = processed["model"]
        adata = processed["adata"]
        p = _mom_params(model, adata)
        assert len(p) == model.get_num_params()

    def test_mom_params_within_bounds(self, processed):
        model = processed["model"]
        adata = processed["adata"]
        lb = np.array(model.bio_bounds["phys_lb"])
        ub = np.array(model.bio_bounds["phys_ub"])
        p = _mom_params(model, adata)
        assert np.all(p >= lb)
        assert np.all(p <= ub)

    def test_mom_params_finite(self, processed):
        model = processed["model"]
        adata = processed["adata"]
        p = _mom_params(model, adata)
        assert np.all(np.isfinite(p))

    def test_mom_params_snapshot(self, processed):
        model = processed["model"]
        adata = processed["adata"]
        p = _mom_params(model, adata)
        check_snapshot(_snap(processed["bio_model"], "mom_params"), p)

    def test_pss_sums_to_one(self, processed):
        model = processed["model"]
        adata = processed["adata"]
        p = _mom_params(model, adata)
        limits = _limits(adata)
        pss = model.eval_model_pss(p, limits)
        np.testing.assert_allclose(pss.sum(), 1.0, rtol=1e-4)

    def test_pss_nonnegative(self, processed):
        model = processed["model"]
        adata = processed["adata"]
        p = _mom_params(model, adata)
        limits = _limits(adata)
        pss = model.eval_model_pss(p, limits)
        assert np.all(pss >= 0)

    def test_pss_shape_matches_limits(self, processed):
        model = processed["model"]
        adata = processed["adata"]
        p = _mom_params(model, adata)
        limits = _limits(adata)
        pss = model.eval_model_pss(p, limits)
        assert list(pss.shape) == limits

    def test_pss_snapshot(self, processed):
        model = processed["model"]
        adata = processed["adata"]
        p = _mom_params(model, adata)
        limits = _limits(adata)
        pss = model.eval_model_pss(p, limits)
        check_snapshot(_snap(processed["bio_model"], "pss_mom"), pss)


# ===========================================================================
# 3. eval_model_kld at MOM estimates
# ===========================================================================


class TestKLDAtMOMAllModels:
    def test_kld_finite(self, processed):
        model = processed["model"]
        adata = processed["adata"]
        p = _mom_params(model, adata)
        limits = _limits(adata)
        coords, freqs = _histogram(adata)
        kld = model.eval_model_kld(p, limits, None, (coords, freqs), "unique")
        assert np.isfinite(kld)

    def test_kld_nonnegative(self, processed):
        model = processed["model"]
        adata = processed["adata"]
        p = _mom_params(model, adata)
        limits = _limits(adata)
        coords, freqs = _histogram(adata)
        kld = model.eval_model_kld(p, limits, None, (coords, freqs), "unique")
        assert kld >= 0.0

    def test_kld_snapshot(self, processed):
        model = processed["model"]
        adata = processed["adata"]
        p = _mom_params(model, adata)
        limits = _limits(adata)
        coords, freqs = _histogram(adata)
        kld = model.eval_model_kld(p, limits, None, (coords, freqs), "unique")
        check_snapshot(_snap(processed["bio_model"], "kld_mom"), np.array([kld]))


# ===========================================================================
# 4. Rust / Python parity on real data limits
# ===========================================================================


class TestRustPythonParityData:
    """Verify Rust and Python eval_model_pss agree on real data grid sizes (~[81, 44]).

    These tests complement the unit-level parity tests in test_cme_toolbox.py
    by running at the grid dimensions that arise from actual scRNA-seq data,
    where the allocations and FFT plan caching paths differ from the [20, 20] case.
    """

    def test_pss_parity_at_mom_params(self, processed):
        model = processed["model"]
        adata = processed["adata"]
        p = _mom_params(model, adata)
        limits = _limits(adata)

        rust = model.eval_model_pss(p, limits)
        with unittest.mock.patch.object(_cme_toolbox_module, "_HAS_RUST", False):
            py = model.eval_model_pss(p, limits)

        np.testing.assert_allclose(
            rust, py, rtol=1e-5, atol=1e-10,
            err_msg=f"{processed['bio_model']}/None Rust vs Python mismatch on real data limits {limits}",
        )


class TestRustPythonParityDataPoisson:
    """Verify Rust and Python agree for seq_model='Poisson' at real data grid sizes.

    Only Bursty and CIR have a Rust fast-path for Poisson technical noise.
    Uses a fixed representative samp rather than an optimised value so the
    test doesn't depend on inference results.
    """

    def test_pss_parity_poisson(self, processed_poisson):
        model = processed_poisson["model"]
        adata = processed_poisson["adata"]
        p = _mom_params(model, adata)
        limits = _limits(adata)

        rust = model.eval_model_pss(p, limits, samp=_POISSON_SAMP)
        with unittest.mock.patch.object(_cme_toolbox_module, "_HAS_RUST", False):
            py = model.eval_model_pss(p, limits, samp=_POISSON_SAMP)

        np.testing.assert_allclose(
            rust, py, rtol=1e-5, atol=1e-10,
            err_msg=(
                f"{processed_poisson['bio_model']}/Poisson Rust vs Python mismatch "
                f"on real data limits {limits}"
            ),
        )
