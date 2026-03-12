"""
End-to-end inference regression tests for gaba_example.h5ad.

Runs the full Bursty+Poisson pipeline on 5 GABAergic neuron genes and
validates fitted parameters against the gaba_results.h5ad reference.

Target genes: Eif5b, Xrcc5, Klhl12, Rgs7, Thsd7b

Reference run metadata (from gaba_results.h5ad):
  samp_optimum     = (-6.8, -1.5)
  samp_optimum_ind = 17
  param columns    = ['$b$', '$\\beta$', '$\\gamma$'] (log10)

Checks
------
1. extract_data   — correct genes, log_lengths injected, M shape
2. Inference      — grid dimensions, samp_optimum index recoverable
3. Quality        — KLD at fitted params ≤ KLD at reference params (at grid pt 17)
4. Snapshot       — fitted params at ref samp_optimum_ind=17 match stored snapshot
"""

import os
import sys
import unittest.mock
import warnings

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
import anndata as ad

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "monod"))

import cme_toolbox as _cme_toolbox_module
from conftest import check_snapshot
from cme_toolbox import CMEModel
from extract_data import extract_data
from inference import (
    InferenceParameters,
    searchdata_from_adata,
    get_hist_type,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(__file__)
GABA_EXAMPLE_PATH = os.path.join(_HERE, "..", "example_h5ad", "gaba_example.h5ad")
GABA_RESULTS_PATH = os.path.join(_HERE, "..", "example_h5ad", "gaba_results.h5ad")

GENES = ["Eif5b", "Xrcc5", "Klhl12", "Rgs7", "Thsd7b"]
REF_SAMP_IND = 17
REF_SAMP_OPTIMUM = (-6.8, -1.5)
REF_PARAM_COLS = ["$b$", "$\\beta$", "$\\gamma$"]

# Gradient settings — enough to converge near the reference without being slow.
GRADIENT_PARAMS = {
    "max_iterations": 25,
    "init_pattern": "moments",
    "num_restarts": 2,
}


# ---------------------------------------------------------------------------
# Module-scoped fixtures (run once for all tests in this file)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ref_adata():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ad.read_h5ad(GABA_RESULTS_PATH)


@pytest.fixture(scope="module")
def gaba_fit(ref_adata):
    """Run the full Bursty+Poisson inference pipeline once for the module."""
    model = CMEModel("Bursty", "Poisson")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata = extract_data(
            GABA_EXAMPLE_PATH,
            model,
            dataset_name="gaba_test",
            modality_name_dict={"unspliced": "unspliced", "spliced": "spliced"},
            n_genes=5,
            genes_to_fit=GENES,
            hist_type="unique",
            viz=False,
        )

    # Inject gene lengths from the reference file (no transcriptome .csv available).
    ref_log_lengths = ref_adata.var.loc[GENES, "log_lengths"]
    adata.var["log_lengths"] = ref_log_lengths.reindex(adata.var.index).values

    sd = searchdata_from_adata(adata)

    ip = InferenceParameters(
        "gaba_test",
        model,
        use_lengths=True,
        gradient_params=GRADIENT_PARAMS,
        save=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sr = ip.fit_all_grid_points(sd, num_cores=1, save=False)

    gene_idx = {g: list(adata.var.index).index(g) for g in GENES}

    return {
        "sr": sr,
        "sd": sd,
        "model": model,
        "adata": adata,
        "ip": ip,
        "gene_idx": gene_idx,
    }


# ---------------------------------------------------------------------------
# 1. extract_data
# ---------------------------------------------------------------------------


class TestGabaExtract:
    def test_correct_genes_extracted(self, gaba_fit):
        assert set(gaba_fit["adata"].var.index) == set(GENES)

    def test_n_genes(self, gaba_fit):
        assert gaba_fit["adata"].n_vars == len(GENES)

    def test_log_lengths_present(self, gaba_fit):
        assert "log_lengths" in gaba_fit["adata"].var.columns

    def test_log_lengths_finite(self, gaba_fit):
        ll = gaba_fit["adata"].var["log_lengths"].values
        assert np.all(np.isfinite(ll))

    def test_M_shape(self, gaba_fit):
        M = gaba_fit["adata"].uns["M"]
        assert M.shape == (2, len(GENES))

    def test_M_positive(self, gaba_fit):
        assert np.all(gaba_fit["adata"].uns["M"] > 0)


# ---------------------------------------------------------------------------
# 2. Inference pipeline structure
# ---------------------------------------------------------------------------


class TestGabaInferencePipeline:
    def test_grid_size(self, gaba_fit):
        ip = gaba_fit["ip"]
        assert ip.gridsize == [6, 7]
        assert ip.n_grid_points == 42

    def test_ref_samp_ind_in_range(self, gaba_fit):
        assert REF_SAMP_IND < gaba_fit["ip"].n_grid_points

    def test_ref_samp_values_match(self, gaba_fit):
        vals = gaba_fit["ip"].sampl_vals[REF_SAMP_IND]
        assert vals[0] == pytest.approx(REF_SAMP_OPTIMUM[0])
        assert vals[1] == pytest.approx(REF_SAMP_OPTIMUM[1])

    def test_param_estimates_shape(self, gaba_fit):
        sr = gaba_fit["sr"]
        n_gp = gaba_fit["ip"].n_grid_points
        n_phys = gaba_fit["ip"].n_phys_pars
        assert sr.param_estimates.shape == (n_gp, len(GENES), n_phys)

    def test_obj_func_shape(self, gaba_fit):
        assert gaba_fit["sr"].obj_func.shape == (gaba_fit["ip"].n_grid_points,)

    def test_obj_func_finite(self, gaba_fit):
        assert np.all(np.isfinite(gaba_fit["sr"].obj_func))


# ---------------------------------------------------------------------------
# 3. Parameter quality — fitted KLD ≤ reference KLD at grid point 17
# ---------------------------------------------------------------------------


class TestGabaParameterQuality:
    @pytest.fixture(scope="class")
    def kld_comparison(self, ref_adata, gaba_fit):
        """Return per-gene (kld_ref, kld_fit) evaluated at REF_SAMP_IND."""
        sr = gaba_fit["sr"]
        sd = gaba_fit["sd"]
        model = gaba_fit["model"]
        gene_idx = gaba_fit["gene_idx"]
        hist_type = get_hist_type(sd)

        ref_params = ref_adata.var.loc[GENES, REF_PARAM_COLS]

        sr.set_sampling_optimum(REF_SAMP_IND)

        results = {}
        for g in GENES:
            idx = gene_idx[g]
            samp = sr.regressor_optimum[idx]
            fitted_p = sr.param_estimates[REF_SAMP_IND, idx]
            ref_p = ref_params.loc[g].values
            kld_fit = model.eval_model_kld(fitted_p, sd.M[:, idx], samp, sd.hist[idx], hist_type)
            kld_ref = model.eval_model_kld(ref_p,    sd.M[:, idx], samp, sd.hist[idx], hist_type)
            results[g] = (kld_ref, kld_fit)
        return results

    @pytest.mark.parametrize("gene", GENES)
    def test_fitted_kld_le_reference(self, kld_comparison, gene):
        kld_ref, kld_fit = kld_comparison[gene]
        # Fitted params must achieve KLD no worse than the reference (small
        # tolerance for floating-point / optimizer noise).
        assert kld_fit <= kld_ref + 1e-3, (
            f"{gene}: fitted KLD {kld_fit:.4f} > ref KLD {kld_ref:.4f}"
        )

    @pytest.mark.parametrize("gene", GENES)
    def test_fitted_kld_finite(self, kld_comparison, gene):
        _, kld_fit = kld_comparison[gene]
        assert np.isfinite(kld_fit)


# ---------------------------------------------------------------------------
# 4. Parameter snapshot at ref samp_optimum_ind = 17
# ---------------------------------------------------------------------------


class TestGabaParamSnapshot:
    @pytest.fixture(scope="class")
    def params_at_ref(self, gaba_fit):
        sr = gaba_fit["sr"]
        gene_idx = gaba_fit["gene_idx"]
        sr.set_sampling_optimum(REF_SAMP_IND)
        return np.array([sr.phys_optimum[gene_idx[g]] for g in GENES])

    def test_params_shape(self, params_at_ref):
        assert params_at_ref.shape == (len(GENES), 3)

    def test_params_finite(self, params_at_ref):
        assert np.all(np.isfinite(params_at_ref))

    def test_params_within_bounds(self, gaba_fit, params_at_ref):
        lb = gaba_fit["ip"].phys_lb
        ub = gaba_fit["ip"].phys_ub
        assert np.all(params_at_ref >= lb - 1e-6)
        assert np.all(params_at_ref <= ub + 1e-6)

    def test_params_snapshot(self, params_at_ref):
        check_snapshot(
            "gaba_bursty_poisson_params",
            params_at_ref,
            rtol=0.05,   # 5% relative — accounts for optimizer variation
            atol=0.1,    # 0.1 log10 units absolute
        )


# ---------------------------------------------------------------------------
# 5. Rust / Python parity on gaba data
# ---------------------------------------------------------------------------
# The main gaba_fit fixture uses Bursty+Poisson (seq_model != "None"), which
# bypasses the Rust fast-path.  These parity tests extract a single gaba gene
# with each seq_model="None" model to exercise the Rust path at the larger
# grid dimensions that arise from GABAergic neuron expression data.

_PARITY_MODELS = [
    ("Bursty", "None"),
    ("Constitutive", "None"),
    ("Extrinsic", "None"),
    ("Delay", "None"),
    ("CIR", "None"),
    ("DelayedSplicing", "None"),
    # Poisson: only Bursty and CIR have a Rust fast-path for seq_model="Poisson".
    ("Bursty", "Poisson"),
    ("CIR", "Poisson"),
]
_PARITY_MODEL_IDS = [f"{m}_{s}" for m, s in _PARITY_MODELS]
_PARITY_GENE = GENES[0]  # "Eif5b"
_DUMMY_SAMP = np.zeros(2)          # safe dummy for seq_model="None" get_MoM calls
_POISSON_SAMP = np.array(list(REF_SAMP_OPTIMUM))  # realistic Poisson samp from reference


@pytest.fixture(scope="module", params=_PARITY_MODELS, ids=_PARITY_MODEL_IDS)
def gaba_parity_processed(request):
    """Extract one gaba gene per model for lightweight Rust/Python parity checks."""
    bio_model, seq_model = request.param
    model = CMEModel(bio_model, seq_model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata = extract_data(
            GABA_EXAMPLE_PATH,
            model,
            dataset_name=f"gaba_parity_{bio_model}_{seq_model}",
            modality_name_dict={"unspliced": "unspliced", "spliced": "spliced"},
            n_genes=1,
            genes_to_fit=[_PARITY_GENE],
            hist_type="unique",
            viz=False,
        )
    return {"bio_model": bio_model, "seq_model": seq_model, "model": model, "adata": adata}


class TestRustPythonParityGaba:
    """Verify Rust and Python eval_model_pss agree on gaba_example.h5ad grid sizes.

    Covers seq_model="None" for all six 2D models and seq_model="Poisson" for
    Bursty and CIR (the only two with a Rust Poisson fast-path).
    """

    def test_pss_parity_at_mom_params(self, gaba_parity_processed):
        model = gaba_parity_processed["model"]
        adata = gaba_parity_processed["adata"]
        seq_model = gaba_parity_processed["seq_model"]

        moments = {
            col: float(adata.var[col].values[0])
            for col in adata.var.columns if col.startswith("MOM_")
        }
        lb = np.array(model.bio_bounds["phys_lb"])
        ub = np.array(model.bio_bounds["phys_ub"])
        p = model.get_MoM(moments, lb, ub, samp=_DUMMY_SAMP)
        limits = adata.uns["M"][:, 0].tolist()
        samp = _POISSON_SAMP if seq_model == "Poisson" else None

        rust = model.eval_model_pss(p, limits, samp=samp)
        with unittest.mock.patch.object(_cme_toolbox_module, "_HAS_RUST", False):
            py = model.eval_model_pss(p, limits, samp=samp)

        np.testing.assert_allclose(
            rust, py, rtol=1e-5, atol=1e-10,
            err_msg=(
                f"{gaba_parity_processed['bio_model']}/{seq_model} Rust vs Python mismatch "
                f"on gaba data limits {limits}"
            ),
        )
