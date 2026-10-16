"""Input validation raises clear ValueErrors instead of panics / opaque errors
(audit fix 5, 2026-10-15).

Python-side checks live in CMEModel._check_pss_args and inference._validate_bounds;
Rust-side checks live at every monod_core entry point and must surface as
ValueError (never PanicException).
"""
import os
import numpy as np
import pytest

from monod.cme_toolbox import CMEModel
from monod.inference import InferenceParameters
import monod.mminference as mminference

from monod._core import _mc  # None unless the compiled extension is really installed

needs_rust = pytest.mark.skipif(_mc is None, reason="monod_core not installed")


# ---------------------------------------------------------------------------
# Python side
# ---------------------------------------------------------------------------

class TestEvalModelPssArgs:
    def test_wrong_param_count(self):
        with pytest.raises(ValueError, match="expected 3 log10 parameters"):
            CMEModel("Bursty", "None").eval_model_pss(np.array([0.0, 0.0]), [10, 10])

    def test_wrong_limits_count(self):
        with pytest.raises(ValueError, match="one grid size per species"):
            CMEModel("Bursty", "None").eval_model_pss(np.array([0.3, 0.0, -0.1]), [10])

    def test_zero_limit(self):
        with pytest.raises(ValueError, match="positive integers"):
            CMEModel("Bursty", "None").eval_model_pss(np.array([0.3, 0.0, -0.1]), [10, 0])

    def test_float_limits_are_coerced(self):
        m = CMEModel("Bursty", "None")
        a = m.eval_model_pss(np.array([0.3, 0.0, -0.1]), [12.0, 12.0])
        b = m.eval_model_pss(np.array([0.3, 0.0, -0.1]), [12, 12])
        np.testing.assert_allclose(a, b)

    def test_non_finite_params(self):
        with pytest.raises(ValueError, match="finite"):
            CMEModel("Bursty", "None").eval_model_pss(np.array([0.3, np.nan, -0.1]), [10, 10])


class TestInferenceParametersBounds:
    def test_inverted_phys_bounds(self):
        with pytest.raises(ValueError, match=r"phys_lb\[0\]"):
            InferenceParameters("ds", CMEModel("Bursty", "None"),
                                phys_lb=[2, -2, -2], phys_ub=[0, 2, 2])

    def test_wrong_phys_length(self):
        with pytest.raises(ValueError, match="has 3 parameters"):
            InferenceParameters("ds", CMEModel("Bursty", "None"), phys_lb=[-1, -1], phys_ub=[1, 1])

    def test_gridsize_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            InferenceParameters("ds", CMEModel("Bursty", "Poisson"), gridsize=[3])

    def test_gridsize_zero(self):
        with pytest.raises(ValueError, match="integers >= 1"):
            InferenceParameters("ds", CMEModel("Bursty", "Poisson"), gridsize=[0, 2])

    def test_inverted_samp_bounds(self):
        with pytest.raises(ValueError, match=r"samp_lb\[1\]"):
            InferenceParameters("ds", CMEModel("Bursty", "Poisson"),
                                samp_lb=[-8, 0], samp_ub=[-5, -3])

    def test_mek_means_shares_validation(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match=r"phys_lb\[0\]"):
            mminference.MEKMeansParameters("ds", CMEModel("Bursty", "None"), k=2, epochs=1,
                                           phys_lb=[2, -2, -2], phys_ub=[0, 2, 2])


# ---------------------------------------------------------------------------
# Rust side
# ---------------------------------------------------------------------------

def _tiny_sd(n_cells=30, n_genes=3, padding=5, seed=0):
    rng = np.random.default_rng(seed)
    layers = [np.ascontiguousarray(rng.poisson(2.0, (n_cells, n_genes)).astype(np.int64)) for _ in range(2)]
    coords, freqs = _mc.make_state_dist(layers)
    limits = np.ascontiguousarray(np.stack([l.max(axis=0) + padding for l in layers]).astype(np.int64))
    sd = _mc.searchdata_from_arrays(layers, ["unspliced", "spliced"], limits, coords, freqs,
                                    [f"g{i}" for i in range(n_genes)], n_cells, "unique",
                                    None, None, None, None)
    return sd, layers, limits


@needs_rust
class TestRustEntryPoints:
    P = [0.3, 0.0, -0.1]

    def test_unknown_model_is_value_error(self):
        with pytest.raises(ValueError, match="Unknown bio_model"):
            _mc.eval_model_pss_2d("bursty", self.P, [8, 8], 10.0, 60)

    def test_short_limits(self):
        with pytest.raises(ValueError, match="expected 2 grid limits"):
            _mc.eval_model_pss_2d("Bursty", self.P, [8], 10.0, 60)

    def test_zero_limit(self):
        with pytest.raises(ValueError, match="every limit must be >= 1"):
            _mc.eval_model_pss_2d("Bursty", self.P, [8, 0], 10.0, 60)

    def test_wrong_param_count(self):
        with pytest.raises(ValueError, match="takes 3 parameters"):
            _mc.eval_model_pss_2d("Bursty", [0.0, 0.0], [8, 8], 10.0, 60)

    def test_kld_unknown_model_is_value_error(self):
        with pytest.raises(ValueError, match="Unknown bio_model"):
            _mc.eval_kld_2d("bursty", self.P, [5, 5], [0], [0], [1.0], 10.0, 60, None, 1e-15, "None")

    def test_kld_coordinate_at_limit(self):
        with pytest.raises(ValueError, match="increase padding"):
            _mc.eval_kld_2d("Bursty", self.P, [5, 5], [5], [0], [1.0], 10.0, 60, None, 1e-15, "None")

    def test_protein_limit_zero(self):
        with pytest.raises(ValueError, match="protein_limit must be > 0"):
            _mc.eval_model_pss_protein_bursty([0.3, 0, 0, 0, 0], [4, 4, 4], True, 0.0, 0.1, 10.0)

    def test_optimizer_inverted_bounds(self):
        sd, _, _ = _tiny_sd()
        with pytest.raises(ValueError, match=r"lb\[0\]=2 exceeds ub\[0\]=0"):
            _mc.optimize_genes_2d_sd(sd, "Bursty", [[self.P]] * 3, [2, -2, -2], [0, 2, 2],
                                     10.0, 60, seq_model="None")

    def test_optimizer_x0_gene_count(self):
        sd, _, _ = _tiny_sd()
        with pytest.raises(ValueError, match="x0_list has 1 genes"):
            _mc.optimize_genes_2d_sd(sd, "Bursty", [[self.P]], [-2, -2, -2], [2, 2, 2],
                                     10.0, 60, seq_model="None")

    def test_optimizer_status_opt_in(self):
        sd, _, _ = _tiny_sd()
        two = _mc.optimize_genes_2d_sd(sd, "Bursty", [[self.P]] * 3, [-2, -2, -2], [2, 2, 2],
                                       10.0, 60, maxiter=50, seq_model="None")
        three = _mc.optimize_genes_2d_sd(sd, "Bursty", [[self.P]] * 3, [-2, -2, -2], [2, 2, 2],
                                         10.0, 60, maxiter=50, seq_model="None", return_status=True)
        assert len(two) == 2 and len(three) == 3
        np.testing.assert_allclose(two[1], three[1])
        assert set(three[2]) <= {0, 1, 2, 3}

    def test_mom_x0_inverted_bounds(self):
        sd, _, _ = _tiny_sd()
        with pytest.raises(ValueError, match="exceeds"):
            sd.mom_x0_all("Bursty", "None", "None", [2, -2, -2], [0, 2, 2])

    def test_fortran_order_rejected(self):
        a = np.asfortranarray(np.arange(12, dtype=np.int64).reshape(3, 4))
        with pytest.raises(ValueError, match="C-contiguous"):
            _mc.compute_moments([a], ["unspliced"])
        with pytest.raises(ValueError, match="C-contiguous"):
            _mc.make_state_dist([a])

    def test_negative_counts_rejected(self):
        a = np.ascontiguousarray(np.array([[0, 1], [-1, 2]], dtype=np.int64))
        with pytest.raises(ValueError, match="negative count"):
            _mc.make_state_dist([a])

    def test_searchdata_coordinate_outside_limits(self):
        sd, layers, limits = _tiny_sd()
        coords, freqs = _mc.make_state_dist(layers)
        bad_limits = np.ascontiguousarray(np.stack([l.max(axis=0) for l in layers]).astype(np.int64))  # padding 0
        with pytest.raises(ValueError, match="increase padding"):
            _mc.searchdata_from_arrays(layers, ["unspliced", "spliced"], bad_limits, coords, freqs,
                                       ["g0", "g1", "g2"], 30, "unique", None, None, None, None)

    def test_searchdata_wrong_n_cells(self):
        sd, layers, limits = _tiny_sd()
        coords, freqs = _mc.make_state_dist(layers)
        with pytest.raises(ValueError, match="expected \\(29, 3\\)"):
            _mc.searchdata_from_arrays(layers, ["unspliced", "spliced"], limits, coords, freqs,
                                       ["g0", "g1", "g2"], 29, "unique", None, None, None, None)

    def test_partition_assigns_length(self):
        sd, _, _ = _tiny_sd()
        with pytest.raises(ValueError, match="assigns has 5 entries"):
            _mc.partition_searchdata_2d(sd, [0] * 5, 2)

    def test_h5ad_padding_zero(self, tmp_path):
        with pytest.raises(ValueError, match="padding must be >= 1"):
            _mc.searchdata_from_h5ad(str(tmp_path / "x.h5ad"), ["unspliced", "spliced"], padding=0)


class TestReviewRegressions:
    """Regressions caught by the post-fix code review (2026-10-15)."""

    def test_ambient_param_count_is_total(self):
        m = CMEModel("Bursty", "None", amb_model="Equal")
        assert m.get_num_params() == 4
        p, lim = m._check_pss_args(np.zeros(4), [8, 8, 4])
        assert p.shape == (4,) and lim == [8, 8, 4]
        with pytest.raises(ValueError, match="expected 4 log10 parameters \\(3 biological \\+ 1 ambient\\)"):
            m._check_pss_args(np.zeros(5), [8, 8, 4])
        with pytest.raises(ValueError, match="expected 4"):
            m._check_pss_args(np.zeros(3), [8, 8, 4])

    def test_bounds_validated_before_scipy(self):
        with pytest.raises(ValueError, match="must be 1-D with equal length"):
            InferenceParameters("ds", CMEModel("Bursty", "None"), phys_lb=[-1, -1, -1], phys_ub=[2, 2])

    def test_mek_means_bounds_validated_before_scipy(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="must be 1-D with equal length"):
            mminference.MEKMeansParameters("ds", CMEModel("Bursty", "None"), k=2, epochs=1,
                                           phys_lb=[-1, -1, -1], phys_ub=[2, 2])


@needs_rust
class TestLeftoverValidation:
    """e_step_2d argument checks, chi-square dof handling, MoM model check, absolute run dirs."""

    def test_e_step_weights_length(self):
        sd, layers, limits = _tiny_sd()
        n_genes = 3
        params = [[[0.3, 0.0, -0.1]] * n_genes] * 2
        lims = [[int(limits[0, g]), int(limits[1, g])] for g in range(n_genes)]
        u = [layers[0][:, g].tolist() for g in range(n_genes)]
        s = [layers[1][:, g].tolist() for g in range(n_genes)]
        with pytest.raises(ValueError, match="weights has 1 entries"):
            _mc.e_step_2d("Bursty", params, lims, u, s, [1.0], 10.0, 60, seq_model="None")
        with pytest.raises(ValueError, match=r"params_per_k\[0\] has 1 genes"):
            _mc.e_step_2d("Bursty", [[[0.3, 0.0, -0.1]]] * 2, lims, u, s, [0.5, 0.5], 10.0, 60, seq_model="None")
        Q, lb, q = _mc.e_step_2d("Bursty", params, lims, u, s, [0.5, 0.5], 10.0, 60, seq_model="None")
        assert np.asarray(Q).shape == (30, 2)

    def test_chisquare_no_dof_gives_nan_not_rejection(self):
        """A gene with too few bins for the parameter count must yield NaN, as scipy does."""
        sd, layers, limits = _tiny_sd(n_cells=12, n_genes=1, padding=2)
        phys = [[0.3, 0.0, -0.1]]
        csq, pval, hel = _mc.chisquare_testing_2d_sd(sd, "Bursty", phys, None, 10.0, 60, 3, 1e9, 1e-15, "None")
        # grouping_thr=1e9 collapses everything into one bin -> dof = 1 - 1 - 3 < 0
        assert np.isnan(pval[0]) and np.isfinite(csq[0])

    def test_mom_x0_unknown_model(self):
        sd, _, _ = _tiny_sd()
        with pytest.raises(ValueError, match="unknown bio_model"):
            sd.mom_x0_all("bursty", "None", "None", [-2, -2, -2], [2, 2, 2])

    def test_run_directory_is_absolute(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ip = InferenceParameters("ds", CMEModel("Bursty", "None"))
        assert os.path.isabs(ip.inference_string) and ip.inference_string.startswith(str(tmp_path))
        p = mminference.MEKMeansParameters("ds", CMEModel("Bursty", "None"), k=2, epochs=1)
        assert os.path.isabs(p.inference_string)


    def test_python_chisquare_honours_grouping_thr_and_tiny_data(self):
        """The Python chi-square path must use grouping_thr (not a hard-coded 5) and
        must not crash when no bin ever closes; it should agree with Rust."""
        import unittest.mock, warnings
        import monod.inference as inf
        from monod.extract_data import extract_data
        from monod.inference import InferenceParameters, searchdata_from_adata
        import anndata as ad, os
        warnings.filterwarnings("ignore")
        here = os.path.dirname(__file__)
        model = CMEModel("Bursty", "None")
        madata = extract_data(ad.read_h5ad(os.path.join(here, "..", "example_h5ad", "example_adata.h5ad")),
                              model, dataset_name="x", n_genes=3, viz=False)
        sd = searchdata_from_adata(madata)
        sr = InferenceParameters("x", model, gradient_params={"max_iterations": 5, "num_restarts": 1}).fit_all_grid_points(sd)
        sr.find_sampling_optimum()
        csq_r, p_r, h_r = sr.chisquare_testing(sd, grouping_thr=20)
        with unittest.mock.patch.object(inf, "_mc", None):
            csq_p, p_p, h_p = sr.chisquare_testing(sd, grouping_thr=20)
        np.testing.assert_allclose(csq_p, csq_r, rtol=1e-6)
        np.testing.assert_allclose(h_p, h_r, rtol=1e-6)
        both_nan = np.isnan(p_p) & np.isnan(p_r)
        np.testing.assert_allclose(p_p[~both_nan], p_r[~both_nan], rtol=1e-6)
        # absurd threshold: a single bin on both paths, NaN p-value, no crash
        with unittest.mock.patch.object(inf, "_mc", None):
            csq1, p1, _ = sr.chisquare_testing(sd, grouping_thr=1e9)
        assert np.all(np.isnan(p1)) and np.all(np.isfinite(csq1))
