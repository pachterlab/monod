"""
Regression (snapshot) tests for cme_toolbox.CMEModel.

All numerical tests use a two-phase approach:
  - First run  : snapshot .npy files are written to tests/snapshots/; tests pass.
  - Later runs : computed values are compared against the stored snapshots.

Delete tests/snapshots/ and re-run pytest to regenerate all snapshots.
"""

import unittest.mock

import numpy as np
import pytest

import cme_toolbox as _cme_toolbox_module
from conftest import check_snapshot
from cme_toolbox import CMEModel


# ---------------------------------------------------------------------------
# Shared fixtures (module-scoped so the heavy __init__ runs only once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bursty_none():
    return CMEModel("Bursty", "None")


@pytest.fixture(scope="module")
def bursty_poisson():
    return CMEModel("Bursty", "Poisson")


@pytest.fixture(scope="module")
def bursty_bernoulli():
    return CMEModel("Bursty", "Bernoulli")


@pytest.fixture(scope="module")
def constitutive_none():
    return CMEModel("Constitutive", "None")


@pytest.fixture(scope="module")
def constitutive_poisson():
    return CMEModel("Constitutive", "Poisson")


@pytest.fixture(scope="module")
def extrinsic_none():
    return CMEModel("Extrinsic", "None")


@pytest.fixture(scope="module")
def delay_none():
    return CMEModel("Delay", "None")


@pytest.fixture(scope="module")
def cir_none():
    return CMEModel("CIR", "None")


@pytest.fixture(scope="module")
def delayed_splicing_none():
    return CMEModel("DelayedSplicing", "None")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _pss_to_unique_data(pss):
    """Convert a PMF array to the (coords, freqs) unique-histogram format."""
    coords = np.array(np.where(pss > 0)).T
    freqs = pss[tuple(coords.T)]
    freqs = freqs / freqs.sum()
    return coords, freqs


# ---------------------------------------------------------------------------
# get_num_params
# ---------------------------------------------------------------------------


class TestGetNumParams:
    def test_constitutive(self, constitutive_none):
        assert constitutive_none.get_num_params() == 2

    def test_bursty(self, bursty_none):
        assert bursty_none.get_num_params() == 3

    def test_extrinsic(self, extrinsic_none):
        assert extrinsic_none.get_num_params() == 3

    def test_delay(self, delay_none):
        assert delay_none.get_num_params() == 3

    def test_cir(self, cir_none):
        assert cir_none.get_num_params() == 3

    def test_delayed_splicing(self, delayed_splicing_none):
        assert delayed_splicing_none.get_num_params() == 3

    def test_bursty_equal_amb(self):
        m = CMEModel("Bursty", "None", amb_model="Equal")
        assert m.get_num_params() == 4

    def test_bursty_unequal_amb(self):
        m = CMEModel("Bursty", "None", amb_model="Unequal")
        assert m.get_num_params() == 5


# ---------------------------------------------------------------------------
# get_log_name_str
# ---------------------------------------------------------------------------


class TestGetLogNameStr:
    def test_constitutive_count(self, constitutive_none):
        assert len(constitutive_none.get_log_name_str()) == 2

    def test_bursty_count(self, bursty_none):
        assert len(bursty_none.get_log_name_str()) == 3

    def test_extrinsic_count(self, extrinsic_none):
        assert len(extrinsic_none.get_log_name_str()) == 3

    def test_delay_count(self, delay_none):
        assert len(delay_none.get_log_name_str()) == 3

    def test_cir_count(self, cir_none):
        assert len(cir_none.get_log_name_str()) == 3

    def test_delayed_splicing_count(self, delayed_splicing_none):
        assert len(delayed_splicing_none.get_log_name_str()) == 3

    def test_constitutive_names(self, constitutive_none):
        assert constitutive_none.get_log_name_str() == [
            r"$\log_{10} \beta$",
            r"$\log_{10} \gamma$",
        ]

    def test_bursty_names(self, bursty_none):
        assert bursty_none.get_log_name_str() == [
            r"$\log_{10} b$",
            r"$\log_{10} \beta$",
            r"$\log_{10} \gamma$",
        ]

    def test_extrinsic_names(self, extrinsic_none):
        assert extrinsic_none.get_log_name_str() == [
            r"$\log_{10} \alpha$",
            r"$\log_{10} \beta$",
            r"$\log_{10} \gamma$",
        ]

    def test_delay_names(self, delay_none):
        assert delay_none.get_log_name_str() == [
            r"$\log_{10} b$",
            r"$\log_{10} \beta$",
            r"$\log_{10} \tau^{-1}$",
        ]

    def test_cir_names(self, cir_none):
        assert cir_none.get_log_name_str() == [
            r"$\log_{10} b$",
            r"$\log_{10} \beta$",
            r"$\log_{10} \gamma$",
        ]

    def test_delayed_splicing_names(self, delayed_splicing_none):
        assert delayed_splicing_none.get_log_name_str() == [
            r"$\log_{10} b$",
            r"$\log_{10} \tau^{-1}$",
            r"$\log_{10} \gamma$",
        ]

    def test_equal_amb_appends_param(self):
        m = CMEModel("Bursty", "None", amb_model="Equal")
        names = m.get_log_name_str()
        assert len(names) == 4
        assert names[-1] == r"$\log_{10} p$"

    def test_unequal_amb_appends_params(self):
        m = CMEModel("Bursty", "None", amb_model="Unequal")
        names = m.get_log_name_str()
        assert len(names) == 5
        assert names[-2] == r"$\log_{10} p_N$"
        assert names[-1] == r"$\log_{10} p_M$"


# ---------------------------------------------------------------------------
# eval_model_pss — normalization, non-negativity, and snapshots
# ---------------------------------------------------------------------------


class TestEvalModelPSS:
    LIMITS = [20, 20]

    # --- Constitutive / None ---
    def test_constitutive_none_sums_to_one(self, constitutive_none):
        pss = constitutive_none.eval_model_pss(np.array([0.0, 0.0]), self.LIMITS)
        np.testing.assert_allclose(pss.sum(), 1.0, rtol=1e-5)

    def test_constitutive_none_nonnegative(self, constitutive_none):
        pss = constitutive_none.eval_model_pss(np.array([0.0, 0.0]), self.LIMITS)
        assert np.all(pss >= 0)

    def test_constitutive_none_snapshot(self, constitutive_none):
        pss = constitutive_none.eval_model_pss(np.array([0.0, 0.0]), self.LIMITS)
        check_snapshot("eval_model_pss_constitutive_none", pss)

    # --- Bursty / None ---
    def test_bursty_none_sums_to_one(self, bursty_none):
        pss = bursty_none.eval_model_pss(np.array([0.3, 0.0, -0.1]), self.LIMITS)
        np.testing.assert_allclose(pss.sum(), 1.0, rtol=1e-5)

    def test_bursty_none_nonnegative(self, bursty_none):
        pss = bursty_none.eval_model_pss(np.array([0.3, 0.0, -0.1]), self.LIMITS)
        assert np.all(pss >= 0)

    def test_bursty_none_snapshot(self, bursty_none):
        pss = bursty_none.eval_model_pss(np.array([0.3, 0.0, -0.1]), self.LIMITS)
        check_snapshot("eval_model_pss_bursty_none", pss)

    # --- Bursty / Poisson ---
    def test_bursty_poisson_sums_to_one(self, bursty_poisson):
        pss = bursty_poisson.eval_model_pss(
            np.array([0.3, 0.0, -0.1]), self.LIMITS, samp=np.array([-6.0, -6.0])
        )
        np.testing.assert_allclose(pss.sum(), 1.0, rtol=1e-5)

    def test_bursty_poisson_nonnegative(self, bursty_poisson):
        pss = bursty_poisson.eval_model_pss(
            np.array([0.3, 0.0, -0.1]), self.LIMITS, samp=np.array([-6.0, -6.0])
        )
        assert np.all(pss >= 0)

    def test_bursty_poisson_snapshot(self, bursty_poisson):
        pss = bursty_poisson.eval_model_pss(
            np.array([0.3, 0.0, -0.1]), self.LIMITS, samp=np.array([-6.0, -6.0])
        )
        check_snapshot("eval_model_pss_bursty_poisson", pss)

    # --- Bursty / Bernoulli ---
    def test_bursty_bernoulli_sums_to_one(self, bursty_bernoulli):
        pss = bursty_bernoulli.eval_model_pss(
            np.array([0.3, 0.0, -0.1]), self.LIMITS, samp=np.array([-1.0, -1.0])
        )
        np.testing.assert_allclose(pss.sum(), 1.0, rtol=1e-5)

    def test_bursty_bernoulli_snapshot(self, bursty_bernoulli):
        pss = bursty_bernoulli.eval_model_pss(
            np.array([0.3, 0.0, -0.1]), self.LIMITS, samp=np.array([-1.0, -1.0])
        )
        check_snapshot("eval_model_pss_bursty_bernoulli", pss)

    # --- Extrinsic / None ---
    def test_extrinsic_none_sums_to_one(self, extrinsic_none):
        pss = extrinsic_none.eval_model_pss(np.array([0.5, 0.0, 0.0]), self.LIMITS)
        np.testing.assert_allclose(pss.sum(), 1.0, rtol=1e-5)

    def test_extrinsic_none_snapshot(self, extrinsic_none):
        pss = extrinsic_none.eval_model_pss(np.array([0.5, 0.0, 0.0]), self.LIMITS)
        check_snapshot("eval_model_pss_extrinsic_none", pss)

    # --- Delay / None ---
    def test_delay_none_sums_to_one(self, delay_none):
        pss = delay_none.eval_model_pss(np.array([0.3, 0.0, 0.0]), self.LIMITS)
        np.testing.assert_allclose(pss.sum(), 1.0, rtol=1e-5)

    def test_delay_none_snapshot(self, delay_none):
        pss = delay_none.eval_model_pss(np.array([0.3, 0.0, 0.0]), self.LIMITS)
        check_snapshot("eval_model_pss_delay_none", pss)

    # --- CIR / None ---
    def test_cir_none_sums_to_one(self, cir_none):
        pss = cir_none.eval_model_pss(np.array([0.3, 0.0, 0.0]), self.LIMITS)
        np.testing.assert_allclose(pss.sum(), 1.0, rtol=1e-5)

    def test_cir_none_snapshot(self, cir_none):
        pss = cir_none.eval_model_pss(np.array([0.3, 0.0, 0.0]), self.LIMITS)
        check_snapshot("eval_model_pss_cir_none", pss)

    # --- DelayedSplicing / None ---
    def test_delayed_splicing_none_sums_to_one(self, delayed_splicing_none):
        pss = delayed_splicing_none.eval_model_pss(
            np.array([0.3, 0.0, 0.0]), self.LIMITS
        )
        np.testing.assert_allclose(pss.sum(), 1.0, rtol=1e-5)

    def test_delayed_splicing_none_snapshot(self, delayed_splicing_none):
        pss = delayed_splicing_none.eval_model_pss(
            np.array([0.3, 0.0, 0.0]), self.LIMITS
        )
        check_snapshot("eval_model_pss_delayed_splicing_none", pss)


# ---------------------------------------------------------------------------
# eval_model_kld — self-KLD should be ~0; cross-parameter KLD snapshotted
# ---------------------------------------------------------------------------


class TestEvalModelKLD:
    LIMITS = [20, 20]

    def test_bursty_self_kld_near_zero(self, bursty_none):
        p = np.array([0.3, 0.0, -0.1])
        pss = bursty_none.eval_model_pss(p, self.LIMITS)
        data = _pss_to_unique_data(pss)
        kld = bursty_none.eval_model_kld(p, self.LIMITS, None, data)
        np.testing.assert_allclose(kld, 0.0, atol=1e-5)

    def test_constitutive_self_kld_near_zero(self, constitutive_none):
        p = np.array([0.0, 0.0])
        pss = constitutive_none.eval_model_pss(p, self.LIMITS)
        data = _pss_to_unique_data(pss)
        kld = constitutive_none.eval_model_kld(p, self.LIMITS, None, data)
        np.testing.assert_allclose(kld, 0.0, atol=1e-5)

    def test_extrinsic_self_kld_near_zero(self, extrinsic_none):
        p = np.array([0.5, 0.0, 0.0])
        pss = extrinsic_none.eval_model_pss(p, self.LIMITS)
        data = _pss_to_unique_data(pss)
        kld = extrinsic_none.eval_model_kld(p, self.LIMITS, None, data)
        np.testing.assert_allclose(kld, 0.0, atol=1e-5)

    def test_bursty_kld_nonnegative(self, bursty_none):
        p_true = np.array([0.3, 0.0, -0.1])
        p_fit = np.array([0.2, 0.1, 0.0])
        pss = bursty_none.eval_model_pss(p_true, self.LIMITS)
        data = _pss_to_unique_data(pss)
        kld = bursty_none.eval_model_kld(p_fit, self.LIMITS, None, data)
        assert kld >= 0.0

    def test_bursty_cross_kld_snapshot(self, bursty_none):
        p_true = np.array([0.3, 0.0, -0.1])
        p_fit = np.array([0.2, 0.1, 0.0])
        pss = bursty_none.eval_model_pss(p_true, self.LIMITS)
        data = _pss_to_unique_data(pss)
        kld = bursty_none.eval_model_kld(p_fit, self.LIMITS, None, data)
        check_snapshot("eval_model_kld_bursty_none_cross", np.array([kld]))

    def test_constitutive_cross_kld_snapshot(self, constitutive_none):
        p_true = np.array([0.0, 0.0])
        p_fit = np.array([0.1, -0.1])
        pss = constitutive_none.eval_model_pss(p_true, self.LIMITS)
        data = _pss_to_unique_data(pss)
        kld = constitutive_none.eval_model_kld(p_fit, self.LIMITS, None, data)
        check_snapshot("eval_model_kld_constitutive_none_cross", np.array([kld]))

    def test_extrinsic_cross_kld_snapshot(self, extrinsic_none):
        p_true = np.array([0.5, 0.0, 0.0])
        p_fit = np.array([0.3, 0.1, -0.1])
        pss = extrinsic_none.eval_model_pss(p_true, self.LIMITS)
        data = _pss_to_unique_data(pss)
        kld = extrinsic_none.eval_model_kld(p_fit, self.LIMITS, None, data)
        check_snapshot("eval_model_kld_extrinsic_none_cross", np.array([kld]))


# ---------------------------------------------------------------------------
# eval_model_logL — snapshot tests
# ---------------------------------------------------------------------------


class TestEvalModelLogL:
    LIMITS = [20, 20]
    N_CELLS = 500

    def test_bursty_logl_snapshot(self, bursty_none):
        p = np.array([0.3, 0.0, -0.1])
        pss = bursty_none.eval_model_pss(p, self.LIMITS)
        data = _pss_to_unique_data(pss)
        logl = bursty_none.eval_model_logL(p, self.LIMITS, None, data, self.N_CELLS)
        check_snapshot("eval_model_logL_bursty_none", np.array([logl]))

    def test_constitutive_logl_snapshot(self, constitutive_none):
        p = np.array([0.0, 0.0])
        pss = constitutive_none.eval_model_pss(p, self.LIMITS)
        data = _pss_to_unique_data(pss)
        logl = constitutive_none.eval_model_logL(
            p, self.LIMITS, None, data, self.N_CELLS
        )
        check_snapshot("eval_model_logL_constitutive_none", np.array([logl]))

    def test_extrinsic_logl_snapshot(self, extrinsic_none):
        p = np.array([0.5, 0.0, 0.0])
        pss = extrinsic_none.eval_model_pss(p, self.LIMITS)
        data = _pss_to_unique_data(pss)
        logl = extrinsic_none.eval_model_logL(
            p, self.LIMITS, None, data, self.N_CELLS
        )
        check_snapshot("eval_model_logL_extrinsic_none", np.array([logl]))

    def test_bursty_poisson_logl_snapshot(self, bursty_poisson):
        p = np.array([0.3, 0.0, -0.1])
        samp = np.array([-6.0, -6.0])
        pss = bursty_poisson.eval_model_pss(p, self.LIMITS, samp=samp)
        data = _pss_to_unique_data(pss)
        logl = bursty_poisson.eval_model_logL(
            p, self.LIMITS, samp, data, self.N_CELLS
        )
        check_snapshot("eval_model_logL_bursty_poisson", np.array([logl]))

    def test_self_logl_equals_neg_entropy(self, bursty_none):
        """LogL evaluated at the true params equals -N * H(p) where H is entropy."""
        p = np.array([0.3, 0.0, -0.1])
        pss = bursty_none.eval_model_pss(p, self.LIMITS)
        data = _pss_to_unique_data(pss)
        coords, freqs = data
        expected = self.N_CELLS * np.sum(freqs * np.log(freqs))
        logl = bursty_none.eval_model_logL(p, self.LIMITS, None, data, self.N_CELLS)
        np.testing.assert_allclose(logl, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# eval_model_noise — fractions sum to 1; snapshots
# ---------------------------------------------------------------------------


class TestEvalModelNoise:
    def test_constitutive_fracs_sum_to_one(self, constitutive_none):
        fracs = constitutive_none.eval_model_noise(np.array([0.0, 0.0]))
        np.testing.assert_allclose(
            np.sum(np.array(fracs), axis=0), np.ones(2), atol=1e-10
        )

    def test_bursty_fracs_sum_to_one(self, bursty_none):
        fracs = bursty_none.eval_model_noise(np.array([0.3, 0.0, -0.1]))
        np.testing.assert_allclose(
            np.sum(np.array(fracs), axis=0), np.ones(2), atol=1e-10
        )

    def test_extrinsic_fracs_sum_to_one(self, extrinsic_none):
        fracs = extrinsic_none.eval_model_noise(np.array([0.5, 0.0, 0.0]))
        np.testing.assert_allclose(
            np.sum(np.array(fracs), axis=0), np.ones(2), atol=1e-10
        )

    def test_cir_fracs_sum_to_one(self, cir_none):
        fracs = cir_none.eval_model_noise(np.array([0.3, 0.0, 0.0]))
        np.testing.assert_allclose(
            np.sum(np.array(fracs), axis=0), np.ones(2), atol=1e-10
        )

    def test_bursty_poisson_fracs_sum_to_one(self, bursty_poisson):
        fracs = bursty_poisson.eval_model_noise(
            np.array([0.3, 0.0, -0.1]), samp=np.array([-6.0, -6.0])
        )
        np.testing.assert_allclose(
            np.sum(np.array(fracs), axis=0), np.ones(2), atol=1e-10
        )

    def test_bursty_bernoulli_fracs_sum_to_one(self, bursty_bernoulli):
        fracs = bursty_bernoulli.eval_model_noise(
            np.array([0.3, 0.0, -0.1]), samp=np.array([-1.0, -1.0])
        )
        np.testing.assert_allclose(
            np.sum(np.array(fracs), axis=0), np.ones(2), atol=1e-10
        )

    def test_constitutive_none_has_no_extrinsic(self, constitutive_none):
        """Constitutive model has zero extrinsic noise."""
        fracs = constitutive_none.eval_model_noise(np.array([0.0, 0.0]))
        noise_ext = np.array(fracs)[1]  # index 1 = extrinsic fraction
        np.testing.assert_allclose(noise_ext, np.zeros(2), atol=1e-10)

    def test_constitutive_none_snapshot(self, constitutive_none):
        fracs = constitutive_none.eval_model_noise(np.array([0.0, 0.0]))
        check_snapshot("eval_model_noise_constitutive_none", np.array(fracs))

    def test_bursty_none_snapshot(self, bursty_none):
        fracs = bursty_none.eval_model_noise(np.array([0.3, 0.0, -0.1]))
        check_snapshot("eval_model_noise_bursty_none", np.array(fracs))

    def test_extrinsic_none_snapshot(self, extrinsic_none):
        fracs = extrinsic_none.eval_model_noise(np.array([0.5, 0.0, 0.0]))
        check_snapshot("eval_model_noise_extrinsic_none", np.array(fracs))

    def test_cir_none_snapshot(self, cir_none):
        fracs = cir_none.eval_model_noise(np.array([0.3, 0.0, 0.0]))
        check_snapshot("eval_model_noise_cir_none", np.array(fracs))

    def test_bursty_poisson_snapshot(self, bursty_poisson):
        fracs = bursty_poisson.eval_model_noise(
            np.array([0.3, 0.0, -0.1]), samp=np.array([-6.0, -6.0])
        )
        check_snapshot("eval_model_noise_bursty_poisson", np.array(fracs))

    def test_bursty_bernoulli_snapshot(self, bursty_bernoulli):
        fracs = bursty_bernoulli.eval_model_noise(
            np.array([0.3, 0.0, -0.1]), samp=np.array([-1.0, -1.0])
        )
        check_snapshot("eval_model_noise_bursty_bernoulli", np.array(fracs))


# ---------------------------------------------------------------------------
# get_MoM — output length, bounds adherence, and snapshots
#
# Note: get_MoM always calls `samp = 10**samp` regardless of seq_model due to
# a condition bug (`if self.seq_model == "Poisson" or "Bernoulli":` is always
# True).  A dummy samp array is passed for all models to avoid a TypeError
# when samp=None.  For seq_model="None" models the samp value does not affect
# b/beta/gamma outputs because the samp-scaling branches are gated on seq_model.
# ---------------------------------------------------------------------------

_DUMMY_SAMP = np.array([-6.0, -6.0])  # ignored for seq_model="None"


def _bursty_moments():
    return {
        "MOM_unspliced_mean": 2.0,
        "MOM_unspliced_var": 8.0,
        "MOM_spliced_mean": 3.0,
        "MOM_spliced_var": 12.0,
    }


def _constitutive_moments():
    return {
        "MOM_unspliced_mean": 2.0,
        "MOM_unspliced_var": 2.0,
        "MOM_spliced_mean": 3.0,
        "MOM_spliced_var": 3.0,
    }


def _extrinsic_moments():
    return {
        "MOM_unspliced_mean": 2.0,
        "MOM_unspliced_var": 8.0,
        "MOM_spliced_mean": 3.0,
        "MOM_spliced_var": 12.0,
    }


class TestGetMoM:
    def test_bursty_none_output_length(self, bursty_none):
        m = bursty_none
        lb = np.array(m.bio_bounds["phys_lb"])
        ub = np.array(m.bio_bounds["phys_ub"])
        x0 = m.get_MoM(_bursty_moments(), lb, ub, samp=_DUMMY_SAMP)
        assert len(x0) == m.get_num_params()

    def test_constitutive_none_output_length(self, constitutive_none):
        m = constitutive_none
        lb = np.array(m.bio_bounds["phys_lb"])
        ub = np.array(m.bio_bounds["phys_ub"])
        x0 = m.get_MoM(_constitutive_moments(), lb, ub, samp=_DUMMY_SAMP)
        assert len(x0) == m.get_num_params()

    def test_extrinsic_none_output_length(self, extrinsic_none):
        m = extrinsic_none
        lb = np.array(m.bio_bounds["phys_lb"])
        ub = np.array(m.bio_bounds["phys_ub"])
        x0 = m.get_MoM(_extrinsic_moments(), lb, ub, samp=_DUMMY_SAMP)
        assert len(x0) == m.get_num_params()

    def test_bursty_none_within_bounds(self, bursty_none):
        m = bursty_none
        lb = np.array(m.bio_bounds["phys_lb"])
        ub = np.array(m.bio_bounds["phys_ub"])
        x0 = m.get_MoM(_bursty_moments(), lb, ub, samp=_DUMMY_SAMP)
        assert np.all(x0 >= lb)
        assert np.all(x0 <= ub)

    def test_constitutive_none_within_bounds(self, constitutive_none):
        m = constitutive_none
        lb = np.array(m.bio_bounds["phys_lb"])
        ub = np.array(m.bio_bounds["phys_ub"])
        x0 = m.get_MoM(_constitutive_moments(), lb, ub, samp=_DUMMY_SAMP)
        assert np.all(x0 >= lb)
        assert np.all(x0 <= ub)

    def test_bursty_none_finite(self, bursty_none):
        m = bursty_none
        lb = np.array(m.bio_bounds["phys_lb"])
        ub = np.array(m.bio_bounds["phys_ub"])
        x0 = m.get_MoM(_bursty_moments(), lb, ub, samp=_DUMMY_SAMP)
        assert np.all(np.isfinite(x0))

    def test_bursty_none_snapshot(self, bursty_none):
        m = bursty_none
        lb = np.array(m.bio_bounds["phys_lb"])
        ub = np.array(m.bio_bounds["phys_ub"])
        x0 = m.get_MoM(_bursty_moments(), lb, ub, samp=_DUMMY_SAMP)
        check_snapshot("get_mom_bursty_none", x0)

    def test_constitutive_none_snapshot(self, constitutive_none):
        m = constitutive_none
        lb = np.array(m.bio_bounds["phys_lb"])
        ub = np.array(m.bio_bounds["phys_ub"])
        x0 = m.get_MoM(_constitutive_moments(), lb, ub, samp=_DUMMY_SAMP)
        check_snapshot("get_mom_constitutive_none", x0)

    def test_extrinsic_none_snapshot(self, extrinsic_none):
        m = extrinsic_none
        lb = np.array(m.bio_bounds["phys_lb"])
        ub = np.array(m.bio_bounds["phys_ub"])
        x0 = m.get_MoM(_extrinsic_moments(), lb, ub, samp=_DUMMY_SAMP)
        check_snapshot("get_mom_extrinsic_none", x0)

    def test_bursty_poisson_snapshot(self, bursty_poisson):
        m = bursty_poisson
        lb = np.array(m.bio_bounds["phys_lb"])
        ub = np.array(m.bio_bounds["phys_ub"])
        samp = np.array([-6.0, -6.0])
        x0 = m.get_MoM(_bursty_moments(), lb, ub, samp=samp)
        check_snapshot("get_mom_bursty_poisson", x0)


# ---------------------------------------------------------------------------
# Rust / Python parity — Rust and Python paths must agree to rtol=1e-5
# ---------------------------------------------------------------------------


def _eval_pss_python(model, p, limits, samp=None):
    """Call eval_model_pss with Rust disabled, returning the pure-Python result."""
    with unittest.mock.patch.object(_cme_toolbox_module, "_HAS_RUST", False):
        return model.eval_model_pss(p, limits, samp=samp)


class TestRustPythonParity:
    """Verify that the Rust fast-path produces the same PSS as the Python path.

    seq_model="None" Rust paths: Bursty, CIR (quadrature models only;
    analytical models use Python/scipy which is faster at small grid sizes).
    seq_model="Poisson" Rust paths: Bursty, CIR.
    seq_model="Bernoulli": no Rust path (Python only).
    """

    LIMITS = [20, 20]
    _SAMP_POISSON = np.array([-6.0, -6.0])

    def test_constitutive_none_parity(self, constitutive_none):
        p = np.array([0.0, 0.0])
        rust = constitutive_none.eval_model_pss(p, self.LIMITS)
        py = _eval_pss_python(constitutive_none, p, self.LIMITS)
        np.testing.assert_allclose(rust, py, rtol=1e-5, atol=1e-10,
                                   err_msg="Constitutive/None Rust vs Python mismatch")

    def test_bursty_none_parity(self, bursty_none):
        p = np.array([0.3, 0.0, -0.1])
        rust = bursty_none.eval_model_pss(p, self.LIMITS)
        py = _eval_pss_python(bursty_none, p, self.LIMITS)
        np.testing.assert_allclose(rust, py, rtol=1e-5, atol=1e-10,
                                   err_msg="Bursty/None Rust vs Python mismatch")

    def test_extrinsic_none_parity(self, extrinsic_none):
        p = np.array([0.5, 0.0, 0.0])
        rust = extrinsic_none.eval_model_pss(p, self.LIMITS)
        py = _eval_pss_python(extrinsic_none, p, self.LIMITS)
        np.testing.assert_allclose(rust, py, rtol=1e-5, atol=1e-10,
                                   err_msg="Extrinsic/None Rust vs Python mismatch")

    def test_delay_none_parity(self, delay_none):
        p = np.array([0.3, 0.0, 0.0])
        rust = delay_none.eval_model_pss(p, self.LIMITS)
        py = _eval_pss_python(delay_none, p, self.LIMITS)
        np.testing.assert_allclose(rust, py, rtol=1e-5, atol=1e-10,
                                   err_msg="Delay/None Rust vs Python mismatch")

    def test_cir_none_parity(self, cir_none):
        p = np.array([0.3, 0.0, 0.0])
        rust = cir_none.eval_model_pss(p, self.LIMITS)
        py = _eval_pss_python(cir_none, p, self.LIMITS)
        np.testing.assert_allclose(rust, py, rtol=1e-5, atol=1e-10,
                                   err_msg="CIR/None Rust vs Python mismatch")

    def test_delayed_splicing_none_parity(self, delayed_splicing_none):
        p = np.array([0.3, 0.0, 0.0])
        rust = delayed_splicing_none.eval_model_pss(p, self.LIMITS)
        py = _eval_pss_python(delayed_splicing_none, p, self.LIMITS)
        np.testing.assert_allclose(rust, py, rtol=1e-5, atol=1e-10,
                                   err_msg="DelayedSplicing/None Rust vs Python mismatch")

    # --- Poisson seq_model (Rust path: Bursty and CIR only) ---

    def test_bursty_poisson_parity(self, bursty_poisson):
        p = np.array([0.3, 0.0, -0.1])
        rust = bursty_poisson.eval_model_pss(p, self.LIMITS, samp=self._SAMP_POISSON)
        py = _eval_pss_python(bursty_poisson, p, self.LIMITS, samp=self._SAMP_POISSON)
        np.testing.assert_allclose(rust, py, rtol=1e-5, atol=1e-10,
                                   err_msg="Bursty/Poisson Rust vs Python mismatch")

    def test_cir_poisson_parity(self):
        model = CMEModel("CIR", "Poisson")
        p = np.array([0.3, 0.0, 0.0])
        rust = model.eval_model_pss(p, self.LIMITS, samp=self._SAMP_POISSON)
        py = _eval_pss_python(model, p, self.LIMITS, samp=self._SAMP_POISSON)
        np.testing.assert_allclose(rust, py, rtol=1e-5, atol=1e-10,
                                   err_msg="CIR/Poisson Rust vs Python mismatch")
