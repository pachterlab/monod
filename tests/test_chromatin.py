"""The chromatin (Markov-modulated transcription) kernel against the authors'
reference code (tests/fgp_reference.py, ported from pachterlab/FGP_2023)."""
import numpy as np
import pytest

from monod import chromatin as ch
from monod.cme_toolbox import CMEModel
import fgp_reference as ref


def _find_koff(e, kon, p1):
    # FGP's helper: the shared k_off giving each of two coupled sites probability p1 of being open
    return 1 / p1 * 0.5 * (e * kon - 2 * e * kon * p1 + ((e * kon - 2 * e * kon * p1) ** 2 + 4 * p1 * (kon ** 2 - kon ** 2 * p1)) ** 0.5)


def _perm_to_fgp(n, combos):
    """Index into monod's binary-order configurations for each FGP basis entry."""
    return np.array([int(sum(bit << j for j, bit in enumerate(c))) for c in combos])


TWO_GENE = dict(k_ons=[0.1, 0.1], epsilon=0.3, bs=[10.0, 15.0], ds=[5.0, 7.0])
TWO_GENE["k_off"] = _find_koff(0.3, 0.1, 0.4)


class TestLocus:
    @pytest.mark.parametrize("n,eps,periodic", [(2, 0.3, False), (3, 0.5, False), (4, 1.0, False), (4, 0.7, True)])
    def test_transition_matrix_and_steady_state_match_fgp(self, n, eps, periodic):
        rng = np.random.default_rng(n)
        k_on = rng.uniform(0.2, 2.0, n); k_off = 0.8
        H_ref, combos = ref.construct_H(list(k_on), k_off, eps, periodic)
        perm = _perm_to_fgp(n, combos)
        H = ch.transition_matrix(k_on, k_off, eps, periodic)
        np.testing.assert_allclose(H[np.ix_(perm, perm)], H_ref, rtol=1e-12, atol=1e-12)
        w_ref, _ = ref.ising_steady_state(list(k_on), k_off, eps, periodic)
        np.testing.assert_allclose(ch.ising_steady_state(k_on, k_off, eps, periodic)[perm], w_ref, rtol=1e-12)
        np.testing.assert_allclose(ch.stationary_distribution(H), ch.ising_steady_state(k_on, k_off, eps, periodic), atol=1e-12)

    def test_dropout_matrix_is_stochastic_and_thins_means(self):
        n, p = 3, 0.3
        D = ch.dropout_matrix(n, p)
        np.testing.assert_allclose(D.sum(1), 1.0)
        w = ch.ising_steady_state([0.5, 1.0, 2.0], 1.0, 0.6)
        configs = ch.configurations(n)
        np.testing.assert_allclose((w @ D) @ configs, (1 - p) * (w @ configs), rtol=1e-12)


class TestJointDistribution:
    def test_matches_fgp_ode_solution(self):
        H_ref, combos = ref.construct_H(TWO_GENE["k_ons"], TWO_GENE["k_off"], TWO_GENE["epsilon"])
        A_ref = ref.create_A(combos, TWO_GENE["bs"])
        C = -np.diag(TWO_GENE["ds"])
        mx = [12, 14]
        T = 60.0 / min(TWO_GENE["ds"])
        pss_ref = ref.pss_wrapper_n(H_ref, A_ref, C, T, mx)          # (12, 14, N), FGP order
        H, A, C2 = ch.fgp_system(TWO_GENE["k_ons"], TWO_GENE["k_off"], TWO_GENE["epsilon"], TWO_GENE["bs"], TWO_GENE["ds"])
        pss = ch.eval_joint_pss(H, A, C2, mx)                        # (N, 12, 14), binary order
        perm = _perm_to_fgp(2, combos)
        np.testing.assert_allclose(np.moveaxis(pss[perm], 0, -1), pss_ref / pss_ref.sum(), atol=2e-6)

    def test_matches_fgp_closed_form_moments(self):
        H_ref, combos = ref.construct_H(TWO_GENE["k_ons"], TWO_GENE["k_off"], TWO_GENE["epsilon"])
        m_ref = ref.calc_moments_generic(H_ref, combos, TWO_GENE["bs"], TWO_GENE["ds"], (0, 1))
        H, A, C = ch.fgp_system(TWO_GENE["k_ons"], TWO_GENE["k_off"], TWO_GENE["epsilon"], TWO_GENE["bs"], TWO_GENE["ds"])
        pss = ch.eval_joint_pss(H, A, C, [40, 40])
        mu, cov = ch.moments(pss, 2)              # variables: site0, site1, x0, x1
        assert mu[0] == pytest.approx(m_ref["site_mean_1"], rel=1e-6)
        assert mu[1] == pytest.approx(m_ref["site_mean_2"], rel=1e-6)
        assert cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]) == pytest.approx(m_ref["site_corr"], rel=1e-6)
        assert mu[2] == pytest.approx(m_ref["transcript_mean_1"], rel=1e-5)
        assert mu[3] == pytest.approx(m_ref["transcript_mean_2"], rel=1e-5)
        assert cov[2, 2] == pytest.approx(m_ref["transcript_var_1"], rel=1e-4)
        assert cov[3, 3] == pytest.approx(m_ref["transcript_var_2"], rel=1e-4)
        assert cov[2, 3] == pytest.approx(m_ref["transcript_covar"], rel=1e-3, abs=1e-6)

    def test_configuration_marginal_is_the_ising_steady_state(self):
        H, A, C = ch.fgp_system([0.3, 0.6, 0.2], 0.5, 0.4, [4.0, 2.0, 3.0], [1.0, 1.0, 1.0])
        pss = ch.eval_joint_pss(H, A, C, [20, 20, 20])
        np.testing.assert_allclose(pss.sum(axis=(1, 2, 3)), ch.ising_steady_state([0.3, 0.6, 0.2], 0.5, 0.4), atol=1e-6)

    def test_gillespie(self):
        H, A, C = ch.fgp_system(TWO_GENE["k_ons"], TWO_GENE["k_off"], TWO_GENE["epsilon"], TWO_GENE["bs"], TWO_GENE["ds"])
        rng = np.random.default_rng(0)
        n_cells = 3000
        states, counts = ref.gillespie(H, A, np.asarray(TWO_GENE["ds"]), 6.0 / min(TWO_GENE["ds"]), n_cells, rng)
        pss = ch.eval_joint_pss(H, A, C, [25, 25])
        mu, cov = ch.moments(pss, 2)
        configs = ch.configurations(2)
        sim = np.column_stack([configs[states], counts]).astype(float)
        se = sim.std(0) / np.sqrt(n_cells)
        assert np.all(np.abs(sim.mean(0) - mu) < 4 * se + 1e-9), (sim.mean(0), mu, se)
        # configuration frequencies against the joint marginal
        freq = np.bincount(states, minlength=4) / n_cells
        assert np.abs(freq - pss.sum(axis=(1, 2))).max() < 4 * np.sqrt(0.25 / n_cells)


class TestLimits:
    def test_one_always_open_site_is_the_constitutive_model(self):
        beta, gamma = 0.7, 0.3
        H, A, C = ch.nascent_mature_system([1.0], 1e-9, 1.0, lambda s: 1.0 * s.sum(), beta, gamma)
        pss = ch.eval_joint_pss(H, A, C, [30, 40])
        ref_pss = CMEModel("Constitutive", "None").eval_model_pss(np.log10([beta, gamma]), [30, 40])
        np.testing.assert_allclose(pss.sum(0), ref_pss, atol=1e-6)
        assert pss[1].sum() > 1 - 1e-6                                  # the open configuration carries the mass

    def test_one_always_open_site_with_bursts_is_the_bursty_model(self):
        """Mean production rate b at burst size b is burst frequency 1, monod's
        Bursty model in its own time unit."""
        b, beta, gamma = 3.0, 0.7, 0.3
        H, A, C = ch.nascent_mature_system([1.0], 1e-9, 1.0, lambda s: b * s.sum(), beta, gamma)
        ref_pss = CMEModel("Bursty", "None").eval_model_pss(np.log10([b, beta, gamma]), [40, 50])
        pss = ch.eval_joint_pss(H, A, C, [40, 50], burst=b, backend="python")
        np.testing.assert_allclose(pss.sum(0), ref_pss, atol=1e-6)

    def test_small_bursts_are_the_constitutive_model_and_large_ones_overdisperse(self):
        H, A, C = ch.nascent_mature_system([0.5, 0.8], 1.0, 0.6, "additive", 0.8, 0.3, rule_params=[4.0, 2.0])
        base = ch.eval_joint_pss(H, A, C, [40, 50], backend="python")
        small = ch.eval_joint_pss(H, A, C, [40, 50], burst=1e-4, backend="python")
        np.testing.assert_allclose(small, base, atol=1e-5)
        big = ch.eval_joint_pss(H, A, C, [40, 50], burst=3.0, backend="python")
        mu0, cov0 = ch.moments(base, 2)
        mu1, cov1 = ch.moments(big, 2)
        np.testing.assert_allclose(mu1, mu0, rtol=2e-2)                 # same configuration and RNA means
        assert cov1[2, 2] > 1.5 * cov0[2, 2] and cov1[3, 3] > 1.5 * cov0[3, 3]

    def test_single_site_is_the_telegraph_model(self):
        k_on, k_off, b, d = 0.4, 0.9, 6.0, 1.0
        H, A, C = ch.fgp_system([k_on], k_off, 1.0, [b], [d])
        pss = ch.eval_joint_pss(H, A, C, [60])
        mu, cov = ch.moments(pss, 1)
        mean = b / d * k_on / (k_on + k_off)
        fano = 1 + b * k_off / ((k_on + k_off) * (k_on + k_off + d))
        assert mu[1] == pytest.approx(mean, rel=1e-6)
        assert cov[1, 1] / mu[1] == pytest.approx(fano, rel=1e-5)

    def test_schedule_replays_exactly(self):
        H, A, C = ch.fgp_system([0.3, 0.6], 0.5, 0.4, [4.0, 2.0], [1.0, 1.0])
        pss, sched = ch.eval_joint_pss(H, A, C, [20, 20], return_schedule=True)
        assert 20 < sched.size < 2000
        again = ch.eval_joint_pss(H, A, C, [20, 20], fixed_schedule=sched)
        np.testing.assert_allclose(pss, again, atol=1e-14)   # the replayed horizon is a float sum of the steps

    def test_replay_with_perturbed_parameters_reaches_time_zero(self):
        """A replayed schedule integrates over its own horizon, so a stencil
        evaluation at perturbed parameters matches a free one closely."""
        H, A, C = ch.nascent_mature_system([0.5, 0.8], 1.0, 0.6, "additive", 0.8, 0.3, rule_params=[4.0, 2.0])
        base, sched = ch.eval_joint_pss(H, A, C, [20, 30], return_schedule=True)
        H2, A2, C2 = ch.nascent_mature_system([0.5, 0.8], 1.0, 0.6, "additive", 0.8, 0.3 * 10 ** 1e-3, rule_params=[4.0, 2.0])
        pinned = ch.eval_joint_pss(H2, A2, C2, [20, 30], fixed_schedule=sched)
        free = ch.eval_joint_pss(H2, A2, C2, [20, 30], tol=1e-9)
        assert np.abs(pinned - free).max() < 1e-6
        # and the perturbation itself is resolved: the pinned difference tracks the free one
        d_pinned = pinned - base
        d_free = free - ch.eval_joint_pss(H, A, C, [20, 30], tol=1e-9)
        assert np.abs(d_pinned - d_free).max() < 0.05 * np.abs(d_free).max()

    def test_dropout_per_site(self):
        H, A, C = ch.fgp_system([0.3, 0.6, 0.2], 0.5, 0.4, [4.0, 2.0, 3.0], [1.0, 1.0, 1.0])
        pss = ch.eval_joint_pss(H, A, C, [12, 12, 12])
        p = np.array([0.1, 0.4, 0.25])
        obs = ch.apply_dropout(pss, 3, p)
        assert obs.sum() == pytest.approx(1.0)
        mu, _ = ch.moments(pss, 3); mo, _ = ch.moments(obs, 3)
        np.testing.assert_allclose(mo[:3], (1 - p) * mu[:3], rtol=1e-10)
        np.testing.assert_allclose(mo[3:], mu[3:], rtol=1e-10)

    def test_dropout_on_the_joint(self):
        H, A, C = ch.fgp_system([0.3, 0.6], 0.5, 0.4, [4.0, 2.0], [1.0, 1.0])
        pss = ch.eval_joint_pss(H, A, C, [20, 20])
        obs = ch.apply_dropout(pss, 2, 0.25)
        assert obs.sum() == pytest.approx(1.0)
        mu, _ = ch.moments(pss, 2); mo, _ = ch.moments(obs, 2)
        np.testing.assert_allclose(mo[:2], 0.75 * mu[:2], rtol=1e-10)
        np.testing.assert_allclose(mo[2:], mu[2:], rtol=1e-10)         # RNA untouched


class TestTranscriptionRules:
    """The three rules for how open sites set a gene's transcription rate, and
    the free per-configuration table that nests them."""

    def test_rules_and_nesting(self):
        n = 3
        configs = ch.configurations(n)
        w = np.array([1.0, 2.5, 0.4])
        add = ch.transcription_rates("additive", n, w)
        np.testing.assert_allclose(add, configs @ w)
        sw = ch.transcription_rates("switch", n, 3.0)
        np.testing.assert_allclose(sw, np.where(configs.any(1), 3.0, 0.0))
        assert sw[0] == 0 and add[0] == 0                              # closed locus is silent
        # the table rule reproduces both, parameter counts as documented
        np.testing.assert_array_equal(ch.transcription_rates("table", n, add), add)
        np.testing.assert_array_equal(ch.rule_as_table("switch", n, [3.0]), sw)
        assert [ch.n_rule_params(r, n) for r in ch.RULES] == [3, 1, 8]
        # a callable and a raw table are accepted too
        np.testing.assert_array_equal(ch.transcription_rates(lambda s: 2.0 * s.sum(), n), 2 * configs.sum(1))
        np.testing.assert_array_equal(ch.transcription_rates(add, n), add)
        with pytest.raises(ValueError):
            ch.transcription_rates("additive", n, [1.0, 2.0])
        with pytest.raises(ValueError):
            ch.transcription_rates("gate", n, [1.0])

    def test_one_site_rules_coincide(self):
        k_on, k_off, b, beta, gamma = 0.4, 1.0, 5.0, 0.8, 0.3
        out = []
        for rule, prm in [("additive", [b]), ("switch", [b]), ("table", [0.0, b])]:
            H, A, C = ch.nascent_mature_system([k_on], k_off, 1.0, rule, beta, gamma, rule_params=prm)
            out.append(ch.eval_joint_pss(H, A, C, [30, 40]))
        np.testing.assert_allclose(out[0], out[1], atol=1e-12)
        np.testing.assert_allclose(out[0], out[2], atol=1e-12)

    def test_rules_differ_with_two_sites(self):
        k_on, k_off, eps, b, beta, gamma = [0.5, 0.5], 1.0, 0.6, 4.0, 0.8, 0.3
        H, A, C = ch.nascent_mature_system(k_on, k_off, eps, "additive", beta, gamma, rule_params=[b, b])
        p_add = ch.eval_joint_pss(H, A, C, [40, 50])
        H, A, C = ch.nascent_mature_system(k_on, k_off, eps, "switch", beta, gamma, rule_params=[b])
        p_sw = ch.eval_joint_pss(H, A, C, [40, 50])
        mu_a, _ = ch.moments(p_add, 2); mu_s, _ = ch.moments(p_sw, 2)
        np.testing.assert_allclose(mu_a[:2], mu_s[:2])                 # same chromatin
        assert mu_a[2] > mu_s[2] and mu_a[3] > mu_s[3]                 # additive transcribes more when both open
        assert np.abs(p_add - p_sw).max() > 1e-3

    def test_paper_model_summed_is_the_additive_rule(self):
        """Summing the paper's per-site transcripts with one degradation rate
        gives the additive rule for a single degrading species."""
        k_on, k_off, eps, bs, d = [0.3, 0.6], 0.5, 0.4, [4.0, 2.0], 1.0
        H, A, C = ch.fgp_system(k_on, k_off, eps, bs, [d, d])
        joint = ch.eval_joint_pss(H, A, C, [40, 40])
        N, l1, l2 = joint.shape
        summed = np.zeros((N, l1 + l2 - 1))
        for i in range(l1):
            summed[:, i:i + l2] += joint[:, i, :]
        H, A, C = ch.transition_matrix(k_on, k_off, eps), None, None
        A = ch.transcription_rates("additive", 2, bs)[:, None]
        single = ch.eval_joint_pss(H, A, -np.array([[d]]), [l1 + l2 - 1])
        np.testing.assert_allclose(summed, single, atol=1e-6)


needs_rust = pytest.mark.skipif(not ch._rust_available(), reason="monod_core without eval_chromatin_pgf")


@needs_rust
class TestRustBackend:
    """The Rust kernel follows the NumPy one step for step."""

    @pytest.mark.parametrize("case", ["fgp2", "nm3", "nm1"])
    def test_matches_numpy(self, case):
        if case == "fgp2":
            H, A, C = ch.fgp_system(TWO_GENE["k_ons"], TWO_GENE["k_off"], TWO_GENE["epsilon"], TWO_GENE["bs"], TWO_GENE["ds"])
            limits = [24, 30]
        elif case == "nm3":
            H, A, C = ch.nascent_mature_system([0.3, 0.6, 0.2], 1.0, 0.4, "additive", 0.8, 0.3, rule_params=[4.0, 2.0, 3.0])
            limits = [40, 60]
        else:
            H, A, C = ch.nascent_mature_system([0.4], 1.0, 1.0, "switch", 0.8, 0.3, rule_params=[5.0])
            limits = [30, 40]
        p_py, s_py = ch.eval_joint_pss(H, A, C, limits, backend="python", return_schedule=True)
        p_rs, s_rs = ch.eval_joint_pss(H, A, C, limits, backend="rust", return_schedule=True)
        assert s_rs.size == s_py.size
        np.testing.assert_allclose(np.asarray(s_rs), np.asarray(s_py), rtol=1e-6)   # controller rounding
        np.testing.assert_allclose(p_rs, p_py, atol=1e-12)

    @pytest.mark.parametrize("case", ["switch", "additive", "table", "independent"])
    def test_bursty_matches_numpy_in_every_mode(self, case):
        """Bursts through the distinct-rate, factored, general and Kronecker paths."""
        eps = 1.0 if case == "independent" else 0.6
        rule = "additive" if case == "independent" else case
        rates = {"switch": [5.0], "additive": [4.0, 2.0], "table": [0.3, 4.0, 2.0, 1.0]}[rule]
        H, A, C = ch.nascent_mature_system([0.5, 0.8], 1.0, eps, rule, 0.8, 0.3, rule_params=rates)
        p_py, s_py = ch.eval_joint_pss(H, A, C, [30, 40], burst=2.5, backend="python", return_schedule=True)
        p_rs, s_rs = ch.eval_joint_pss(H, A, C, [30, 40], burst=2.5, backend="rust", return_schedule=True)
        assert s_rs.size == s_py.size
        np.testing.assert_allclose(np.asarray(s_rs), np.asarray(s_py), rtol=1e-6)
        np.testing.assert_allclose(p_rs, p_py, atol=1e-12)
        replay = ch.eval_joint_pss(H, A, C, [30, 40], burst=2.5, fixed_schedule=s_py, backend="rust")
        np.testing.assert_allclose(replay, p_py, atol=1e-12)
        with pytest.raises(ValueError):
            ch.eval_joint_pss(H, A, C, [30, 40], burst=-1.0, backend="rust")

    def test_default_backend_is_rust_and_replays_python_schedule(self):
        H, A, C = ch.fgp_system([0.3, 0.6], 0.5, 0.4, [4.0, 2.0], [1.0, 1.0])
        p_py, s_py = ch.eval_joint_pss(H, A, C, [20, 20], backend="python", return_schedule=True)
        p_default = ch.eval_joint_pss(H, A, C, [20, 20])
        np.testing.assert_allclose(p_default, p_py, atol=1e-12)
        replay = ch.eval_joint_pss(H, A, C, [20, 20], fixed_schedule=s_py, backend="rust")
        np.testing.assert_allclose(replay, p_py, atol=1e-12)

    def test_parallel_path_matches(self):
        """A grid large enough for the parallel blocks gives the same result."""
        H, A, C = ch.nascent_mature_system([0.3, 0.6, 0.2, 0.5], 1.0, 0.5, "additive", 0.8, 0.3, rule_params=[2.0, 1.0, 1.5, 0.5])
        limits = [30, 40]                                   # 16 states × 30 × 21 points > 4096
        p_py = ch.eval_joint_pss(H, A, C, limits, backend="python")
        p_rs = ch.eval_joint_pss(H, A, C, limits, backend="rust")
        np.testing.assert_allclose(p_rs, p_py, atol=1e-12)

    @pytest.mark.parametrize("case", ["switch", "additive", "table", "fgp_per_site", "independent", "independent_additive"])
    def test_structured_paths_match_dense(self, case):
        """Distinct-rate, factored and Kronecker paths agree with the general NumPy path."""
        if case == "switch":
            H, A, C = ch.nascent_mature_system([0.5, 0.8, 0.3], 1.0, 0.6, "switch", 0.8, 0.3, rule_params=[5.0])
            expect = (None, None)
        elif case == "additive":
            H, A, C = ch.nascent_mature_system([0.5, 0.8, 0.3], 1.0, 0.6, "additive", 0.8, 0.3, rule_params=[4.0, 2.0, 1.0])
            expect = ("W", None)
        elif case == "table":
            H, A, C = ch.nascent_mature_system([0.5, 0.8], 1.0, 0.6, "table", 0.8, 0.3, rule_params=[0.0, 3.0, 3.0, 1.0])
            expect = (None, None)
        elif case == "fgp_per_site":
            H, A, C = ch.fgp_system([0.3, 0.6], 0.5, 0.4, [4.0, 2.0], [1.0, 1.5])
            expect = ("W", None)
        elif case == "independent":
            H, A, C = ch.nascent_mature_system([0.5, 0.8, 0.3, 1.2], 1.0, 1.0, "table", 0.8, 0.3, rule_params=np.sqrt(np.arange(16)))
            expect = (None, "K")
        else:
            H, A, C = ch.nascent_mature_system([0.5, 0.8, 0.3, 1.2], 1.0, 1.0, "additive", 0.8, 0.3, rule_params=[1.0, 2.0, 0.5, 1.5])
            expect = ("W", "K")
        sw, ind = ch.locus_structure(H, A)
        assert (sw is not None) == (expect[0] is not None)
        assert (ind is not None) == (expect[1] is not None)
        limits = [30, 40]
        p_py = ch.eval_joint_pss(H, A, C, limits, backend="python")
        p_rs = ch.eval_joint_pss(H, A, C, limits, backend="rust")
        np.testing.assert_allclose(p_rs, p_py, atol=1e-12)

    def test_locus_structure_rejects_coupled_sites(self):
        H, A, C = ch.nascent_mature_system([0.5, 0.8], 1.0, 0.6, "additive", 0.8, 0.3, rule_params=[4.0, 2.0])
        sw, ind = ch.locus_structure(H, A)
        assert sw is not None and ind is None

    @pytest.mark.parametrize("rule, eps, params", [
        ("additive", 0.6, [4.0, 2.0, 1.0]), ("additive", 1.0, [1.0, 2.0, 0.5]),
        ("switch", 0.6, [5.0]), ("switch", 1.0, [5.0]),
        ("table", 0.6, [0.0, 3.0, 3.0, 1.0, 2.0, 1.0, 0.5, 4.0]), ("table", 1.0, np.sqrt(np.arange(8))),
    ])
    def test_rule_structure_matches_detection(self, rule, eps, params):
        """What the rule fixes agrees with what locus_structure detects from the matrices."""
        k_on = [0.5, 0.8, 0.3]
        H, A, C = ch.nascent_mature_system(k_on, 1.0, eps, rule, 0.8, 0.3, rule_params=params)
        sw_d, ind_d = ch.locus_structure(H, A)
        sw_r, ind_r = ch.rule_structure(H, A, k_on, 1.0, eps, rule, params)
        assert (sw_r is None) == (sw_d is None) and (ind_r is None) == (ind_d is None)
        if sw_d is not None:
            np.testing.assert_allclose(sw_r, sw_d, atol=1e-12)
        if ind_d is not None:
            np.testing.assert_allclose(ind_r, ind_d, rtol=1e-12)
        p_a = ch.eval_joint_pss(H, A, C, [20, 24], backend="rust")
        G, _ = ch.eval_joint_pgf(H, A, C, ch._mesh([20, 24])[0], structure=(sw_r, ind_r))
        p_b = ch.invert_joint_pgf(G, ch._mesh([20, 24])[1], (20, 24))
        np.testing.assert_allclose(p_b, p_a, atol=1e-13)

    def test_shape_errors(self):
        with pytest.raises(ValueError):
            ch._mc.eval_chromatin_pgf(2, 1, [0.0] * 4, [0.0] * 3, [0.0], [10], [0.5, 0.5], 1.0, 0.1, 1e-9, 1e-6)
