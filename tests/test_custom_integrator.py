"""The exponential integrator behind custom reaction networks.

Its cost must not grow with the spread of the rates, its result must agree
with closed forms, the recorded step schedule must replay exactly (gradient
and identifiability stencils depend on it), delays shorter than a step must
behave, and the Rust and Python kernels must agree to rounding.
"""
import numpy as np
import pytest

import monod.cme_toolbox as ct
from monod._core import _mc
from monod.cme_toolbox import CMEModel
from monod.reaction_network import ReactionNetwork

BURSTY = "0 -> B x U -> S -> 0"
UDEG = "0 -> B x U -> S -> 0; U -> 0"
CATALYTIC = "0 -> B x U -> S -> 0; S -> S + P; P -> 0"
DELAYED = "0 -> B x U => S -> 0"


def _custom(net):
    return CMEModel("Custom", "None", network=net, normalize_production_rate=True)


def _grid(l0, l1):
    mx = (l0, l1 // 2 + 1)
    u0 = np.exp(-2j * np.pi * np.arange(mx[0]) / l0) - 1
    u1 = np.exp(-2j * np.pi * np.arange(mx[1]) / l1) - 1
    g0, g1 = np.meshgrid(u0, u1, indexing="ij")
    return [g0.ravel(), g1.ravel()]


def _synthetic_data(model, p, limits):
    pss = model.eval_model_pss(p, limits)
    idx = np.argwhere(pss > 1e-6)
    f = pss[pss > 1e-6]
    return idx, f / f.sum()


class TestAccuracy:
    @pytest.mark.parametrize("p", [[3, 0.4, 0.25], [3, 100, 0.1], [3, 0.1, 100], [3, 30, 30]])
    def test_matches_closed_form_bursty(self, p):
        ref = CMEModel("Bursty", "None", quad_method="quad_vec", quad_vec_T=100)
        pss_ref = ref.eval_model_pss(np.log10(p), [60, 80])
        pss = _custom(BURSTY).eval_model_pss(np.log10(p), [60, 80])
        np.testing.assert_allclose(pss, pss_ref, atol=1e-6)

    def test_constitutive_network_gives_independent_poissons(self):
        from scipy.stats import poisson
        m = CMEModel("Custom", "None", network="0 -> U -> S -> 0")
        pss = m.eval_model_pss(np.log10([1.0, 0.4, 0.25]), [40, 60])
        np.testing.assert_allclose(pss.sum(1), poisson.pmf(np.arange(40), 2.5), atol=1e-6)
        np.testing.assert_allclose(pss.sum(0), poisson.pmf(np.arange(60), 4.0), atol=1e-6)

    def test_step_count_independent_of_rate_spread(self):
        m = _custom(BURSTY)
        counts = []
        for p in ([3, 0.4, 0.25], [3, 10, 0.01], [3, 100, 0.001], [3, 0.001, 100]):
            m.eval_model_pss(np.log10(p), [40, 50])
            counts.append(m._last_schedule.size)
        assert max(counts) < 400, counts


class TestSchedule:
    def test_replay_reproduces_result_exactly(self):
        m = _custom(UDEG)
        p, lim = np.log10([3, 0.4, 0.25, 0.15]), [30, 40]
        a = m.eval_model_pss(p, lim)
        sched = m._last_schedule.copy()
        assert sched.size > 10 and np.all(np.asarray(sched) > 0)
        m._fixed_schedule = sched
        try:
            b = m.eval_model_pss(p, lim)
        finally:
            m._fixed_schedule = None
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(m._last_schedule, sched)

    def test_python_eval_pgf_records_and_replays(self):
        net = ReactionNetwork(BURSTY, normalize_production_rate=True)
        g = _grid(12, 10)
        p = np.log10([2.0, 0.5, 0.8])
        phi = net.eval_pgf(p, g)
        sched = net.last_schedule.copy()
        half = net.eval_pgf(p, g, fixed_schedule=sched[: sched.size // 2])
        assert not np.allclose(half, phi)
        np.testing.assert_array_equal(net.eval_pgf(p, g, fixed_schedule=sched), phi)

    def test_identifiability_stencil_resolves_exact_degeneracy(self):
        # Every rate free: the time unit is unidentifiable.  With the schedule
        # pinned across the stencil the singular value sits at rounding level
        # instead of at the integrator's discretisation error.
        m = CMEModel("Custom", "None", network=BURSTY)
        r = m.check_identifiability([0.5, 0.0, 0.0, -0.3], [30, 25], warn=False)
        assert not r["identifiable"]
        assert r["singular_values"][-1] < 1e-6
        assert r["singular_values"][-2] > 1e-2

    def test_fd_gradient_matches_central_difference(self):
        m = _custom(UDEG)
        p0, lim = np.log10([3, 0.4, 0.25, 0.15]), [40, 50]
        data = _synthetic_data(m, p0 + [0.1, -0.1, 0.05, 0.1], lim)
        kld, grad = m.eval_model_kld_and_grad(p0, lim, None, data, 1e-15, eps=1e-6)
        central = []
        for i in range(4):
            e = np.zeros(4)
            e[i] = 1e-5
            central.append((m.eval_model_kld(p0 + e, lim, None, data, 1e-15)
                            - m.eval_model_kld(p0 - e, lim, None, data, 1e-15)) / 2e-5)
        np.testing.assert_allclose(grad, central, rtol=1e-3, atol=1e-6)


class TestDelays:
    def test_short_delay_approaches_immediate_network(self):
        now = _custom(BURSTY).eval_model_pss(np.log10([5, 1, 0.5]), [40, 50])
        m = _custom(DELAYED)
        diffs = []
        for tau in (1e-1, 1e-2, 1e-3):
            pss = m.eval_model_pss(np.log10([5, 1 / tau, 1, 0.5]), [40, 50])
            diffs.append(np.abs(pss - now).max())
            assert m._last_schedule.size < 500
        assert diffs[0] > diffs[1] > diffs[2] and diffs[2] < 5e-5, diffs

    def test_steps_land_on_the_delay(self):
        m = _custom(DELAYED)
        m.eval_model_pss(np.log10([5, 0.5, 1, 0.5]), [20, 20])    # tau = 2
        times = np.cumsum(m._last_schedule)
        assert np.min(np.abs(times - 2.0)) < 1e-9
        assert m._last_schedule.breakpoints == (2.0,)

    def test_replayed_schedule_is_warped_onto_the_new_delay(self):
        m = _custom(DELAYED)
        p, lim = np.log10([5, 0.5, 1, 0.5]), [20, 20]
        with m.pinned_schedule(p, lim):
            m.eval_model_pss(p + [0, 0.02, 0, 0], lim)            # tau = 2 / 10**0.02
            times = np.cumsum(m._last_schedule)
        assert np.min(np.abs(times - 2.0 / 10 ** 0.02)) < 1e-9

    def test_replayed_kld_is_smooth_in_the_delay(self):
        # Without the warp a replayed result is continuous but not
        # differentiable at the base delay, which a forward difference with a
        # 1e-6 step turns into an O(0.1) gradient error.
        m = _custom(DELAYED)
        p, lim = np.log10([14.0, 0.574, 10.8, 21.3]), [40, 50]
        data = _synthetic_data(m, p + [0.05, 0.1, -0.05, 0.05], lim)
        K = lambda q: m.eval_model_kld(q, lim, None, data, 1e-15)
        with m.pinned_schedule(p, lim):
            def slope(d):
                e = np.array([0, d, 0, 0.0])
                return (K(p + e) - K(p - e)) / (2 * d)
            s_small, s_big = slope(1e-6), slope(1e-3)
        assert abs(s_small - s_big) < 1e-3 * max(1.0, abs(s_big)), (s_small, s_big)


@pytest.mark.skipif(_mc is None, reason="monod_core not available")
class TestRustParity:
    @pytest.mark.parametrize("net,p,limits", [
        (CATALYTIC, np.log10([2, 1.5, 1, 1.2, 0.8]), [8, 8, 10]),
        (CATALYTIC, np.log10([2, 50, 1, 30, 0.05]), [6, 6, 8]),
        (UDEG, np.log10([3, 100, 0.001, 0.5]), [16, 14]),
        ("0 -> 3 x U -> S -> 0", np.log10([1, 0.3, 0.2]), [16, 14]),
    ])
    def test_rust_matches_python(self, net, p, limits, monkeypatch):
        m = CMEModel("Custom", "None", network=net, normalize_production_rate=("B x" in net))
        rust = m.eval_model_pss(p, limits)
        n_rust = m._last_schedule.size
        monkeypatch.setattr(ct, "_HAS_RUST", False)
        py = m.eval_model_pss(p, limits)
        np.testing.assert_allclose(rust, py, rtol=1e-9, atol=1e-12)
        assert m._last_schedule.size == n_rust

    def test_kernel_returns_schedule_and_replays_it(self):
        net = ReactionNetwork(BURSTY, normalize_production_rate=True)
        topo = net.rust_topology()
        params = [2.0, 0.5, 0.8, 1.0]                       # B, k1, k2, k0 (normalised)
        args = (2, [12, 10], topo["rxn_kinds"], topo["rxn_rate_idxs"], topo["rxn_extra1"],
                topo["rxn_extra2"], topo["prod_sp"], topo["prod_st"], topo["prod_off"], params)
        re1, im1, shape, steps = _mc.eval_custom_network_pgf(*args)
        assert len(steps) > 10 and shape == [12, 6]
        re2, im2, _, steps2 = _mc.eval_custom_network_pgf(*args, fixed_schedule=steps)
        assert steps2 == steps and re2 == re1 and im2 == im1


class TestPinnedSchedule:
    def test_block_pins_and_restores(self):
        m = _custom(UDEG)
        p, lim = np.log10([3, 0.4, 0.25, 0.15]), [30, 40]
        with m.pinned_schedule(p, lim):
            sched = m._fixed_schedule.copy()
            m.eval_model_pss(p + 0.05, lim)
            np.testing.assert_array_equal(m._last_schedule, sched)
            data = _synthetic_data(m, p, lim)
            m.eval_model_kld_and_grad(p + 0.02, lim, None, data, 1e-15, eps=1e-6)
            np.testing.assert_array_equal(m._fixed_schedule, sched)   # the gradient keeps an outer pin
        assert m._fixed_schedule is None
        m.eval_model_pss(p + 0.05, lim)
        assert m._last_schedule.size != sched.size or not np.array_equal(m._last_schedule, sched)

    def test_noop_for_closed_form_models(self):
        m = CMEModel("Bursty", "None")
        with m.pinned_schedule(np.log10([3, 0.4, 0.25]), [20, 20]):
            assert not m._schedule_is_pinned()

    def test_hessian_is_positive_definite_at_the_truth_for_a_delayed_network(self):
        # Central differences of the finite-difference gradient at the
        # generating parameters: clean only if every evaluation shares the
        # base point's schedule.
        m = _custom(DELAYED)
        p, lim = np.log10([5, 1, 1, 0.5]), [40, 50]
        data = _synthetic_data(m, p, lim)
        h = 1e-2
        with m.pinned_schedule(p, lim):
            cols = []
            for j in range(4):
                e = np.zeros(4); e[j] = h
                _, gp = m.eval_model_kld_and_grad(p + e, lim, None, data, 1e-15, eps=1e-6)
                _, gm = m.eval_model_kld_and_grad(p - e, lim, None, data, 1e-15, eps=1e-6)
                cols.append((gp - gm) / (2 * h))
        H = np.column_stack(cols)
        H = 0.5 * (H + H.T)
        assert np.all(np.linalg.eigvalsh(H) > 0), np.linalg.eigvalsh(H)


class TestDelayAfterDecay:
    def test_delayed_pulse_after_the_state_has_decayed_is_integrated(self):
        # Fast rates, long delay: u decays to nothing before t = tau, then the
        # delayed product term revives it.  Compare with a brute-force replay
        # of a dense schedule that lands on the delay and reaches well past it.
        m = _custom(DELAYED)
        tau = 1 / 0.574
        p, lim = np.log10([14.0, 0.574, 10.8, 21.3]), [40, 50]
        pss = m.eval_model_pss(p, lim)
        times = np.cumsum(m._last_schedule)
        assert times[-1] > tau + 0.5, times[-1]
        dense = np.concatenate([np.full(4000, tau / 4000), np.full(6000, (4.0 - tau) / 6000)])
        m._fixed_schedule = dense
        try:
            ref = m.eval_model_pss(p, lim)
        finally:
            m._fixed_schedule = None
        np.testing.assert_allclose(pss, ref, atol=5e-5)
        # the same parameters on the immediate network give a different joint
        now = _custom(BURSTY).eval_model_pss(p[[0, 2, 3]], lim)
        assert np.abs(pss - now).max() > 1e-3


class TestThreadLocalSchedules:
    def test_pins_do_not_leak_between_threads(self):
        # compute_sigma evaluates genes in a thread pool on one shared model:
        # a schedule pinned for one gene's grid must not be replayed for another.
        from concurrent.futures import ThreadPoolExecutor
        m = _custom(UDEG)
        p = np.log10([3, 0.4, 0.25, 0.15])
        ref = {lim: m.eval_model_pss(p, list(lim)) for lim in ((20, 30), (36, 44), (28, 50), (40, 24))}

        def work(lim):
            lim = list(lim)
            with m.pinned_schedule(p, lim):
                out = []
                for _ in range(5):
                    out.append(m.eval_model_pss(p + 0.01, lim))
                    assert m._fixed_schedule is not None
                    assert m._last_schedule.size == m._fixed_schedule.size
                return out
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(work, ref))
        for lim, outs in zip(ref, results):
            for o in outs:
                assert o.shape == ref[lim].shape
                assert np.abs(o - ref[lim]).max() < 1e-2      # a replay of another grid's schedule would not even match in shape
        assert m._fixed_schedule is None
