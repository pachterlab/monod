"""The contract every error-controlled kernel owes its consumers.

The kernels with adaptive step schedules (custom reaction networks,
ProteinBursty, chromatin) are each validated against references at single
points elsewhere.  The defects that reached the optimiser were never in a
single value; they were in relations between evaluations: a replayed
schedule at perturbed parameters, a gradient taken across schedule changes,
a pin shared between threads, a nesting model fitted worse than the model it
nests.  This module states those relations once and runs them against every
kernel through a small adapter, so a lesson learned on one kernel is checked
on all of them.

Checks, per kernel:

1. replay: the base point's schedule replays exactly at the base point, and
   at nearby parameters it lands within O(perturbation) of a free evaluation,
   with the perturbation itself resolved;
2. gradient: the kernel's pinned gradient agrees with free central
   differences at a wide step;
3. threads: the same gradient from a thread pool equals the serial one;
4. MLE sanity: on data drawn from a known truth, a fit from a displaced start
   never ends above the objective at the truth;
5. nesting: a nesting model started from the nested optimum never ends worse,
   where a nested pair exists.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from scipy.optimize import minimize

from monod import chromatin as ch
from monod import chromatin_fit as cf
from monod.cme_toolbox import CMEModel

N_CELLS = 1000


def _sample_counts(pss, n_cells, rng):
    flat = pss.ravel() / pss.sum()
    idx = rng.choice(flat.size, size=n_cells, p=flat)
    return np.column_stack(np.unravel_index(idx, pss.shape))


def _histogram_data(cells):
    coords, counts = np.unique(cells, axis=0, return_counts=True)
    return coords.astype(np.int64), counts / counts.sum()


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

class CMEKernel:
    """CMEModel kernels with ``pinned_schedule`` (Custom, ProteinBursty)."""

    nested = None            # (nested adapter, embed) when a nested pair exists

    def __init__(self, name, model, theta, limits, step=0.2, nested=None):
        self.name, self.m, self.theta, self.limits, self.step = name, model, np.asarray(theta, float), limits, step
        self.nested = nested

    def pss(self, theta):
        return self.m.eval_model_pss(np.asarray(theta, float), self.limits)

    def _last(self):
        attr = "_last_custom_schedule" if self.m.bio_model == "Custom" else "_last_protein_schedule"
        last = getattr(self.m, attr, None)
        return None if last is None else np.asarray(last, float)

    def pss_pinned(self, theta_base, theta_eval):
        """``(pss, steps used)`` for ``theta_eval`` on ``theta_base``'s schedule."""
        with self.m.pinned_schedule(np.asarray(theta_base, float), self.limits):
            out = self.m.eval_model_pss(np.asarray(theta_eval, float), self.limits)
            return out, self._last()

    def schedule(self, theta):
        self.pss(theta)
        return self._last()

    def make_data(self, theta, rng):
        return _histogram_data(_sample_counts(self.pss(theta), N_CELLS, rng))

    def objective(self, theta, data):
        return float(self.m.eval_model_kld(np.asarray(theta, float), self.limits, None, data, 1e-15))

    def pinned_gradient(self, theta, data):
        f, g = self.m.eval_model_kld_and_grad(np.asarray(theta, float), self.limits, None, data, 1e-15, eps=1e-6)
        return float(f), np.asarray(g, float)

    def free_gradient(self, theta, data, h=1e-3):
        theta = np.asarray(theta, float)
        g = np.zeros(theta.size)
        for i in range(theta.size):
            e = np.zeros(theta.size); e[i] = h
            g[i] = (self.objective(theta + e, data) - self.objective(theta - e, data)) / (2 * h)
        return g

    def bounds(self):
        return [(t - 1.5, t + 1.5) for t in self.theta]

    def fit(self, data, theta0, bounds=None):
        res = minimize(lambda x: self.pinned_gradient(x, data), np.asarray(theta0, float), jac=True,
                       method="L-BFGS-B", bounds=bounds or self.bounds(), options=dict(maxiter=150, ftol=1e-8, gtol=1e-2))
        return res.x, float(res.fun)

    def start(self):
        return self.theta + self.step * np.array([1, -1, 1, -1, 1][:self.theta.size])


class ChromatinKernel:
    """The chromatin kernel through the per-locus likelihood."""

    def __init__(self, rule, theta, nested=None):
        self.name, self.rule, self.theta, self.nested = f"chromatin-{rule}", rule, np.asarray(theta, float), nested
        self.lik = None
        self.tol = 1e-5

    def _system(self, theta):
        k_on, eps, rates, beta, gamma = cf.split_theta(theta, self.rule, 2)
        return ch.nascent_mature_system(k_on, 1.0, eps, self.rule, beta, gamma, rule_params=rates)

    def _limits(self):
        return [40, 60] if self.lik is None else self.lik.limits

    def pss(self, theta, tol=None):
        H, A, C = self._system(theta)
        return ch.eval_joint_pss(H, A, C, self._limits(), tol=tol or self.tol)

    def pss_pinned(self, theta_base, theta_eval):
        H, A, C = self._system(theta_base)
        _, sched = ch.eval_joint_pss(H, A, C, self._limits(), tol=self.tol, return_schedule=True)
        H, A, C = self._system(theta_eval)
        out, used = ch.eval_joint_pss(H, A, C, self._limits(), fixed_schedule=sched, return_schedule=True)
        return out, np.asarray(used, float)

    def schedule(self, theta):
        H, A, C = self._system(theta)
        return np.asarray(ch.eval_joint_pss(H, A, C, self._limits(), tol=self.tol, return_schedule=True)[1], float)

    def make_data(self, theta, rng):
        cells = _sample_counts(self.pss(theta), N_CELLS, rng)
        lim = [int(cells[:, 1].max() * 1.3) + 3, int(cells[:, 2].max() * 1.3) + 3]
        hist = np.zeros((4, lim[0], lim[1]))
        np.add.at(hist, (cells[:, 0], cells[:, 1], cells[:, 2]), 1.0)
        self.lik = cf.LocusLikelihood(hist, self.rule)
        return hist

    def objective(self, theta, data, tol=None):
        return self.lik.nll(np.asarray(theta, float), tol or self.tol) / self.lik.n_cells

    def pinned_gradient(self, theta, data):
        f, g = self.lik.nll_grad(np.asarray(theta, float), self.tol)
        return f / self.lik.n_cells, g / self.lik.n_cells

    def free_gradient(self, theta, data, h=1e-3):
        theta = np.asarray(theta, float)
        g = np.zeros(theta.size)
        for i in range(theta.size):
            e = np.zeros(theta.size); e[i] = h
            g[i] = (self.objective(theta + e, data, 1e-8) - self.objective(theta - e, data, 1e-8)) / (2 * h)
        return g

    def bounds(self):
        return cf.bounds(self.rule, 2)

    def fit(self, data, theta0, bounds=None):
        res = minimize(lambda x: self.pinned_gradient(x, data), np.asarray(theta0, float), jac=True,
                       method="L-BFGS-B", bounds=bounds or self.bounds(), options=dict(maxiter=150, ftol=1e-8, gtol=1e-2))
        return res.x, float(res.fun)

    def start(self):
        return self.theta + 0.15 * np.array([1, -1, 1, -1, 1, -1, 1, -1, 1][:self.theta.size])


def _custom(net):
    return CMEModel("Custom", "None", network=net, normalize_production_rate=True)


def _kernels():
    bursty = CMEKernel("custom-bursty", _custom("0 -> B x U -> S -> 0"), np.log10([3, 0.4, 0.25]), [30, 40])
    # U -> 0 at the lower bound of its box embeds the bursty network in the one with nascent degradation
    udeg = CMEKernel("custom-udeg", _custom("0 -> B x U -> S -> 0; U -> 0"), np.log10([3, 0.4, 0.25, 0.15]), [30, 40],
                     nested=(bursty, lambda th: np.append(th, np.log10(0.15) - 1.5)))
    udeg_bounds = udeg.bounds
    udeg.bounds = lambda: udeg_bounds()[:3] + [(np.log10(0.15) - 1.5, np.log10(0.15) + 1.5)]
    protein = CMEKernel("protein", CMEModel("ProteinBursty", "None", fit_unspliced=True, protein_limit=float("inf")),
                        [0.5, -0.3, -0.6, 0.2, -0.7], [14, 12, 20], step=0.15)
    switch = ChromatinKernel("switch", cf.join_theta([0.5, 0.8], 0.6, [5.0], 0.8, 0.3))
    table = ChromatinKernel("table", cf.join_theta([0.5, 0.8], 0.6, [0.01, 5.0, 5.0, 5.0], 0.8, 0.3),
                            nested=(switch, lambda th: cf.embed_in_table(th, "switch", 2)))
    return [udeg, protein, switch, table]


KERNELS = _kernels()


@pytest.fixture(scope="module", params=KERNELS, ids=[k.name for k in KERNELS])
def kernel(request):
    return request.param


@pytest.fixture(scope="module")
def data(kernel):
    return kernel.make_data(kernel.theta, np.random.default_rng(11))


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------

class TestReplay:
    def test_schedule_is_recorded_and_replays_exactly_at_the_base_point(self, kernel, data):
        free = kernel.pss(kernel.theta)
        sched = kernel.schedule(kernel.theta)
        assert sched is not None and sched.ndim == 1 and sched.size > 10 and np.all(sched > 0)
        again, used = kernel.pss_pinned(kernel.theta, kernel.theta)
        np.testing.assert_allclose(again, free, atol=1e-13)
        np.testing.assert_array_equal(used, sched)

    def test_replay_at_perturbed_parameters_tracks_the_free_result(self, kernel, data):
        delta = np.zeros(kernel.theta.size); delta[-1] = 1e-3          # the slowest rate, which sets the horizon
        base = kernel.pss(kernel.theta)
        sched = kernel.schedule(kernel.theta)
        free = kernel.pss(kernel.theta + delta)
        pinned, used = kernel.pss_pinned(kernel.theta, kernel.theta + delta)
        np.testing.assert_array_equal(used, sched)                      # the pin was honoured, not silently dropped
        d_free, d_pinned = free - base, pinned - base
        assert np.abs(pinned - free).max() < 1e-5, np.abs(pinned - free).max()
        assert np.abs(d_pinned - d_free).max() < 0.1 * np.abs(d_free).max(), (np.abs(d_pinned - d_free).max(), np.abs(d_free).max())


class TestGradient:
    def test_pinned_gradient_matches_free_central_differences(self, kernel, data):
        f, g = kernel.pinned_gradient(kernel.theta, data)
        assert f == pytest.approx(kernel.objective(kernel.theta, data), rel=1e-6, abs=1e-9)
        g_free = kernel.free_gradient(kernel.theta, data)
        scale = max(np.abs(g_free).max(), 1e-3)
        np.testing.assert_allclose(g, g_free, rtol=3e-2, atol=3e-2 * scale)

    def test_gradient_is_identical_from_a_thread_pool(self, kernel, data):
        thetas = [kernel.theta + 0.05 * k for k in range(4)]
        serial = [kernel.pinned_gradient(t, data) for t in thetas]
        with ThreadPoolExecutor(max_workers=4) as pool:
            pooled = list(pool.map(lambda t: kernel.pinned_gradient(t, data), thetas))
        for (fs, gs), (fp, gp) in zip(serial, pooled):
            assert fs == fp
            np.testing.assert_array_equal(gs, gp)


class TestOuterPin:
    def test_gradient_inside_a_pinned_block_uses_the_outer_schedule(self, kernel, data):
        """Hessians by differences of gradients (identifiability checks, standard
        errors) wrap the stencil in one ``pinned_schedule`` block; every gradient
        inside it must be taken on that block's schedule, whichever backend runs."""
        if not isinstance(kernel, CMEKernel):
            pytest.skip("schedules are passed explicitly for this kernel")
        m, lim, theta = kernel.m, kernel.limits, kernel.theta
        e = np.zeros(theta.size); e[0] = 1e-2
        h = 1e-4
        with m.pinned_schedule(theta, lim):
            sched = kernel._last().copy() if m.bio_model != "Custom" else np.asarray(m._fixed_custom_schedule, float).copy()
            _, g = m.eval_model_kld_and_grad(theta + e, lim, None, data, 1e-15, eps=1e-6)
            ref = np.zeros(theta.size)
            for i in range(theta.size):
                d = np.zeros(theta.size); d[i] = h
                ref[i] = (m.eval_model_kld(theta + e + d, lim, None, data, 1e-15)
                          - m.eval_model_kld(theta + e - d, lim, None, data, 1e-15)) / (2 * h)
            used = kernel._last()
        np.testing.assert_array_equal(used, sched)                      # the block's schedule, start to finish
        scale = max(np.abs(ref).max(), 1e-3)
        np.testing.assert_allclose(g, ref, rtol=1e-2, atol=1e-2 * scale)


class TestOptimiserContracts:
    def test_fit_from_a_displaced_start_is_never_worse_than_the_truth(self, kernel, data):
        f_truth = kernel.objective(kernel.theta, data)
        theta_hat, f_hat = kernel.fit(data, kernel.start())
        assert f_hat <= f_truth + 1e-6 * max(1.0, abs(f_truth)), (f_hat, f_truth, theta_hat)

    def test_nesting_model_started_from_the_nested_optimum_is_never_worse(self, kernel, data):
        if kernel.nested is None:
            pytest.skip("no nested pair for this kernel")
        nested, embed = kernel.nested
        # data from the nested truth, fitted by the nested model
        rng = np.random.default_rng(5)
        nested_data = nested.make_data(nested.theta, rng)
        theta_n, f_n = nested.fit(nested_data, nested.start())
        f_n = nested.objective(theta_n, nested_data)
        # the nesting model sees the same cells
        if isinstance(kernel, ChromatinKernel):
            kernel.lik = cf.LocusLikelihood(nested_data, kernel.rule)
            data_for_kernel = nested_data
        else:
            data_for_kernel = nested_data
        f_embedded = kernel.objective(embed(theta_n), data_for_kernel)
        assert f_embedded <= f_n + 1e-3 * max(1.0, abs(f_n)), (f_embedded, f_n)
        theta_k, f_k = kernel.fit(data_for_kernel, embed(theta_n))
        f_k = kernel.objective(theta_k, data_for_kernel)
        assert f_k <= f_embedded + 1e-6 * max(1.0, abs(f_embedded)), (f_k, f_embedded)
        if isinstance(kernel, ChromatinKernel):
            kernel.lik = cf.LocusLikelihood(data, kernel.rule)
