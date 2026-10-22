"""Staged per-locus fitting of the chromatin model (monod.chromatin_fit)."""
import numpy as np
import pytest

from monod import chromatin as ch
from monod import chromatin_fit as cf


def _sample_hist(rule, prm, n_cells, rng, k_on=(0.5, 0.8), eps=0.6, beta=0.8, gamma=0.3):
    H, A, C = ch.nascent_mature_system(k_on, 1.0, eps, rule, beta, gamma, rule_params=prm)
    pss = ch.eval_joint_pss(H, A, C, [40, 60])
    flat = pss.ravel() / pss.sum()
    idx = rng.choice(flat.size, size=n_cells, p=flat)
    cells = np.column_stack(np.unravel_index(idx, pss.shape))
    lim = [int(cells[:, 1].max() * 1.3) + 3, int(cells[:, 2].max() * 1.3) + 3]
    hist = np.zeros((pss.shape[0], lim[0], lim[1]))
    np.add.at(hist, (cells[:, 0], cells[:, 1], cells[:, 2]), 1.0)
    return hist


class TestChainMarginal:
    def test_recovers_chain_from_expected_counts(self):
        k_on, eps = np.array([0.5, 0.8, 0.3]), 0.6
        counts = 5000 * ch.ising_steady_state(k_on, 1.0, eps)
        fit = cf.fit_chain_marginal(counts)
        np.testing.assert_allclose(fit.k_on, k_on, rtol=1e-3)
        assert fit.eps == pytest.approx(eps, rel=1e-3)
        assert fit.p_drop is None and fit.n_params == 4

    def test_recovers_shared_dropout(self):
        k_on, eps, p = np.array([0.5, 0.8, 0.3]), 0.6, 0.2
        w = ch.ising_steady_state(k_on, 1.0, eps) @ ch.dropout_matrix(3, p)
        fit = cf.fit_chain_marginal(20000 * w, dropout="shared")
        np.testing.assert_allclose(fit.k_on, k_on, rtol=2e-2)
        assert fit.eps == pytest.approx(eps, rel=2e-2)
        np.testing.assert_allclose(fit.p_drop, p, atol=2e-2)
        assert fit.n_params == 5


class TestParameterVectors:
    def test_split_join_embed(self):
        theta = cf.join_theta([0.5, 0.8], 0.6, [4.0, 2.0], 0.8, 0.3)
        k_on, eps, rates, beta, gamma = cf.split_theta(theta, "additive", 2)
        np.testing.assert_allclose(k_on, [0.5, 0.8]); assert eps == pytest.approx(0.6)
        np.testing.assert_allclose(rates, [4.0, 2.0]); assert (beta, gamma) == pytest.approx((0.8, 0.3))
        tab = cf.split_theta(cf.embed_in_table(theta, "additive", 2), "table", 2)[2]
        np.testing.assert_allclose(tab, [10 ** cf.BOUNDS["rate"][0], 4.0, 2.0, 6.0])
        assert cf.n_params("table", 2) == 9 and cf.n_params("switch", 2) == 6
        assert len(cf.bounds("additive", 3)) == cf.n_params("additive", 3)


class TestLikelihood:
    def test_pinned_gradient_is_correct_and_schedule_independent(self):
        rng = np.random.default_rng(1)
        hist = _sample_hist("switch", [5.0], 800, rng)
        lik = cf.LocusLikelihood(hist, "switch")
        theta = cf.join_theta([0.5, 0.8], 0.6, [5.0], 0.8, 0.3)
        f, g = lik.nll_grad(theta, 1e-5)
        assert f == pytest.approx(lik.nll(theta, 1e-5))
        # against free (adaptive) differences at a tight tolerance and a wide step
        g_free = np.zeros(theta.size)
        for i in range(theta.size):
            e = np.zeros(theta.size); e[i] = 1e-3
            g_free[i] = (lik.nll(theta + e, 1e-8) - lik.nll(theta - e, 1e-8)) / 2e-3
        np.testing.assert_allclose(g, g_free, rtol=2e-2, atol=0.05)
        # and against the pinned gradient on a finer schedule
        f7, g7 = lik.nll_grad(theta, 1e-7)
        assert f7 == pytest.approx(f, abs=1e-3)
        np.testing.assert_allclose(g7, g, rtol=1e-3, atol=1e-2)
        # the Hessian is symmetric positive definite near the truth
        Hm = lik.hessian(theta, 1e-5)
        assert np.all(np.linalg.eigvalsh(Hm) > 0)


class TestFitLocus:
    def test_switch_truth_is_recovered(self):
        rng = np.random.default_rng(3)
        hist = _sample_hist("switch", [5.0], 1000, rng)
        res = cf.fit_locus(hist, rules=("switch", "additive", "table"), table="score")
        rows = res.table()
        assert set(rows) >= {"switch", "additive"}
        assert res.best_by_bic() == "switch"
        # the switch fits its own data better than the additive rule by a clear margin
        assert rows["additive"]["nll"] - rows["switch"]["nll"] > 3.0
        # recovered parameters are close to the truth (log10)
        prm = res.fits["switch"].params(2)
        assert prm["rates"][0] == pytest.approx(5.0, rel=0.25)
        assert prm["gamma"] == pytest.approx(0.3, rel=0.25)
        np.testing.assert_allclose(res.chain.k_on, [0.5, 0.8], rtol=0.3)
        # the score test ran; if it asked for the table, the table cannot beat the switch by much
        assert res.score is not None
        if "table" in rows:
            assert rows["switch"]["lr_vs_table"] < 7.81 + 2.0
        # staged fitting keeps evaluations few
        assert all(f.nfev < 600 for f in res.fits.values()), {r: f.nfev for r, f in res.fits.items()}
