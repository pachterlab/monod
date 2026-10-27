"""Comparing two CMEModels on one gene: the nestings monod knows
(analysis.nesting_embedding), the nested test without a hand-written
embedding, and the non-nested comparison (analysis.compare_models)."""
import numpy as np
import pytest

from monod import chromatin_fit as cf
from monod.analysis import compare_models, compare_nested_models, nesting_embedding
from monod.cme_toolbox import CMEModel


def _net(s, **kw):
    kw.setdefault("normalize_production_rate", True)
    return CMEModel("Custom", "None", network=s, **kw)


def _chrom(rule, **kw):
    return CMEModel("Chromatin", "None", n_sites=2, rule=rule, dropout_bounds=(0.0, 0.0, 1), **kw)


def _hist(cells):
    coords, counts = np.unique(cells, axis=0, return_counts=True)
    return coords.astype(np.int64), counts / counts.sum()


def _draw(model, theta, limits, n, rng, samp=None):
    pss = np.asarray(model.eval_model_pss(np.asarray(theta, float), list(limits), samp))
    flat = pss.ravel() / pss.sum()
    idx = rng.choice(flat.size, size=n, p=flat)
    return np.column_stack(np.unravel_index(idx, pss.shape))


class TestNestingEmbedding:
    def test_chromatin_rule_and_emission(self):
        x = cf.join_theta([0.5, 0.8], 0.6, [4.0], 0.8, 0.3)
        e = nesting_embedding(_chrom("switch"), _chrom("table"))
        np.testing.assert_array_equal(e(x), cf.embed_in_table(x, "switch", 2))
        e = nesting_embedding(_chrom("switch"), _chrom("switch", bursty=True))
        np.testing.assert_array_equal(e(x), np.append(x, -3.0))
        e = nesting_embedding(_chrom("additive"), _chrom("table", bursty=True))
        xa = cf.join_theta([0.5, 0.8], 0.6, [4.0, 1.0], 0.8, 0.3)
        out = e(xa)
        assert out.size == 10 and out[-1] == -3.0
        np.testing.assert_array_equal(out[:-1], cf.embed_in_table(xa, "additive", 2))
        # the embedded point is the same model
        small, big = _chrom("additive"), _chrom("table", bursty=True)
        np.testing.assert_allclose(big.eval_model_pss(out, [20, 30, 4]), small.eval_model_pss(xa, [20, 30, 4]), atol=1e-5)
        with pytest.raises(ValueError):
            nesting_embedding(_chrom("table"), _chrom("switch"))
        with pytest.raises(ValueError):
            nesting_embedding(_chrom("switch", bursty=True), _chrom("table"))

    def test_network_adds_a_burst_a_reaction_or_a_delay(self):
        const = _net("0 -> U -> S -> 0")
        bursty = _net("0 -> B x U -> S -> 0", burst_mean_rate=True)
        e = nesting_embedding(const, bursty)
        x = np.log10([0.4, 0.25])
        np.testing.assert_array_equal(e(x), [-3.0, x[0], x[1]])              # B first, at its floor
        np.testing.assert_allclose(bursty.eval_model_pss(e(x), [30, 40]), const.eval_model_pss(x, [30, 40]), atol=5e-4)
        with pytest.raises(ValueError, match="burst_mean_rate"):
            nesting_embedding(const, _net("0 -> B x U -> S -> 0"))
        with pytest.raises(ValueError):
            nesting_embedding(bursty, const)
        udeg = _net("0 -> B x U -> S -> 0; U -> 0", burst_mean_rate=True)
        e = nesting_embedding(bursty, udeg)
        xb = np.log10([2.0, 0.4, 0.25])
        np.testing.assert_array_equal(e(xb), np.append(xb, udeg.bio_bounds["phys_lb"][-1]))
        delayed = _net("0 -> B x U => S -> 0", burst_mean_rate=True)
        e = nesting_embedding(bursty, delayed)
        out = e(xb)
        assert out.size == 4 and out[1] == delayed.bio_bounds["phys_ub"][1]   # tauinv0 at its ceiling
        with pytest.raises(ValueError):
            nesting_embedding(delayed, bursty)

    def test_builtin_pairs(self):
        with pytest.raises(ValueError, match="burst_mean_rate"):
            nesting_embedding(CMEModel("Constitutive", "None"), CMEModel("Bursty", "None"))
        with pytest.raises(ValueError, match="compare_models"):
            nesting_embedding(CMEModel("Bursty", "None"), CMEModel("Delay", "None"))
        e = nesting_embedding(CMEModel("Bursty", "None"), CMEModel("Bursty", "None"))
        np.testing.assert_array_equal(e([0.5, 0.0, -0.3]), [0.5, 0.0, -0.3])


@pytest.fixture(scope="module")
def bursty_data():
    rng = np.random.default_rng(3)
    b, beta, gamma = 5.0, 0.5, 0.3
    truth = np.log10([b, beta, gamma])
    lim = [40, 70]
    cells = _draw(CMEModel("Bursty", "None"), truth, lim, 3000, rng)
    return truth, lim, _hist(cells), 3000


class TestComparisons:
    def test_vuong_prefers_the_bursty_model_and_is_zero_for_identical_fits(self, bursty_data):
        truth, lim, data, n = bursty_data
        b, beta, gamma = 10 ** truth
        # the moment-matched constitutive fit: production 1, the same means
        x_const = np.log10([beta / b, gamma / b])
        r = compare_models(CMEModel("Bursty", "None"), CMEModel("Constitutive", "None"), truth, x_const, lim, None, data, n)
        assert r["prefer"] == "a" and r["vuong_z"] > 3 and r["p_vuong"] < 1e-3
        assert r["aic_a"] < r["aic_b"] and r["bic_a"] < r["bic_b"]
        same = compare_models(CMEModel("Bursty", "None"), CMEModel("Bursty", "None"), truth, truth, lim, None, data, n)
        assert same["vuong_z"] == 0.0 and same["prefer"] is None and same["kld_a"] == same["kld_b"]

    def test_nested_test_without_an_embedding_on_networks(self, bursty_data):
        truth, lim, data, n = bursty_data
        b, beta, gamma = 10 ** truth
        const = _net("0 -> U -> S -> 0")
        bursty = _net("0 -> B x U -> S -> 0", burst_mean_rate=True)
        x_const = np.log10([beta / b, gamma / b])
        r = compare_nested_models(const, bursty, None, x_const, lim, None, data, n, table="always", maxiter=60)
        assert r["fitted"] and r["prefer_big"] and r["lr"] > 100
        assert r["kld_big"] <= r["kld_small"]
        assert r["aic_big"] < r["aic_small"]
        assert 10 ** r["x_big"][0] > 1.0                                      # a burst size well above the floor
        # constitutive data, with the constitutive model at its own optimum: the bursty network gains nothing
        import scipy.optimize
        rng = np.random.default_rng(5)
        data0 = _hist(_draw(const, x_const, lim, 3000, rng))
        fit0 = scipy.optimize.minimize(lambda x: const.eval_model_kld_and_grad(x, lim, None, data0), x_const, jac=True,
                                       method="L-BFGS-B", options=dict(maxiter=100, ftol=1e-10, gtol=1e-6))
        r0 = compare_nested_models(const, bursty, None, fit0.x, lim, None, data0, 3000, table="always", maxiter=60, alpha=0.01)
        assert r0["lr"] < 6.0 and not r0["prefer_big"]

    def test_nudged_starts_leave_the_flat_floor(self):
        """With large bursts the embedded start (burst size at its floor) sits on a
        flat stretch of the divergence and a gradient fitter stays there; the
        nudged start reaches the optimum the independent bursty fit finds."""
        import scipy.optimize
        rng = np.random.default_rng(11)
        truth = np.log10([20.0, 0.5, 0.3])
        lim = [60, 120]
        n = 2000
        data = _hist(_draw(CMEModel("Bursty", "None"), truth, lim, n, rng))
        const = _net("0 -> U -> S -> 0")
        bursty = _net("0 -> B x U -> S -> 0", burst_mean_rate=True)
        x_c = scipy.optimize.minimize(lambda x: const.eval_model_kld_and_grad(x, lim, None, data), np.log10([0.5 / 20, 0.3 / 20]),
                                      jac=True, method="L-BFGS-B",
                                      bounds=scipy.optimize.Bounds(const.bio_bounds["phys_lb"], const.bio_bounds["phys_ub"]),
                                      options=dict(maxiter=100, ftol=1e-10, gtol=1e-6)).x
        kld_ref = float(CMEModel("Bursty", "None").eval_model_kld(truth, lim, None, data))
        stuck = compare_nested_models(const, bursty, None, x_c, lim, None, data, n, table="always", maxiter=100, nudge=False)
        moved = compare_nested_models(const, bursty, None, x_c, lim, None, data, n, table="always", maxiter=100)
        assert moved["n_starts"] >= 2 and moved["kld_big"] <= stuck["kld_big"]   # the constitutive gamma sits at its floor too
        assert moved["kld_big"] < kld_ref + 0.02 and 10 ** moved["x_big"][0] > 5.0
        # an explicit start in the big model's own coordinates does the same job
        b, beta, gamma = 10 ** truth
        given = compare_nested_models(const, bursty, None, x_c, lim, None, data, n, table="always", maxiter=100, nudge=False,
                                      extra_starts=[np.log10([b, beta / b, gamma / b])])
        assert given["kld_big"] < kld_ref + 0.02

    def test_needs_the_data_arguments(self):
        with pytest.raises(TypeError):
            compare_nested_models(_chrom("switch"), _chrom("table"))
