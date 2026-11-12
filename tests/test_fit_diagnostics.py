"""The goodness-of-fit statistic's sampling floor and the multi-tail droplet filter
(docs/chromatin-protein-design.md, broad-antibody diagnosis, 2026-11-10)."""
import numpy as np
import pytest

from monod.cme_toolbox import CMEModel
from monod.extract_data import tail_droplets
from monod.inference import hellinger_floor, hellinger_statistic


class TestHellingerFloor:
    def test_statistic_is_the_legacy_formula(self):
        rng = np.random.default_rng(0)
        pss = rng.dirichlet(np.ones(40)).reshape(5, 8)
        counts = rng.multinomial(60, pss.ravel()).reshape(5, 8).astype(float)
        # chisquare_testing before 0.4: occupied states plus one bin holding the expected mass left over
        n = counts.sum()
        occ = counts.ravel() > 0
        e = np.concatenate((pss.ravel()[occ] * n, [n - (pss.ravel()[occ] * n).sum()]))
        o = np.concatenate((counts.ravel()[occ], [0.0]))
        legacy = ((np.sqrt(e / n) - np.sqrt(o / n)) ** 2).sum() / np.sqrt(2)
        assert hellinger_statistic(pss * n, counts) == pytest.approx(legacy)
        assert hellinger_statistic(pss * n, counts * 1.0) >= 0
        assert hellinger_statistic(counts, counts) == pytest.approx(0.0)                 # a perfect fit scores zero

    def test_floor_falls_with_the_cell_count_and_rises_with_the_grid(self):
        pss = CMEModel("Bursty", "None").eval_model_pss(np.log10([3.0, 1.0, 0.3]), [40, 60])
        f_small = hellinger_floor(pss, 50)
        f_big = hellinger_floor(pss, 20_000)
        assert f_small > 0.1 > f_big > 0.0
        # one cell per state: the statistic saturates however good the fit
        wide = np.full((200, 200), 1.0 / 40_000)
        assert hellinger_floor(wide, 300) > 0.6
        assert hellinger_floor(pss, 500, seed=1) == pytest.approx(hellinger_floor(pss, 500, seed=1))

    def test_excess_is_near_zero_for_data_from_the_model(self):
        pss = CMEModel("Bursty", "None").eval_model_pss(np.log10([3.0, 1.0, 0.3]), [40, 60])
        rng = np.random.default_rng(7)
        n = 400
        counts = rng.multinomial(n, pss.ravel()).reshape(pss.shape)
        raw = hellinger_statistic(pss * n, counts)
        floor = hellinger_floor(pss, n, reps=20)
        assert raw > 0.05                                               # the raw statistic would reject this perfect model
        assert abs(raw - floor) < 0.03                                  # the excess does not


class TestTailDroplets:
    def test_flags_cells_high_on_many_features(self):
        rng = np.random.default_rng(0)
        X = rng.gamma(2.0, 1.5, size=(5000, 100))
        clean, hits = tail_droplets(X, quantile=0.99, min_hits=5)
        assert hits.mean() == pytest.approx(1.0, abs=0.05)                         # independent features: a hit per cell on average
        assert 5 <= clean.sum() <= 40                                               # Poisson(1) reaches five 0.37% of the time
        X[:20] *= 8                                                     # aggregates: high on every feature
        drop, hits = tail_droplets(X, quantile=0.99, min_hits=5)
        assert hits.shape == (5000,) and drop[:20].all() and (hits[:20] > 50).all()
        assert drop[20:].sum() <= 40
        # integer counts: ties at the quantile make the rule conservative (fewer hits), never looser
        drop_i, hits_i = tail_droplets(rng.poisson(3.0, size=(5000, 100)), quantile=0.99, min_hits=5)
        assert hits_i.mean() < 1.0 and drop_i.sum() <= 40
        from scipy.sparse import csr_matrix
        drop_sp, hits_sp = tail_droplets(csr_matrix(X))
        assert np.array_equal(drop_sp, drop) and np.array_equal(hits_sp, hits)
        with pytest.raises(ValueError):
            tail_droplets(X[0])
