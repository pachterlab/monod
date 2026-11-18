"""The goodness-of-fit statistic's sampling floor and the multi-tail droplet filter
(notes/chromatin-protein-log.md, broad-antibody diagnosis, 2026-11-10)."""
import os
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


_GABA = os.path.join(os.path.dirname(__file__), "..", "example_h5ad", "gaba_example.h5ad")   # gitignored; see test_gaba_inference
_GENES = ["Eif5b", "Xrcc5", "Klhl12"]
_GP = {"max_iterations": 5, "init_pattern": "moments", "num_restarts": 1, "num_gene_cores": 1, "n_jac_jobs": 1}


def _bursty_fit(workdir, gp):
    """A quick 3-gene Bursty/None fit of the gaba example, run inside ``workdir``."""
    import warnings
    from monod.inference import perform_inference
    warnings.filterwarnings("ignore")
    cwd = os.getcwd(); os.chdir(workdir)
    try:
        return perform_inference(_GABA, CMEModel("Bursty", "None"), n_genes=3, genes_to_fit=_GENES, viz=False,
                                 modality_name_dict={"unspliced": "unspliced", "spliced": "spliced"},
                                 gradient_params=gp, seed=1, use_lengths=None)
    finally:
        os.chdir(cwd)


@pytest.fixture(scope="module")
def cold(tmp_path_factory):
    if not os.path.exists(_GABA):
        pytest.skip("gaba_example.h5ad not available")
    """The moments-started fit the warm starts are measured against."""
    return _bursty_fit(tmp_path_factory.mktemp("cold"), dict(_GP))


class TestWarmStart:
    """``init_pattern="given"``: a fit can start where an earlier one ended."""

    GP = _GP

    def _fit(self, workdir, gp):
        return _bursty_fit(workdir, gp)

    def test_warm_start_maps_gene_names_to_optima(self, cold):
        from monod.inference import search_result_from_uns, warm_start
        x0 = warm_start(cold)
        sr = search_result_from_uns(cold.uns["search_result"])
        assert set(x0) == set(cold.var_names)
        for i, g in enumerate(cold.var_names):
            np.testing.assert_array_equal(x0[g], sr.phys_optimum[i])

    def test_warm_start_from_path_and_without_a_result(self, cold, tmp_path):
        import anndata as ad
        from monod.inference import warm_start
        from monod.extract_data import write_h5ad
        path = str(tmp_path / "cold.h5ad"); write_h5ad(cold, path)
        again = warm_start(path)
        for g in cold.var_names:
            np.testing.assert_array_equal(again[g], warm_start(cold)[g])
        bare = ad.AnnData(X=np.zeros((2, 2)))
        with pytest.raises(ValueError, match="search_result"):
            warm_start(bare)

    def test_a_warm_start_at_the_optimum_stays_there(self, cold, tmp_path):
        """Started at the cold fit's optimum, the fit reproduces it: a contract
        between the start and the optimizer, not a new optimum."""
        from monod.inference import search_result_from_uns, warm_start
        gp = dict(self.GP, init_pattern="given", x0=warm_start(cold))
        warm = self._fit(tmp_path, gp)
        s0, s1 = search_result_from_uns(cold.uns["search_result"]), search_result_from_uns(warm.uns["search_result"])
        assert list(warm.var_names) == list(cold.var_names) and cold.n_vars == 3
        k0, k1 = np.asarray(s0.klds).ravel(), np.asarray(s1.klds).ravel()
        assert np.all(k1 <= k0 + 1e-6)
        np.testing.assert_allclose(s1.phys_optimum, s0.phys_optimum, atol=0.05)

    def test_missing_genes_take_the_moments_start(self, cold, tmp_path):
        """A mapping that lacks a gene, or an array row with a NaN, falls back to
        the moments start for that gene and only that gene."""
        from monod.inference import search_result_from_uns, warm_start
        x0 = warm_start(cold); g_drop = list(cold.var_names)[0]; del x0[g_drop]
        (tmp_path / "a").mkdir()
        warm = self._fit(tmp_path / "a", gp=dict(self.GP, init_pattern="given", x0=x0))
        s0, s1 = search_result_from_uns(cold.uns["search_result"]), search_result_from_uns(warm.uns["search_result"])
        # the dropped gene repeats the cold (moments-started) fit exactly; the others sit at the warm optimum
        np.testing.assert_allclose(s1.phys_optimum[0], s0.phys_optimum[0], atol=1e-8)
        np.testing.assert_allclose(s1.phys_optimum[1:], s0.phys_optimum[1:], atol=0.05)
        arr = np.asarray(s0.phys_optimum, float).copy(); arr[1, 0] = np.nan
        (tmp_path / "b").mkdir()
        warm_b = self._fit(tmp_path / "b", gp=dict(self.GP, init_pattern="given", x0=arr))
        s2 = search_result_from_uns(warm_b.uns["search_result"])
        np.testing.assert_allclose(s2.phys_optimum[1], s0.phys_optimum[1], atol=1e-8)

    def test_bad_x0_raises(self, cold, tmp_path):
        with pytest.raises(ValueError, match="x0"):
            self._fit(tmp_path, gp=dict(self.GP, init_pattern="given"))
        (tmp_path / "shape").mkdir()
        with pytest.raises(ValueError, match="shape"):
            self._fit(tmp_path / "shape", gp=dict(self.GP, init_pattern="given", x0=np.zeros((3, 2))))
