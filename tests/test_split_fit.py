"""perform_inference(split_by=...): several groups of cells from one AnnData,
fitted separately, with the per-group tables in varm (2026-10-29)."""
import os
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from monod import analysis
from monod.cme_toolbox import CMEModel
from monod.inference import (perform_inference, reject_genes, search_result_list_from_uns, split_info,
                             find_shared_sampling_optimum)

GP = {"max_iterations": 15, "init_pattern": "moments", "num_restarts": 1, "num_gene_cores": 1}
PERMISSIVE = {"min_means": [0, 0], "max_maxes": [10_000, 10_000], "min_maxes": [0, 0]}
NAMES = [p.replace(r"\log_{10} ", "") for p in CMEModel("Bursty", "None").param_str]


def _two_populations(n_cells=240, n_genes=4, seed=0, third=0):
    """Two Poisson populations with different means, labelled in obs['pop'];
    ``third`` extra cells carry a label that is left out of the fits."""
    rng = np.random.default_rng(seed)
    half = n_cells // 2
    means = np.linspace(0.8, 3.0, n_genes)
    U = np.vstack([rng.poisson(means, (half, n_genes)), rng.poisson(4 * means, (half, n_genes)),
                   rng.poisson(2 * means, (third, n_genes))])
    S = np.vstack([rng.poisson(2 * means, (half, n_genes)), rng.poisson(6 * means, (half, n_genes)),
                   rng.poisson(3 * means, (third, n_genes))])
    a = ad.AnnData(X=U + S, layers={"unspliced": U, "spliced": S})
    a.var_names = [f"g{i}" for i in range(n_genes)]
    a.obs_names = [f"c{i}" for i in range(a.n_obs)]
    a.obs["pop"] = pd.Categorical(["low cells"] * half + ["high"] * half + ["other"] * third,
                                  categories=["low cells", "high", "other"])
    return a


@pytest.fixture(scope="module")
def split_fit(tmp_path_factory):
    warnings.filterwarnings("ignore")
    cwd = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("split"))
    try:
        a = _two_populations(third=20)
        out = perform_inference(a, CMEModel("Bursty", "None"), split_by="pop", groups=["low cells", "high"],
                                n_genes=4, filt_param=PERMISSIVE, gradient_params=GP, viz=False,
                                exclude_sigma=False, dataset_string="split")
        yield a, out
    finally:
        os.chdir(cwd)


class TestSplitFit:
    def test_layout(self, split_fit):
        a, out = split_fit
        assert out.n_obs == 240 and set(out.obs["pop"].astype(str)) == {"low cells", "high"}   # 'other' dropped
        info = split_info(out)
        assert (info["column"], info["groups"], info["keys"]) == ("pop", ["low cells", "high"], ["low_cells", "high"])
        assert info["batch_column"] is None and info["batch"] == [None, None] and info["split"] == info["groups"]
        assert "search_result_list" in out.uns and "search_result" not in out.uns
        for key in ("low_cells", "high"):
            tab = out.varm[key]
            assert isinstance(tab, pd.DataFrame) and list(tab.index) == list(out.var_names)
            for name in NAMES:
                assert name in tab.columns and f"error_{name}" in tab.columns
            assert "AIC" in tab.columns and "error_unreliable" in tab.columns
            assert np.all(np.isfinite(tab[NAMES + ["AIC"]].to_numpy()))
        assert not any(name in out.var.columns for name in NAMES)      # nothing spilled into var
        srs = search_result_list_from_uns(out.uns["search_result_list"])
        assert [sr.group for sr in srs] == ["low cells", "high"] and [sr.assigns for sr in srs] == [0, 1]
        masks = [np.asarray(sr.filt, bool) for sr in srs]
        assert masks[0].sum() == 120 and masks[1].sum() == 120 and not np.any(masks[0] & masks[1])
        assert srs[0].n_cells == 120

    def test_groups_fit_like_separate_objects(self, split_fit):
        a, out = split_fit
        sub = a[a.obs["pop"] == "high"].copy()
        sep = perform_inference(sub, CMEModel("Bursty", "None"), n_genes=4, genes_to_fit=list(out.var_names),
                                filt_param=PERMISSIVE, gradient_params=GP, viz=False, exclude_sigma=False,
                                dataset_string="sep")
        assert list(sep.var_names) == list(out.var_names)
        for name in NAMES:
            np.testing.assert_allclose(out.varm["high"][name].to_numpy(), sep.var[name].to_numpy(), rtol=1e-6, atol=1e-8)
            np.testing.assert_allclose(out.varm["high"][f"error_{name}"].to_numpy(), sep.var[f"error_{name}"].to_numpy(),
                                       rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(out.varm["high"]["AIC"].to_numpy(), sep.var["AIC"].to_numpy(), rtol=1e-6)
        # the two populations really differ
        assert not np.allclose(out.varm["high"][NAMES].to_numpy(), out.varm["low_cells"][NAMES].to_numpy())

    def test_group_helpers(self, split_fit):
        a, out = split_fit
        sr = analysis.group_result(out, "high")
        assert sr.group == "high" and analysis.group_result(out, "low_cells").group == "low cells"
        assert analysis.group_table(out, "low cells") is out.varm["low_cells"]
        sd = analysis.group_search_data(out, "high")
        assert sd.n_cells == 120 and sd.n_genes == out.n_vars
        with pytest.raises(KeyError):
            analysis.group_result(out, "nope")

    def test_reject_and_de_write_into_tables(self, split_fit):
        a, out = split_fit
        reject_genes(out, viz=False)
        for key in ("low_cells", "high"):
            for c in ("rejected_genes", "csq", "pval", "hellinger"):
                assert c in out.varm[key].columns
            assert out.varm[key]["rejected_genes"].dtype == bool
        assert "rejected_genes" not in out.var.columns
        gn, gf, offs, resid = analysis.DE_parameters(out, viz=False, use_sigma=True, discard_rejected=False)
        assert list(gn) == [("low cells", "high")]
        for name in NAMES:
            assert f"FC_{name}_vs_low_cells" in out.varm["high"].columns
            assert f"adj_FC_{name}_vs_high" in out.varm["low_cells"].columns
            np.testing.assert_allclose(out.varm["high"][f"FC_{name}_vs_low_cells"].to_numpy(),
                                       -out.varm["low_cells"][f"FC_{name}_vs_high"].to_numpy())

    def test_h5ad_round_trip(self, split_fit, tmp_path):
        a, out = split_fit
        path = tmp_path / "split.h5ad"
        out.write_h5ad(path)
        back = ad.read_h5ad(path)
        assert split_info(back) == split_info(out)
        pd.testing.assert_frame_equal(back.varm["high"], out.varm["high"])
        sr = analysis.group_result(back, "high")
        np.testing.assert_allclose(sr.phys_optimum, analysis.group_result(out, "high").phys_optimum)
        assert np.asarray(sr.filt, bool).sum() == 120 and sr.group == "high"


class TestSplitArguments:
    def test_bad_arguments(self):
        warnings.filterwarnings("ignore")
        a = _two_populations()
        with pytest.raises(KeyError, match="split_by"):
            perform_inference(a, CMEModel("Bursty", "None"), split_by="nope", n_genes=4, viz=False)
        with pytest.raises(ValueError, match="do not occur"):
            perform_inference(a, CMEModel("Bursty", "None"), split_by="pop", groups=["high", "ghost"], n_genes=4, viz=False)
        with pytest.raises(ValueError, match="cannot be combined"):
            perform_inference(a, CMEModel("Bursty", "None"), split_by="pop", mek_means_params=(2, 2), n_genes=4, viz=False)
        a.obs["one"] = "x"
        with pytest.raises(ValueError, match="single group"):
            perform_inference(a, CMEModel("Bursty", "None"), split_by="one", n_genes=4, viz=False)

    def test_default_groups_follow_the_categories(self, tmp_path, monkeypatch):
        warnings.filterwarnings("ignore")
        monkeypatch.chdir(tmp_path)
        a = _two_populations(n_cells=120)
        out = perform_inference(a, CMEModel("Bursty", "None"), split_by="pop", n_genes=4, filt_param=PERMISSIVE,
                                gradient_params=GP, viz=False, dataset_string="d")
        assert split_info(out)["groups"] == ["low cells", "high"]        # category order, absent 'other' skipped
        assert "error_$b$" not in out.varm["high"].columns         # exclude_sigma default


class TestSharedSamplingOptimum:
    """With a sequencing-noise model the groups of a split fit share one grid point."""

    @pytest.fixture(scope="class")
    def poisson_fits(self, tmp_path_factory):
        warnings.filterwarnings("ignore")
        cwd = os.getcwd()
        os.chdir(tmp_path_factory.mktemp("shared"))
        try:
            a = _two_populations(n_cells=200, seed=3)
            kw = dict(split_by="pop", n_genes=4, filt_param=PERMISSIVE, gradient_params=GP, viz=False,
                      samp_lb=[-1.0, -1.0], samp_ub=[0.0, 0.0], gridsize=[2, 2], dataset_string="shared")
            shared = perform_inference(a.copy(), CMEModel("Bursty", "Poisson"), **kw)
            own = perform_inference(a.copy(), CMEModel("Bursty", "Poisson"), shared_samp=False, **kw)
            yield shared, own
        finally:
            os.chdir(cwd)

    def test_shared_point_is_the_cell_weighted_argmin(self, poisson_fits):
        shared, _ = poisson_fits
        srs = search_result_list_from_uns(shared.uns["search_result_list"])
        assert shared.uns["split_by"]["shared_samp"]
        idx = {int(sr.samp_optimum_ind) for sr in srs}
        assert len(idx) == 1
        total = sum(sr.n_cells * np.asarray(sr.klds).sum(axis=1) for sr in srs)
        assert idx == {int(np.argmin(total))}
        for sr in srs:
            assert int(sr.samp_optimum_shared) == int(np.argmin(total))
            np.testing.assert_allclose(sr.phys_optimum, sr.param_estimates[int(np.argmin(total))])
            assert list(sr.samp_optimum) == list(sr.sp.sampl_vals[int(np.argmin(total))])

    def test_find_sampling_optimum_keeps_the_shared_point(self, poisson_fits):
        shared, _ = poisson_fits
        sr = analysis.group_result(shared, "high")
        idx = int(sr.samp_optimum_shared)
        sr.find_sampling_optimum()
        assert int(sr.samp_optimum_ind) == idx
        sr.samp_optimum_shared = None
        sr.find_sampling_optimum()
        assert int(sr.samp_optimum_ind) == int(np.argmin(sr.obj_func))

    def test_own_optima_without_sharing(self, poisson_fits):
        _, own = poisson_fits
        assert not own.uns["split_by"]["shared_samp"]
        for sr in search_result_list_from_uns(own.uns["search_result_list"]):
            assert getattr(sr, "samp_optimum_shared", None) is None
            assert int(sr.samp_optimum_ind) == int(np.argmin(sr.obj_func))

    def test_shared_point_survives_h5ad(self, poisson_fits, tmp_path):
        shared, _ = poisson_fits
        path = tmp_path / "shared.h5ad"
        shared.write_h5ad(path)
        back = analysis.group_result(ad.read_h5ad(path), "low cells")
        assert int(back.samp_optimum_shared) == int(analysis.group_result(shared, "low cells").samp_optimum_shared)

    def test_different_grids_are_refused(self, poisson_fits, tmp_path, monkeypatch):
        shared, _ = poisson_fits
        monkeypatch.chdir(tmp_path)
        warnings.filterwarnings("ignore")
        other = perform_inference(_two_populations(n_cells=100, seed=4), CMEModel("Bursty", "Poisson"), n_genes=4,
                                  genes_to_fit=list(shared.var_names), filt_param=PERMISSIVE, gradient_params=GP,
                                  viz=False, samp_lb=[-1.0, -1.0], samp_ub=[0.0, 0.0], gridsize=[1, 1], dataset_string="g1")
        from monod.inference import search_result_from_uns
        with pytest.raises(ValueError, match="different sampling grids"):
            find_shared_sampling_optimum([analysis.group_result(shared, "high"),
                                          search_result_from_uns(other.uns["search_result"])])


def _two_pops_two_batches(n_cells=320, seed=5):
    """Two populations, each split over two experiments; experiment B is thinned
    to half the counts so the two have different capture rates."""
    a = _two_populations(n_cells=n_cells, seed=seed)
    rng = np.random.default_rng(seed + 1)
    batch = np.where(np.arange(a.n_obs) % 2 == 0, "A", "B")
    thin = batch == "B"
    for layer in ("unspliced", "spliced"):
        X = np.asarray(a.layers[layer]).copy()
        X[thin] = rng.binomial(X[thin], 0.5)
        a.layers[layer] = X
    a.X = a.layers["unspliced"] + a.layers["spliced"]
    a.obs["batch"] = pd.Categorical(batch, categories=["A", "B"])
    return a


class TestBatches:
    """split_by x batch_by: one group per (cell type, experiment), the sampling
    optimum shared within an experiment only."""

    @pytest.fixture(scope="class")
    def fit(self, tmp_path_factory):
        warnings.filterwarnings("ignore")
        cwd = os.getcwd()
        os.chdir(tmp_path_factory.mktemp("batches"))
        try:
            out = perform_inference(_two_pops_two_batches(), CMEModel("Bursty", "Poisson"), split_by="pop", batch_by="batch",
                                    n_genes=4, filt_param=PERMISSIVE, gradient_params=GP, viz=False,
                                    samp_lb=[-1.0, -1.0], samp_ub=[0.0, 0.0], gridsize=[2, 2], dataset_string="b")
            yield out
        finally:
            os.chdir(cwd)

    def test_groups_and_keys(self, fit):
        info = split_info(fit)
        assert info["groups"] == ["low cells | A", "high | A", "low cells | B", "high | B"]
        assert info["keys"] == ["low_cells__A", "high__A", "low_cells__B", "high__B"]
        assert info["split"] == ["low cells", "high", "low cells", "high"] and info["batch"] == ["A", "A", "B", "B"]
        assert info["column"] == "pop" and info["batch_column"] == "batch"
        for key in info["keys"]:
            assert key in fit.varm and list(fit.varm[key].index) == list(fit.var_names)
        srs = search_result_list_from_uns(fit.uns["search_result_list"])
        assert [sr.group for sr in srs] == info["groups"]
        assert [int(np.asarray(sr.filt, bool).sum()) for sr in srs] == [80, 80, 80, 80]

    def test_optimum_shared_within_an_experiment(self, fit):
        srs = search_result_list_from_uns(fit.uns["search_result_list"])
        for members in (srs[:2], srs[2:]):
            total = sum(sr.n_cells * np.asarray(sr.klds).sum(axis=1) for sr in members)
            assert {int(sr.samp_optimum_ind) for sr in members} == {int(np.argmin(total))}
            assert {int(sr.samp_optimum_shared) for sr in members} == {int(np.argmin(total))}
        # the two experiments were thinned differently, so their landscapes differ
        assert not np.allclose(srs[0].obj_func, srs[2].obj_func)

    def test_group_lookup_by_pair(self, fit):
        assert analysis.group_result(fit, ("high", "B")).group == "high | B"
        assert analysis.group_result(fit, "high__A").group == "high | A"
        assert analysis.group_table(fit, ("low cells", "A")) is fit.varm["low_cells__A"]
        with pytest.raises(KeyError):
            analysis.group_result(fit, ("high", "C"))

    def test_default_pairs_stay_within_type_or_experiment(self, fit):
        reject_genes(fit, viz=False)
        gn, gf, offs, resid = analysis.DE_parameters(fit, viz=False, use_sigma=False, discard_rejected=False)
        assert set(gn) == {("low cells | A", "high | A"), ("low cells | B", "high | B"),
                           ("low cells | A", "low cells | B"), ("high | A", "high | B")}
        assert "FC_$b$_vs_low_cells__A" in fit.varm["high__A"].columns
        assert "FC_$b$_vs_high__A" in fit.varm["high__B"].columns
        assert not any(c.endswith("_vs_low_cells__A") for c in fit.varm["high__B"].columns)   # mixed pair not compared
        one = analysis.DE_parameters(fit, viz=False, use_sigma=False, discard_rejected=False,
                                     pairs=[(("high", "A"), ("high", "B"))], save_cluster_FCs=False)
        assert list(one[0]) == [("high | A", "high | B")]

    def test_batch_only_and_restrictions(self, tmp_path, monkeypatch):
        warnings.filterwarnings("ignore")
        monkeypatch.chdir(tmp_path)
        a = _two_pops_two_batches(n_cells=200)
        kw = dict(n_genes=4, filt_param=PERMISSIVE, gradient_params=GP, viz=False,
                  samp_lb=[-1.0, -1.0], samp_ub=[0.0, 0.0], gridsize=[2, 2], dataset_string="bo")
        out = perform_inference(a.copy(), CMEModel("Bursty", "Poisson"), batch_by="batch", **kw)
        info = split_info(out)
        assert info["groups"] == ["A", "B"] and info["keys"] == ["A", "B"] and info["column"] is None
        for sr in search_result_list_from_uns(out.uns["search_result_list"]):
            assert getattr(sr, "samp_optimum_shared", None) is None          # nothing to share across experiments
            assert int(sr.samp_optimum_ind) == int(np.argmin(sr.obj_func))
        out = perform_inference(a.copy(), CMEModel("Bursty", "Poisson"), split_by="pop", batch_by="batch", batches=["B"], **kw)
        assert split_info(out)["groups"] == ["low cells | B", "high | B"] and out.n_obs == 100
        with pytest.raises(KeyError, match="batch_by"):
            perform_inference(a.copy(), CMEModel("Bursty", "Poisson"), split_by="pop", batch_by="nope", **kw)
