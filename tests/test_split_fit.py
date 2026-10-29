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
from monod.inference import perform_inference, reject_genes, search_result_list_from_uns, split_info

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
        assert split_info(out) == ("pop", ["low cells", "high"], ["low_cells", "high"])
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
        assert split_info(out)[1] == ["low cells", "high"]        # category order, absent 'other' skipped
        assert "error_$b$" not in out.varm["high"].columns         # exclude_sigma default
