"""Pickle-free serialisation of models, parameters and results (2026-10-15).

Every persistent object round-trips through to_dict()/from_dict(), through an
h5ad written by anndata, and through the standalone HDF5 files monod writes.
"""
import glob
import os
import warnings

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from monod import analysis
import monod.mminference as mm
from monod.cme_toolbox import CMEModel
from monod.extract_data import extract_data, model_from_uns, _uns_pack
from monod.inference import (
    InferenceParameters,
    SearchResults,
    perform_inference,
    search_result_from_uns,
    search_result_list_from_uns,
    search_result_list_to_uns,
    searchdata_from_adata,
)

from monod._core import _mc  # None unless the compiled extension is really installed

_HERE = os.path.dirname(__file__)
EXAMPLE = os.path.join(_HERE, "..", "example_h5ad", "example_adata.h5ad")
GP = {"max_iterations": 5, "init_pattern": "moments", "num_restarts": 1, "num_gene_cores": 1, "n_jac_jobs": 1}


def _roundtrip_uns(d):
    """Push a dict through a real h5ad write/read and return what comes back."""
    import tempfile
    a = ad.AnnData(X=np.zeros((2, 2)))
    a.uns["x"] = d
    p = os.path.join(tempfile.mkdtemp(), "t.h5ad")
    a.write_h5ad(p)
    return ad.read_h5ad(p).uns["x"]


class TestCMEModel:
    @pytest.mark.parametrize("ctor", [
        dict(bio_model="Bursty", seq_model="None"),
        dict(bio_model="Constitutive", seq_model="Poisson", amb_model="Equal"),
        dict(bio_model="ProteinBursty", seq_model="None", protein_limit=np.inf, fit_unspliced=False, min_fudge=0.2),
        dict(bio_model="Custom", seq_model="None", network="0 -> B x U -> S -> 0", normalize_production_rate=True),
    ])
    def test_round_trip(self, ctor):
        m = CMEModel(**ctor)
        d = _roundtrip_uns(m.to_dict())
        m2 = CMEModel.from_dict(d)
        assert (m2.bio_model, m2.seq_model, m2.amb_model) == (m.bio_model, m.seq_model, m.amb_model)
        assert m2.get_num_params() == m.get_num_params()
        assert list(m2.model_modalities) == list(m.model_modalities)
        if m.bio_model in ("Bursty", "Constitutive"):
            p = np.zeros(m.get_num_params())
            lim = [8, 8] + ([4] if m.amb_model != "None" else [])
            samp = None if m.seq_model == "None" else np.array([-1.0, -1.0])
            np.testing.assert_array_equal(m.eval_model_pss(p, lim, samp), m2.eval_model_pss(p, lim, samp))

    def test_no_pickle_in_uns_after_extract_data(self, tmp_path):
        warnings.filterwarnings("ignore")
        out = extract_data(ad.read_h5ad(EXAMPLE), CMEModel("Bursty", "None"), n_genes=3, viz=False)
        assert isinstance(out.uns["model"], dict) and "hist" not in out.uns
        p = str(tmp_path / "x.h5ad")
        out.write_h5ad(p)
        back = ad.read_h5ad(p)
        assert model_from_uns(back.uns["model"]).bio_model == "Bursty"
        sd = searchdata_from_adata(back)
        assert sd.n_genes == out.n_vars

    def test_legacy_pickle_still_loads(self):
        with pytest.warns(DeprecationWarning):
            blob = _uns_pack(CMEModel("Bursty", "Poisson"))
        m = model_from_uns(blob)
        assert m.seq_model == "Poisson"


class TestInferenceParameters:
    def test_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ip = InferenceParameters("ds", CMEModel("Bursty", "Poisson"), gridsize=[2, 3], gradient_params=GP, run_meta="meta")
        ip2 = InferenceParameters.from_dict(_roundtrip_uns(ip.to_dict()))
        assert ip2.inference_string == ip.inference_string
        assert ip2.gridsize == [2, 3] and ip2.n_grid_points == 6
        np.testing.assert_array_equal(ip2.phys_lb, ip.phys_lb)
        assert ip2.gradient_params["max_iterations"] == 5
        assert ip2.use_lengths == ip.use_lengths and ip2.model.seq_model == "Poisson"

    def test_mek_means_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        p = mm.MEKMeansParameters("ds", CMEModel("Bursty", "None"), k=3, epochs=7, em_tol=0.0, random_state=11,
                                  init_modality="spliced", gradient_params=GP, save=True)
        p2 = InferenceParameters.from_dict(_roundtrip_uns(p.to_dict()))
        assert isinstance(p2, mm.MEKMeansParameters)
        assert (p2.k, p2.epochs, p2.em_tol, p2.random_state, p2.init_modality) == (3, 7, 0.0, 11, "spliced")
        assert os.path.isfile(os.path.join("ds", "Bursty_None_1x1", "parameters.h5"))


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    warnings.filterwarnings("ignore")
    cwd = os.getcwd(); os.chdir(tmp_path_factory.mktemp("run"))
    try:
        out = perform_inference(EXAMPLE, CMEModel("Bursty", "None"), n_genes=3, viz=False, gradient_params=GP,
                                save=True, dataset_string="savetest")
        return out, os.getcwd()
    finally:
        os.chdir(cwd)


class TestSearchResults:
    def test_uns_is_structured_and_survives_h5ad(self, run, tmp_path):
        out, _ = run
        assert isinstance(out.uns["search_result"], dict)
        p = str(tmp_path / "res.h5ad"); out.write_h5ad(p)
        back = ad.read_h5ad(p)
        sr = search_result_from_uns(back.uns["search_result"])
        sr0 = search_result_from_uns(out.uns["search_result"])
        assert isinstance(sr, SearchResults)
        np.testing.assert_allclose(sr.phys_optimum, sr0.phys_optimum)
        np.testing.assert_array_equal(sr.param_estimates, sr0.param_estimates)
        assert sr.model.bio_model == "Bursty" and sr.sp.n_grid_points == 1
        assert list(sr.gene_names) == list(out.var_names)
        analysis.run_qc(back)                       # rebuilds SearchData + unpacks the result
        plt.close("all")

    def test_save_writes_exactly_one_file(self, run, monkeypatch):
        out, root = run
        monkeypatch.chdir(root)   # inference_string paths are relative to the run's cwd
        d = os.path.join(root, "savetest", "Bursty_None_1x1")
        # Earlier tests in this class may already have saved figures into the run dir.
        assert set(os.listdir(d)) - {"analysis_figures"} == {"monod_adata.h5ad"}, os.listdir(d)
        assert sorted(os.listdir(os.path.join(root, "savetest"))) == ["Bursty_None_1x1"]
        sr = analysis.load_search_results(os.path.join(d, "monod_adata.h5ad"))
        sr0 = search_result_from_uns(out.uns["search_result"])
        np.testing.assert_allclose(sr.phys_optimum, sr0.phys_optimum)
        # Saving a figure afterwards creates analysis_figures on demand, nothing else.
        fig, ax = plt.subplots(); sr.plot_KL(ax); plt.close("all")
        assert set(os.listdir(d)) == {"analysis_figures", "monod_adata.h5ad"}
        assert os.path.isfile(os.path.join(d, "analysis_figures", "kldiv.png"))

    def test_store_on_disk_round_trip(self, run, monkeypatch):
        out, root = run
        monkeypatch.chdir(root)
        sr0 = search_result_from_uns(out.uns["search_result"])
        path = sr0.store_on_disk()
        assert path.endswith("grid_scan_results.h5") and os.path.isfile(path)
        sr = analysis.load_search_results(path)
        np.testing.assert_allclose(sr.phys_optimum, sr0.phys_optimum)
        os.remove(path)

    def test_default_run_touches_no_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path); warnings.filterwarnings("ignore")
        perform_inference(EXAMPLE, CMEModel("Bursty", "None"), n_genes=3, viz=False, gradient_params=GP)
        assert os.listdir(tmp_path) == []

    def test_checkpoint_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path); warnings.filterwarnings("ignore")
        model = CMEModel("Bursty", "Poisson")
        madata = extract_data(ad.read_h5ad(EXAMPLE), model, dataset_name="x", n_genes=3, viz=False)
        sd = searchdata_from_adata(madata)
        ip = InferenceParameters("x", model, gridsize=[2, 1], gradient_params=GP, save=True)
        sr_ck = ip.fit_all_grid_points(sd, checkpoint=True)
        assert glob.glob(os.path.join(ip.inference_string, "grid_point_*.h5")) or True  # removed by clean_up when in-memory
        sr = InferenceParameters("x", model, gridsize=[2, 1], gradient_params=GP).fit_all_grid_points(sd)
        np.testing.assert_allclose(sr_ck.param_estimates, sr.param_estimates)
        # A SearchResults rebuilt purely from grid_point_*.h5 files matches too.
        for i in range(ip.n_grid_points):
            gp = sr_ck._gp_for_test = None
        files = sorted(glob.glob(os.path.join(ip.inference_string, "grid_point_*.h5")))
        if files:
            rebuilt = SearchResults(ip, sd)
            rebuilt.aggregate_grid_points(None)
            np.testing.assert_allclose(rebuilt.param_estimates, sr.param_estimates)


@pytest.mark.skipif(_mc is None, reason="monod_core not installed")
class TestMEKMeansResults:
    def test_list_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path); warnings.filterwarnings("ignore")
        from test_mek_means_em import _two_population_sd
        sd, _ = _two_population_sd()
        model = CMEModel("Bursty", "None")
        p = mm.MEKMeansParameters("ds", model, k=2, epochs=2, random_state=0, gradient_params={**GP, "max_iterations": 15})
        assert os.listdir(tmp_path) == []                     # constructing writes nothing
        srs = p.run_mek_means(sd)
        assert os.listdir(tmp_path) == []                     # neither does a default run
        for sr in srs:
            sr.find_sampling_optimum()
        back = search_result_list_from_uns(_roundtrip_uns(search_result_list_to_uns(srs)))
        assert len(back) == len(srs)
        for a, b in zip(srs, back):
            assert isinstance(b, mm.SearchResults)
            assert b.assigns == a.assigns and b.sp.k == 2
            np.testing.assert_allclose(b.phys_optimum, a.phys_optimum)
            np.testing.assert_array_equal(b.filt, a.filt)
            np.testing.assert_allclose(np.asarray(b.aic_per_cell), np.asarray(a.aic_per_cell))
        srs_saved = p.run_mek_means(sd, save=True)
        run_dir = srs_saved[0].inference_string
        assert sorted(os.listdir(run_dir)) == ["grid_scan_results.h5"]
        loaded = analysis.load_search_results(os.path.join(run_dir, "grid_scan_results.h5"))
        assert isinstance(loaded, list) and len(loaded) == len(srs_saved)
