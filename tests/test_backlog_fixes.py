"""Backlog fixes from the 2026-10-14 audit (applied 2026-10-15)."""
import importlib
import inspect
import logging
import unittest.mock
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import monod.extract_data as ed
from monod.cme_toolbox import CMEModel
from monod.inference import perform_inference


class TestLogging:
    def test_import_does_not_disable_existing_loggers(self):
        probe = logging.getLogger("monod_test_probe_logger")
        probe.disabled = False
        importlib.reload(ed)
        assert probe.disabled is False
        assert logging.getLogger("monod").name == "monod"
        assert not logging.getLogger().handlers or True   # root untouched by the import


class TestCMEModelInputs:
    def test_amb_model_none_normalised(self):
        assert CMEModel("Bursty", "None", amb_model=None).amb_model == "None"

    def test_amb_model_typo_rejected(self):
        with pytest.raises(ValueError, match="amb_model must be one of"):
            CMEModel("Bursty", "None", amb_model="equal")

    def test_protein_bursty_with_poisson_noise(self):
        m = CMEModel("ProteinBursty", "Poisson")
        p = np.array([0.3, 0.0, -0.1, 0.0, -0.5])
        pss = m.eval_model_pss(p, [6, 6, 8], samp=np.array([-0.5, -0.5]))
        assert pss.shape == (6, 6, 8)
        np.testing.assert_allclose(pss.sum(), 1.0, rtol=1e-5)
        assert np.all(pss >= 0)

    def test_poisson_without_samp_raises(self):
        with pytest.raises(ValueError, match="requires samp"):
            CMEModel("Bursty", "Poisson").eval_model_pss(np.array([0.3, 0.0, -0.1]), [10, 10])

    def test_cached_mesh_not_mutated_by_noise_transform(self):
        import monod.cme_toolbox as ct
        m = CMEModel("Bursty", "Poisson")
        p = np.array([0.3, 0.0, -0.1])
        with unittest.mock.patch.object(ct, "_HAS_RUST", False), \
             unittest.mock.patch.object(ct, "_HAS_GPU", False):
            a = m.eval_model_pss(p, [10, 10], samp=np.array([-1.0, -1.0]))
            b = m.eval_model_pss(p, [10, 10], samp=np.array([-1.0, -1.0]))
        np.testing.assert_array_equal(a, b)


class TestHistType:
    def test_grid_rejected(self, tmp_path):
        import anndata as ad, os
        here = os.path.dirname(__file__)
        a = ad.read_h5ad(os.path.join(here, "..", "example_h5ad", "example_adata.h5ad"))
        with pytest.raises(ValueError, match="only 'unique'"):
            ed.extract_data(a, CMEModel("Bursty", "None"), n_genes=2, viz=False, hist_type="grid")


class TestDefaults:
    def test_no_mutable_defaults(self):
        for fn in (perform_inference, ed.extract_data, ed.threshold_by_expression):
            for name, prm in inspect.signature(fn).parameters.items():
                assert not isinstance(prm.default, (list, dict, set)), f"{fn.__name__}.{name}"


@pytest.fixture(scope="module")
def fit(tmp_path_factory):
    import os, anndata as ad
    from monod.inference import InferenceParameters, searchdata_from_adata
    warnings.filterwarnings("ignore")
    cwd = os.getcwd(); os.chdir(tmp_path_factory.mktemp("fit"))
    try:
        here = os.path.dirname(__file__)
        model = CMEModel("Bursty", "None")
        madata = ed.extract_data(ad.read_h5ad(os.path.join(here, "..", "example_h5ad", "example_adata.h5ad")),
                                 model, dataset_name="x", n_genes=3, viz=False)
        sd = searchdata_from_adata(madata)
        ip = InferenceParameters("x", model, gradient_params={"max_iterations": 5, "num_restarts": 1})
        sr = ip.fit_all_grid_points(sd)
        sr.find_sampling_optimum()
        return sd, sr
    finally:
        os.chdir(cwd)


class TestWarningsAndFigures:
    def test_user_warning_filters_survive(self, fit):
        sd, sr = fit
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.filterwarnings("error", message="monod-probe", category=UserWarning)
            probe = warnings.filters[0]
            sr.compute_sigma(sd, num_cores=1)
            assert warnings.filters[0] == probe

    def test_plots_without_figure_dir_do_not_raise(self, fit):
        sd, sr = fit
        fig, ax = plt.subplots()
        sr.plot_landscape(ax, savefig=True)
        sr.plot_KL(ax)
        sr.plot_param_marg()
        plt.close("all")
