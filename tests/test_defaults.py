"""Defaults must not crash the simplest calls (audit fix 2, 2026-10-15).

Covers: extract_data without dataset_name, the use_lengths="auto" default in
InferenceParameters / MEKMeansParameters, explicit-vs-auto behaviour when the
search data carries no gene lengths, and the legacy bool handling.
"""
import os

import anndata as ad
import numpy as np
import pytest

from monod.cme_toolbox import CMEModel
from monod.extract_data import extract_data
from monod.inference import (
    GradientInference,
    InferenceParameters,
    _normalize_use_lengths,
    searchdata_from_adata,
)
import monod.mminference as mminference
_HERE = os.path.dirname(__file__)
EXAMPLE = os.path.join(_HERE, "..", "example_h5ad", "example_adata.h5ad")


@pytest.fixture(scope="module")
def example_adata():
    return ad.read_h5ad(EXAMPLE)


class TestExtractDataDefaults:
    def test_no_dataset_name(self, example_adata):
        out = extract_data(example_adata.copy(), CMEModel("Bursty", "None"), n_genes=3, viz=False, ambiguous="drop")
        assert out.n_vars >= 1


class TestUseLengthsNormalisation:
    @pytest.mark.parametrize(
        "seq,expected", [("None", None), ("Poisson", "unspliced"), ("Bernoulli", None)]
    )
    def test_auto_resolution(self, seq, expected):
        mode, explicit = _normalize_use_lengths("auto", CMEModel("Bursty", seq))
        assert mode == expected
        assert explicit is False

    def test_legacy_bools(self):
        m = CMEModel("Bursty", "Poisson")
        assert _normalize_use_lengths(True, m) == ("unspliced", True)
        assert _normalize_use_lengths(False, m) == (None, True)

    def test_explicit_on_none_model_raises(self):
        with pytest.raises(ValueError, match="requires seq_model='Poisson'"):
            _normalize_use_lengths("unspliced", CMEModel("Bursty", "None"))

    def test_explicit_on_bernoulli_raises(self):
        with pytest.raises(ValueError, match="requires seq_model='Poisson'"):
            _normalize_use_lengths(True, CMEModel("Bursty", "Bernoulli"))

    def test_bad_value_raises(self):
        with pytest.raises(ValueError, match="use_lengths must be"):
            _normalize_use_lengths("nascent", CMEModel("Bursty", "Poisson"))


class TestInferenceParametersDefaults:
    def test_none_model_default_constructs(self):
        ip = InferenceParameters("ds", CMEModel("Bursty", "None"))
        assert ip.use_lengths is None

    def test_poisson_model_default(self):
        ip = InferenceParameters("ds", CMEModel("Bursty", "Poisson"))
        assert ip.use_lengths == "unspliced"
        assert ip.use_lengths_explicit is False

    def test_none_model_explicit_raises_at_construction(self):
        with pytest.raises(ValueError, match="requires seq_model='Poisson'"):
            InferenceParameters("ds", CMEModel("Bursty", "None"), use_lengths="unspliced")


class TestMEKMeansDefaults:
    def test_none_model_default_constructs(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        p = mminference.MEKMeansParameters("ds", CMEModel("Bursty", "None"), k=2, epochs=1)
        assert p.use_lengths is None


@pytest.fixture(scope="module")
def poisson_sd(example_adata):
    m = CMEModel("Bursty", "Poisson")
    madata = extract_data(example_adata.copy(), m, dataset_name="x", n_genes=3, viz=False, ambiguous="drop")
    sd = searchdata_from_adata(madata)
    if getattr(sd, "gene_log_lengths", None) is not None:
        pytest.skip("example data unexpectedly carries gene lengths")
    return m, sd


class TestGradientInferenceLengthFallback:
    def test_auto_without_lengths_falls_back(self, poisson_sd):
        m, sd = poisson_sd
        ip = InferenceParameters("ds", m, gridsize=[1, 1])
        gi = GradientInference(ip, m, sd, 0)
        assert np.all(np.isfinite(gi.regressor))

    def test_explicit_without_lengths_raises_clearly(self, poisson_sd):
        m, sd = poisson_sd
        ip = InferenceParameters("ds", m, use_lengths="unspliced", gridsize=[1, 1])
        with pytest.raises(ValueError, match="no gene lengths"):
            GradientInference(ip, m, sd, 0)


def test_core_loader_rejects_namespace_lookalike(monkeypatch):
    """A bare `monod_core/` source dir imports as an empty namespace package; the loader must say no."""
    import sys, types
    from monod import _core
    monkeypatch.setitem(sys.modules, "monod_core", types.ModuleType("monod_core"))
    assert _core.load() is None
    monkeypatch.delitem(sys.modules, "monod_core")
    monkeypatch.setitem(sys.modules, "monod_core", None)   # makes `import monod_core` raise ImportError
    assert _core.load() is None
