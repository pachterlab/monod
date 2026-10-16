"""Monod: fitting and analysing stochastic models of transcription and sequencing.

Submodules
----------
cme_toolbox       CMEModel: biological + technical noise models, PSS evaluation
extract_data      AnnData ingestion, gene filtering, histogram construction
inference         InferenceParameters / SearchResults: per-gene grid-scan inference
mminference       MEKMeansParameters: meK-Means mixture-model inference
analysis          QC and differential-expression helpers on fitted results
reaction_network  ReactionNetwork: user-defined CME networks for CMEModel("Custom")

Submodules are imported lazily: ``import monod.inference`` and ``monod.inference``
both work, and ``import monod`` alone stays cheap.
"""
import importlib as _importlib
from importlib.metadata import PackageNotFoundError as _PNF, version as _version

try:
    __version__ = _version("monod")
except _PNF:  # running from a checkout without pip install
    __version__ = "0.0.0+unknown"

__all__ = [
    "cme_toolbox",
    "extract_data",
    "inference",
    "mminference",
    "analysis",
    "reaction_network",
    "plot_aesthetics",
]


def __getattr__(name):
    if name in __all__:
        return _importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
