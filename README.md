# *Monod*: a package for CME inference from seq data

[![PyPI version](https://badge.fury.io/py/monod.svg?branch=main&kill_cache=1)](https://badge.fury.io/py/monod)

This repository contains the current version of the *Monod* package, which implements procedures for fitting and analyzing stochastic models of transcription and sequencing. The description of the package's functionality and several applications are given in the [paper](https://www.nature.com/articles/s41592-025-02832-x).

Examples and tutorials are given at the [example repository](https://github.com/pachterlab/monod_examples). For theoretical background, see [Gorin & Pachter 2022](https://www.cell.com/biophysreports/fulltext/S2667-0747(22)00054-4). For usage instructions, see the [readthedocs website](https://monod-examples.readthedocs.io/en/latest/index.html). For a sample workflow applied a small dataset, see the [Colaboratory notebook](https://github.com/pachterlab/monod/blob/main/Monod_demo_new.ipynb).

## Installation

Python 3.9 or newer is required.

```bash
git clone https://github.com/pachterlab/monod.git
cd monod
pip install -e .
```

This installs the pure-Python package and its dependencies (numpy, scipy, numba, anndata, scanpy, scikit-learn, and friends). Everything works without the Rust extension below, just more slowly.

### Optional: Rust acceleration (`monod_core`)

The `monod_core` extension replaces the hot loops (PSS evaluation, histogramming, L-BFGS-B, chi-square and log-likelihood scoring) with multithreaded Rust. Monod detects it automatically at import time and falls back to Python when it is missing.

Requirements: a Rust toolchain (`rustup`), a C compiler, a system HDF5 library discoverable via `pkg-config` or `HDF5_DIR`, and [maturin](https://www.maturin.rs/).

```bash
pip install maturin
cd monod_core
maturin build --release --compatibility linux -i "$(which python)"   # drop --compatibility on macOS
pip install --force-reinstall --no-deps target/wheels/monod_core-*.whl
cd ..
python -c "import monod_core; print('Rust extension OK, meK-Means init:', hasattr(monod_core, 'initialize_q_kmeans'))"
```

The `kmeans` cargo feature (the Rust k-means initializer used by meK-Means) is on by default; building with `--no-default-features` drops it, in which case meK-Means initializes through scikit-learn instead. The `ruanndata` crate is fetched from GitHub at a pinned commit by cargo; if you are developing ruanndata alongside monod, see the commented `[patch]` block at the end of `monod_core/Cargo.toml`.

On Apple Silicon with an x86_64 (Rosetta) Homebrew HDF5, set `HDF5_NO_RUNTIME_CHECK=1` for the build.

### Optional: GPU

If PyTorch is installed and a CUDA device with float64 support is available, the two-species PSS evaluation runs on the GPU. Apple MPS does not support float64 and is not used.

## Quick start

```python
import anndata as ad
from monod.cme_toolbox import CMEModel
from monod.inference import perform_inference

model = CMEModel("Bursty", "Poisson")        # biology, technical noise
adata = perform_inference("example_h5ad/example_adata.h5ad", model, n_genes=100)
adata.var[["$b$", r"$\beta$", r"$\gamma$", "AIC"]].head()
```

`perform_inference` filters genes, builds the per-gene count histograms, scans a grid of technical-noise parameters, fits every gene by maximum likelihood at each grid point, and writes the log10 parameter estimates and AIC into `adata.var`. Parameter columns are named with the model's TeX labels minus the log prefix (`$b$`, `$\beta$`, `$\gamma$`; see `model.param_str`), and fold-change columns written by `monod.analysis.DE_parameters` follow the same convention (`FC_$b$`, `adj_FC_$b$`). For finer control use `extract_data`, `searchdata_from_adata`, and `InferenceParameters.fit_all_grid_points` directly; the `demo_*.ipynb` notebooks walk through each of these, the reaction-network and protein models, and the Rust and GPU paths.

### meK-Means (mixture-model clustering)

`monod.mminference.MEKMeansParameters` fits K clusters jointly with per-cluster CME parameters (the meK-Means algorithm of Carilli et al.). It takes the same bounds and `gradient_params` as `InferenceParameters` plus `k`, `epochs`, and three controls that are not in the paper's Algorithm 1: `random_state` seeds the k-means-plus-perturbation initialization so reruns are reproducible; `init_modality` chooses the counts k-means sees (`"sum"` = U + S, the default; `"spliced"` = the paper's description); and `em_tol` / `em_patience` stop EM early once the per-cell log-likelihood bound plateaus. Pass `em_tol=0` to run a fixed number of epochs exactly as published. The per-run `aic_per_cell` on the results follows the paper's definition (lower is better; pick K with `argmin`). The old `aic` attribute still reads but warns, because its sign convention was the opposite.

Input expectations: an AnnData whose `layers` hold integer counts for the model's modalities (`unspliced` and `spliced` by default; map other names with `modality_name_dict`). Gene lengths, if used, come from a transcriptome file or `adata.var['log_lengths']`.

### Logging and stored objects

monod logs through the `"monod"` logger and never reconfigures Python logging itself. To see progress messages, configure logging in your application, for example `logging.basicConfig(); logging.getLogger("monod").setLevel(logging.INFO)`.

### Files on disk

By default monod writes nothing. Results live in memory and in the returned AnnData: `adata.uns['model']` and `adata.uns['search_result']` (or `search_result_list` for meK-Means) are nested dicts of arrays and scalars, not pickles, so the AnnData writes to a plain h5ad with `adata.write_h5ad()` and reads back anywhere. Rebuild the Python objects with `CMEModel.from_dict`, `monod.inference.search_result_from_uns`, or `search_result_list_from_uns`.

When you do ask for files:

| Call | What is written |
|---|---|
| `perform_inference(..., save=True)` | one file, `<dataset>/<model>_<grid>/monod_adata.h5ad`, holding data, model, parameters, results and cluster labels |
| `SearchResults.store_on_disk()` or `run_mek_means(save=True)` | one `grid_scan_results.h5` in the run directory |
| `InferenceParameters(..., save=True)` | `parameters.h5` in the run directory |
| `fit_all_grid_points(..., checkpoint=True)` | one `grid_point_<i>.h5` per grid point, for crash recovery |
| any plot method with `savefig=True` | `analysis_figures/<name>.png`, created on first use, only if the run directory exists |

`monod.analysis.load_search_results(path)` reads any of the h5ad or h5 files above (and, with a warning, legacy pickled `.res` files). Run directories are resolved to absolute paths at construction, from the working directory at that time.

## Releasing

Continuous integration lives in `.github/workflows/`:

- `tests.yml` runs the test suite on every push and pull request, with and without the Rust extension.
- `wheels.yml` runs on every `v*` tag (or by hand from the Actions tab). It builds self-contained `monod-core` wheels (Linux x86_64, macOS arm64 and x86_64; Python 3.9 to 3.12; HDF5 linked statically) plus the `monod_anndata` sdist and wheel, creates a GitHub release with those files attached and auto-generated notes, and publishes both packages to PyPI.

To cut a release:

1. Bump the version in `setup.cfg`, `monod_core/Cargo.toml`, and `monod_core/pyproject.toml`, and commit.
2. `git tag vX.Y.Z && git push origin vX.Y.Z`.

One-time setup before the first tagged release: create a `pypi` environment in the repository settings (Settings, Environments), and on PyPI add a *trusted publisher* for each project (`monod-core` and `monod_anndata`) pointing at `pachterlab/monod`, workflow `wheels.yml`, environment `pypi`. With that in place no API token is stored anywhere; the workflow authenticates with a short-lived token from GitHub.

## Development

```bash
pip install -e .[test]
python -m pytest tests/ -q
```

The tests run against `src/` directly (see `tests/conftest.py`), so an editable install is optional. Tests that need the large `gaba_example.h5ad` data are skipped when the file is absent.
