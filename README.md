# Monod

[![PyPI version](https://badge.fury.io/py/monod.svg?branch=main&kill_cache=1)](https://badge.fury.io/py/monod)

Monod fits mechanistic models of transcription to single-cell sequencing data. Instead of summarising each gene by its mean expression, it treats the unspliced and spliced counts of every gene as draws from a stochastic model of bursting, splicing and degradation, and infers the model's rate parameters gene by gene. The result is a biophysical description of each gene, with the technical noise of the sequencing process modelled explicitly rather than normalised away.

The method is described in the [Monod paper](https://www.nature.com/articles/s41592-025-02832-x) (Nature Methods, 2025), with the theory in [Gorin & Pachter (2022)](https://www.cell.com/biophysreports/fulltext/S2667-0747(22)00054-4). meK-Means, Monod's model-based clustering of cells, is described in [Chari, Gorin & Pachter (2024)](https://www.nature.com/articles/s43588-024-00689-2).

## Installation

```bash
pip install monod monod-core
```

Python 3.9 or newer is required. `monod-core` is an optional accelerator written in Rust, with prebuilt wheels for Linux and Apple-silicon Macs. Monod uses it automatically when it is present and runs in pure Python otherwise, with identical results and more patience required. If `monod-core` has no wheel for your platform, `pip install monod` alone is enough to get started.

## Quick start

Monod works on an [AnnData](https://anndata.readthedocs.io) object whose `layers` hold integer unspliced and spliced counts, as produced by kallisto|bustools, velocyto or similar tools.

```python
from monod.cme_toolbox import CMEModel
from monod.inference import perform_inference

model = CMEModel("Bursty", "Poisson")   # bursty transcription, Poisson sampling noise
adata = perform_inference("example_h5ad/example_adata.h5ad", model, n_genes=100)

adata.var[["$b$", r"$\beta$", r"$\gamma$", "AIC"]].head()
```

This selects the 100 most informative genes, fits each one, and writes the inferred burst size, splicing rate and degradation rate (all log10) into `adata.var`, alongside a goodness-of-fit score. Everything stays in the returned AnnData, which saves to an ordinary `.h5ad` file; nothing is written to disk unless you ask.

From there you can:

- compare parameters between conditions with `monod.analysis`,
- cluster cells by their transcriptional kinetics with meK-Means (`monod.mminference`),
- swap in other models of transcription, sequencing noise, or ambient contamination,
- add a protein layer or define your own reaction network.

## Learning more

- [`demos/monod_walkthrough.ipynb`](https://github.com/pachterlab/monod/blob/main/demos/monod_walkthrough.ipynb): an end-to-end walkthrough on a small dataset, runnable in Colab.
- The other notebooks in [`demos/`](https://github.com/pachterlab/monod/tree/main/demos): one topic each, covering the available models, reaction networks, the protein model, the analysis steps after a fit, meK-Means, and the Rust and GPU paths.
- The [example repository](https://github.com/pachterlab/monod_examples) and the [documentation site](https://monod-examples.readthedocs.io/en/latest/index.html).
- The [documentation pages](https://github.com/pachterlab/monod/tree/main/docs) in this repository: installation and the Rust accelerator, running an analysis and reading the results, and meK-Means.

## Citing

If you use Monod, please cite the [Monod paper](https://www.nature.com/articles/s41592-025-02832-x) (Nature Methods, 2025), and for meK-Means, [Chari, Gorin & Pachter (2024)](https://www.nature.com/articles/s43588-024-00689-2).
