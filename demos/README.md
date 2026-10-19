# Demo notebooks

Each notebook runs from this folder or from the repository root; the first cell
changes to the repository root so that data (`example_h5ad/`) and figure paths
resolve. The PBMC notebooks need `example_h5ad/processed_pbmc_10k_raw.h5ad`,
which is not in the repository.

| Notebook | What it shows | Runtime |
|---|---|---|
| [`monod_walkthrough.ipynb`](monod_walkthrough.ipynb) | End-to-end analysis on a small dataset; opens in Colab | minutes |
| [`analysis_workflow.ipynb`](analysis_workflow.ipynb) | After the fit: gene rejection, standard errors, parameter-level differential expression between two PBMC cell types, saving results | under a minute |
| [`models.ipynb`](models.ipynb) | The six two-species transcription models, their parameters and distributions, and a fit to real data | minutes |
| [`reaction_networks.ipynb`](reaction_networks.ipynb) | Defining custom reaction networks with the reaction-string syntax | minutes |
| [`protein_bursty.ipynb`](protein_bursty.ipynb) | The three-species protein model: moments, 3-D distribution, L-BFGS-B vs Nelder-Mead fits | about a minute |
| [`mek_means.ipynb`](mek_means.ipynb) | meK-Means clustering on PBMC 10k with a confusion matrix against known cell types | about a minute |
| [`mek_means_rust.ipynb`](mek_means_rust.ipynb) | The Rust fast paths inside meK-Means, step by step, with timings | minutes |
| [`mek_means_estep_backends.ipynb`](mek_means_estep_backends.ipynb) | The GPU, Rust and Python backends for the meK-Means E-step | minutes |
| [`rust_extension.ipynb`](rust_extension.ipynb) | Checking the Rust accelerator is active and comparing results with and without it | minutes |
| [`rust_pipeline.ipynb`](rust_pipeline.ipynb) | The all-Rust path from h5ad file to fitted parameters, with and without gene-length correction | minutes |
| [`rust_optimizer.ipynb`](rust_optimizer.ipynb) | Benchmarks of the Rust KLD, gradient and L-BFGS-B entry points against scipy, up to 1000 genes | about 20 minutes |
| [`rust_vs_python_benchmark.ipynb`](rust_vs_python_benchmark.ipynb) | End-to-end speed and result concordance of the Rust and Python paths | over an hour |
| [`inference_speed_pbmc.ipynb`](inference_speed_pbmc.ipynb) | Inference time against core count, gene count and cell count | about 20 minutes |

Figures are written to `figures/` at the repository root, which is ignored by git.
