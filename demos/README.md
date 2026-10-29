# Demo notebooks

Each notebook runs from this folder or from the repository root; the first cell
changes to the repository root so that data (`example_h5ad/`) and figure paths
resolve. The PBMC notebooks need `example_h5ad/processed_pbmc_10k_raw.h5ad`,
which is not in the repository, and `sequencing_noise.ipynb` needs the Allen
cortex data and the poly(A) length reference from the
[monod_examples](https://github.com/pachterlab/monod_examples) repository (see its first cells).

| Notebook | What it shows | Runtime |
|---|---|---|
| [`monod_walkthrough.ipynb`](monod_walkthrough.ipynb) | End-to-end analysis on a small dataset; opens in Colab | minutes |
| [`analysis_workflow.ipynb`](analysis_workflow.ipynb) | After the fit: gene rejection, standard errors, parameter-level differential expression between two PBMC cell types, saving results | under a minute |
| [`sequencing_noise.ipynb`](sequencing_noise.ipynb) | The Poisson and Bernoulli sequencing models, gene-length bias, the sampling landscape and the burst-size trade-off, noise decomposition, shared optima for two cell types, ambiguous assignments | about 15 minutes |
| [`models.ipynb`](models.ipynb) | The six two-species transcription models, their parameters and distributions, and a fit to real data | minutes |
| [`reaction_networks.ipynb`](reaction_networks.ipynb) | Defining custom reaction networks with the reaction-string syntax, fitting one to data, checking that a network is identifiable, and time-delayed reactions, compared with the immediate network across a panel of 40 genes | about 10 minutes |
| [`protein_bursty.ipynb`](protein_bursty.ipynb) | The three-species protein model: moments, 3-D distribution, L-BFGS-B vs Nelder-Mead fits | about a minute |
| [`model_comparison.ipynb`](model_comparison.ipynb) | Comparing two models on a gene: which pairs nest and where, the mean-rate burst parametrisation that makes constitutive a point of bursty, the nested likelihood-ratio test with its warm start on a 40-gene panel, and Vuong's test for a pair that does not nest | about ten minutes, needs `gaba_example.h5ad` |
| [`chromatin.ipynb`](chromatin.ipynb) | The chromatin model: locus configurations driving nascent and mature RNA, the three transcription rules, constitutive versus bursty emission, building the configuration layer from a peak matrix, fitting with ATAC dropout on the grid, and comparing rules and emissions with the nested test | a few minutes |
| [`mek_means.ipynb`](mek_means.ipynb) | meK-Means clustering on PBMC 10k with a confusion matrix against known cell types | about a minute |
| [`mek_means_rust.ipynb`](mek_means_rust.ipynb) | The Rust fast paths inside meK-Means, step by step, with timings | minutes |
| [`mek_means_estep_backends.ipynb`](mek_means_estep_backends.ipynb) | The Rust and Python backends for the meK-Means E-step, with timings | minutes |
| [`gpu.ipynb`](gpu.ipynb) | Which backend runs and why; GPU against Rust and Python for one evaluation and for batches of up to 1000 genes | a few minutes, needs a CUDA GPU |
| [`rust_extension.ipynb`](rust_extension.ipynb) | Checking the Rust accelerator is active and comparing results with and without it | minutes |
| [`rust_pipeline.ipynb`](rust_pipeline.ipynb) | The all-Rust path from h5ad file to fitted parameters, with and without gene-length correction | minutes |
| [`rust_benchmarks.ipynb`](rust_benchmarks.ipynb) | Speed of the Rust KLD kernels and the Rust L-BFGS-B optimizer against scipy and Python, on synthetic genes and on PBMC 10k across gene, core and cell counts, and whether the two optimizers reach the same optima across the sampling grid | about 15 minutes |

Figures are written to `figures/` at the repository root, which is ignored by git.
