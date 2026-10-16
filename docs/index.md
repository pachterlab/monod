# Monod documentation

Monod fits mechanistic models of transcription to single-cell sequencing data and infers each gene's rate parameters, with the technical noise of sequencing modelled explicitly. The [README](../README.md) has the short introduction and a quick start; these pages go into more depth.

- [Installation](installation.md): installing Monod and the optional Rust accelerator, the GPU path, and building the accelerator from source.
- [Running an analysis](usage.md): input data, choosing a model, running inference, reading the results, and what is written to disk.
- [meK-Means](mek-means.md): model-based clustering of cells by transcriptional kinetics.

Worked examples live in the `demo_*.ipynb` notebooks at the repository root and in the [example repository](https://github.com/pachterlab/monod_examples). The method is described in the [Monod paper](https://www.nature.com/articles/s41592-025-02832-x) (Nature Methods, 2025); the theory in [Gorin & Pachter (2022)](https://www.cell.com/biophysreports/fulltext/S2667-0747(22)00054-4); meK-Means in [Chari, Gorin & Pachter (2024)](https://www.nature.com/articles/s43588-024-00689-2).
