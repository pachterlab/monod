# meK-Means

meK-Means ([Chari, Gorin & Pachter, 2024](https://www.nature.com/articles/s43588-024-00689-2)) clusters cells by their transcriptional kinetics: it fits K clusters jointly with a set of CME parameters per cluster, alternating between assigning cells to the cluster whose model explains them best and refitting each cluster's parameters.

```python
from monod.mminference import MEKMeansParameters
params = MEKMeansParameters(dataset_string, model, k=4, epochs=20, random_state=0)
results = params.run_mek_means(search_data)     # a list of SearchResults, one per cluster
```

`MEKMeansParameters` takes the same arguments, bounds and `gradient_params` as `InferenceParameters`, plus `k` and `epochs`. Three further controls are not part of the paper's Algorithm 1 and default to behaviour that matches it unless noted:

- `random_state` seeds the k-means-plus-perturbation initialization so reruns are reproducible.
- `init_modality` chooses which counts the initial k-means sees: `"sum"` (unspliced + spliced, the default) or `"spliced"` (the paper's description).
- `em_tol` and `em_patience` stop EM early once the per-cell log-likelihood bound plateaus. Pass `em_tol=0` to run a fixed number of epochs exactly as published.

## Choosing K

Each run exposes `aic_per_cell`, defined as in the paper; lower is better, so fit a range of K and take the `argmin`. The older `aic` attribute still reads but warns, because its sign convention was the opposite.

`mminference.cell_assignments(results)` turns the list of per-cluster results into one cluster index per cell. `analysis.plot_cluster_confusion(labels, assigns)` draws that against an annotation with the adjusted Rand index, and `analysis.match_clusters(reference, assigns)` aligns the arbitrary cluster ids of two runs before comparing them cell by cell.

The `mek_means` notebook in `demos/` shows a complete run, with and without the Rust accelerator; `mek_means_rust` times each accelerated step on its own and shows the two-stage procedure used with a technical-noise model (the technical parameters are fitted once on the whole population, then meK-Means runs at that fixed point).
