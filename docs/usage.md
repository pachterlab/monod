# Running an analysis

## Input data

Monod works on an [AnnData](https://anndata.readthedocs.io) object whose `layers` hold integer counts for the model's modalities: `unspliced` and `spliced` by default, as produced by kallisto|bustools, velocyto or similar tools. Layers with other names are mapped with `modality_name_dict`. Gene lengths, if a model uses them, come from a transcriptome file or `adata.var['log_lengths']`.

## Choosing a model

A `CMEModel` combines three ingredients:

```python
from monod.cme_toolbox import CMEModel
model = CMEModel("Bursty", "Poisson")                      # transcription, sequencing noise
model = CMEModel("Bursty", "Bernoulli", amb_model="Equal")  # plus ambient contamination
```

- **Transcription** (`bio_model`): `Constitutive`, `Bursty`, `CIR`, `Extrinsic`, `Delay`, `DelayedSplicing`, `ProteinBursty` (adds a protein layer), or a custom reaction network passed as `network=`. `Bursty` is the usual choice for mammalian and bacterial cells.
- **Sequencing noise** (`seq_model`): `None`, `Poisson`, or `Bernoulli`.
- **Ambient contamination** (`amb_model`): `None`, `Equal`, or `Unequal`.

The `models`, `reaction_networks` and `protein_bursty` notebooks in `demos/` show each family.

### A note on the protein model

`ProteinBursty` adds protein translation (`k_p`) and degradation (`γ_p`) to the bursty model. Two properties of this model are worth knowing before fitting it:

- **`k_p` and `γ_p` are only separately identifiable when protein turnover is comparable to or slower than mRNA turnover.** When protein degrades much faster than mRNA, the protein count tracks the mRNA count instantly and the data only constrain the ratio `k_p/γ_p`; the likelihood is then flat along the `k_p + γ_p` direction and the fitted pair can sit anywhere along it with the same quality of fit. Inspect the two estimates together, and treat their ratio as the robust quantity.
- **Evaluation time grows with the spread between the fastest and slowest rate.** The steady state is obtained by integrating an ordinary differential equation with a step set by the fastest rate over a horizon set by the slowest, so a gene whose fitted rates span four orders of magnitude costs a hundred times more per evaluation than one whose rates are within a decade of each other. Tight parameter bounds keep fits fast.

Because the finite-difference gradient is unreliable on the flat valley, the protein model is fitted in two stages by default: a short Nelder-Mead search from the moments estimate, then the usual L-BFGS-B polish. This is controlled by `gradient_params["presearch"]` (`"auto"`, `"nelder-mead"` or `None`) and `gradient_params["presearch_maxiter"]` (default 300).

## Running inference

```python
from monod.inference import perform_inference
adata = perform_inference("example_h5ad/example_adata.h5ad", model, n_genes=100)
```

`perform_inference` filters genes, builds the per-gene count histograms, scans a grid of technical-noise parameters, fits every gene by maximum likelihood at each grid point, and writes the results into the returned AnnData. For finer control, the same steps are available individually: `extract_data`, `searchdata_from_adata`, and `InferenceParameters.fit_all_grid_points`; the `rust_pipeline` notebook in `demos/` walks through them.

## Reading the results

Per-gene log10 parameter estimates and the AIC land in `adata.var`. Parameter columns are named with the model's TeX labels minus the log prefix, so for the bursty model they are `$b$` (burst size), `$\beta$` (splicing rate) and `$\gamma$` (degradation rate); `model.param_str` lists them for any model. Fold-change columns written by `monod.analysis.DE_parameters` follow the same convention (`FC_$b$`, `adj_FC_$b$`).

The full results objects are stored in `adata.uns`:

- `adata.uns['model']`: the model definition;
- `adata.uns['search_result']` (or `search_result_list` for meK-Means): fits at every grid point, the chosen optimum, and quality metrics.

These are nested dictionaries of arrays and scalars, not pickles, so the AnnData writes to a plain `.h5ad` with `adata.write_h5ad()` and reads back anywhere. To get the Python objects back:

```python
from monod.cme_toolbox import CMEModel
from monod.inference import search_result_from_uns
model = CMEModel.from_dict(adata.uns["model"])
result = search_result_from_uns(adata.uns["search_result"])
```

Plots, gene rejection, noise decomposition and differential-parameter tests are methods on the result object and functions in `monod.analysis`; the `monod_walkthrough` notebook in `demos/` shows the typical sequence.

## What is written to disk

By default, nothing. Results live in memory and in the returned AnnData. Files are written only on request:

| Call | What is written |
|---|---|
| `perform_inference(..., save=True)` | one file, `<dataset>/<model>_<grid>/monod_adata.h5ad`, holding data, model, parameters, results and cluster labels |
| `SearchResults.store_on_disk()` or `run_mek_means(save=True)` | one `grid_scan_results.h5` in the run directory |
| `InferenceParameters(..., save=True)` | `parameters.h5` in the run directory |
| `fit_all_grid_points(..., checkpoint=True)` | one `grid_point_<i>.h5` per grid point, for crash recovery |
| any plot method with `savefig=True` | `analysis_figures/<name>.png`, created on first use, only if the run directory exists |

`monod.analysis.load_search_results(path)` reads any of these (and, with a warning, legacy pickled `.res` files from earlier versions). Run directories are resolved to absolute paths when the parameters object is created, relative to the working directory at that time.

## Logging

Monod reports progress through the standard `"monod"` logger and never reconfigures Python logging itself. To see messages:

```python
import logging
logging.basicConfig()
logging.getLogger("monod").setLevel(logging.INFO)
```
