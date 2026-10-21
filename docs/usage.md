# Running an analysis

## Input data

Monod works on an [AnnData](https://anndata.readthedocs.io) object whose `layers` hold integer counts for the model's modalities: `unspliced` and `spliced` by default, as produced by kallisto|bustools, velocyto or similar tools. Layers with other names are mapped with `modality_name_dict`. Gene lengths, if a model uses them, come from a transcriptome file or `adata.var['log_lengths']`.

## Choosing a model

A `CMEModel` combines three ingredients:

```python
from monod.cme_toolbox import CMEModel
model = CMEModel("Bursty", "Poisson")                      # transcription, sequencing noise
model = CMEModel("Bursty", "Bernoulli", amb_model="Equal")  # plus ambiguous read assignment
```

- **Transcription** (`bio_model`): `Constitutive`, `Bursty`, `CIR`, `Extrinsic`, `Delay`, `DelayedSplicing`, `ProteinBursty` (adds a protein layer), or a custom reaction network passed as `network=`, including time-delayed reactions written with `=>`. All of these run on the Rust accelerator when it is installed. `Bursty` is the usual choice for mammalian and bacterial cells.
- **Sequencing noise** (`seq_model`): `None`, `Poisson`, or `Bernoulli`. The `sequencing_noise` notebook in `demos/` shows what the models do, how the sampling landscape is scanned, and how the fitted variance splits into intrinsic, extrinsic and technical parts.
- **Ambiguous assignment** (`amb_model`): `None`, `Equal`, or `Unequal`. Models reads that cannot be assigned to the unspliced or spliced transcript and are reported in an `ambiguous` layer; each molecule is counted as ambiguous with a per-gene probability (one shared, or one per species).

The `models`, `reaction_networks` and `protein_bursty` notebooks in `demos/` show each family.

### A note on the protein model

`ProteinBursty` adds protein translation (`k_p`) and degradation (`γ_p`) to the bursty model. Two properties of this model are worth knowing before fitting it:

- **`k_p` and `γ_p` are only separately identifiable when protein turnover is comparable to or slower than mRNA turnover.** When protein degrades much faster than mRNA, the protein count tracks the mRNA count instantly and the data only constrain the ratio `k_p/γ_p`; the likelihood is then flat along the `k_p + γ_p` direction and the fitted pair can sit anywhere along it with the same quality of fit. Inspect the two estimates together, and treat their ratio as the robust quantity.
- **Evaluation cost is nearly independent of the rates.** The steady state is obtained by integrating the characteristic equations with an exponential integrator whose steps are chosen by error control, so a gene whose fitted rates span four orders of magnitude costs about the same as one whose rates are within a decade of each other (80 to 300 steps either way, accurate to a few parts per million on the generating function). Releases up to 0.3.2 used a fixed-step integrator whose cost grew linearly with the rate spread.

Because the finite-difference gradient is unreliable on the flat valley, the protein model is fitted in two stages by default: a short Nelder-Mead search from the moments estimate, then the usual L-BFGS-B polish. This is controlled by `gradient_params["presearch"]` (`"auto"`, `"nelder-mead"` or `None`) and `gradient_params["presearch_maxiter"]` (default 300).

### A note on custom networks

A reaction network is parsed, not vetted: nothing stops you from writing one whose parameters the data cannot tell apart. Monod checks for the common cases at three points.

- **When the network is built.** `ReactionNetwork` warns (a `ReactionNetworkWarning`) about a delayed reaction with no product (its delay changes nothing observable), two reactions with the same reactant and products (only their combined rate is identifiable), a species that is never produced (its steady state is zero) or never removed (it has no steady state), and an autocatalytic reaction. `network.structural_issues()` returns the same findings as a list, and `network.validate(warn=False)` runs the checks silently.
- **When inference starts.** The steady state of any network is unchanged when every rate constant (and inverse delay) is scaled together, so a network with all rates free has an exactly flat direction. Normalisation is off by default; `InferenceParameters` warns when a `Custom` model is fitted without `normalize_production_rate=True`. At the first grid point, `GradientInference` also runs `CMEModel.check_identifiability` at the first gene's starting point: a numerical rank test of the steady state with respect to the parameters that reports any flat direction by name (an `IdentifiabilityWarning`). `gradient_params["check_identifiability"]` is `"auto"` (custom networks only), `True` (every model) or `False`. You can call `model.check_identifiability(p, limits)` yourself at any parameter vector; the result holds the relative singular values and the directions.
Two practical notes. `perform_inference` accepts a `Custom` model directly: pass `modality_name_dict` to map species names to layers (for example `{"U": "unspliced", "S": "spliced"}`), and override `phys_lb`/`phys_ub` and `filt_param` if the defaults (burst sizes 0.1 to 16000 and rates 0.016 to 3200 in log10, as for the built-in models, and the mRNA filters for every species) do not suit. There is no moments estimator for an arbitrary network, so every gene starts from the middle of the parameter box; give the optimizer a few random restarts (`gradient_params={"num_restarts": 3}`), and when a network nests a simpler one check that its divergence is not higher than the simpler fit's, which marks a gene that ended in the wrong basin. Gradients of custom networks are central differences with a step of 1e-4 in log10, wide enough to stay clear of the noise floor that histogram bins deep in the tail put on the divergence. The network integrator propagates the linear part of the characteristic equations exactly through a matrix exponential and chooses its steps by error control, like the protein integrator, so an evaluation costs about the same whether the rates span one decade or five (60 to 300 steps on a two-species network, accurate to a few parts per million on the generating function). Delayed reactions add a history lookup and catalytic steps such as `S -> S + P` a nonlinear remainder; both are handled by the same scheme at some extra cost. Finite-difference gradients, the identifiability check and the Hessian behind the standard errors all replay the base point's step schedule (`CMEModel.pinned_schedule`), so the integrator's tolerance does not leak into differences taken over small steps.

- **After the fit.** `compute_sigma` flags genes whose Hessian at the optimum is not positive definite or gives a standard error wider than half the parameter box, warns with the count and the flattest direction, and stores the flags in `sigma_fail` (and `adata.var["error_unreliable"]` through `perform_inference`). The flagged genes' standard errors are replaced with the mean over the other genes so that downstream steps keep working; `sigma_flat_direction` holds the least-constrained direction for every gene. This is the only one of the three that sees the data, so it is the one that catches value-dependent problems such as the protein model's `k_p`/`γ_p` valley.

## Running inference

```python
from monod.inference import perform_inference
adata = perform_inference("example_h5ad/example_adata.h5ad", model, n_genes=100)
```

`perform_inference` filters genes, builds the per-gene count histograms, scans a grid of technical-noise parameters, fits every gene by maximum likelihood at each grid point, and writes the results into the returned AnnData. For finer control, the same steps are available individually: `extract_data`, `searchdata_from_adata`, and `InferenceParameters.fit_all_grid_points`; the `rust_pipeline` notebook in `demos/` walks through them.

When the Rust accelerator is installed the per-gene fits run in its L-BFGS-B kernel, all genes in parallel; on the paper's GABAergic data a 250-gene scan of the 42-point sampling grid takes 25 s against 250 s with the scipy optimizer, with the same optimum and the same per-gene results to within optimizer noise. `gradient_params={"use_rust_lbfgsb": False}` selects scipy. Each gene's grid is its maximum count plus a padding of 10, rounded up to the next length whose only prime factors are 2, 3 and 5; the inverse FFT that turns the generating function into a distribution is fastest on those lengths and slowest on primes. Standard errors (`exclude_sigma=False`) come from a Hessian built by differencing the model's gradient, six gradient evaluations per gene for a three-parameter model, and use the same `num_cores` as the fit.

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

Plots, gene rejection, noise decomposition and differential-parameter tests are methods on the result object and functions in `monod.analysis`; the `analysis_workflow` notebook in `demos/` walks through them on two PBMC cell types, and `monod_walkthrough` shows the same sequence end to end.

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
