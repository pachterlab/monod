# Running an analysis

## Input data

Monod works on an [AnnData](https://anndata.readthedocs.io) object whose `layers` hold integer counts for the model's modalities: `unspliced` and `spliced` by default, as produced by kallisto|bustools, velocyto or similar tools. Layers with other names are mapped with `modality_name_dict`. Gene lengths, if a model uses them, come from a transcriptome file or `adata.var['log_lengths']`. An `ambiguous` layer, reads that could belong to either transcript, is never used silently: `extract_data` and `perform_inference` take `ambiguous=` with the name of a channel to fold it into (`'spliced'` or `'unspliced'` for the built-in models, a species name for a custom network), `'drop'`, or `'model'` together with a model that has `amb_model` (fit it as a third channel, see below), and they fail when the layer is present and no choice was made. Some quantifications already fold the ambiguous reads into `spliced`; then `'drop'` is the right choice, and folding again would count them twice. The choice is recorded in `adata.uns['ambiguous']`.

## Choosing a model

A `CMEModel` combines three ingredients:

```python
from monod.cme_toolbox import CMEModel
model = CMEModel("Bursty", "Poisson")                      # transcription, sequencing noise
model = CMEModel("Bursty", "Bernoulli", amb_model="Equal")  # plus ambiguous read assignment
```

- **Transcription** (`bio_model`): `Constitutive`, `Bursty`, `CIR`, `Extrinsic`, `Delay`, `DelayedSplicing`, `ProteinBursty` (adds a protein layer), or a custom reaction network passed as `network=`, including time-delayed reactions written with `=>`. All of these run on the Rust accelerator when it is installed. `Bursty` is the usual choice for mammalian and bacterial cells.

The families are kernels, not separate mechanisms: each is a corner of one mechanism built from a transcription mode, what regulates it, the RNA lifetimes and an optional protein layer. `CMEModel.compose` names those pieces and picks the kernel:

```python
CMEModel.compose("bursty")                                   # Bursty
CMEModel.compose("bursty", protein=True)                     # ProteinBursty
CMEModel.compose("bursty", regulation="cir")                 # CIR
CMEModel.compose("bursty", lifetime="delayed_degradation")   # Delay
CMEModel.compose("constitutive", protein=True)               # no dedicated kernel: a reaction network
CMEModel.compose("bursty", locus=dict(n_sites=2, rule="switch"), seq_model="Poisson")   # Chromatin, bursty=True
```

| pieces | kernel |
|---|---|
| constitutive or bursty, exponential lifetimes | `Constitutive`, `Bursty` |
| bursty + `regulation="cir"` or `"extrinsic"` | `CIR`, `Extrinsic` |
| bursty + `lifetime="delayed_splicing"` or `"delayed_degradation"` (fixed lifetimes) | `DelayedSplicing`, `Delay` |
| bursty + `protein=True` | `ProteinBursty` |
| any transcription + `locus=...` | `Chromatin` (`bursty` set from the transcription mode) |
| other combinations without a fixed lifetime, e.g. constitutive + protein | a `Custom` network with species named `unspliced`, `spliced`, `protein` |

`prefer_network=True` builds the network form even where a kernel exists. A locus with a protein layer (`locus=..., protein=...`) is the chromatin model with `protein=True`, see the chromatin note below. The modalities are the same in every form, so one `modality_name_dict` serves all of them, and `seq_model`, `amb_model` and `amb_species` apply as in the constructor. The request is kept in `model.composition`.
- **Sequencing noise** (`seq_model`): `None`, `Poisson`, or `Bernoulli`. The `sequencing_noise` notebook in `demos/` shows what the models do, how the sampling landscape is scanned, and how the fitted variance splits into intrinsic, extrinsic and technical parts.
- **Ambiguous assignment** (`amb_model`): `None`, `Equal`, or `Unequal`. Reads that cannot be assigned to the unspliced or the spliced transcript are a third observed channel: each nascent count lands in it with a per-gene probability `p_N` and each mature count with `p_M` (`Unequal`), or both with one `p` (`Equal`). The channel is a modality like the other two, so the AnnData needs an `ambiguous` layer (map other names with `modality_name_dict`, for example `{"unspliced": "nascent", "spliced": "mature", "ambiguous": "ambiguous"}` for `kb count --workflow nac` output), the probabilities are fitted per gene and written to `var` as `$p$` or `$p_N$`, `$p_M$`, and the moments start takes `p` from the pooled ambiguous share. Any model takes the channel: the built-in two-species models through a Rust kernel, the protein, chromatin and custom-network models by evaluating the model without the channel on a grid enlarged by the ambiguous range and splitting the counts afterwards, which costs a factor of the ambiguous grid size per evaluation. `amb_species` names the species the ambiguous reads come from, `unspliced` and `spliced` unless told otherwise, and a custom network must name them (`amb_species=("U", "S")`). Not available in meK-Means. The alternative is to fold the ambiguous layer into one channel before the fit, which fixes the answer in advance.

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
Two practical notes. `perform_inference` accepts a `Custom` model directly: pass `modality_name_dict` to map species names to layers (for example `{"U": "unspliced", "S": "spliced"}`), and override `phys_lb`/`phys_ub` and `filt_param` if the defaults (burst sizes 0.1 to 16000 and rates 0.016 to 3200 in log10, as for the built-in models, and the mRNA filters for every species) do not suit. There is no moments estimator for an arbitrary network, so every gene starts from the middle of the parameter box, where a four-rate network can wander into a corner with fast and slow rates together that costs the integrator a hundred times more per evaluation; the network form a `compose` call builds is the exception, because its species and reactions are a family's, so it starts from that family's moments estimate. For a hand-written network give the optimizer a few random restarts (`gradient_params={"num_restarts": 3}`), and when a network nests a simpler one check that its divergence is not higher than the simpler fit's, which marks a gene that ended in the wrong basin. Gradients of custom networks are central differences with a step of 1e-4 in log10, wide enough to stay clear of the noise floor that histogram bins deep in the tail put on the divergence. The network integrator propagates the linear part of the characteristic equations exactly through a matrix exponential and chooses its steps by error control, like the protein integrator, so an evaluation costs about the same whether the rates span one decade or five (60 to 300 steps on a two-species network, accurate to a few parts per million on the generating function). Delayed reactions add a history lookup and catalytic steps such as `S -> S + P` a nonlinear remainder; both are handled by the same scheme at some extra cost. Finite-difference gradients, the identifiability check and the Hessian behind the standard errors all replay the base point's step schedule (`CMEModel.pinned_schedule`), so the integrator's tolerance does not leak into differences taken over small steps.

- **After the fit.** `compute_sigma` flags genes whose Hessian at the optimum is not positive definite or gives a standard error wider than half the parameter box, warns with the count and the flattest direction, and stores the flags in `sigma_fail` (and `adata.var["error_unreliable"]` through `perform_inference`). The flagged genes' standard errors are replaced with the mean over the other genes so that downstream steps keep working; `sigma_flat_direction` holds the least-constrained direction for every gene. This is the only one of the three that sees the data, so it is the one that catches value-dependent problems such as the protein model's `k_p`/`γ_p` valley.

### A note on the chromatin model

`CMEModel("Chromatin", seq_model, n_sites=2, rule="additive")` fits a gene whose nascent and mature RNA are driven by a locus of binary chromatin sites, the model of Felce, Gorin and Pachter (2024): each site opens at its own rate and closes at a shared rate, neighbouring sites couple through `epsilon` (1 means independent sites), and the open sites set the nascent transcription rate through a `rule`: `additive` (one weight per site), `switch` (one rate whenever any site is open) or `table` (a free rate per configuration, which nests both). Parameters are `k_on` per site, `epsilon`, the rule's rates, `beta` and `gamma`, all in units of the closing rate. The data needs three layers: `unspliced`, `spliced` and `configuration`, the last an integer per cell and gene encoding which sites are open (site 0 the least significant bit). `monod.chromatin_data.build_multiome_adata(rna, atac, genes, rule=, n_sites=)` builds it from an RNA AnnData with the two RNA layers, an ATAC AnnData of fragment counts per peak (coordinates in `var` or in peak names such as `chr1:9992-10688`) and a gene annotation (a GTF path or a table with gene, chrom, start, end, strand). Peaks are assigned to genes by one of three rules, each a hypothesis worth comparing: `"tss_window"` (peaks within a window of the start site), `"gene_body"` (peaks from the promoter to the gene's end) or `"loci_promoter"` (the paper's loci, runs of peaks closer than 1 kb, taken whole when one overlaps the promoter). A gene keeps the `n_sites` peaks nearest its start site, ordered along transcription so site 0 is the most upstream, and a site is open at one fragment or more; genes with fewer peaks are left out. The assignment is kept in `uns["chromatin_sites"]`, and `site_summary` lists each site's peak and how often it is open. With `protein=True` the model also fits a `protein` layer of antibody counts downstream of the mature RNA, the translation and degradation of `ProteinBursty` (`k_p`, `gamma_p` last in the parameter vector, `protein_limit` as there; modalities `unspliced`, `spliced`, `protein`, `configuration`). The protein axis is coarse-grained relative to its grid limit, and the limit is the largest count in the data, so remove cells carrying antibody aggregates before extracting (a count above three times the antibody's 99th percentile is a reasonable rule; `extract_data` warns when a maximum ten times the 99th percentile sets a limit) and keep `protein_limit` at or above the typical maximum of the low-abundance antibodies, 50 for a CITE-seq panel; `build_multiome_adata(..., protein=adt, protein_map={"CD14": "CD14"})` writes that layer from an AnnData of antibody counts on the same cells (TEA-seq, DOGMA-seq), dropping genes without an antibody. The protein kernel runs on the Rust accelerator like the rest of the chromatin model (about half a second per evaluation on a PTPRC-sized grid, with the Nelder-Mead pre-search of `ProteinBursty` on by default), with a NumPy fallback that is twenty times slower; the simulator, the generative checks and the contract suite cover it, and `docs/chromatin-protein-design.md` records the design.

`bursty=True` makes the emission geometric bursts of a shared mean size `b` (one more parameter, listed last) instead of single molecules; the rule's rates stay mean production rates, so the constitutive model is the small-`b` limit and the two can be compared on a gene. This matters because a constitutive emission has only one way to make an overdispersed RNA distribution, transcribing from a rare configuration, so without `b` a rule comparison can mistake burst kinetics for a gate.

ATAC dropout, an open site read as closed, is a sampling parameter on the configuration axis, scanned on the grid together with the sequencing parameters (`dropout_bounds=(log10 low, log10 high, grid points)`, default three points between 0.01 and 0.9; a value of 0 means none). It is therefore global per dataset, like the capture rates, and does not count against the per-gene parameters. With a protein layer, `protein_background=(log10 low, log10 high, grid points)` adds a fourth sampling parameter of the same kind: a Poisson background on the protein channel (nonspecific antibody binding, one to two counts per cell on a CITE-seq panel), shared by every gene and set by the antibodies whose target the cells do not carry. The method-of-moments start fits the chain from the configuration marginal, which is exact whatever the transcription does, and the rates from the nascent means per configuration; the optimiser searches at a coarse integrator tolerance and reports at the fine one. To compare rules on a gene, fit the simpler rule and call `monod.analysis.compare_nested_models` with the `table` model and `monod.chromatin_fit.embed_in_table`: it runs a score test at the simpler optimum and fits the table, warm-started from it, only when the test asks for it.

### Comparing two models on a gene

`monod.analysis.compare_nested_models(model_small, model_big, embed, x_small, limits, samp, data, n_cells)` tests a fitted model against one that nests it: it embeds the small model's optimum in the big model's parameter space, fits the big model from there (so it can never end worse, which also catches a cold-started fit that landed in the wrong basin), and reports the likelihood ratio `2 n_cells (KLD_small - KLD_big)` with its chi-square p-value, a Rao score test at the embedded point when asked, and the AIC and BIC of both fits. `embed=None` uses the nestings monod knows (`monod.analysis.nesting_embedding`): a chromatin `switch` or `additive` rule inside the `table` rule and a constitutive emission inside a bursty one, and a reaction network inside one that adds a reaction (its rate at the floor), a geometric burst (its size at the floor) or a delay (its inverse delay at the ceiling, which makes that one approximate). Pairs with no nesting raise with the reason. Two cautions: when the extra parameter sits at a bound in the embedding, as a burst size or an extra rate does, the score test cannot see it and the chi-square p-value is conservative, so pass `table="always"`; and the test treats the cells as independent, as every monod fit does.

The built-in `Bursty` model is written in units of the burst frequency and `Constitutive` in units of the production rate, so neither is a point of the other's box. To test whether a gene's transcription is bursty, write both as reaction networks with `burst_mean_rate=True`: `CMEModel("Custom", "None", network="0 -> U -> S -> 0", normalize_production_rate=True)` and the same with `"0 -> B x U -> S -> 0"`. Under that flag the rate of a burst production is the mean production rate (bursts of mean size `B` at frequency `k/B`), the burst-size box reaches down to 1e-3, and the single-molecule network is the small-`B` limit at the same rates, so the constitutive fit embeds exactly. The flag changes the parametrisation only; at `k_freq = k_mean / B` the two forms are the same network.

For two models that do not nest, such as `Bursty` against `Delay` or `Extrinsic`, `monod.analysis.compare_models(model_a, model_b, x_a, x_b, limits, samp, data, n_cells)` takes both fitted optima and reports the divergences, AIC and BIC, and Vuong's test on the per-cell log-likelihood ratios read off the histogram (`vuong_z`, positive favouring the first model, the Schwarz-corrected `vuong_z_bic` and its two-sided `p_vuong`). The demo `model_comparison.ipynb` runs both tests on a panel of real genes.

## Running inference

```python
from monod.inference import perform_inference
adata = perform_inference("example_h5ad/example_adata.h5ad", model, n_genes=100)
```

`perform_inference` filters genes, builds the per-gene count histograms, scans a grid of technical-noise parameters, fits every gene by maximum likelihood at each grid point, and writes the results into the returned AnnData. For finer control, the same steps are available individually: `extract_data`, `searchdata_from_adata`, and `InferenceParameters.fit_all_grid_points`; the `rust_pipeline` notebook in `demos/` walks through them.

To fit several groups of cells from one object, name the `obs` column that separates them:

```python
adata = perform_inference(adata, model, split_by="subclass_label", groups=["Tcells", "Monocytes"], exclude_sigma=False)
```

The genes are selected once on all the cells, so the groups line up gene for gene; each group is then fitted on its own cells, with its own grid limits, sampling optimum and standard errors. `groups` restricts the fit to some labels and fixes their order (by default every label, in category order). Each group's per-gene table goes to `adata.varm[<key>]`, the label with runs of non-alphanumeric characters replaced by `_`, so `adata.varm["Tcells"]["$b$"]` is the burst size per gene in T cells; the result objects go to `adata.uns["search_result_list"]` and `adata.uns["split_by"]` records the column, the labels and the keys. `analysis.group_result(adata, "Tcells")` returns one group's `SearchResults`, `analysis.group_search_data` the histograms of its cells. `reject_genes(adata)` writes its flags into every group's table and `analysis.DE_parameters(adata)` compares every pair of groups, recording `FC_<param>_vs_<other>` and `adj_FC_<param>_vs_<other>` in both tables (second group minus first), so comparing two cell types needs one object rather than two. With a sequencing-noise model the groups share one sampling optimum: the capture efficiencies belong to the experiment, not to the cell type, so the grid point is the one that minimises the groups' summed divergence, each group weighted by its cell count. `shared_samp=False` lets each group choose its own, and `inference.find_shared_sampling_optimum([sr1, sr2])` applies the same rule to two objects fitted separately from one experiment. When one object holds several experiments, name their column too: `perform_inference(adata, model, split_by="subclass_label", batch_by="experiment")` fits every (cell type, experiment) pair that occurs, named `"Tcells | exp1"` with table key `Tcells__exp1`, and shares the sampling optimum among the cell types of each experiment only; `batches=[...]` restricts the experiments, and `DE_parameters` then compares the cell types within each experiment and each cell type across experiments unless `pairs=[...]` says otherwise. `analysis.group_result(adata, ("Tcells", "exp1"))` looks a group up by its pair of labels.

When the Rust accelerator is installed the per-gene fits run in its L-BFGS-B kernel, all genes in parallel; on the paper's GABAergic data a 250-gene scan of the 42-point sampling grid takes 25 s against 250 s with the scipy optimizer, with the same optimum and the same per-gene results to within optimizer noise. `gradient_params={"use_rust_lbfgsb": False}` selects scipy. Each gene's grid is its maximum count plus a padding of 10, rounded up to the next length whose only prime factors are 2, 3 and 5; the inverse FFT that turns the generating function into a distribution is fastest on those lengths and slowest on primes. Standard errors (`exclude_sigma=False`) come from a Hessian built by differencing the model's gradient, six gradient evaluations per gene for a three-parameter model, and use the same `num_cores` as the fit.

## Simulating from a model

`monod.simulate.simulate_counts(model, theta, n_cells, samp)` draws cells from the model's process without using its kernel: molecule lifelines for the linear families (`Constitutive`, `Bursty`, `Extrinsic`, `Delay`, `DelayedSplicing`, `ProteinBursty`), a Gillespie simulation for the chromatin and network models, then the observation layers on each cell: Poisson or Bernoulli capture, the ambiguous split per read, ATAC dropout per site. `simulate_adata(model, thetas, n_cells, samp)` wraps several genes into an AnnData with one layer per modality, ready for `perform_inference`. `CIR` has no simulator: its kernel is the limiting law of an inverse-Gaussian driven production process whose finite-rate parameters are not part of the model. The test suite uses this to check every family against its own simulation (moments, histogram divergence, and parameter recovery through the pipeline), which is what ties the separate kernels to one generative model.

## Reading the results

Per-gene log10 parameter estimates and the AIC land in `adata.var`. Parameter columns are named with the model's TeX labels minus the log prefix, so for the bursty model they are `$b$` (burst size), `$\beta$` (splicing rate) and `$\gamma$` (degradation rate); `model.param_str` lists them for any model. Fold-change columns written by `monod.analysis.DE_parameters` follow the same convention (`FC_$b$`, `adj_FC_$b$`). For a fit with `split_by`, the same columns sit in one table per group in `adata.varm`.

The full results objects are stored in `adata.uns`:

- `adata.uns['model']`: the model definition;
- `adata.uns['search_result']` (or `search_result_list` for meK-Means and split fits, with `split_by` naming the groups): fits at every grid point, the chosen optimum, and quality metrics.

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
