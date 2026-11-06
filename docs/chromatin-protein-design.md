# Design: a joint chromatin, RNA and protein model

Status: all five stages built, 2026-11-02 to 2026-11-03 (NumPy kernel, model
definition, limit tests in `tests/test_chromatin_protein.py`; protein
reactions in the chromatin simulator, generative checks in
`tests/test_generative_checks.py`; the kernel in the contract suite
`tests/test_kernel_contract.py`; the pipeline and the `protein` layer of
`build_multiome_adata`; the Rust kernel, "How stage 5 does it" below). The "Why the kernel composes" section describes the design; "How
stage 1 does it" what was built.

## What the model is

One gene, four observed modalities per cell: nascent RNA, mature RNA, protein and
the chromatin configuration of a locus of `n` binary sites. The mechanism is the
existing `Chromatin` model with the existing protein step added downstream of the
mature RNA:

```
sites open/close        k_on[i], k_off = 1, Ising coupling epsilon        (Chromatin)
configuration -> nascent  transcription rule: switch / additive / table    (Chromatin)
                          single molecules, or geometric bursts of mean b  (Chromatin, bursty=True)
nascent -> mature         beta                                             (Chromatin)
mature -> 0               gamma                                            (Chromatin)
mature -> mature + P      k_p                                              (ProteinBursty)
P -> 0                    gamma_p                                          (ProteinBursty)
```

Nothing couples the protein back to the chromatin or the RNA, so the protein
inherits the configuration's influence only through the mature RNA. That is the
reason the kernel composes (next section) and also the reason to expect the
configuration to carry little information about the protein unless the sites
switch slowly compared with the protein lifetime.

Technical noise is the union of what the two models already have: Poisson or
Bernoulli capture on the two RNA channels (with the gene-length offset), ATAC
dropout on the configuration axis as the third sampling parameter, nothing on
the protein channel (as in `ProteinBursty`). The ambiguous channel works without
further work: it enlarges the two RNA axes of the latent model and splits the
counts afterwards, which is model-agnostic.

## Why the kernel composes

The chromatin kernel integrates the vector generating function
`G[c, p] = E[prod_j (1 + u_j)^{x_j}; configuration c]` by Strang splitting with
three exact sub-flows per step:

1. production, diagonal in `c`: `G[c, p] *= exp(A[c] * integral of phi(U_nascent(s, p)) ds)`;
2. configuration transitions: `G <- exp(H^T h) G`;
3. production again over the second half step.

`U(s, p)` is the characteristic of the RNA system started at the mesh point `p`:
today `U(s) = exp(C s) u` with `C` the splicing/degradation matrix, in closed
form, and identical for every configuration.

The protein model's characteristics, in the variables `v = u_U + 1`,
`w = u_S + 1`, `u_P`:

```
u_P(t) = u_P(0) exp(-gamma_p t)                       closed form
v'     = beta (w - v)
w'     = a(t) w + gamma,   a(t) = -gamma + k_p u_P(t)
```

are a linear system with one time-varying coefficient, and `protein_pgf`
propagates the homogeneous part of `w` exactly per step and treats the forcing
and the `v` coupling with the step-average coefficient. They still do not depend
on the configuration, because the configuration only sets the nascent production
rate `A[c]`.

So the joint kernel is the chromatin stepper with one substitution: the
characteristic provider. Where `_production_exponent` evaluates
`exp(C s) u` (constitutive, closed form) or at three Simpson nodes (bursty), it
instead evaluates the protein characteristic at those times. Sub-flow 2 is
unchanged. The production exponent still only reads the nascent characteristic
`U_nascent(s, p)`, through `phi(U) = U` or `U / (1 - b U)`.

Consequences:

- The step-doubling error control of the chromatin kernel covers the
  characteristic integration as well, provided the characteristic is advanced
  with the same step `h` the stepper is trying (one full step against two half
  steps), as the protein integrator does on its own.
- The pinned-schedule contract (replay the base point's steps at every stencil
  point) carries over unchanged: the schedule is the chromatin stepper's.
- The Rust kernel `eval_chromatin_pgf` needs a protein branch: today it takes
  `C` and builds `exp(C s)`; it would take `(beta, gamma, k_p, gamma_p)` and the
  protein mesh row, and run the protein characteristic update inside its step.
  The Python mirror comes first.

## How stage 1 does it

The stepper walks from the horizon `T` down to zero, so it needs the
characteristics at the far end first, while they are known at `s = 0` (the mesh
point). Advancing them alongside the step, as the design above assumed, is
therefore not possible: `chromatin._ProteinCharacteristics` tabulates `v(s)`,
`w(s)` once over `[0, T]` with the protein integrator's step under step
doubling, and the stepper reads them at any `s` by cubic Hermite interpolation
between the half-step nodes with the derivatives the equations give. The step
control measures the interpolation error too (the half-step value against the
cubic through the step's ends): when `k_p` vanishes every step is exact for the
equations and the control would otherwise take steps far too long for the
cubic. `_LinearCharacteristics` wraps `exp(C s) u` for the plain model, so
`_production_exponent` reads one interface.

Checked: the (RNA, configuration) marginal equals the `Chromatin` kernel to
3e-8; a one-site `table` with equal rates equals `ProteinBursty` to 5e-8 and,
constitutive, the `compose("constitutive", protein=True)` network to 6e-8;
`k_p -> 0` puts the protein at zero; `gamma_p = 300` gives `E[P | S] = r S`
within 5%; Poisson capture, dropout, `protein_limit` (the same truncation as
`ProteinBursty`, equal to it under the same setting) and the ambiguous channel
go through the existing routes; pinned schedules replay to 1e-13 and the
central-difference gradient matches a wide plain difference.

Cost in NumPy: 5 to 15 s per evaluation on a two-site locus with a 20 x 24 x 40
grid (about 350 steps); the table holds `2 x nodes x P` complex numbers, some
hundred megabytes on that grid. The Rust kernel avoids the table by the
affine-map form ("How stage 5 does it"); the NumPy path stays as the
reference.

## How stage 5 does it

Every step of the protein integrator is affine in `(v, w)`, with coefficients
that depend on the mesh point only through the protein argument `u_P`, and
the Hermite read is linear in the node values and derivatives. The Rust
kernel (`eval_chromatin_pgf(..., protein=(k_p, gamma_p, h_first, h_min,
tol))`) therefore tabulates the affine map from `s = 0` to each node, five
numbers per node per distinct `u_P` (the protein axis of the mesh, a handful
of values), and reads `v(s) = m11 v0 + m12 w0 + b1` at the point inside the
diagonal flow: the same numbers as the NumPy table over every mesh point, at a
negligible fraction of the memory. The tabulation schedule is decided as in
NumPy, by the exact maximum over the mesh points of the step-doubling
difference and of the Hermite midpoint check, so both kernels take the same
steps; they agree to 1e-16 at every mesh point in every mode (bursty and
constitutive, switch, additive, table, independent sites, Poisson capture,
coarse protein axis, replayed schedules; `TestRustBackend`). The stepper is
untouched: the protein enters only through the production exponent's three
Simpson nodes, so the structured paths (distinct rates, factored rates,
Kronecker chain) and the parallel blocks all apply.

Timing on bio (24 cores), bursty two-site switch, Poisson capture and dropout:

| grid | Rust | NumPy |
|---|---|---|
| 32 x 42 x 35 x 4 | 0.53 s | 11.1 s |
| 23 x 27 x 21 x 4 | 0.17 s | 2.9 s |
| 10 x 10 x 10 x 4 | 0.09 s | 0.45 s |
| 81 x 44 x 86 x 4, `protein_limit=20` (PTPRC-like) | 0.68 s | |

A pinned gradient on the 23 x 27 x 21 grid takes 3.5 s against 44 s. The
Nelder-Mead pre-search (`presearch="auto"`) is on for this model when the
Rust kernel is installed, as for `ProteinBursty`; without it the model falls
back to NumPy and the pre-search stays off.

## Model definition

```python
CMEModel("Chromatin", "Poisson", n_sites=2, rule="switch", bursty=True,
         protein=True, protein_limit=20)
```

or, through the compositional constructor, which is where the model naturally
lives once built (today that call raises NotImplementedError pointing here):

```python
CMEModel.compose("bursty", locus=dict(n_sites=2, rule="switch"), protein=dict(protein_limit=20),
                 seq_model="Poisson")
```

- Modalities: `['unspliced', 'spliced', 'protein', 'configuration']`. The
  configuration stays last (sampling vector indices unchanged: `samp[0:2]` RNA
  capture, `samp[2]` ATAC dropout), protein sits where `ProteinBursty` puts it.
- Parameters: `[k_on_1..n, epsilon, rule rates, beta, gamma, (b), k_p, gamma_p]`,
  log10. The protein pair goes last so that every existing chromatin index is
  unchanged; `split_theta`, `join_theta`, `n_params`, `burst_size` and
  `embed_in_table` take a `protein` flag. Bounds for `k_p` as in
  `ProteinBursty` (`[-1.8, 3.5]`); `gamma_p` reaches down to -3.0, and `beta`,
  `gamma` to -2.5, because the chromatin box is in the locus clock (`k_off =
  1`) while Bursty's -1.8 is in units of the burst frequency (see validation
  item 6 for the data that forced this and the cost it carries).
- `protein_limit` coarse-grains the protein axis of the mesh exactly as
  `ProteinBursty` does; the inversion reuses the protein model's zero-padded
  3-D inverse transform, applied per configuration.
- `fit_unspliced=False` is not offered: the nascent channel is where the
  configuration acts.
- Serialisation: `protein` joins `_CTOR_KEYS`; `to_dict` writes it with the
  chromatin block.

## Fitting

- Optimiser: the scipy path with central finite differences and pinned
  schedules, as for `Chromatin` today. The Nelder-Mead pre-search that
  `ProteinBursty` uses by default (`presearch="auto"`) would suit
  `protein=True` too, because the `k_p`/`gamma_p` valley is the same valley;
  it is on when the Rust kernel is installed (stage 5).
- Moments start, in `chromatin_fit.chromatin_start`:
  1. chain from the configuration marginal (sum over the three count axes),
     dropout fixed from the sampling vector, as now;
  2. rule rates from the conditional nascent means per configuration, `gamma`
     from the mature mean, `beta` from the short scan, as now, on the histogram
     with the protein axis summed out;
  3. `k_p / gamma_p` from `E[P] = (k_p / gamma_p) E[S]`, `gamma_p` from a short
     scan (`{0.1, 1, 10} * gamma`) under the model's own divergence, the pattern
     `beta` already follows. The covariance-based formula of `ProteinBursty` is
     an option but it is the step that failed most often there.
- Standard errors: `compute_sigma` unchanged; expect the valley flag to fire on
  genes where protein turns over faster than mRNA.
- Nested comparisons: `compare_nested_models` works as now between rules and
  emissions at fixed `protein=True`; `protein=False` against `protein=True` is
  not a nesting (a different data space), it is a different model of the same
  cells only once the protein layer is dropped, so no test is offered there.

## Data path

`chromatin_data.build_multiome_adata(rna, atac, genes, ..., protein=None,
protein_map=None)`:

- `protein` is an AnnData of antibody counts (ADT) on the same cells;
  `protein_map` maps gene name to antibody name (`{"CD14": "CD14"}`); genes
  without an antibody are dropped from the fit with a log message, since a
  protein model needs a protein count.
- Writes an integer `protein` layer next to `unspliced`, `spliced` and
  `configuration`; `extract_data` then needs nothing new beyond the model's
  modality list (filters: the protein entry of `ProteinBursty`'s filter box).
- Cells are matched by barcode; the three assays must come from the same
  cell. TEA-seq and DOGMA-seq produce this; the BMMC multiome does not have
  protein.

## Cost

Mesh points: `lu * ls * (lp_coarse // 2 + 1)` with `N = 2**n` configurations
each. For a gene like PTPRC (limits 81, 44, 86 with `protein_limit=20` so
`lp_coarse` about 5) that is about 11 k points by 4 configurations, the size of
the two-site chromatin fits that run today in tens of milliseconds per
evaluation in Rust. The protein characteristic adds one closed-form update per
mesh point per sub-step. In Python expect a few seconds per evaluation, enough
to validate on synthetic data, not to fit panels; the Rust branch is what makes
real data feasible.

## Validation plan, in order

1. **Python kernel, limits.** Done. A one-site `table` locus transcribing at the
   same rate in both configurations (the locus invisible; `k_on >> 1` would
   leave a closed share of `1 / (1 + k_on)`) reproduces `ProteinBursty(b, beta,
   gamma, k_p, gamma_p)` and, constitutive, the network form; `k_p -> 0`
   reproduces `Chromatin` with a delta at protein zero; `gamma_p -> infinity`
   with `k_p / gamma_p` fixed gives a protein that tracks the mature count
   (its conditional mean `(k_p / gamma_p) S`).
2. **Simulation.** Done. `simulate._chromatin_ssa` carries translation and
   degradation as two more reactions of the vectorised Gillespie loop (the
   burn-in horizon counts the protein lifetime); the observation layer leaves
   the protein uncaptured and reads the dropout from the third sampling entry.
   `tests/test_generative_checks.py` checks moments and histogram divergence
   with capture, dropout and the ambiguous split on. The pipeline recovery
   (`TestRecovery::test_chromatin_protein`, one gene, 30 iterations) took 27
   minutes on the NumPy kernel (a pinned gradient about 45 s) and was gated
   until stage 5; on the Rust kernel it takes a few minutes and runs with the
   suite.
3. **Contract tests.** Done. `chromatin-switch-protein` and
   `chromatin-table-protein` (bursty, two sites, a 10 x 10 x 10 grid, the
   switch embedded in the table by `embed_in_table(..., protein=True)`) run
   every contract: replay, gradient, thread pool, outer pin, the fit from a
   displaced start and the nesting. About 10 minutes for the pair, most of it
   the thread-pool gradients and the fits.
4. **Pipeline.** Done (`tests/test_chromatin_protein.py::TestPipeline`,
   `tests/test_chromatin_data.py`). `perform_inference` on simulated four-layer
   data with a three-point dropout grid under Poisson capture picks the true
   dropout and recovers the chain and `k_p / gamma_p`; the ambiguous channel
   runs through the generic route with its probability in `var`; the model
   definition (protein flag, dropout grid, ambiguous model) survives the h5ad
   round trip. `build_multiome_adata(protein=, protein_map=)` writes the
   `protein` layer, intersects the cells with the antibody data, drops genes
   without an antibody with a log message and records the antibody in `var`.
   Nothing in `extract_data` or `inference` needed a change: the modality
   list drives both. The two pipeline tests take about two minutes each on
   the NumPy kernel even with one to three optimiser iterations (the fixed
   cost is the gradients and the fine-tolerance evaluations per grid point),
   and one gene cannot tell dropout from its own opening rates, so the grid
   is checked as plumbing; the recovery proper is the generative test. The Nelder-Mead pre-search stays off for this model
   until the Rust kernel exists (the design above said on; at seconds per
   evaluation it is not affordable). Stage 5 turned it on.
5. **Rust.** Done ("How stage 5 does it"): the protein characteristics as an
   affine-map table per distinct protein argument inside `eval_chromatin_pgf`,
   Python against Rust at every mesh point to 1e-16 in every mode, 17 to 20
   times faster on fitting grids.

Stages 1 to 4 were Python only and self-contained; stage 5 was the port.

6. **Real data, first pass.** DOGMA-seq DIG control (Mimitou 2021,
   GSE156478), CD4 T cells, every gene with an antibody and a two-site
   `tss_window` locus open in at least 2% of cells: 8,526 cells, 28 genes,
   switch rule, bursty, dropout grid of four points, 60 iterations.
   Chromatin without protein 10 min, `ProteinBursty` 13 min, the joint model
   6 h on 12 cores of bio (about 1 h per grid point on an idle machine). The
   locus parameters do not move when the protein is added (`k_on`, `epsilon`
   within 0.02 log10 of the chromatin-only fit); `a`, `beta`, `gamma` move
   0.2 to 0.3 log10 on the fifteen genes with a mean mature count above 0.05
   and 0.5 to 0.6 on the rest. `k_p / gamma_p` agrees with `ProteinBursty`
   to 0.03 log10 on the covered genes (CD4 4.1, CD27 3.0, ITGB7 2.7, ICOS
   2.6, ITGA4 and IL2RA 2.0, PTPRC 0.35). The dropout is not identified
   here: the summed divergence is flat over dropout 0.001 to 0.1 and worse
   only at 0.9. Two things to fix before the next pass. `gamma_p` sits at
   its lower bound of -1.8 for 18 of the 28 genes: the protein box is
   `ProteinBursty`'s, whose clock is the burst frequency, while the chromatin
   clock is `k_off = 1`, under which the RNA `gamma` itself lands at -1 to
   -1.8, so protein lifetimes need room down to about -3 (the pass-2 box:
   `beta`, `gamma` from -2.5, `gamma_p` from -3.0). The price is the horizon,
   `log(1 / tol) / min(gamma, gamma_p)`: one Rust evaluation on a 40 x 30 x
   50 x 4 mesh took 2.0 s at `gamma_p = -1.8`, 5.5 s at -2.5, 12 s at -3.0
   and 64 s at -3.5 (8 to 21 s with `beta, gamma` at -2.5, -2.8). The Strang
   error does not shrink with the forcing's slow decay, so the step count
   scales with the horizon; an integrator that treats the slow protein tail
   with longer steps is the next kernel item if the fits keep going to the
   low end. Pass 2 therefore fixes the dropout at the chromatin-only optimum
   (an 8-point grid on the cheap model sets it; the protein does not inform
   it) and runs with standard errors and the rejection test. And the fit script
   must write its tables before any reporting line (the first attempt lost
   four hours to a `KeyError` on a rejection count that only exists after
   the rejection test; `perform_inference(checkpoint=True)` now keeps every
   grid point).

   **Pass 2 (2026-11-04, wider box, errors, rejection test).** Chromatin on
   the 8-point dropout grid 37 min (optimum 0.43, the divergence flat below
   it), `ProteinBursty` 5 min, the joint model 2 h 40 min with dropout fixed.
   The locus parameters again do not move (0.01 log10); `gamma_p` now reaches
   -3.0 on six genes, so the slow tail is real. The chromatin-only fit passes
   the rejection test on 22 genes (Hellinger 0.004 to 0.07; the six failures
   are five at a bound and one at Hellinger 0.052) but every gene fails under
   both protein models, with Hellinger 0.3 to 1.1 and the fitted protein mean
   10 to 200 times the observed one. That was the data, not the optimiser:
   the protein axis is coarse-grained relative to the grid limit, the limit
   is the largest count in the panel, and the largest count is an antibody
   aggregate (CCR10: 99th percentile 19, maximum 1829, so at
   `protein_limit=20` the model resolved the protein at 92 counts per step and
   could only smear). Dropping the cells with any antibody above three times
   its 99th percentile removes 155 of 8,526 cells (1.8%) and takes the Fano
   factor of the lineage-negative antibodies from 30 to 400 down to 1.4 to 3,
   which is what a Poisson background with a little biology looks like;
   `protein_limit=50` then resolves 23 of the 28 antibodies at full
   resolution and CD4 (mean 824) at 50 counts, at about 2.4 times the mesh.
   Pass 2b reruns the three fits so. Two fixes came out of the pass:
   `chisquare_testing` indexed two or three modalities and could not take
   four, and `SearchResults.gene_pss` collapsed the configuration axis for a
   marginal (the kernel needs all `2**n_sites` states) and dropped the
   chromatin sampling vector; both now go through the joint PMF. The standard
   errors were flagged unreliable on 27 of 28 genes under the joint model (8
   of 28 without the protein). The flattest Hessian direction says why: the
   burst size alone on the RNA-poor genes (CCR10, CCR2, CCR4, CD63, CD79A,
   LAG3, NCR3, PDCD1), and on the rest an equal-weight combination of `a`,
   `beta`, `gamma`, `k_p`, `gamma_p`, the common speed of RNA and protein
   against the locus clock, which the data constrain only when that speed is
   comparable to the switching. Treat the flagged genes' error columns as
   missing; the reported values are the mean over the unflagged genes.

   **Pass 2b (2026-11-05, aggregate-free cells, `protein_limit=50`).**
   Chromatin 38 min (dropout optimum unchanged), `ProteinBursty` 7 min, the
   joint model 7.7 h (the wider protein mesh, and the full test suite shared
   the machine for part of it). The summed divergence of the protein models
   fell from 47 and 50 to 13 and 18, and the rejection test now separates the
   genes: the joint model passes on nine (CCR2, CCR4, CD226, CD79A, CTLA4,
   FASLG, LAG3, NOTCH1, PDCD1), `ProteinBursty` on six, and the two agree on
   `k_p / gamma_p` to 0.05 log10 wherever neither sits at a bound. The genes
   that pass are the low-abundance antibodies whose counts are background,
   and on them the joint fit moves `a`, `beta`, `gamma` a full decade from
   the chromatin-only values (median 1.0 log10, against 0.01 for `b`): the
   model is spending the RNA parameters to translate background counts,
   which is the case for the background term pass 3 scans. The genes that
   fail (Hellinger 0.4 to 0.7) are the abundant antibodies, CD4, CD27, ICOS,
   IL2RA, ITGA4, ITGB7, CD52, CD63, CD9: real signal whose spread the model
   does not produce (CD4 has a Fano factor of 290 with the aggregates gone),
   so after the background the open question is extrinsic protein
   variation. Standard errors unreliable on 16 of 28 (10 under
   `ProteinBursty`, 8 without protein); where they are reliable, `k_p` and
   `gamma_p` carry 0.1 to 0.4 log10.

   **Pass 3 (2026-11-05, shared Poisson background on the grid).** Joint
   model only, background scanned at 1, 1.6 and 2.5 counts, 5.2 h on an idle
   bio (1.9 h per grid point). The grid picks 1.6, inside the box and at the
   lineage-negative level, but the summed divergence does not move (18.08
   without, 18.23 with) and the rejection test gets worse (25 of 28 fail, 19
   before). Ten genes improve by more than 0.1 nats (CCR10, FAS, TNFRSF4,
   NCR3, TNFSF10 among them) and ten worsen by as much (CCR2, LAG3, NOTCH1,
   PDCD1, CD4). The hypothesis that the background would hand the RNA
   parameters back to the chromatin-only values fails: on the sixteen
   antibodies below three counts the median distance of `a`, `beta`, `gamma`
   from the chromatin-only fit goes from 1.0, 0.8, 0.9 to 0.9, 1.3, 1.4
   log10, and `gamma_p` runs to its upper bound on the antibodies whose
   counts are all background (CCR2, CD79A, CTLA4, LAG3, TNFRSF4), where the
   protein channel then says nothing and `k_p / gamma_p` is free. Two
   readings. One shared rate is the wrong shape: the nonspecific level is
   antibody-specific (0.5 to 3 counts on this panel), so a single value
   over-corrects half the antibodies and under-corrects the other half, and
   the per-gene result follows that split. And the RNA parameters on these
   genes are not set by the RNA at all: with a mature mean of 0.01 to 0.05
   the chain's common-speed direction is flat, so whatever the protein
   channel says, background or not, decides `a`, `beta`, `gamma`. Standard
   errors unreliable on 20 of 28. The background term stays in the code (it
   is correct and tested) but is not the fix for this panel; what would be is
   a per-antibody rate held fixed per gene, read off a cell type in the same
   run that does not carry the target (the sample has B cells, CD8 T cells
   and pDCs), and that is the next design decision, together with extrinsic
   protein variation for the abundant antibodies.

   **Pass-4 diagnostics (2026-11-05).** Both candidates were tested before
   building. The background by cell type (mean count per antibody on each
   type with ≥ 150 cells, aggregates removed within the type) is
   antibody-specific and large where it matters: the CD4 antibody puts 243
   counts on B cells and 287 on CD8 T cells, a third of the CD4 T mean; CD27
   26, CD49d 17, integrin beta7 15, ICOS 9, CD25 5; the low antibodies 0.5 to
   3. That is a per-gene constant, not a grid axis: `adata.var
   ['log10_protein_background']` now rides on the background entry of every
   gene's sampling vector (`InferenceParameters(gene_samp_offsets=...)`,
   filled by `perform_inference`), the grid scanning a shared factor on top.
   Where the least-expressing type still carries over half the CD4 T level
   and the antibody is abundant (CD45, CD52, CD63, DNAM-1, Notch1, Fas, and
   CD25, CD49d, integrin beta7, which the "other" cluster does not resolve)
   there is no negative population and the floor of one count stands in;
   the table is `adt_background_used.csv` next to the data. The abundant
   antibodies' misfit turned out not to be extrinsic noise. The pass-2b
   protein marginals are qualitatively wrong (CD27: quantiles 1, 11, 144,
   396, 415, 416 against 14, 41, 65, 91, 147, 208 in the data, mean 193
   against 71, a tenth of the mass piled in the last grid bins by the
   transform's wrap-around). The reason is RNA capture: with `seq_model=
   "None"` the observed mature count, 0.09 per cell, is the true one, so
   each rare long-lived mRNA makes one giant protein burst and the protein
   is bimodal whatever `gamma_p` does. Poisson capture at 0.1 on the RNA
   channels (true mRNA ten times the observed), `a` raised and `k_p` lowered
   to hold the means, gives CD27 quantiles 15, 39, 63, 95, 152, 205, mean 71,
   Fano 28 against 26, divergence 0.15 against 1.5, at `gamma_p = -3.0`;
   capture 0.3 needs `gamma_p` below -3 for the same. The protein channel
   therefore identifies the RNA capture that the RNA alone cannot (`a` and
   the capture are degenerate there), and the joint fit must run with the
   Poisson sequencing model and a capture grid. The slow end has a hard
   limit: the chromatin stepper's cap of 20,000 steps is reached at
   `gamma_p = -3.5` on CD27's grid (10,565 steps at -3.0), and a capped
   integration returns a truncated distribution without saying so (at -4.0
   the divergence jumps to 3.8); the kernel now warns when it hits the cap,
   and the `gamma_p` box stays at -3.0 until the slow-tail integrator exists.
   The same marginals showed a third defect, in the cells rather than the
   model: the CD4 antibody is bimodal within the "CD4 T" cluster, with a
   fifth of the cells at 60 to 130 counts against a mode near 900, and those
   cells carry CD8a at a median of 270 against 28 (mislabelled CD8 T cells
   and a few NK cells), so every fit so far ran on a mixture. The refined set
   keeps CD4 at or above 250 and CD8a below 250 ADT counts (6,508 cells; the
   CD4 Fano factor falls from 288 to 127). A
   left-skewed marginal under a model that can only skew right is the
   signature to look for. Pass 4: the refined cells, chromatin-only on the
   dropout grid first, then the joint model with Poisson capture on a tied
   grid (c_U = c_S at 0.1, 0.3, 0.03, one fit each), per-antibody background
   fixed, `protein_limit` 50.

   **Pass 4 (2026-11-06) and the start that was not one.** The chromatin-only
   stage on the refined cells reproduces pass 2b (dropout optimum -0.371,
   shared rates within 0.2 decades). The joint fit at capture 0.1 brought the
   summed divergence from 18.1 to 12.6 and the median Hellinger from 0.27 to
   0.07, with twelve genes below 0.02 and `gamma_p` off its bound on every
   gene; capture 0.3 gave 14.6. But the abundant antibodies got worse (CD4
   Hellinger 0.46 to 0.97, CD27 0.67 to 0.72, CD45 0.11 to 0.67) and sat in
   the fast-protein regime (CD27: `k_p` -1.7, `gamma_p` -0.1). Evaluating the
   divergence at the fitted optimum, at the moments start and at candidate
   slow points (chromatin-only RNA rates, `a` raised by the capture factor,
   `k_p` from the protein mean less the background, `gamma_p` swept) showed
   why: every gene's moments start was the middle of the box. The RNA twin
   that `get_MoM` builds for a protein model dropped the protein layer but
   kept `protein_background`, the constructor refused it, and the `except`
   that makes the start "a convenience, never a failure" swallowed the error
   into a `log.info`. From the middle of the box the optimizer only reaches
   the fast basin (CD27 divergence 3.2 to 2.6), while the slow candidates
   sit at 0.24 (CD27, `gamma_p` -2.5), 0.82 (CD4, -2.5), 0.58 (CD45, -1.0)
   and 0.15 (Fas, -1.0): the model fits these antibodies, the search did
   not. CD49d and CD25 are the exception: no slow candidate beats the fit
   (0.74, 0.62), their marginals being broader than any single-rate protein
   layer with the floor background allows. Fixed (the twin drops the
   background; `TestStart::test_moments_start_with_a_protein_background`
   checks the start is not the box centre and that the background comes off
   `k_p`). Pass 5 reruns the three joint fits with the real start, dropout
   fixed at -0.371, everything else as pass 4.

## Open decisions

- **Protein measurement noise.** `ProteinBursty` has none, and ADT counts carry
  a known background. The DOGMA-seq DIG control shows it matters for the
  low-abundance antibodies: in CD4 T cells the rat isotype controls average
  0.05 and 1.1 counts, antibodies against targets the cells do not carry
  (CD79a, CD68, CD146, PD-1) 1.1 to 2.4 counts with medians of 0 to 2, so for
  the dozen antibodies whose mean is 1 to 3 the whole signal is background,
  for CD25 (mean 9) it is about a fifth, for CD49d, integrin beta7, CD27 and
  CD4 it is negligible. The level does not follow the ambient pool's
  composition (empty droplets are 22% CD4, 7% CD8a; a composition-scaled
  ambient would put 0.3 counts of CD79a on a cell, not 1.9), so it is
  nonspecific binding, antibody-specific and roughly uniform at one to two
  counts, with heavy tails from aggregates (Fano factors 6 to 400, single
  cells at hundreds of counts). The aggregates are cell-level outliers and
  must go before the fit: they set the protein grid limit, and the coarse
  protein axis then loses the bulk of the distribution (validation item 6,
  pass 2). Without them the background's Fano factor is 1.4 to 3. A
  shared background rate as a fourth sampling parameter on the grid, in the
  mould of the ATAC dropout, is therefore the right first model: the
  low-abundance antibodies pin it across genes. Per-antibody rates would need
  a negative population per antibody, which a 210-plex panel with two isotype
  controls does not give. Built 2026-11-04 as `protein_background=(log10 low,
  log10 high, grid points)` on `CMEModel("Chromatin", protein=True)`: the
  fourth sampling entry, scanned on the grid like the dropout. In the kernel
  it is one factor `exp(lambda (g_P - 1))` on the protein frequencies the
  mesh keeps (the counts add independently, so the generating functions
  multiply; the coarse protein axis is unaffected), the simulator adds a
  Poisson draw to the protein, the moments start takes `lambda` off the
  protein mean. Pass 3 scanned it on the aggregate-free cells and it did not
  help (validation item 6): one rate shared across antibodies is the wrong
  shape. The open choice is a per-antibody rate fixed per gene from a
  negative cell type in the same run, which the model would take as a
  per-gene constant rather than a grid axis.
- **Emission default.** The BMMC analysis made `bursty=True` the sensible
  default for real data; the constitutive emission stays available for the
  limits and the nested comparisons.
- **Name.** A flag on `Chromatin` (`protein=True`) keeps one family and one
  parameter convention; a separate `bio_model` name would duplicate the
  chromatin options.
- **Fitting the protein pair at all.** Decided 2026-11-03: no prior selection.
  Fit every gene that has an antibody and a locus, and let the identifiability
  checks (the valley flag, the standard errors, the rejection tests) say where
  the protein pair is not determined. The alternative was to start with genes
  whose locus switches slowly compared with the protein lifetime, where the
  configuration can inform the protein through the mature RNA; on other loci
  the protein adds a layer downstream of an uninformative one. With the Rust
  kernel at under a second per evaluation a panel of tens of antibodies is
  affordable either way, and the per-configuration means the chromatin
  notebook recommends remain the right first look at each gene.
