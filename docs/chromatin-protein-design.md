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
  `embed_in_table` take a `protein` flag. Bounds for `k_p`, `gamma_p` as in
  `ProteinBursty` (`[-1.8, 3.5]`).
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

## Open decisions

- **Protein measurement noise.** `ProteinBursty` has none, and ADT counts carry
  a known ambient background. A per-dataset background rate on the protein
  channel would be a fourth sampling parameter on the grid, in the mould of the
  ATAC dropout. Deferred unless a dataset shows it matters.
- **Emission default.** The BMMC analysis made `bursty=True` the sensible
  default for real data; the constitutive emission stays available for the
  limits and the nested comparisons.
- **Name.** A flag on `Chromatin` (`protein=True`) keeps one family and one
  parameter convention; a separate `bio_model` name would duplicate the
  chromatin options.
- **Fitting the protein pair at all.** On loci where the configuration carries
  no information about the RNA, the protein adds a layer downstream of an
  uninformative one. The first fits should be on genes with a surface-protein
  readout and a locus that switches slowly, and they should start with the
  check the chromatin notebook recommends: mean RNA and mean protein per
  configuration before any fit.
