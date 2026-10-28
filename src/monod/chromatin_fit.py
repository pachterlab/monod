"""Starting points for the Chromatin model.

:class:`monod.cme_toolbox.CMEModel` with ``bio_model="Chromatin"`` is fitted
by the ordinary inference pipeline; this module supplies what its
method-of-moments hook needs, which the per-modality moments do not carry:

1. **The chain from the ATAC marginal.** The configuration marginal is the
   Ising steady state whatever the transcription does, and dropout acts on
   that axis alone, so ``k_on`` (in units of ``k_off = 1``) and ``epsilon``
   have a closed-form multinomial likelihood (:func:`fit_chain_marginal`), no
   kernel involved.
2. **The RNA parameters from conditional moments.** The nascent mean in each
   configuration gives the transcription rule's rates up to the scale
   ``beta``; the mature mean gives ``gamma``; a short scan of the likelihood
   fixes ``beta`` (:func:`moment_start`).

The parameter vector is ``log10 [k_on (n), epsilon, rates (k), beta, gamma]``,
with the burst size ``b`` appended for a bursty model
(:func:`monod.chromatin.split_theta`, :func:`monod.chromatin.burst_size`,
:func:`monod.chromatin.join_theta`); :func:`embed_in_table` maps the
``switch`` and ``additive`` rules into the ``table`` rule's space, for nested
comparisons (:func:`monod.analysis.compare_nested_models`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.optimize import minimize

from . import chromatin as ch

K_OFF = 1.0
RATE_FLOOR = 1e-4        # a configuration whose nested rate is zero sits here in the table's space


# ---------------------------------------------------------------------------
# Parameter vectors
# ---------------------------------------------------------------------------

from .chromatin import n_params, split_theta, burst_size, join_theta  # noqa: F401  (the vector layout lives there)


def embed_in_table(theta: np.ndarray, rule: str, n_sites: int, floor: float = RATE_FLOOR) -> np.ndarray:
    """The parameters of ``rule`` as a point of the ``table`` rule's space
    (a burst size is carried over)."""
    k_on, eps, rates, beta, gamma = split_theta(theta, rule, n_sites)
    tab = ch.rule_as_table(rule, n_sites, rates)
    return join_theta(k_on, eps, np.maximum(tab, floor), beta, gamma, burst_size(theta, rule, n_sites))


# ---------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------

def dense_histogram(hist, limits) -> np.ndarray:
    """Counts (or frequencies) on the full grid from monod's sparse
    ``(coords, freqs)`` histogram, or the array itself if already dense."""
    if isinstance(hist, np.ndarray) and hist.ndim == len(limits):
        return np.asarray(hist, dtype=float)
    coords, freqs = hist
    coords = np.asarray(coords, dtype=np.int64)
    dense = np.zeros([int(l) for l in limits])
    np.add.at(dense, tuple(coords.T), np.asarray(freqs, dtype=float))
    return dense


def sparse_histogram(dense):
    """monod's ``(coords, freqs)`` from a dense array of counts or frequencies."""
    dense = np.asarray(dense, dtype=float)
    coords = np.argwhere(dense > 0)
    freqs = dense[dense > 0]
    return coords.astype(np.int64), freqs / freqs.sum()


# ---------------------------------------------------------------------------
# The chain from the ATAC marginal
# ---------------------------------------------------------------------------

@dataclass
class ChainFit:
    k_on: np.ndarray
    eps: float
    p_drop: Optional[np.ndarray]
    nll: float
    n_params: int
    nfev: int


def chain_marginal_nll(x, counts, n_sites, p_drop, periodic):
    k_on = 10.0 ** x[:n_sites]
    eps = 10.0 ** x[n_sites]
    w = ch.ising_steady_state(k_on, K_OFF, eps, periodic)
    if p_drop is not None:
        w = w @ ch.dropout_matrix(n_sites, p_drop)
    return -float((counts * np.log(np.maximum(w, 1e-300))).sum())


def fit_chain_marginal(config_counts, dropout: Optional[str] = None, periodic: bool = False, fix_eps: bool = False,
                       p_drop=None, k_on_bounds=(-2.0, 2.0), eps_bounds=(-1.5, 0.5)) -> ChainFit:
    """Maximum-likelihood ``k_on`` (units of ``k_off``) and ``epsilon`` from the
    counts of cells in each configuration.  ``p_drop`` holds a known dropout
    (scalar or per site); ``dropout="shared"`` or ``"per_site"`` fits it
    instead.  Closed form, no kernel."""
    counts = np.asarray(config_counts, dtype=float)
    n = int(counts.size).bit_length() - 1
    configs = ch.configurations(n)
    frac = np.clip((counts @ configs) / counts.sum(), 1e-3, 1 - 1e-3)
    x0 = list(np.clip(np.log10(frac / (1 - frac)), k_on_bounds[0], k_on_bounds[1])) + [0.0]
    bd = [tuple(k_on_bounds)] * n + [tuple(eps_bounds) if not fix_eps else (0.0, 0.0)]
    n_drop = {None: 0, "shared": 1, "per_site": n}[dropout]
    fixed = None if p_drop is None else np.broadcast_to(np.asarray(p_drop, dtype=float), (n,)).copy()
    if n_drop:
        x0 += [0.1] * n_drop
        bd += [(0.0, 0.95)] * n_drop

    def f(x):
        pd = np.broadcast_to(x[n + 1:], (n,)) if n_drop else fixed
        return chain_marginal_nll(x, counts, n, pd, periodic)

    res = minimize(f, np.array(x0), method="L-BFGS-B", bounds=bd, options=dict(maxiter=500, ftol=1e-13, gtol=1e-8))
    pd = np.broadcast_to(res.x[n + 1:], (n,)).copy() if n_drop else fixed
    return ChainFit(10.0 ** res.x[:n], float(10.0 ** res.x[n]), pd, float(res.fun),
                    n + (0 if fix_eps else 1) + n_drop, int(res.nfev))


# ---------------------------------------------------------------------------
# RNA parameters from conditional moments
# ---------------------------------------------------------------------------

def moment_start(hist: np.ndarray, rule: str, n_sites: int, chain: ChainFit, objective: Callable[[np.ndarray], float],
                 rate_bounds=(-2.0, 2.0), gamma_bounds=(-1.8, 3.5), betas: Sequence[float] = (0.3, 1.0, 3.0),
                 rna_scale=(1.0, 1.0), bursts: Optional[Sequence[float]] = None) -> np.ndarray:
    """Starting ``theta`` for ``rule`` from a dense histogram over
    (configuration, nascent, mature): rates from the conditional nascent means
    per configuration (scaled by ``beta``), ``gamma`` from the mature mean,
    ``beta`` from the best of ``betas`` under ``objective``.  ``rna_scale`` is
    the sequencing model's expected capture per RNA modality, divided out.
    For a bursty model ``bursts`` lists the burst sizes to try with each
    ``beta``; the rates are mean production rates either way."""
    hist = np.asarray(hist, dtype=float)
    n = n_sites
    per_config = hist.sum(axis=(1, 2))
    n_cells = float(hist.sum())
    u = np.arange(hist.shape[1]); s = np.arange(hist.shape[2])
    m_c = np.where(per_config > 0, (hist.sum(2) @ u) / np.maximum(per_config, 1e-300), 0.0) / rna_scale[0]   # E[U | c]
    mu_s = float((hist.sum(1) @ s).sum() / n_cells) / rna_scale[1]
    w = per_config / per_config.sum()
    configs = ch.configurations(n).astype(float)
    lo, hi = 10.0 ** rate_bounds[0], 10.0 ** rate_bounds[1]

    def rates_for(beta):
        a = beta * m_c                                               # rate per configuration
        if rule == "table":
            r = a
        elif rule == "additive":
            r, *_ = np.linalg.lstsq(configs[1:] * np.sqrt(w[1:, None]), a[1:] * np.sqrt(w[1:]), rcond=None)
        else:
            open_ = configs.any(1)
            r = np.array([(w[open_] @ a[open_]) / max(w[open_].sum(), 1e-12)])
        return np.clip(r, lo, hi)

    best = None
    for beta in betas:
        rates = rates_for(beta)
        mean_rate = float(w @ ch.transcription_rates(rule, n, rates))
        gamma = np.clip(mean_rate / max(mu_s, 1e-6), 10.0 ** gamma_bounds[0], 10.0 ** gamma_bounds[1])
        for b in ([None] if bursts is None else list(bursts)):
            theta = join_theta(chain.k_on, chain.eps, rates, beta, gamma, b)
            f = float(objective(theta))
            if best is None or f < best[0]:
                best = (f, theta)
    return best[1]


def chromatin_start(model, hist, limits, samp, lb_log, ub_log) -> np.ndarray:
    """The Chromatin model's starting point for one gene: chain from the
    configuration marginal (dropout fixed at the sampling vector's), RNA
    parameters from conditional moments with ``beta`` chosen by the model's
    own divergence at the search tolerance.  ``hist`` is monod's sparse
    ``(coords, freqs)`` on ``limits`` in the model's modality order
    (unspliced, spliced, configuration)."""
    n = model.n_sites
    lb_log = np.asarray(lb_log, float); ub_log = np.asarray(ub_log, float)
    dense = np.moveaxis(dense_histogram(hist, limits), -1, 0)        # (N, lu, ls)
    chain = fit_chain_marginal(dense.sum(axis=(1, 2)), periodic=model.periodic, p_drop=model.chromatin_dropout(samp),
                               k_on_bounds=(lb_log[0], ub_log[0]), eps_bounds=(lb_log[n], ub_log[n]))
    scale = (1.0, 1.0)
    if model.seq_model in ("Poisson", "Bernoulli") and samp is not None:
        lam = 10.0 ** np.asarray(samp, dtype=float).ravel()[:2]
        scale = (float(lam[0]), float(lam[1]))
    data = hist if not (isinstance(hist, np.ndarray) and hist.ndim == 3) else sparse_histogram(hist)

    def objective(theta):
        with model.search_tolerance():
            return model.eval_model_kld(np.clip(theta, lb_log, ub_log), limits, samp, data)

    bursty = bool(getattr(model, "bursty", False))
    i_gamma = n + 2 + ch.n_rule_params(model.rule, n)
    bursts = None
    if bursty:
        lo, hi = 10.0 ** lb_log[-1], 10.0 ** ub_log[-1]
        bursts = [float(np.clip(b, lo, hi)) for b in (0.5, 2.0, 8.0)]
    return moment_start(dense, model.rule, n, chain, objective,
                        rate_bounds=(lb_log[n + 1], ub_log[n + 1]), gamma_bounds=(lb_log[i_gamma], ub_log[i_gamma]),
                        rna_scale=scale, bursts=bursts)
