"""Per-locus maximum-likelihood fitting of the chromatin model.

This is the inference prototype for the ``Chromatin`` model family: one gene
whose nascent and mature RNA are driven by a locus of ``n`` binary chromatin
sites (:mod:`monod.chromatin`).  The data for a gene is a histogram of cells
over (configuration, nascent, mature).  The parameters, all in units of the
shared closing rate ``k_off = 1`` and fitted in log10, are

    k_on[n], epsilon, rates[k], beta, gamma

where ``rates`` are the parameters of the transcription rule (``switch``: one
rate; ``additive``: one weight per site; ``table``: one rate per
configuration, nesting both) and ATAC dropout ``p_drop`` is taken from the
chromatin marginal and held fixed.

The fit is staged to keep kernel evaluations few:

1. **Chain from the ATAC marginal.** The configuration marginal is the Ising
   steady state whatever the transcription does, and dropout acts on that
   axis alone, so ``k_on``, ``epsilon`` and ``p_drop`` have a closed-form
   multinomial likelihood (:func:`fit_chain_marginal`), no kernel involved.
2. **Moments for the RNA parameters.** Conditional nascent means per
   configuration give the rule's rates up to the scale ``beta``, which a
   short scan fixes together with ``gamma`` (:func:`rna_moment_start`).
3. **Search at a coarse tolerance.** L-BFGS-B over the RNA parameters with
   the chain fixed, then over everything, at step tolerance ``tol_search``;
   gradients are central differences on the base point's pinned step
   schedule, so the integrator's adaptivity does not leak into them.
4. **Final evaluation at the fine tolerance** for the reported likelihood.

:func:`fit_locus` fits the nested rules first and the table only where a
score test at the nested optimum says the free rates would move, warm-started
from that optimum (the nested-model check); it reports likelihood ratios
against the table and the BIC of each rule.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

from . import chromatin as ch

FD_EPS = 1e-4            # central-difference step in log10
TOL_SEARCH = 1e-4        # step tolerance during the search (error on P about 1e-6)
TOL_FINAL = 1e-5         # step tolerance for reported likelihoods
BOUNDS = dict(k_on=(-2.0, 2.0), eps=(-1.5, 0.5), rate=(-2.0, 2.0), beta=(-1.5, 1.5), gamma=(-1.5, 1.5))
K_OFF = 1.0


# ---------------------------------------------------------------------------
# Parameter vectors
# ---------------------------------------------------------------------------

def n_params(rule: str, n_sites: int) -> int:
    return n_sites + 1 + ch.n_rule_params(rule, n_sites) + 2


def split_theta(theta: np.ndarray, rule: str, n_sites: int):
    """``theta`` (log10) -> ``(k_on, eps, rates, beta, gamma)`` on the linear scale."""
    n, k = n_sites, ch.n_rule_params(rule, n_sites)
    lin = 10.0 ** np.asarray(theta, dtype=float)
    return lin[:n], float(lin[n]), lin[n + 1:n + 1 + k], float(lin[n + 1 + k]), float(lin[n + 2 + k])


def join_theta(k_on, eps, rates, beta, gamma) -> np.ndarray:
    return np.log10(np.concatenate([np.atleast_1d(k_on), [eps], np.atleast_1d(rates), [beta, gamma]]).astype(float))


def bounds(rule: str, n_sites: int):
    k = ch.n_rule_params(rule, n_sites)
    return [BOUNDS["k_on"]] * n_sites + [BOUNDS["eps"]] + [BOUNDS["rate"]] * k + [BOUNDS["beta"], BOUNDS["gamma"]]


def embed_in_table(theta: np.ndarray, rule: str, n_sites: int) -> np.ndarray:
    """The parameters of ``rule`` as a point of the ``table`` rule's space."""
    k_on, eps, rates, beta, gamma = split_theta(theta, rule, n_sites)
    tab = ch.rule_as_table(rule, n_sites, rates)
    lo = 10.0 ** BOUNDS["rate"][0]
    return join_theta(k_on, eps, np.maximum(tab, lo), beta, gamma)


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------

class LocusLikelihood:
    """Negative log-likelihood of a (configuration, nascent, mature) histogram
    under a transcription rule, with the kernel evaluated on the histogram's
    grid and a pinned-schedule central-difference gradient."""

    def __init__(self, hist: np.ndarray, rule: str, p_drop=None, periodic: bool = False):
        self.hist = np.asarray(hist, dtype=float)
        self.n_sites = int(self.hist.shape[0]).bit_length() - 1
        if 2 ** self.n_sites != self.hist.shape[0]:
            raise ValueError("the histogram's first axis must have 2**n_sites entries")
        self.rule = rule
        self.limits = [int(s) for s in self.hist.shape[1:]]
        self.periodic = periodic
        self.D = None if p_drop is None else ch.dropout_matrix(self.n_sites, p_drop)
        self.n_cells = float(self.hist.sum())
        self.nfev = 0

    def system(self, theta):
        k_on, eps, rates, beta, gamma = split_theta(theta, self.rule, self.n_sites)
        return ch.nascent_mature_system(k_on, K_OFF, eps, self.rule, beta, gamma, periodic=self.periodic, rule_params=rates)

    def pss(self, theta, tol, fixed_schedule=None, return_schedule=False):
        H, A, C = self.system(theta)
        out = ch.eval_joint_pss(H, A, C, self.limits, tol=tol, fixed_schedule=fixed_schedule, return_schedule=return_schedule)
        pss, sched = out if return_schedule else (out, None)
        if self.D is not None:
            pss = np.tensordot(self.D.T, pss, axes=(1, 0))
        self.nfev += 1
        return (pss, sched) if return_schedule else pss

    def nll(self, theta, tol=TOL_FINAL, fixed_schedule=None):
        p = self.pss(theta, tol, fixed_schedule)
        return -float((self.hist * np.log(np.maximum(p, 1e-300))).sum())

    def nll_grad(self, theta, tol, free=None):
        """``(nll, grad)`` at ``theta``; ``free`` restricts the stencil to those indices."""
        theta = np.asarray(theta, dtype=float)
        p, sched = self.pss(theta, tol, return_schedule=True)
        f0 = -float((self.hist * np.log(np.maximum(p, 1e-300))).sum())
        idx = range(theta.size) if free is None else free
        g = np.zeros(theta.size)
        for i in idx:
            e = np.zeros(theta.size); e[i] = FD_EPS
            g[i] = (self.nll(theta + e, tol, sched) - self.nll(theta - e, tol, sched)) / (2 * FD_EPS)
        return f0, g

    def hessian(self, theta, tol):
        """Observed information by central second differences on the pinned schedule."""
        theta = np.asarray(theta, dtype=float)
        p_, sched = self.pss(theta, tol, return_schedule=True)
        f0 = -float((self.hist * np.log(np.maximum(p_, 1e-300))).sum())
        d = theta.size
        fp = np.zeros(d); fm = np.zeros(d)
        E = np.eye(d) * FD_EPS
        for i in range(d):
            fp[i] = self.nll(theta + E[i], tol, sched)
            fm[i] = self.nll(theta - E[i], tol, sched)
        Hm = np.zeros((d, d))
        for i in range(d):
            Hm[i, i] = (fp[i] - 2 * f0 + fm[i]) / FD_EPS ** 2
            for j in range(i):
                fpp = self.nll(theta + E[i] + E[j], tol, sched)
                fmm = self.nll(theta - E[i] - E[j], tol, sched)
                Hm[i, j] = Hm[j, i] = (fpp - fp[i] - fp[j] + 2 * f0 - fm[i] - fm[j] + fmm) / (2 * FD_EPS ** 2)
        return Hm


# ---------------------------------------------------------------------------
# Stage 1: the chain from the ATAC marginal
# ---------------------------------------------------------------------------

@dataclass
class ChainFit:
    k_on: np.ndarray
    eps: float
    p_drop: Optional[np.ndarray]
    nll: float
    n_params: int
    nfev: int


def chain_marginal_nll(x, counts, n_sites, dropout, periodic):
    k_on = 10.0 ** x[:n_sites]
    eps = 10.0 ** x[n_sites]
    w = ch.ising_steady_state(k_on, K_OFF, eps, periodic)
    if dropout:
        w = w @ ch.dropout_matrix(n_sites, x[n_sites + 1:])
    return -float((counts * np.log(np.maximum(w, 1e-300))).sum())


def fit_chain_marginal(config_counts, dropout: Optional[str] = None, periodic: bool = False, fix_eps: bool = False) -> ChainFit:
    """Maximum-likelihood ``k_on`` (units of ``k_off``), ``epsilon`` and, if
    ``dropout`` is ``"shared"`` or ``"per_site"``, ``p_drop`` from the counts
    of cells in each configuration.  Closed form, no kernel."""
    counts = np.asarray(config_counts, dtype=float)
    n = int(counts.size).bit_length() - 1
    configs = ch.configurations(n)
    frac = np.clip((counts @ configs) / counts.sum(), 1e-3, 1 - 1e-3)
    x0 = list(np.log10(frac / (1 - frac))) + [0.0]
    bd = [BOUNDS["k_on"]] * n + [BOUNDS["eps"] if not fix_eps else (0.0, 0.0)]
    n_drop = {None: 0, "shared": 1, "per_site": n}[dropout]
    if n_drop:
        x0 += [0.1] * n_drop
        bd += [(0.0, 0.95)] * n_drop

    def f(x):
        xx = np.concatenate([x[:n + 1], np.broadcast_to(x[n + 1:], (n,))]) if n_drop == 1 else x
        return chain_marginal_nll(xx, counts, n, bool(n_drop), periodic)

    res = minimize(f, np.array(x0), method="L-BFGS-B", bounds=bd, options=dict(maxiter=500, ftol=1e-13, gtol=1e-8))
    p_drop = None if not n_drop else np.broadcast_to(res.x[n + 1:], (n,)).copy()
    return ChainFit(10.0 ** res.x[:n], float(10.0 ** res.x[n]), p_drop, float(res.fun),
                    n + (0 if fix_eps else 1) + n_drop, int(res.nfev))


# ---------------------------------------------------------------------------
# Stage 2: moments for the RNA parameters
# ---------------------------------------------------------------------------

def rna_moment_start(lik: LocusLikelihood, chain: ChainFit, betas: Sequence[float] = (0.3, 1.0, 3.0)) -> np.ndarray:
    """Starting ``theta`` for ``lik.rule``: rates from the conditional nascent
    means per configuration (scaled by ``beta``), ``gamma`` from the mature
    mean, ``beta`` from a short scan of the likelihood at the coarse tolerance."""
    hist, n = lik.hist, lik.n_sites
    N = hist.shape[0]
    per_config = hist.sum(axis=(1, 2))
    u = np.arange(hist.shape[1]); s = np.arange(hist.shape[2])
    m_c = np.where(per_config > 0, (hist.sum(2) @ u) / np.maximum(per_config, 1), 0.0)   # E[U | c]
    mu_u = float((hist.sum(2) @ u).sum() / lik.n_cells)
    mu_s = float((hist.sum(1) @ s).sum() / lik.n_cells)
    w = per_config / per_config.sum()
    configs = ch.configurations(n).astype(float)
    lo, hi = 10.0 ** BOUNDS["rate"][0], 10.0 ** BOUNDS["rate"][1]

    def rates_for(beta):
        a = beta * m_c                                               # rate per configuration
        if lik.rule == "table":
            r = a
        elif lik.rule == "additive":
            r, *_ = np.linalg.lstsq(configs[1:] * np.sqrt(w[1:, None]), a[1:] * np.sqrt(w[1:]), rcond=None)
        else:
            open_ = configs.any(1)
            r = np.array([(w[open_] @ a[open_]) / max(w[open_].sum(), 1e-12)])
        return np.clip(r, lo, hi)

    best = None
    for beta in betas:
        rates = rates_for(beta)
        mean_rate = float(w @ ch.transcription_rates(lik.rule, n, rates))
        gamma = np.clip(mean_rate / max(mu_s, 1e-6), 10.0 ** BOUNDS["gamma"][0], 10.0 ** BOUNDS["gamma"][1])
        theta = join_theta(chain.k_on, chain.eps, rates, beta, gamma)
        theta = np.clip(theta, [b[0] for b in bounds(lik.rule, n)], [b[1] for b in bounds(lik.rule, n)])
        f = lik.nll(theta, TOL_SEARCH)
        if best is None or f < best[0]:
            best = (f, theta)
    return best[1]


# ---------------------------------------------------------------------------
# Stage 3 and 4: search and final evaluation
# ---------------------------------------------------------------------------

@dataclass
class RuleFit:
    rule: str
    theta: np.ndarray
    nll: float
    n_params: int
    nfev: int
    seconds: float
    converged: bool
    hessian: Optional[np.ndarray] = None
    extra: Dict = field(default_factory=dict)

    def params(self, n_sites):
        k_on, eps, rates, beta, gamma = split_theta(self.theta, self.rule, n_sites)
        return dict(k_on=k_on, eps=eps, rates=rates, beta=beta, gamma=gamma)

    def bic(self, n_cells):
        return 2 * self.nll + self.n_params * np.log(n_cells)


def _lbfgs(lik: LocusLikelihood, theta0, free, tol, maxiter):
    theta0 = np.asarray(theta0, dtype=float)
    bd = bounds(lik.rule, lik.n_sites)
    free = list(free)

    def fg(x):
        th = theta0.copy(); th[free] = x
        f, g = lik.nll_grad(th, tol, free=free)
        return f, g[free]

    # ftol is relative to |f| (about 1e4 nats), so 1e-8 stops at changes of 1e-4 nats; a gradient of
    # 1e-2 nats per log10 unit against curvatures of order 1e4 locates parameters to 1e-6 in log10
    res = minimize(fg, theta0[free], jac=True, method="L-BFGS-B", bounds=[bd[i] for i in free],
                   options=dict(maxiter=maxiter, ftol=1e-8, gtol=1e-2))
    th = theta0.copy(); th[free] = res.x
    return th, float(res.fun), bool(res.success)


def fit_rule(lik: LocusLikelihood, chain: ChainFit, theta0=None, tol_search=TOL_SEARCH, tol_final=TOL_FINAL,
             maxiter=200, refine_chain=True, with_hessian=False) -> RuleFit:
    """Fit ``lik.rule``: from the moments start, the RNA parameters first
    with the chain fixed at the marginal fit, then everything jointly; from a
    given ``theta0`` (a warm start whose chain is already refined) the joint
    stage alone.  The search runs at ``tol_search``; the reported likelihood
    is at ``tol_final``."""
    t0 = time.time(); nfev0 = lik.nfev
    n, k = lik.n_sites, ch.n_rule_params(lik.rule, lik.n_sites)
    rna = list(range(n + 1, n + 3 + k))
    if theta0 is None:
        theta = rna_moment_start(lik, chain)
        theta, f, ok = _lbfgs(lik, theta, rna, tol_search, maxiter)
    else:
        theta = np.asarray(theta0, dtype=float)
    if refine_chain or theta0 is not None:
        theta, f, ok = _lbfgs(lik, theta, range(theta.size), tol_search, maxiter)
    nll = lik.nll(theta, tol_final)
    fit = RuleFit(lik.rule, theta, nll, n_params(lik.rule, n), lik.nfev - nfev0, time.time() - t0, ok)
    if with_hessian:
        fit.hessian = lik.hessian(theta, tol_final)
    return fit


def score_test(lik_table: LocusLikelihood, nested: RuleFit, n_sites: int, tol=TOL_SEARCH):
    """Rao score test of ``nested`` against the table rule at the nested
    optimum embedded in the table's space: ``S = g^T I^{-1} g`` with the
    observed information ``I``, on ``df`` = extra parameters.  Returns
    ``(statistic, df, p_value, theta_embedded)``; the statistic is ``inf``
    when the information is not positive definite (then the fit is run)."""
    theta = embed_in_table(nested.theta, nested.rule, n_sites)
    f0, g = lik_table.nll_grad(theta, tol)
    I = lik_table.hessian(theta, tol)
    # coordinates pinned at a bound (a configuration whose nested rate is zero sits at the
    # lower rate bound) are flat directions: they are held fixed and do not count as extra
    bd = np.array(bounds("table", n_sites))
    free = np.where((theta > bd[:, 0] + 1e-9) & (theta < bd[:, 1] - 1e-9))[0]
    n_pinned = theta.size - free.size
    df = max(n_params("table", n_sites) - nested.n_params - n_pinned, 1)
    g, I = g[free], I[np.ix_(free, free)]
    try:
        L = np.linalg.cholesky(I)
        y = np.linalg.solve(L, g)
        stat = float(y @ y)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(I)
        if vals.max() <= 0:
            return float("inf"), df, 0.0, theta
        vals = np.maximum(vals, 1e-8 * vals.max())
        stat = float((vecs.T @ g) ** 2 @ (1.0 / vals))
    return stat, df, float(chi2.sf(stat, df)), theta


@dataclass
class LocusResult:
    chain: ChainFit
    fits: Dict[str, RuleFit]
    n_cells: float
    n_sites: int
    score: Optional[Tuple[float, int, float]] = None
    seconds: float = 0.0

    def table(self):
        """Per-rule summary: nll, parameters, LR and p-value against the table, BIC."""
        rows = {}
        tab = self.fits.get("table")
        for rule, f in self.fits.items():
            lr = p = None
            if tab is not None and rule != "table":
                lr = 2 * (f.nll - tab.nll)
                p = float(chi2.sf(max(lr, 0.0), tab.n_params - f.n_params))
            rows[rule] = dict(nll=f.nll, n_params=f.n_params, lr_vs_table=lr, p_vs_table=p,
                              bic=f.bic(self.n_cells), nfev=f.nfev, seconds=f.seconds)
        return rows

    def best_by_bic(self):
        return min(self.fits, key=lambda r: self.fits[r].bic(self.n_cells))


def fit_locus(hist, rules: Sequence[str] = ("switch", "additive", "table"), table: str = "score", alpha: float = 0.05,
              dropout: Optional[str] = None, periodic: bool = False, tol_search=TOL_SEARCH, tol_final=TOL_FINAL,
              maxiter=200) -> LocusResult:
    """Fit the transcription rules to one gene's histogram ``(2**n, l_U, l_S)``.

    ``table`` is ``"score"`` (fit the table only when the score test at the
    best nested fit rejects it at level ``alpha``), ``"always"`` or
    ``"never"``.  The table is warm-started from the best nested fit.
    """
    t0 = time.time()
    hist = np.asarray(hist, dtype=float)
    chain = fit_chain_marginal(hist.sum(axis=(1, 2)), dropout=dropout, periodic=periodic)
    p_drop = chain.p_drop
    fits: Dict[str, RuleFit] = {}
    for rule in rules:
        if rule == "table":
            continue
        lik = LocusLikelihood(hist, rule, p_drop, periodic)
        fits[rule] = fit_rule(lik, chain, tol_search=tol_search, tol_final=tol_final, maxiter=maxiter)
    score = None
    if "table" in rules and table != "never":
        lik = LocusLikelihood(hist, "table", p_drop, periodic)
        nested = min(fits.values(), key=lambda f: f.nll) if fits else None
        run = True
        theta0 = None
        if nested is not None:
            if table == "score":
                stat, df, p, theta0 = score_test(lik, nested, lik.n_sites, tol_search)
                score = (stat, df, p)
                run = p < alpha
            else:
                theta0 = embed_in_table(nested.theta, nested.rule, lik.n_sites)
        if run:
            fits["table"] = fit_rule(lik, chain, theta0=theta0, tol_search=tol_search, tol_final=tol_final, maxiter=maxiter)
            if nested is not None and fits["table"].nll > nested.nll:
                # a nesting model can never fit worse: keep the nested optimum as the table's
                fits["table"] = RuleFit("table", embed_in_table(nested.theta, nested.rule, lik.n_sites), lik.nll(embed_in_table(nested.theta, nested.rule, lik.n_sites), tol_final),
                                        n_params("table", lik.n_sites), fits["table"].nfev, fits["table"].seconds, False,
                                        extra=dict(fell_back_to=nested.rule))
    return LocusResult(chain, fits, float(hist.sum()), int(hist.shape[0]).bit_length() - 1, score, time.time() - t0)
