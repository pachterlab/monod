"""Markov-modulated transcription: a locus of binary chromatin sites driving RNA.

The model is the one of Felce, Gorin and Pachter, *Biophysical model for joint
analysis of chromatin and RNA sequencing data* (Phys. Rev. E, 2024; code at
github.com/pachterlab/FGP_2023).  A locus has ``n`` sites, each open (1) or
closed (0), so the chromatin state is one of ``N = 2**n`` configurations.  Site
``j`` opens at rate ``k_on[j]`` and closes at rate ``k_off[j]``; every flip rate
out of a configuration is divided by ``epsilon`` raised to the number of
mismatched neighbouring pairs in that configuration, which gives an Ising
steady state with neighbour correlation when ``epsilon < 1``.  While the locus
is in configuration ``c`` it transcribes species ``j`` at rate ``A[c, j]``; the
RNA species then follow any first-order scheme (conversions and degradation),
written as the generator ``C`` of their characteristics.

The joint steady state of configuration and counts comes from the vector
generating function ``G(u) = (G_c(u))_c``, ``G_c(u) = E[prod_j (1+u_j)^{x_j};
state = c]``.  Along the characteristics ``u(tau)`` of the RNA part
(``du/dtau = C u``, ``tau`` = time to go) it obeys

    dG/dsigma = H^T G + diag(A u(tau)) G,      sigma = T - tau,

integrated forward from a horizon ``T`` beyond which the forcing is negligible,
starting from the chain's stationary distribution (no RNA in the far past).
:func:`eval_joint_pss` integrates this with a Strang splitting of two exact
sub-flows, the diagonal transcription term (its integral over a step is
analytic) and ``exp(H^T h)`` (one matrix exponential per step shared by the
whole grid), with the step chosen by error control across the grid and
Richardson extrapolation, as in :mod:`monod.reaction_network`.  Each
configuration's component is then inverted by an FFT over the count axes.

The integration runs in Rust (``monod_core.eval_chromatin_pgf``) when the
extension is installed and in NumPy otherwise; both follow the same scheme
step for step, so a schedule from one replays on the other.

This module is the numerical kernel and the transcription rules.  Wiring it
into :class:`CMEModel`, inference and the data path (assigning ATAC peaks to
loci and loci to genes) follows.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.fft import irfftn

from ._core import _mc
from .reaction_network import ReactionNetwork, StepSchedule

_TOL = 1e-5            # per-step error target on the generating function; with the Richardson
                       # extrapolation the error on P is about tol/100 (1e-7 here), tighter than the
                       # custom-network kernel at its 1e-6, at half the steps of tol = 1e-6
_C0 = 0.05             # first step as a fraction of the fastest time scale
_MAX_STEPS = 20_000

_expm = ReactionNetwork._expm


# ---------------------------------------------------------------------------
# Locus: configurations, transition matrix, steady state, dropout
# ---------------------------------------------------------------------------

def configurations(n_sites: int) -> np.ndarray:
    """All ``2**n`` configurations, shape ``(N, n)``; configuration ``c`` has
    site ``j`` open when bit ``j`` of ``c`` is set (site 0 = least significant)."""
    N = 2 ** n_sites
    c = np.arange(N)
    return ((c[:, None] >> np.arange(n_sites)[None, :]) & 1).astype(int)


def mismatches(configs: np.ndarray, periodic: bool = False) -> np.ndarray:
    """Number of neighbouring site pairs in different states, per configuration."""
    m = (configs[:, 1:] != configs[:, :-1]).sum(1)
    if periodic and configs.shape[1] > 2:
        m = m + (configs[:, 0] != configs[:, -1])
    return m


def transition_matrix(k_on, k_off, epsilon: float, periodic: bool = False) -> np.ndarray:
    """Generator ``H`` of the chromatin chain (``H[c, c']`` = rate from ``c`` to
    ``c'``, rows sum to zero), in the convention of ``FGP_2023/construct_H.py``:
    every flip out of ``c`` is divided by ``epsilon ** mismatches(c)``."""
    k_on = np.asarray(k_on, dtype=float)
    n = k_on.size
    k_off = np.broadcast_to(np.asarray(k_off, dtype=float), (n,))
    configs = configurations(n)
    N = configs.shape[0]
    pen = epsilon ** mismatches(configs, periodic).astype(float)
    H = np.zeros((N, N))
    for c in range(N):
        for j in range(n):
            target = c ^ (1 << j)
            rate = k_off[j] if configs[c, j] else k_on[j]
            H[c, target] = rate / pen[c]
    H[np.arange(N), np.arange(N)] = -H.sum(1)
    return H


def ising_steady_state(k_on, k_off, epsilon: float, periodic: bool = False) -> np.ndarray:
    """Stationary distribution of :func:`transition_matrix` in closed form:
    ``w(c) ∝ epsilon ** mismatches(c) * prod_open k_on * prod_closed k_off``."""
    k_on = np.asarray(k_on, dtype=float)
    n = k_on.size
    k_off = np.broadcast_to(np.asarray(k_off, dtype=float), (n,))
    configs = configurations(n)
    w = epsilon ** mismatches(configs, periodic).astype(float)
    w = w * np.where(configs == 1, k_on[None, :], k_off[None, :]).prod(1)
    return w / w.sum()


def stationary_distribution(H: np.ndarray) -> np.ndarray:
    """Left null vector of ``H`` normalised to one (any ergodic generator)."""
    vals, vecs = np.linalg.eig(H.T)
    k = int(np.argmin(np.abs(vals)))
    w = np.real(vecs[:, k])
    w = w / w.sum()
    return np.clip(w, 0.0, None) / np.clip(w, 0.0, None).sum()


def dropout_matrix(n_sites: int, p_drop) -> np.ndarray:
    """Linear map ``D`` from true to observed configuration probabilities when
    each open site is read as closed with probability ``p_drop`` (scalar or per
    site), independently: ``P_obs = P_true @ D``."""
    configs = configurations(n_sites)
    p = np.broadcast_to(np.asarray(p_drop, dtype=float), (n_sites,))
    N = configs.shape[0]
    D = np.zeros((N, N))
    for c in range(N):
        for o in range(N):
            if np.any((configs[o] == 1) & (configs[c] == 0)):
                continue            # a closed site cannot be read as open
            lost = (configs[c] == 1) & (configs[o] == 0)
            kept = (configs[c] == 1) & (configs[o] == 1)
            D[c, o] = np.prod(np.where(lost, p, 1.0)) * np.prod(np.where(kept, 1.0 - p, 1.0))
    return D


# ---------------------------------------------------------------------------
# System builders
# ---------------------------------------------------------------------------

def fgp_system(k_on, k_off, epsilon, b, d, periodic: bool = False):
    """The paper's system: one transcript per site, produced at rate ``b[j]``
    while site ``j`` is open, degraded at ``d[j]``.  Returns ``(H, A, C)``."""
    k_on = np.asarray(k_on, dtype=float)
    n = k_on.size
    configs = configurations(n)
    H = transition_matrix(k_on, k_off, epsilon, periodic)
    A = configs * np.asarray(b, dtype=float)[None, :]
    C = -np.diag(np.asarray(d, dtype=float))
    return H, A, C


# Transcription rules: how the open sites of a locus set one gene's nascent
# transcription rate.  Each rule maps its parameters to a rate per
# configuration (length 2**n).  "table" is the free rule, with one rate per
# configuration, and nests the other two, so a likelihood-ratio test of
# "additive" or "switch" against "table" (and against each other, where one
# nests the other) is how the rules are compared on data.
RULES = ("additive", "switch", "table")


def n_rule_params(rule: str, n_sites: int) -> int:
    """Number of transcription parameters a rule has for ``n_sites`` sites."""
    if rule == "additive":
        return n_sites
    if rule == "switch":
        return 1
    if rule == "table":
        return 2 ** n_sites
    raise ValueError(f"unknown transcription rule {rule!r}; choose from {RULES}")


def n_params(rule: str, n_sites: int, bursty: bool = False) -> int:
    """Length of the Chromatin parameter vector
    ``log10 [k_on (n_sites), epsilon, rates (k), beta, gamma]``, plus one for
    the burst size of a bursty model."""
    return n_sites + 1 + n_rule_params(rule, n_sites) + 2 + int(bursty)


def split_theta(theta: np.ndarray, rule: str, n_sites: int):
    """``theta`` (log10) -> ``(k_on, eps, rates, beta, gamma)`` on the linear
    scale; a trailing burst size is left to :func:`burst_size`."""
    n, k = n_sites, n_rule_params(rule, n_sites)
    lin = 10.0 ** np.asarray(theta, dtype=float)
    return lin[:n], float(lin[n]), lin[n + 1:n + 1 + k], float(lin[n + 1 + k]), float(lin[n + 2 + k])


def burst_size(theta: np.ndarray, rule: str, n_sites: int) -> Optional[float]:
    """The mean burst size (linear) when ``theta`` carries one, else None."""
    theta = np.asarray(theta, dtype=float)
    if theta.size == n_params(rule, n_sites, bursty=True):
        return float(10.0 ** theta[-1])
    if theta.size == n_params(rule, n_sites):
        return None
    raise ValueError(f"theta has {theta.size} entries; rule {rule!r} with {n_sites} sites takes "
                     f"{n_params(rule, n_sites)} or {n_params(rule, n_sites, True)}")


def join_theta(k_on, eps, rates, beta, gamma, burst: Optional[float] = None) -> np.ndarray:
    """The inverse of :func:`split_theta`: linear parts -> log10 vector."""
    tail = [beta, gamma] if burst is None else [beta, gamma, burst]
    return np.log10(np.concatenate([np.atleast_1d(k_on), [eps], np.atleast_1d(rates), tail]).astype(float))


def transcription_rates(rule, n_sites: int, params=None) -> np.ndarray:
    """Nascent transcription rate in each of the ``2**n_sites`` configurations.

    ``rule`` is one of :data:`RULES`, with ``params`` as follows:

    - ``"additive"``: ``params[j]`` is the rate contributed by site ``j`` while
      open; the rate in configuration ``c`` is the sum over its open sites
      (each site a weighted promoter).
    - ``"switch"``: a single rate ``params[0]`` whenever at least one site is
      open, zero when the locus is fully closed.
    - ``"table"``: ``params[c]`` is the rate in configuration ``c`` (binary
      order, site 0 the least significant bit); nests both rules above.

    ``rule`` may also be a callable applied to each 0/1 site vector, or an
    array of length ``2**n_sites`` (taken as a table).
    """
    configs = configurations(n_sites)
    if callable(rule):
        return np.array([float(rule(s)) for s in configs])
    if not isinstance(rule, str):
        rates = np.asarray(rule, dtype=float)
        if rates.shape != (configs.shape[0],):
            raise ValueError(f"a rate table for {n_sites} sites has length {configs.shape[0]}, got {rates.shape}")
        return rates
    params = np.atleast_1d(np.asarray(params, dtype=float))
    if params.size != n_rule_params(rule, n_sites):
        raise ValueError(f"rule {rule!r} with {n_sites} sites takes {n_rule_params(rule, n_sites)} parameters, got {params.size}")
    if rule == "additive":
        return configs @ params
    if rule == "switch":
        return params[0] * configs.any(axis=1)
    return params.copy()                                    # table


def rule_as_table(rule: str, n_sites: int, params) -> np.ndarray:
    """The parameters of ``rule`` expressed as the nesting ``"table"`` rule."""
    return transcription_rates(rule, n_sites, params)


def nascent_mature_system(k_on, k_off, epsilon, rates, beta: float, gamma: float,
                          periodic: bool = False, rule_params=None):
    """One gene with nascent and mature RNA driven by a locus of ``len(k_on)``
    sites: in configuration ``c`` nascent RNA is produced at the rate the
    transcription rule gives (``rates`` is a rule name from :data:`RULES` with
    ``rule_params``, a callable on the 0/1 site vector, or a table of length
    ``2**n``; see :func:`transcription_rates`), converts to mature at ``beta``
    and degrades at ``gamma``.  Returns ``(H, A, C)``."""
    k_on = np.asarray(k_on, dtype=float)
    configs = configurations(k_on.size)
    H = transition_matrix(k_on, k_off, epsilon, periodic)
    A = np.zeros((configs.shape[0], 2))
    A[:, 0] = transcription_rates(rates, k_on.size, rule_params)
    C = np.array([[-beta, beta], [0.0, -gamma]])
    return H, A, C


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

def _mesh(limits: Sequence[int]) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """PGF arguments ``u_j = exp(-2 pi i k / l_j) - 1`` on the half-spectrum mesh
    monod uses for ``irfftn``: shape ``(m, P)`` and the mesh shape."""
    mx = tuple(int(l) if i < len(limits) - 1 else int(l) // 2 + 1 for i, l in enumerate(limits))
    axes = [np.exp(-2j * np.pi * np.arange(mx[i]) / int(limits[i])) - 1 for i in range(len(limits))]
    grids = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.ravel() for g in grids]), mx


def _char_integral(C: np.ndarray, tau_a: float, delta: float) -> np.ndarray:
    """``∫_{tau_a}^{tau_a+delta} exp(C tau') dtau'`` = ``exp(C tau_a) · delta φ1(C delta)``."""
    m = C.shape[0]
    aug = np.zeros((2 * m, 2 * m))
    aug[:m, :m] = C * delta
    aug[:m, m:] = np.eye(m)
    x = _expm(aug)
    return _expm(C * tau_a) @ (delta * x[:m, m:])


def _rust_available() -> bool:
    return _mc is not None and hasattr(_mc, "eval_chromatin_pgf")


def locus_structure(H: np.ndarray, A: np.ndarray, rtol: float = 1e-12):
    """Structure the Rust kernel can exploit, read off the matrices.

    Returns ``(site_weights, independent)``: ``site_weights`` is ``W`` with
    ``A = S W`` for the 0/1 site matrix ``S`` when the rates factorise over
    sites (the additive rule, or one transcript per site), else ``None``;
    ``independent`` is the flat ``(k_on_j, k_off_j)`` list when every
    transition is a single-site flip whose rate does not depend on the other
    sites (``epsilon = 1``), else ``None``.  Both are exact identities, so
    using them changes results only at rounding level.
    """
    N = H.shape[0]
    n = int(N).bit_length() - 1
    if 2 ** n != N or n == 0:
        return None, None
    S = configurations(n).astype(float)
    scale = max(1.0, float(np.abs(A).max()))
    W, *_ = np.linalg.lstsq(S, A, rcond=None)
    site_weights = W if np.abs(S @ W - A).max() <= rtol * scale else None
    idx = np.arange(N)
    off = H - np.diag(np.diag(H))
    independent = []
    hscale = max(1.0, float(np.abs(H).max()))
    for j in range(n):
        bit = 1 << j
        closed = idx[(idx & bit) == 0]
        k_on = H[closed, closed | bit]
        k_off = H[closed | bit, closed]
        if np.ptp(k_on) > rtol * hscale or np.ptp(k_off) > rtol * hscale:
            independent = None
            break
        independent += [float(k_on[0]), float(k_off[0])]
        off[closed, closed | bit] = 0.0
        off[closed | bit, closed] = 0.0
    if independent is not None and np.abs(off).max() > rtol * hscale:
        independent = None
    return site_weights, independent


def eval_joint_pgf(H: np.ndarray, A: np.ndarray, C: np.ndarray, u: np.ndarray,
                   tol: float = _TOL, fixed_schedule: Optional[StepSchedule] = None,
                   backend: Optional[str] = None, burst: Optional[float] = None):
    """Vector generating function ``G[c, p] = E[prod_j (1+u_j)^{x_j}; state c]``
    at the PGF arguments ``u`` (shape ``(m, P)``), returned with the accepted
    :class:`StepSchedule`.  This is the kernel proper; :func:`eval_joint_pss`
    builds the FFT mesh and inverts, and :class:`monod.cme_toolbox.CMEModel`
    passes arguments already transformed by its sequencing-noise model.

    ``burst`` switches the emission from single molecules to geometric bursts
    of that mean size, shared by all configurations and species; ``A`` keeps
    its meaning as the mean production rate (burst frequency ``A / burst``),
    so ``burst -> 0`` is the constitutive model."""
    H = np.asarray(H, dtype=float); A = np.asarray(A, dtype=float); C = np.asarray(C, dtype=float)
    u = np.asarray(u, dtype=np.complex128)
    N, m = A.shape
    if u.shape[0] != m:
        raise ValueError(f"u must have one row per RNA species ({m}), got shape {u.shape}")
    P = u.shape[1]
    w = stationary_distribution(H)
    lam = np.abs(np.linalg.eigvals(C).real)
    lam_slow = float(lam[lam > 0].min()) if np.any(lam > 0) else 1.0
    lam_fast = max(float(lam.max()), float(np.abs(np.diag(H)).max()), float(2.0 * np.abs(A).max()), 1e-300)
    T = (np.log(1.0 / tol) + np.log1p(2.0 * float(np.abs(A).max()) / lam_slow)) / lam_slow
    if fixed_schedule is not None:
        T = float(np.sum(np.asarray(fixed_schedule, dtype=float)))   # replay over the schedule's own horizon
    h_first = _C0 / lam_fast
    h_min = 1e-12 / lam_fast
    if backend is None:
        backend = "rust" if _rust_available() else "python"
    if backend == "rust":
        if not _rust_available():
            raise RuntimeError("monod_core with eval_chromatin_pgf is not installed")
        site_weights, independent = locus_structure(H, A)
        re, im, _, steps = _mc.eval_chromatin_pgf(
            N, m, H.ravel().tolist(), A.ravel().tolist(), C.ravel().tolist(),
            [1] * m, w.tolist(), float(T), float(h_first), float(h_min), float(tol),
            None if fixed_schedule is None else [float(x) for x in np.asarray(fixed_schedule, dtype=float)],
            None if site_weights is None else np.asarray(site_weights, dtype=float).ravel().tolist(),
            independent,
            u.real.ravel().tolist(), u.imag.ravel().tolist(),
            None if burst is None else float(burst))
        G = (np.asarray(re) + 1j * np.asarray(im)).reshape(P, N).T
        steps = list(steps)
    elif backend == "python":
        HT = H.T
        G = np.repeat(w[:, None].astype(np.complex128), P, axis=1)
        G, steps = _integrate_numpy(HT, A, C, u, G, T, h_first, h_min, tol, fixed_schedule,
                                    None if burst is None else float(burst))
    else:
        raise ValueError(f"backend must be 'rust', 'python' or None, got {backend!r}")
    return G, StepSchedule(steps)


def invert_joint_pgf(G: np.ndarray, mesh_shape: Sequence[int], limits: Sequence[int]) -> np.ndarray:
    """``P[c, x]`` from the half-spectrum generating function of each configuration."""
    N = G.shape[0]
    pss = np.empty((N,) + tuple(int(l) for l in limits))
    for c in range(N):
        pss[c] = irfftn(G[c].reshape(tuple(mesh_shape)), s=[int(l) for l in limits])
    pss = np.abs(pss)
    return pss / pss.sum()


def eval_joint_pss(H: np.ndarray, A: np.ndarray, C: np.ndarray, limits: Sequence[int],
                   tol: float = _TOL, fixed_schedule: Optional[StepSchedule] = None,
                   return_schedule: bool = False, backend: Optional[str] = None,
                   burst: Optional[float] = None):
    """Joint steady-state distribution ``P[c, x_1, ..., x_m]`` of the chromatin
    configuration and the RNA counts on the grid ``limits``.

    Parameters
    ----------
    H : (N, N) chain generator, rows summing to zero (``H[c, c']`` = rate c → c').
    A : (N, m) transcription rate of species ``j`` in configuration ``c``.
    C : (m, m) generator of the RNA characteristics, ``du/dtau = C u``: for a
        species converting to ``k`` at rate ``r``, ``C[j, j] -= r`` and
        ``C[j, k] += r``; degradation at ``g`` is ``C[j, j] -= g``.
    limits : grid size per RNA species.
    tol : per-step error target on the generating function.
    fixed_schedule : replay these steps instead of adapting (stencils).
    return_schedule : also return the accepted :class:`StepSchedule`.
    backend : ``"rust"``, ``"python"`` or ``None`` (Rust when installed).
    burst : mean size of geometric transcription bursts (``A`` stays the mean
        production rate); ``None`` for single molecules.
    """
    u, mx = _mesh(limits)
    G, sched = eval_joint_pgf(H, A, C, u, tol=tol, fixed_schedule=fixed_schedule, backend=backend, burst=burst)
    pss = invert_joint_pgf(G, mx, limits)
    if return_schedule:
        return pss, sched
    return pss


def _production_exponent(A, C, u, tau_a, delta, burst):
    """``X[c, p] = Σ_j A[c, j] ∫_{tau_a}^{tau_a+delta} φ(U_j(s, p)) ds`` with
    ``U(s) = e^{Cs} u`` the characteristic: the diagonal sub-flow over the
    interval multiplies ``G[c, p]`` by ``exp(X[c, p])``.  Constitutive
    production has ``φ(U) = U`` and the integral in closed form; geometric
    bursts of mean size ``burst`` at mean production rate ``A`` have
    ``φ(U) = U / (1 - burst U)`` (burst frequency ``A / burst``), integrated
    by Simpson's rule on the interval, which the step-doubling error control
    covers like the splitting error."""
    if burst is None:
        return (A @ _char_integral(C, tau_a, delta)) @ u
    nodes = (tau_a, tau_a + 0.5 * delta, tau_a + delta)
    weights = (delta / 6.0, 4.0 * delta / 6.0, delta / 6.0)
    I = np.zeros_like(u)
    for s, w in zip(nodes, weights):
        U = _expm(C * s) @ u
        I += w * (U / (1.0 - burst * U))
    return A @ I


def _ops(A, C, u, tau, h, E, burst):
    """Operators of the Strang step from time-to-go ``tau`` to ``tau - h``: the
    production exponents over ``[tau-h/2, tau]`` and ``[tau-h, tau-h/2]`` and
    ``E = exp(H^T h)`` between them."""
    hh = 0.5 * h
    X1 = _production_exponent(A, C, u, tau - hh, hh, burst)
    X2 = _production_exponent(A, C, u, tau - h, hh, burst)
    return X1, E, X2


def _step(G, ops):
    X1, E, X2 = ops
    G = np.exp(X1) * G
    G = E @ G
    return np.exp(X2) * G


def _integrate_numpy(HT, A, C, u, G, T, h_first, h_min, tol, fixed_schedule, burst=None):
    """NumPy mirror of ``monod_core.eval_chromatin_pgf``: Strang steps with step
    doubling and Richardson extrapolation; returns ``(G, steps)``.  The two
    half steps share one ``exp(H^T h/2)``, the full step uses its square, and
    when the step doubles the previous full-step exponential is reused."""
    adaptive = fixed_schedule is None
    steps = []
    tau = T
    h = h_first
    last_E = None                                           # (h, exp(H^T h)) of the last accepted step
    while True:
        if adaptive:
            if tau <= 0 or len(steps) >= _MAX_STEPS:
                break
            h = min(h, tau)
        else:
            if len(steps) >= len(fixed_schedule):
                break
            h = float(np.asarray(fixed_schedule)[len(steps)])
        hh = 0.5 * h
        E_half = last_E[1] if last_E is not None and last_E[0] == hh else _expm(HT * hh)
        full = _ops(A, C, u, tau, h, E_half @ E_half, burst)
        half1 = _ops(A, C, u, tau, hh, E_half, burst)
        half2 = _ops(A, C, u, tau - hh, hh, E_half, burst)
        G1 = _step(G, full)
        G2 = _step(_step(G, half1), half2)
        diff = G2 - G1
        if adaptive:
            err = float(np.abs(diff).max())
            if err > tol and h > h_min:
                h *= max(0.2, 0.9 * (tol / err) ** (1.0 / 3.0))
                continue
            fac = min(2.0, max(0.2, 0.9 * (tol / max(err, 1e-300)) ** (1.0 / 3.0)))
        G = G2 + diff / 3.0                                 # Richardson
        last_E = (h, full[1])
        tau -= h
        steps.append(h)
        if adaptive:
            h *= fac
    return G, steps


def apply_dropout(pss: np.ndarray, n_sites: int, p_drop) -> np.ndarray:
    """Observed joint distribution after ATAC dropout on the configuration axis."""
    D = dropout_matrix(n_sites, p_drop)
    return np.tensordot(D.T, pss, axes=(1, 0))


def moments(pss: np.ndarray, n_sites: int):
    """Means, variances and the full covariance of (site states, RNA counts) under ``pss``."""
    configs = configurations(n_sites)
    N = configs.shape[0]
    m = pss.ndim - 1
    counts = [np.arange(s) for s in pss.shape[1:]]
    # per-cell variables: n site indicators + m counts
    grids = np.meshgrid(np.arange(N), *counts, indexing="ij")
    vars_ = [configs[grids[0], j] for j in range(n_sites)] + [grids[1 + j] for j in range(m)]
    mu = np.array([(v * pss).sum() for v in vars_])
    cov = np.array([[((vi - mi) * (vj - mj) * pss).sum() for vj, mj in zip(vars_, mu)] for vi, mi in zip(vars_, mu)])
    return mu, cov
