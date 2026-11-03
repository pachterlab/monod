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

from ._core import _mc, step_control
from .reaction_network import ReactionNetwork, StepSchedule, _phi_ops

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


def n_params(rule: str, n_sites: int, bursty: bool = False, protein: bool = False) -> int:
    """Length of the Chromatin parameter vector
    ``log10 [k_on (n_sites), epsilon, rates (k), beta, gamma]``, plus one for
    the burst size of a bursty model, plus ``k_p, gamma_p`` (last) with a
    protein layer."""
    return n_sites + 1 + n_rule_params(rule, n_sites) + 2 + int(bursty) + 2 * int(protein)


def split_theta(theta: np.ndarray, rule: str, n_sites: int):
    """``theta`` (log10) -> ``(k_on, eps, rates, beta, gamma)`` on the linear
    scale; a trailing burst size is left to :func:`burst_size`."""
    n, k = n_sites, n_rule_params(rule, n_sites)
    lin = 10.0 ** np.asarray(theta, dtype=float)
    return lin[:n], float(lin[n]), lin[n + 1:n + 1 + k], float(lin[n + 1 + k]), float(lin[n + 2 + k])


def burst_size(theta: np.ndarray, rule: str, n_sites: int, protein: bool = False) -> Optional[float]:
    """The mean burst size (linear) when ``theta`` carries one, else None
    (``protein``: the vector ends with ``k_p, gamma_p``, which sit after it)."""
    theta = np.asarray(theta, dtype=float)
    tail = 2 if protein else 0
    if theta.size == n_params(rule, n_sites, bursty=True, protein=protein):
        return float(10.0 ** theta[-1 - tail])
    if theta.size == n_params(rule, n_sites, protein=protein):
        return None
    raise ValueError(f"theta has {theta.size} entries; rule {rule!r} with {n_sites} sites takes "
                     f"{n_params(rule, n_sites, protein=protein)} or {n_params(rule, n_sites, True, protein)}")


def protein_rates(theta: np.ndarray) -> Tuple[float, float]:
    """``(k_p, gamma_p)`` on the linear scale: the last two entries of a
    protein model's vector."""
    theta = np.asarray(theta, dtype=float)
    return float(10.0 ** theta[-2]), float(10.0 ** theta[-1])


def join_theta(k_on, eps, rates, beta, gamma, burst: Optional[float] = None,
               protein: Optional[Sequence[float]] = None) -> np.ndarray:
    """The inverse of :func:`split_theta`: linear parts -> log10 vector
    (``protein`` is the pair ``(k_p, gamma_p)``, appended last)."""
    tail = [beta, gamma] if burst is None else [beta, gamma, burst]
    if protein is not None:
        tail = tail + [float(protein[0]), float(protein[1])]
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
    monod uses for ``irfftn``: shape ``(m, P)`` and the mesh shape (the cached
    builder of :mod:`monod.cme_toolbox`)."""
    from .cme_toolbox import _build_mesh          # lazy: cme_toolbox imports this module
    limits = tuple(int(l) for l in limits)
    mx = tuple(l if i < len(limits) - 1 else l // 2 + 1 for i, l in enumerate(limits))
    return _build_mesh(limits, mx), mx


def _char_integral(C: np.ndarray, tau_a: float, delta: float) -> np.ndarray:
    """``∫_{tau_a}^{tau_a+delta} exp(C tau') dtau'`` = ``exp(C tau_a) · delta φ1(C delta)``."""
    return _expm(C * tau_a) @ _phi_ops(C, delta, 1)[1]


def _rust_available(protein: bool = False) -> bool:
    """Whether ``monod_core.eval_chromatin_pgf`` is installed (and, with
    ``protein``, whether it is a build that takes the protein layer)."""
    if _mc is None or not hasattr(_mc, "eval_chromatin_pgf"):
        return False
    if not protein:
        return True
    return "protein" in (getattr(_mc.eval_chromatin_pgf, "__text_signature__", None) or "")


class _LinearCharacteristics:
    """The RNA characteristics of a first-order scheme, ``U(s) = exp(C s) u``,
    at the mesh points ``u`` (shape ``(m, P)``): what the production exponent
    of the Strang step reads.  ``values(s)`` is ``U(s)``; ``integral(tau_a,
    delta)`` is ``∫_{tau_a}^{tau_a+delta} U(s) ds``, in closed form here."""

    def __init__(self, C: np.ndarray, u: np.ndarray):
        self.C = np.asarray(C, dtype=float)
        self.u = u

    def values(self, s: float) -> np.ndarray:
        return _expm(self.C * s) @ self.u

    def integral(self, tau_a: float, delta: float) -> np.ndarray:
        return _char_integral(self.C, tau_a, delta) @ self.u


def _phi1(z: np.ndarray) -> np.ndarray:
    """``(exp(z) - 1) / z`` with the series near zero (vectorised, complex)."""
    z = np.asarray(z)
    out = np.empty_like(z)
    big = np.abs(z) > 1e-4
    out[big] = np.expm1(z[big]) / z[big]
    zs = z[~big]
    out[~big] = 1.0 + zs * 0.5 + zs * zs * (1.0 / 6.0)
    return out


class _ProteinCharacteristics:
    """Characteristics of nascent RNA, mature RNA and protein, the system of
    ``CMEModel.protein_pgf``, for a locus whose configuration sets only the
    nascent production rate.  In ``v = 1 + u_U``, ``w = 1 + u_S`` and the
    protein argument ``u_P`` (``s`` = time to go)::

        u_P(s) = u_P exp(-gamma_p s)
        w'     = a(s) w + gamma,        a(s) = -gamma + k_p u_P(s)
        v'     = beta (w - v)

    The stepper of :func:`eval_joint_pgf` walks from the horizon ``T`` down
    to zero and so needs the characteristics at the far end first, while
    they are known at ``s = 0`` (the mesh point): they are therefore
    tabulated once over ``[0, T]`` with the protein integrator's step (the
    homogeneous part of ``w`` exact per step, the forcing and the coupling
    with the step-average coefficient) under step doubling, and read at any
    ``s`` by cubic Hermite interpolation between the accepted nodes with the
    derivatives the equations give there.  The step control measures the
    interpolation error as well (the half-step value against the cubic
    through the step's ends), since a step that is exact for the equations,
    as every step is when ``k_p`` vanishes, can still be too long for the
    cubic; the nodes are the half steps, so what the stepper reads is
    accurate to a sixteenth of that.  Only the two RNA rows are returned:
    the locus does not make protein.

    The table costs ``2 x nodes x P`` complex numbers; a Rust port would
    tabulate per chunk of mesh points under a pinned schedule.
    """

    def __init__(self, beta: float, gamma: float, k_p: float, gamma_p: float, u: np.ndarray, T: float, tol: float):
        self.beta, self.gamma, self.k_p, self.gamma_p = float(beta), float(gamma), float(k_p), float(gamma_p)
        u = np.asarray(u, dtype=np.complex128)
        if u.shape[0] != 3:
            raise ValueError(f"protein characteristics need (u_U, u_S, u_P) rows, got shape {u.shape}")
        self.u_P = u[2]
        self._tabulate(u[0] + 1.0, u[1] + 1.0, float(T), float(tol))

    # one step of the protein integrator from time ``t`` over ``h``
    def _step(self, t, h, v, w):
        beta, gamma, k_p, gamma_p = self.beta, self.gamma, self.k_p, self.gamma_p
        da = -gamma * h + (k_p / gamma_p) * np.exp(-gamma_p * t) * (-np.expm1(-gamma_p * h)) * self.u_P
        e = np.exp(da)
        w_new = e * w + _phi1(da) * (gamma * h)
        abar = da / h
        s_ = abar + beta
        eb = np.exp(-beta * h)
        i0 = h * float(_phi1(np.array([-beta * h]))[0].real)
        z = s_ * h
        i1 = np.empty_like(z)
        big = np.abs(z) > 1e-3
        i1[big] = (e[big] - eb) / s_[big]
        zs = z[~big]
        i1[~big] = (1.0 + zs * 0.5 + zs * zs * (1.0 / 6.0)) * (h * eb)
        goa = gamma / abar
        v_new = v * eb + ((w + goa) * i1 - goa * i0) * beta
        return v_new, w_new

    def _tabulate(self, v, w, T, tol):
        lam_fast = max(self.beta, self.gamma + 2.0 * self.k_p, self.gamma_p)
        h = _C0 / lam_fast
        h_min = 1e-12 / lam_fast
        nodes, V, W = [0.0], [v], [w]
        t = 0.0
        n_steps = 0
        while t < T and n_steps < _MAX_STEPS:
            h = min(h, T - t)
            v1, w1 = self._step(t, h, v, w)
            vm, wm = self._step(t, 0.5 * h, v, w)
            v2, w2 = self._step(t + 0.5 * h, 0.5 * h, vm, wm)
            err = max(float(np.abs(v2 - v1).max()), float(np.abs(w2 - w1).max()))
            # the cubic through the step's ends at its middle, against the half-step value
            a0 = -self.gamma + self.k_p * self.u_P * np.exp(-self.gamma_p * t)
            a2 = -self.gamma + (a0 + self.gamma) * np.exp(-self.gamma_p * h)
            v_mid = 0.5 * (v + v2) + 0.125 * h * (self.beta * (w - v) - self.beta * (w2 - v2))
            w_mid = 0.5 * (w + w2) + 0.125 * h * ((a0 * w + self.gamma) - (a2 * w2 + self.gamma))
            err = max(err, float(np.abs(v_mid - vm).max()), float(np.abs(w_mid - wm).max()))
            accept, fac = step_control(err, h, h_min, tol)
            if not accept:
                h = fac
                continue
            nodes += [t + 0.5 * h, t + h]
            V += [vm, v2]
            W += [wm, w2]
            v, w = v2, w2
            t += h
            n_steps += 1
            h *= fac
        self.nodes = np.asarray(nodes, dtype=float)
        self.V = np.asarray(V)
        self.W = np.asarray(W)
        self.T = float(self.nodes[-1])

    def _derivatives(self, k):
        """``(v', w')`` at node ``k`` from the equations."""
        a = -self.gamma + self.k_p * self.u_P * np.exp(-self.gamma_p * self.nodes[k])
        return self.beta * (self.W[k] - self.V[k]), a * self.W[k] + self.gamma

    def values(self, s: float) -> np.ndarray:
        s = float(s)
        if s <= 0.0:
            v, w = self.V[0], self.W[0]
        elif s >= self.T:
            v, w = self.V[-1], self.W[-1]
        else:
            k = int(np.searchsorted(self.nodes, s, side="right")) - 1
            h = self.nodes[k + 1] - self.nodes[k]
            x = (s - self.nodes[k]) / h
            h00 = (1.0 + 2.0 * x) * (1.0 - x) ** 2
            h10 = x * (1.0 - x) ** 2
            h01 = x * x * (3.0 - 2.0 * x)
            h11 = x * x * (x - 1.0)
            dv0, dw0 = self._derivatives(k)
            dv1, dw1 = self._derivatives(k + 1)
            v = h00 * self.V[k] + h10 * h * dv0 + h01 * self.V[k + 1] + h11 * h * dv1
            w = h00 * self.W[k] + h10 * h * dw0 + h01 * self.W[k + 1] + h11 * h * dw1
        return np.vstack([v - 1.0, w - 1.0])

    def integral(self, tau_a: float, delta: float) -> np.ndarray:
        """Simpson on the interval; the stepper's step doubling covers it."""
        return (delta / 6.0) * (self.values(tau_a) + 4.0 * self.values(tau_a + 0.5 * delta) + self.values(tau_a + delta))


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


def rule_structure(H: np.ndarray, A: np.ndarray, k_on, k_off, epsilon: float, rule, rule_params=None):
    """:func:`locus_structure` of a :func:`nascent_mature_system`, with the
    parts the rule fixes read off directly instead of detected: the additive
    rule's rates factorise over sites (``site_weights`` holds the per-site
    rates), and ``epsilon = 1`` makes every flip independent of the other
    sites.  Anything else is detected from the matrices as before."""
    k_on = np.asarray(k_on, dtype=float)
    n = k_on.size
    k_off = np.broadcast_to(np.asarray(k_off, dtype=float), (n,))
    site_weights = independent = None
    if isinstance(rule, str) and rule == "additive":
        site_weights = np.zeros((n, A.shape[1]))
        site_weights[:, 0] = np.asarray(rule_params, dtype=float)
    if float(epsilon) == 1.0:
        independent = [float(v) for j in range(n) for v in (k_on[j], k_off[j])]
    if site_weights is None or independent is None:
        sw, ind = locus_structure(H, A)
        site_weights = sw if site_weights is None else site_weights
        independent = ind if independent is None else independent
    return site_weights, independent


def eval_joint_pgf(H: np.ndarray, A: np.ndarray, C: np.ndarray, u: np.ndarray,
                   tol: float = _TOL, fixed_schedule: Optional[StepSchedule] = None,
                   backend: Optional[str] = None, burst: Optional[float] = None,
                   structure=None, protein: Optional[Tuple[float, float]] = None):
    """Vector generating function ``G[c, p] = E[prod_j (1+u_j)^{x_j}; state c]``
    at the PGF arguments ``u`` (shape ``(m, P)``), returned with the accepted
    :class:`StepSchedule`.  This is the kernel proper; :func:`eval_joint_pss`
    builds the FFT mesh and inverts, and :class:`monod.cme_toolbox.CMEModel`
    passes arguments already transformed by its sequencing-noise model.

    ``burst`` switches the emission from single molecules to geometric bursts
    of that mean size, shared by all configurations and species; ``A`` keeps
    its meaning as the mean production rate (burst frequency ``A / burst``),
    so ``burst -> 0`` is the constitutive model.

    ``structure`` is the ``(site_weights, independent)`` pair of
    :func:`locus_structure` when the caller already knows it
    (:func:`rule_structure`); the Rust kernel exploits it, and it is detected
    from the matrices when not given.

    ``protein=(k_p, gamma_p)`` adds the protein layer of ``ProteinBursty``
    downstream of the mature RNA (``C`` must be the nascent/mature scheme
    ``[[-beta, beta], [0, -gamma]]``): ``u`` then has a third row, the
    protein's argument, and the characteristics come from
    :class:`_ProteinCharacteristics` instead of ``exp(C s) u``.  The locus
    still only sets the nascent production rate, so the stepper is otherwise
    unchanged.  The Rust kernel tabulates the same characteristics as an
    affine map per distinct protein argument (``monod_core.eval_chromatin_pgf``
    with ``protein=``)."""
    H = np.asarray(H, dtype=float); A = np.asarray(A, dtype=float); C = np.asarray(C, dtype=float)
    u = np.asarray(u, dtype=np.complex128)
    N, m = A.shape
    if u.shape[0] != m + (1 if protein is not None else 0):
        raise ValueError(f"u must have one row per RNA species ({m}){' plus the protein' if protein is not None else ''}, "
                         f"got shape {u.shape}")
    P = u.shape[1]
    w = stationary_distribution(H)
    lam = np.abs(np.linalg.eigvals(C).real)
    lam_slow = float(lam[lam > 0].min()) if np.any(lam > 0) else 1.0
    lam_fast = max(float(lam.max()), float(np.abs(np.diag(H)).max()), float(2.0 * np.abs(A).max()), 1e-300)
    if protein is not None:
        k_p, gamma_p = (float(x) for x in protein)
        if not (k_p > 0.0 and gamma_p > 0.0):
            raise ValueError(f"protein=(k_p, gamma_p) must be positive, got {protein!r}")
        if m != 2 or C[1, 0] != 0.0 or C[0, 1] != -C[0, 0]:
            raise ValueError("the protein layer needs the nascent/mature scheme C = [[-beta, beta], [0, -gamma]]")
        beta, gamma = -C[0, 0], -C[1, 1]
        lam_slow = min(lam_slow, gamma_p)
        lam_fast = max(lam_fast, gamma + 2.0 * k_p, gamma_p)
    T = (np.log(1.0 / tol) + np.log1p(2.0 * float(np.abs(A).max()) / lam_slow)) / lam_slow
    if protein is not None:
        T += max(0.0, np.log(2.0 * k_p / gamma)) / gamma_p          # as the protein integrator's horizon
    if fixed_schedule is not None:
        T = float(np.sum(np.asarray(fixed_schedule, dtype=float)))   # replay over the schedule's own horizon
    h_first = _C0 / lam_fast
    h_min = 1e-12 / lam_fast
    if backend is None:
        backend = "rust" if _rust_available(protein is not None) else "python"
    if backend == "rust":
        if not _rust_available(protein is not None):
            raise RuntimeError("monod_core with eval_chromatin_pgf" + (" taking the protein layer" if protein is not None else "")
                               + " is not installed")
        site_weights, independent = locus_structure(H, A) if structure is None else structure
        prot = None
        if protein is not None:
            lam_p = max(beta, gamma + 2.0 * k_p, gamma_p)              # the tabulation's own controls, as in Python
            prot = (k_p, gamma_p, _C0 / lam_p, 1e-12 / lam_p, 0.1 * tol)
        re, im, _, steps = _mc.eval_chromatin_pgf(
            N, m, H.ravel().tolist(), A.ravel().tolist(), C.ravel().tolist(),
            [1] * m, w.tolist(), float(T), float(h_first), float(h_min), float(tol),
            None if fixed_schedule is None else [float(x) for x in np.asarray(fixed_schedule, dtype=float)],
            None if site_weights is None else np.asarray(site_weights, dtype=float).ravel().tolist(),
            independent,
            u.real.ravel().tolist(), u.imag.ravel().tolist(),
            None if burst is None else float(burst),
            prot)
        G = (np.asarray(re) + 1j * np.asarray(im)).reshape(P, N).T
        steps = list(steps)
    elif backend == "python":
        HT = H.T
        G = np.repeat(w[:, None].astype(np.complex128), P, axis=1)
        if protein is None:
            chars = _LinearCharacteristics(C, u)
        else:
            chars = _ProteinCharacteristics(beta, gamma, k_p, gamma_p, u, T, 0.1 * tol)
        G, steps = _integrate_numpy(HT, A, chars, G, T, h_first, h_min, tol, fixed_schedule,
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
                   burst: Optional[float] = None, protein: Optional[Tuple[float, float]] = None):
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
    protein : ``(k_p, gamma_p)`` for a protein layer; ``limits`` then has a
        third RNA-side entry, the protein axis.
    """
    u, mx = _mesh(limits)
    G, sched = eval_joint_pgf(H, A, C, u, tol=tol, fixed_schedule=fixed_schedule, backend=backend, burst=burst,
                              protein=protein)
    pss = invert_joint_pgf(G, mx, limits)
    if return_schedule:
        return pss, sched
    return pss


def _production_exponent(A, chars, tau_a, delta, burst):
    """``X[c, p] = Σ_j A[c, j] ∫_{tau_a}^{tau_a+delta} φ(U_j(s, p)) ds`` with
    ``U(s)`` the characteristic (``chars``, a :class:`_LinearCharacteristics`
    or :class:`_ProteinCharacteristics`): the diagonal sub-flow over the
    interval multiplies ``G[c, p]`` by ``exp(X[c, p])``.  Constitutive
    production has ``φ(U) = U`` and the provider's integral; geometric bursts
    of mean size ``burst`` at mean production rate ``A`` have
    ``φ(U) = U / (1 - burst U)`` (burst frequency ``A / burst``), integrated
    by Simpson's rule on the interval, which the step-doubling error control
    covers like the splitting error."""
    if burst is None:
        return A @ chars.integral(tau_a, delta)
    nodes = (tau_a, tau_a + 0.5 * delta, tau_a + delta)
    weights = (delta / 6.0, 4.0 * delta / 6.0, delta / 6.0)
    I = None
    for s, w in zip(nodes, weights):
        U = chars.values(s)
        term = w * (U / (1.0 - burst * U))
        I = term if I is None else I + term
    return A @ I


def _ops(A, chars, tau, h, E, burst):
    """Operators of the Strang step from time-to-go ``tau`` to ``tau - h``: the
    production exponents over ``[tau-h/2, tau]`` and ``[tau-h, tau-h/2]`` and
    ``E = exp(H^T h)`` between them."""
    hh = 0.5 * h
    X1 = _production_exponent(A, chars, tau - hh, hh, burst)
    X2 = _production_exponent(A, chars, tau - h, hh, burst)
    return X1, E, X2


def _step(G, ops):
    X1, E, X2 = ops
    G = np.exp(X1) * G
    G = E @ G
    return np.exp(X2) * G


def _integrate_numpy(HT, A, chars, G, T, h_first, h_min, tol, fixed_schedule, burst=None):
    """NumPy mirror of ``monod_core.eval_chromatin_pgf``: Strang steps with step
    doubling and Richardson extrapolation; returns ``(G, steps)``.  The two
    half steps share one ``exp(H^T h/2)``, the full step uses its square, and
    when the step doubles the previous full-step exponential is reused.
    ``chars`` provides the RNA characteristics (see :func:`_production_exponent`)."""
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
        full = _ops(A, chars, tau, h, E_half @ E_half, burst)
        half1 = _ops(A, chars, tau, hh, E_half, burst)
        half2 = _ops(A, chars, tau - hh, hh, E_half, burst)
        G1 = _step(G, full)
        G2 = _step(_step(G, half1), half2)
        diff = G2 - G1
        if adaptive:
            err = float(np.abs(diff).max())
            accept, fac = step_control(err, h, h_min, tol)
            if not accept:
                h = fac
                continue
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
