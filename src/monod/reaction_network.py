"""User-defined reaction networks for automatic CME log-PGF derivation.

Syntax
------
Reactions are separated by ``';'``.  Within a ``';'``-segment, ``'->'``
chains are sugar for sequential reactions — each ``'->'`` or ``'=>'``
becomes one elementary reaction.  Nodes can be:

  ``0``               empty set (∅)
  ``A``               one molecule of species A
  ``N x A``           N molecules of A
                        N integer  → deterministic stoichiometry
                        N name     → geometric-burst parameter
  ``A + B``           sum of terms (multi-product or multi-reactant)

Arrow types
-----------
``->``  Immediate (Markovian) reaction.
``=>``  Time-delayed reaction.  A single molecule of the (sole) reactant
        is committed at rate k; the product appears τ = 1/tauinv time
        units later.  Adds two parameters: the rate constant ``kN`` and the
        inverse delay ``tauinvM`` (both in log₁₀ scale).

        Characteristic DDE contribution for ``A => B`` (rate k, delay τ):

          ``du_A/dt += k · (u_B(t−τ) − u_A(t))``

        Numerically integrated via RK4 with a frozen-delay circular history
        buffer (method-of-steps approach).

The ``N x A`` notation for the *target* of a production reaction (source
= ``0``) controls the type of burst:

* **Named parameter** (e.g. ``B``): geometric distribution with mean B.
  ``dφ/dt += k · B·u_A / (1 − B·u_A)``
* **Integer literal** (e.g. ``2``): deterministic burst of that size.
  ``dφ/dt += k · ((1 + u_A)^N − 1)``

For all other (non-production) reactions a single reactant species is
consumed and any number of product species are created.  The characteristic
ODE/DDE contribution for reactant A is:

  ``du_A/dt += k · (∏_j (1 + u_j)^{n_j} − (1 + u_A))``

where for delayed reactions the product ``z`` is evaluated at the delayed
state ``u(t−τ)``.

Parameters
----------
All parameters are stored and passed in **log₁₀ scale**, matching the
convention used throughout monod.  The ordering of ``all_params`` is:

  1. Named burst-size parameters (e.g. ``B``), in order of first appearance.
  2. Inverse-delay parameters (``tauinv0``, ``tauinv1``, …), in order of
     first appearance (only present when ``=>`` is used).
  3. Rate constants (``k0``, ``k1``, …), one per arrow, in order.

If ``normalize_production_rate=True`` the rate constant of the first
production reaction is fixed to 1 and removed from ``all_params``.

Examples
--------
>>> net = ReactionNetwork("0 -> B x U -> S -> 0")
>>> net.species
['U', 'S']
>>> net.all_params          # B, k0 (burst rate), k1 (β), k2 (γ)
['B', 'k0', 'k1', 'k2']

>>> net2 = ReactionNetwork("0 -> B x U => S -> 0")
>>> net2.all_params         # B, tauinv0, k0 (burst rate), k1 (β), k2 (γ)
['B', 'tauinv0', 'k0', 'k1', 'k2']
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class _Term:
    """One species-quantity in a reaction node (e.g. ``2 x A`` or ``B x A``)."""
    species: str
    stoich: int = 1                  # literal count (ignored for geometric bursts)
    burst_param: Optional[str] = None  # name of a geometric-burst parameter


@dataclass
class _Reaction:
    """A single parsed elementary reaction."""
    reactants: Dict[str, int]          # species → stoich (empty dict ≡ ∅)
    products:  Dict[str, int]          # species → stoich (empty dict ≡ ∅)
    rate_name: str                     # name of the rate constant
    burst_param:   Optional[str] = None  # named geometric-burst parameter …
    burst_species: Optional[str] = None  # … and the species it applies to
    is_delayed: bool = False           # True when the arrow was '=>'
    tau_name:  Optional[str] = None   # inverse-delay param name (if is_delayed)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_term(token: str) -> _Term:
    """Parse a single species term: ``'N x A'``, ``'B x A'``, or ``'A'``."""
    token = token.strip()
    # "prefix x species"
    m = re.fullmatch(
        r'([A-Za-z_][A-Za-z0-9_]*|[0-9]+)\s+x\s+([A-Za-z_][A-Za-z0-9_]*)',
        token,
    )
    if m:
        n_str, species = m.group(1), m.group(2)
        if re.fullmatch(r'[0-9]+', n_str):
            return _Term(species=species, stoich=int(n_str), burst_param=None)
        else:
            return _Term(species=species, stoich=1, burst_param=n_str)
    # plain species name
    if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', token):
        return _Term(species=token, stoich=1, burst_param=None)
    raise ValueError(f"Cannot parse reaction term: {token!r}")


def _parse_node(node_str: str) -> List[_Term]:
    """Parse a reaction node into a list of terms; returns ``[]`` for ``'0'``/``'∅'``."""
    s = node_str.strip()
    if s in ('0', '∅'):
        return []
    return [_parse_term(part) for part in re.split(r'\s*\+\s*', s)]


def _terms_to_stoich(terms: List[_Term], is_source: bool = False) -> Dict[str, int]:
    """Aggregate terms into a stoichiometry dict.

    For *source* nodes the ``N x SPECIES`` multiplier is always treated as 1:
    the ``N`` prefix is a production modifier (burst size) and does not imply
    that N molecules are consumed.  Write ``A + A -> B`` explicitly if you want
    to consume 2 copies of A in a single reaction.
    """
    d: Dict[str, int] = {}
    for t in terms:
        stoich = 1 if is_source else t.stoich
        d[t.species] = d.get(t.species, 0) + stoich
    return d


def _parse_chain(
    segment: str,
    rate_counter: List[int],
    tau_counter: List[int],
) -> List[_Reaction]:
    """Parse one ``';'``-segment (a chain) into a list of reactions.

    Supports both ``'->'`` (immediate) and ``'=>'`` (delayed) arrows.
    """
    # Split on -> and => while keeping the separators.
    # re.split with a capturing group keeps the delimiters in the result.
    parts = re.split(r'(->|=>)', segment)
    # parts = [node0, arrow0, node1, arrow1, node2, ...]
    if len(parts) < 3:
        raise ValueError(f"Segment contains no reaction arrow: {segment!r}")

    node_strs = parts[0::2]   # every even element
    arrows    = parts[1::2]   # every odd element

    reactions: List[_Reaction] = []
    for i, arrow in enumerate(arrows):
        src_terms = _parse_node(node_strs[i])
        tgt_terms = _parse_node(node_strs[i + 1])

        rate_name = f"k{rate_counter[0]}"
        rate_counter[0] += 1

        is_delayed = (arrow == '=>')
        tau_name: Optional[str] = None
        if is_delayed:
            tau_name = f"tauinv{tau_counter[0]}"
            tau_counter[0] += 1

        reactants = _terms_to_stoich(src_terms, is_source=True)
        products  = _terms_to_stoich(tgt_terms, is_source=False)

        # Identify geometric-burst parameter (only valid for production reactions)
        burst_param: Optional[str] = None
        burst_species: Optional[str] = None
        if not reactants:
            burst_terms = [t for t in tgt_terms if t.burst_param is not None]
            if len(burst_terms) > 1:
                raise ValueError(
                    "At most one geometric-burst term is allowed per production "
                    f"reaction; found multiple in: {node_strs[i+1]!r}"
                )
            if burst_terms:
                burst_param   = burst_terms[0].burst_param
                burst_species = burst_terms[0].species

        reactions.append(
            _Reaction(
                reactants=reactants,
                products=products,
                rate_name=rate_name,
                burst_param=burst_param,
                burst_species=burst_species,
                is_delayed=is_delayed,
                tau_name=tau_name,
            )
        )

    return reactions


# ---------------------------------------------------------------------------
# ReactionNetwork
# ---------------------------------------------------------------------------

class ReactionNetwork:
    """Parse a reaction-network string and evaluate the steady-state log-PGF.

    Parameters
    ----------
    network_str : str
        Reaction network definition (see module docstring).
    normalize_production_rate : bool, optional
        If ``True``, the rate constant of the *first* production reaction
        (source = ``0``) is fixed to 1 and excluded from ``all_params``.
        This matches the convention of the built-in Bursty model where the
        burst rate is absorbed into the burst-size parameter.

    Attributes
    ----------
    species : list of str
        Species names, in order of first appearance in the network string.
    named_params : list of str
        Geometric-burst parameter names, in order of first appearance.
    delay_params : list of str
        Inverse-delay parameter names (``'tauinv0'``, …), one per ``'=>'``
        arrow, in order of first appearance.
    rate_names : list of str
        Rate-constant names (``'k0'``, ``'k1'``, …).
        Excludes the normalized production rate when applicable.
    all_params : list of str
        ``named_params + delay_params + rate_names``; the full parameter
        vector accepted by :meth:`eval_pgf` (all in log₁₀ scale).
    reactions : list of _Reaction
        Parsed reaction list (read-only; for inspection / debugging).
    """

    def __init__(
        self,
        network_str: str,
        normalize_production_rate: bool = False,
    ) -> None:
        self.network_str = network_str
        self.normalize_production_rate = normalize_production_rate
        self.reactions: List[_Reaction] = self._parse(network_str)

        # ── collect species and named params in order of first appearance ──
        seen_sp:    Dict[str, int] = {}
        seen_named: Dict[str, int] = {}
        seen_delay: Dict[str, int] = {}
        for rxn in self.reactions:
            for sp in list(rxn.reactants) + list(rxn.products):
                if sp not in seen_sp:
                    seen_sp[sp] = len(seen_sp)
            if rxn.burst_param and rxn.burst_param not in seen_named:
                seen_named[rxn.burst_param] = len(seen_named)
            if rxn.tau_name and rxn.tau_name not in seen_delay:
                seen_delay[rxn.tau_name] = len(seen_delay)

        self.species: List[str]       = list(seen_sp)
        self.named_params: List[str]  = list(seen_named)
        self.delay_params: List[str]  = list(seen_delay)

        # ── build rate-name list ────────────────────────────────────────────
        all_rate_names = [rxn.rate_name for rxn in self.reactions]
        if normalize_production_rate:
            first_prod_rate = next(
                (rxn.rate_name for rxn in self.reactions if not rxn.reactants),
                None,
            )
            self.rate_names: List[str]    = [r for r in all_rate_names if r != first_prod_rate]
            self._norm_rate: Optional[str] = first_prod_rate
        else:
            self.rate_names = all_rate_names
            self._norm_rate = None

        # Order: burst-size params → delay params → rate constants
        self.all_params: List[str] = self.named_params + self.delay_params + self.rate_names

        self._has_delays: bool = bool(self.delay_params)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(network_str: str) -> List[_Reaction]:
        rate_counter: List[int] = [0]
        tau_counter:  List[int] = [0]
        reactions: List[_Reaction] = []
        for seg in network_str.split(';'):
            seg = seg.strip()
            if seg:
                reactions.extend(_parse_chain(seg, rate_counter, tau_counter))
        return reactions

    # ------------------------------------------------------------------
    # Parameter map
    # ------------------------------------------------------------------

    def _param_map(self, p_log: np.ndarray) -> Dict[str, float]:
        """Map ``all_params`` (log₁₀) → linear-scale dict."""
        if len(p_log) != len(self.all_params):
            raise ValueError(
                f"Expected {len(self.all_params)} parameters {self.all_params}, "
                f"got {len(p_log)}"
            )
        m: Dict[str, float] = {
            name: float(10.0 ** p_log[i])
            for i, name in enumerate(self.all_params)
        }
        if self._norm_rate is not None:
            m[self._norm_rate] = 1.0
        return m

    # ------------------------------------------------------------------
    # ODE/DDE right-hand side
    # ------------------------------------------------------------------

    def _rhs(
        self,
        u: List[np.ndarray],
        params: Dict[str, float],
        sp_idx: Dict[str, int],
        u_delayed_by_tau: Optional[Dict[str, List[np.ndarray]]] = None,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Characteristic ODE/DDE RHS for the log-PGF.

        For delayed reactions (``is_delayed=True``) the product term
        ``z^products`` is evaluated at the historically delayed state
        ``u_delayed_by_tau[tau_name]`` instead of the current ``u``.
        Pass ``None`` to fall back to the non-delayed formula (used during
        the initialisation half-step where no history exists yet).

        Returns
        -------
        du : list of arrays (one per species)
            Derivative of the characteristic variables.
        dphi : array
            Source term for the log-PGF accumulation.
        """
        n = len(self.species)
        shape = u[0].shape
        dtype = u[0].dtype

        du   = [np.zeros(shape, dtype=dtype) for _ in range(n)]
        dphi =  np.zeros(shape, dtype=dtype)

        for rxn in self.reactions:
            k = params[rxn.rate_name]

            # ── production (zero-reactant) ──────────────────────────────────
            if not rxn.reactants:
                if rxn.burst_param is not None:
                    # Geometric burst: dφ/dt += k · B·u_A / (1 − B·u_A)
                    b = params[rxn.burst_param]
                    i = sp_idx[rxn.burst_species]
                    dphi += k * b * u[i] / (1.0 - b * u[i])
                else:
                    # Deterministic stoichiometry: dφ/dt += k · (z^products − 1)
                    z_prod = _product_z(u, rxn.products, sp_idx, shape, dtype)
                    dphi  += k * (z_prod - 1.0)

            # ── single-reactant (first-order propensity) ────────────────────
            elif len(rxn.reactants) == 1:
                (src_sp, src_n), = rxn.reactants.items()
                if src_n != 1:
                    raise NotImplementedError(
                        f"Reactant stoichiometry > 1 not yet supported: {rxn}"
                    )
                i = sp_idx[src_sp]
                # For delayed reactions use the historical u for the product term.
                # If no history is available yet (u_delayed_by_tau is None or the
                # key is missing), fall back to current u — equivalent to assuming
                # constant history equal to the initial condition.
                if (
                    rxn.is_delayed
                    and u_delayed_by_tau is not None
                    and rxn.tau_name in u_delayed_by_tau
                ):
                    u_for_prod = u_delayed_by_tau[rxn.tau_name]
                else:
                    u_for_prod = u
                z_prod  = _product_z(u_for_prod, rxn.products, sp_idx, shape, dtype)
                du[i]  += k * (z_prod - (1.0 + u[i]))

            # ── bimolecular (two distinct reactants) ────────────────────────
            else:
                raise NotImplementedError(
                    "Bimolecular reactions (two distinct reactants) are not yet "
                    "supported in the characteristic-ODE framework.  "
                    f"Offending reaction: {rxn}"
                )

        return du, dphi

    # ------------------------------------------------------------------
    # RK4 step
    # ------------------------------------------------------------------

    def _rk4_step(
        self,
        u: List[np.ndarray],
        phi: np.ndarray,
        params: Dict[str, float],
        sp_idx: Dict[str, int],
        dt: float,
        u_delayed_by_tau: Optional[Dict[str, List[np.ndarray]]] = None,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """One RK4 update of the full (u, φ) state.

        ``u_delayed_by_tau`` is frozen across all four stages (method-of-steps
        convention for DDE-RK4 integration).
        """
        n = len(u)
        du1, dp1 = self._rhs(u, params, sp_idx, u_delayed_by_tau)
        u2  = [u[i] + 0.5 * dt * du1[i] for i in range(n)]
        du2, dp2 = self._rhs(u2, params, sp_idx, u_delayed_by_tau)
        u3  = [u[i] + 0.5 * dt * du2[i] for i in range(n)]
        du3, dp3 = self._rhs(u3, params, sp_idx, u_delayed_by_tau)
        u4  = [u[i] + dt * du3[i] for i in range(n)]
        du4, dp4 = self._rhs(u4, params, sp_idx, u_delayed_by_tau)

        u_new   = [u[i] + (dt / 6.0) * (du1[i] + 2*du2[i] + 2*du3[i] + du4[i])
                   for i in range(n)]
        phi_new = phi + (dt / 6.0) * (dp1 + 2*dp2 + 2*dp3 + dp4)
        return u_new, phi_new

    # ------------------------------------------------------------------
    # Delayed-state helpers
    # ------------------------------------------------------------------

    def _build_history(
        self,
        u_init: List[np.ndarray],
        params: Dict[str, float],
        dt: float,
    ) -> Tuple[Deque, Dict[str, int]]:
        """Allocate a circular history deque pre-filled with the initial state.

        Returns
        -------
        u_hist : deque of List[np.ndarray]
            Circular buffer of length ``max_delay_steps + 1``.
            All slots initialised to copies of ``u_init``.
        tau_d_steps : dict  tau_name → int
            Number of integration steps corresponding to each delay τ.
        """
        tau_d_steps: Dict[str, int] = {}
        for rxn in self.reactions:
            if rxn.is_delayed and rxn.tau_name not in tau_d_steps:
                tau = params[rxn.tau_name]  # linear scale, τ = 1/tauinv
                d = max(1, int(round(tau / dt)))
                tau_d_steps[rxn.tau_name] = d

        max_d = max(tau_d_steps.values()) if tau_d_steps else 1
        u_hist: Deque = deque(
            [[arr.copy() for arr in u_init] for _ in range(max_d + 1)],
            maxlen=max_d + 1,
        )
        return u_hist, tau_d_steps

    @staticmethod
    def _delayed_state(
        u_hist: Deque,
        tau_d_steps: Dict[str, int],
    ) -> Dict[str, List[np.ndarray]]:
        """Extract the delayed u state for each tauinv parameter.

        ``u_hist[-d]`` gives the state from ``d`` steps ago (with ``d >= 1``).
        If the buffer is smaller than ``d`` (early steps), the oldest available
        state (initial condition) is used.
        """
        result: Dict[str, List[np.ndarray]] = {}
        hist_len = len(u_hist)
        for tau_name, d in tau_d_steps.items():
            idx = -min(d, hist_len)  # negative index into deque
            result[tau_name] = u_hist[idx]
        return result

    # ------------------------------------------------------------------
    # Public PGF evaluator
    # ------------------------------------------------------------------

    def _dopri5_step(
        self,
        u: List[np.ndarray],
        phi: np.ndarray,
        params: Dict[str, float],
        sp_idx: Dict[str, int],
        h: float,
        u_delayed_by_tau: Optional[Dict[str, List[np.ndarray]]] = None,
    ) -> Tuple[List[np.ndarray], np.ndarray, float]:
        """One Dormand-Prince RK45 step.

        Returns ``(u_new, phi_new, err)`` where *err* is the scalar error
        norm (step is accepted when ``err <= 1.0``).  Both the 5th-order
        solution and the embedded error estimate use the same 7 RHS
        evaluations (6 stages + 1 FSAL evaluation of the new state).
        """
        # Dormand-Prince Butcher tableau (DOPRI5)
        a21 = 1 / 5
        a31, a32 = 3 / 40, 9 / 40
        a41, a42, a43 = 44 / 45, -56 / 15, 32 / 9
        a51, a52, a53, a54 = 19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729
        a61, a62, a63, a64, a65 = (
            9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656
        )
        b1, b3, b4, b5, b6 = 35 / 384, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84
        # Error coefficients e = b5th − b4th
        e1 = 71 / 57600
        e3 = -71 / 16695
        e4 = 71 / 1920
        e5 = -17253 / 339200
        e6 = 22 / 525
        e7 = -1 / 40

        n = len(u)
        d = u_delayed_by_tau  # shorthand

        du1, dp1 = self._rhs(u, params, sp_idx, d)

        u2 = [u[i] + h * a21 * du1[i] for i in range(n)]
        du2, dp2 = self._rhs(u2, params, sp_idx, d)

        u3 = [u[i] + h * (a31 * du1[i] + a32 * du2[i]) for i in range(n)]
        du3, dp3 = self._rhs(u3, params, sp_idx, d)

        u4 = [u[i] + h * (a41 * du1[i] + a42 * du2[i] + a43 * du3[i]) for i in range(n)]
        du4, dp4 = self._rhs(u4, params, sp_idx, d)

        u5 = [u[i] + h * (a51 * du1[i] + a52 * du2[i] + a53 * du3[i] + a54 * du4[i]) for i in range(n)]
        du5, dp5 = self._rhs(u5, params, sp_idx, d)

        u6 = [
            u[i] + h * (a61 * du1[i] + a62 * du2[i] + a63 * du3[i] + a64 * du4[i] + a65 * du5[i])
            for i in range(n)
        ]
        du6, dp6 = self._rhs(u6, params, sp_idx, d)

        # 5th-order solution
        u_new = [u[i] + h * (b1 * du1[i] + b3 * du3[i] + b4 * du4[i] + b5 * du5[i] + b6 * du6[i]) for i in range(n)]
        phi_new = phi + h * (b1 * dp1 + b3 * dp3 + b4 * dp4 + b5 * dp5 + b6 * dp6)

        # 7th stage at u_new — used only for error estimate (FSAL)
        du7, _ = self._rhs(u_new, params, sp_idx, d)

        # Componentwise error norm  max_i max_j |e_{ij}| / sc_{ij}
        atol, rtol = 1e-6, 1e-4
        err = 0.0
        for i in range(n):
            e_i = np.abs(
                h * (e1 * du1[i] + e3 * du3[i] + e4 * du4[i]
                     + e5 * du5[i] + e6 * du6[i] + e7 * du7[i])
            )
            sc = atol + rtol * np.maximum(np.abs(u[i]), np.abs(u_new[i]))
            local = float(np.max(e_i / sc))
            if local > err:
                err = local

        return u_new, phi_new, err

    def eval_pgf(
        self,
        p_log: np.ndarray,
        g: List[np.ndarray],
        T: float,
        n_steps: int,
    ) -> np.ndarray:
        """Evaluate the steady-state log-PGF φ at complex grid points.

        For non-delayed networks uses an adaptive Dormand-Prince RK45
        scheme that automatically tightens the step size where needed and
        expands it in smooth regions.  The initial step size is
        ``T / n_steps`` (derived from the parameter fudge factors), so
        the caller's ``T`` / ``n_steps`` still control the starting scale.

        For delayed networks (``=>``) the fixed-step RK4 integrator is
        used instead, because the history buffer requires a uniform step
        size.

        Parameters
        ----------
        p_log : np.ndarray, shape (n_params,)
            Log₁₀ parameters in the order ``all_params``.
        g : list of np.ndarray
            Complex initial conditions, one per species, all the same shape.
        T : float
            Sets the initial step size as ``h0 = T / n_steps``.
        n_steps : int
            Sets the initial step size as ``h0 = T / n_steps``.

        Returns
        -------
        phi : np.ndarray (complex64)
            Log-PGF values at every grid point.
        """
        if len(g) != len(self.species):
            raise ValueError(
                f"Expected {len(self.species)} grid arrays (species {self.species}), "
                f"got {len(g)}"
            )

        # Delayed networks require uniform step-size for the history buffer.
        if self._has_delays:
            return self._eval_pgf_rk4_fixed(p_log, g, T, n_steps)

        params = self._param_map(p_log)
        sp_idx = {s: i for i, s in enumerate(self.species)}

        u   = [np.array(g[i], dtype=np.complex64) for i in range(len(self.species))]
        phi = np.zeros_like(u[0])

        h     = T / n_steps if n_steps > 0 else 1e-3
        h_max = 10.0 * h
        h_min = h * 1e-8

        # Leading trapezoidal half-step with initial h.
        _, dphi0 = self._rhs(u, params, sp_idx)
        phi += dphi0 * (h / 2.0)

        # Adaptive integration until |u[0]| < 1e-3.
        max_steps = 50 * n_steps + 50_000
        step_count = 0
        while np.max(np.abs(u[0])) > 1e-3 and step_count < max_steps:
            u_new, phi_new, err = self._dopri5_step(u, phi, params, sp_idx, h)
            # PI-controller step-size update (5th-order exponent 0.2).
            factor = float(np.clip(0.9 * (1.0 / max(err, 1e-10)) ** 0.2, 0.1, 5.0))
            if err <= 1.0:
                u, phi = u_new, phi_new
                step_count += 1
            h = float(np.clip(h * factor, h_min, h_max))

        # Trailing trapezoidal half-step.
        _, dphi_f = self._rhs(u, params, sp_idx)
        phi += dphi_f * (h / 2.0)

        return phi

    def _eval_pgf_rk4_fixed(
        self,
        p_log: np.ndarray,
        g: List[np.ndarray],
        T: float,
        n_steps: int,
    ) -> np.ndarray:
        """Fixed-step RK4 integrator (used for delayed networks).

        Maintains a circular history buffer for the ``=>`` DDE reactions.
        The step size is fixed throughout so the buffer index arithmetic
        stays exact.
        """
        params = self._param_map(p_log)
        sp_idx = {s: i for i, s in enumerate(self.species)}

        u   = [np.array(g[i], dtype=np.complex64) for i in range(len(self.species))]
        phi = np.zeros_like(u[0])
        dt  = T / n_steps if n_steps > 0 else T

        u_hist, tau_d_steps = self._build_history(u, params, dt)

        _, dphi0 = self._rhs(u, params, sp_idx)
        phi += dphi0 * (dt / 2.0)

        def _step(u, phi):
            u_del = self._delayed_state(u_hist, tau_d_steps)
            u_new, phi_new = self._rk4_step(u, phi, params, sp_idx, dt, u_del)
            u_hist.append([arr.copy() for arr in u_new])
            return u_new, phi_new

        for _ in range(n_steps):
            u, phi = _step(u, phi)

        max_while_steps = 10 * n_steps + 10_000
        while_steps = 0
        while np.max(np.abs(u[0])) > 1e-3 and while_steps < max_while_steps:
            u, phi = _step(u, phi)
            while_steps += 1

        _, dphi_f = self._rhs(u, params, sp_idx)
        phi += dphi_f * (dt / 2.0)

        return phi

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_log_name_str(self) -> List[str]:
        """TeX-formatted log₁₀ parameter labels (for plot axes)."""
        return [rf"$\log_{{10}}\,{p}$" for p in self.all_params]

    def __repr__(self) -> str:
        lines = [f"ReactionNetwork({self.network_str!r})"]
        lines.append(f"  species    : {self.species}")
        lines.append(f"  all_params : {self.all_params}")
        lines.append("  reactions  :")
        for rxn in self.reactions:
            src = (
                " + ".join(f"{n}×{s}" if n > 1 else s for s, n in rxn.reactants.items())
                or "∅"
            )
            tgt = (
                " + ".join(f"{n}×{s}" if n > 1 else s for s, n in rxn.products.items())
                or "∅"
            )
            burst = f"  [geometric {rxn.burst_param}]" if rxn.burst_param else ""
            if rxn.is_delayed:
                arrow = f"⇒({rxn.rate_name},{rxn.tau_name})"
            else:
                arrow = f"→({rxn.rate_name})"
            lines.append(f"    {src} {arrow} {tgt}{burst}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level helper (used by the ODE RHS)
# ---------------------------------------------------------------------------

def _product_z(
    u: List[np.ndarray],
    products: Dict[str, int],
    sp_idx: Dict[str, int],
    shape: tuple,
    dtype,
) -> np.ndarray:
    """Compute z^products = ∏_j (1 + u_j)^{n_j}."""
    z = np.ones(shape, dtype=dtype)
    for sp, n in products.items():
        i = sp_idx[sp]
        z = z * (1.0 + u[i]) ** n
    return z
