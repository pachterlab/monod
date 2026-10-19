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
  ``A + B``           sum of terms (products only; a reactant node must be
                      ``0`` or a single species, see below)

Arrow types
-----------
``->``  Immediate (Markovian) reaction.
``=>``  Time-delayed reaction.  A single molecule of the (sole) reactant
        is committed at rate k; the product appears τ = 1/tauinv time
        units later.  Adds two parameters: the rate constant ``kN`` and the
        inverse delay ``tauinvM`` (both in log₁₀ scale).

        Characteristic DDE contribution for ``A => B`` (rate k, delay τ):

          ``du_A/dt += k · (u_B(t−τ) − u_A(t))``

        Numerically integrated by fixed-step RK4 (method of steps) with the
        delayed state read from a history buffer by linear interpolation
        between stored steps, so the result depends smoothly on τ.  Before
        t = τ the delayed state is zero: a product that would only appear
        after the observation time contributes nothing (z = 1).

The ``N x A`` notation for the *target* of a production reaction (source
= ``0``) controls the type of burst:

* **Named parameter** (e.g. ``B``): geometric distribution with mean B.
  ``dφ/dt += k · B·u_A / (1 − B·u_A)``
* **Integer literal** (e.g. ``2``): deterministic burst of that size.
  ``dφ/dt += k · ((1 + u_A)^N − 1)``

For all other (non-production) reactions exactly one reactant molecule is
consumed and any number of product species are created.  Reactant
stoichiometry > 1 (``2 x A``, ``A + A``) and bimolecular reactions
(``A + B``) are not supported and are rejected when the network is parsed.  The characteristic
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
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


class ReactionNetworkWarning(UserWarning):
    """A network is parseable but cannot be fitted as written.

    Raised by :meth:`ReactionNetwork.validate` for structural problems:
    parameters the steady state cannot distinguish, species that are never
    produced or never removed. See :meth:`ReactionNetwork.structural_issues`.
    """


class IdentifiabilityWarning(UserWarning):
    """A fitted or evaluated model has a direction the data cannot constrain.

    Raised by :meth:`monod.cme_toolbox.CMEModel.check_identifiability` and by
    :meth:`monod.inference.SearchResults.compute_sigma`.
    """


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


def _terms_to_stoich(
    terms: List[_Term], is_source: bool = False, is_chain_head: bool = True
) -> Dict[str, int]:
    """Aggregate terms into a stoichiometry dict.

    Source (reactant) nodes must be empty (``0``) or a single species with
    stoichiometry 1: the characteristic-ODE solver only supports zero- and
    first-order propensities.  ``2 x A -> B``, ``A + A -> B`` and
    ``A + B -> C`` are rejected here, at parse time, so that every backend
    (Python and Rust) sees the same network.

    Inside a chain (``0 -> B x U -> S``) an interior node is both the product
    of one arrow and the reactant of the next; its ``N x`` prefix belongs to
    the production step and is ignored on the reactant side
    (``is_chain_head=False``).  On the head node of a chain a prefix has no
    production to attach to and is an error.
    """
    d: Dict[str, int] = {}
    for t in terms:
        if is_source and is_chain_head and t.burst_param is not None:
            raise ValueError(
                f"A geometric-burst prefix ({t.burst_param!r}) is only allowed on "
                f"products, not on the reactant {t.species!r}."
            )
        if is_source and is_chain_head and t.stoich != 1:
            raise ValueError(
                f"Reactant stoichiometry > 1 is not supported ({t.stoich} x {t.species}); "
                "only zero- and first-order reactions are available."
            )
        stoich = 1 if is_source else t.stoich
        d[t.species] = d.get(t.species, 0) + stoich
    if is_source and (len(d) > 1 or any(v != 1 for v in d.values())):
        raise ValueError(
            "Only zero- and first-order reactions are supported: a reactant node must "
            f"be '0' or a single species, got {' + '.join(f'{v} x {k}' if v > 1 else k for k, v in d.items())!r}."
        )
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

        reactants = _terms_to_stoich(src_terms, is_source=True, is_chain_head=(i == 0))
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

        # Warn about structural problems now, except the time-unit degeneracy
        # of a network with every rate free: that only matters for fitting and
        # is reported by InferenceParameters (see ``validate(for_inference=True)``).
        self.validate(for_inference=False, stacklevel=3)

    # ------------------------------------------------------------------
    # Structural checks
    # ------------------------------------------------------------------

    def _reaction_str(self, rxn: _Reaction) -> str:
        src = " + ".join(f"{n} x {s}" if n > 1 else s for s, n in rxn.reactants.items()) or "0"
        tgt = " + ".join(f"{n} x {s}" if n > 1 else s for s, n in rxn.products.items()) or "0"
        if rxn.burst_param:
            tgt = f"{rxn.burst_param} x {rxn.burst_species}"
        return f"{src} {'=>' if rxn.is_delayed else '->'} {tgt}"

    def structural_issues(self) -> List[Dict[str, object]]:
        """Problems that can be read off the network without evaluating it.

        Each issue is a dict with ``code``, ``params`` (the parameter names
        involved, possibly empty) and a human-readable ``message``. Codes:

        ``timescale``
            Every rate constant (and inverse delay) is a free parameter. The
            steady state is unchanged when all of them are scaled together, so
            the fit has an exactly flat direction. Pass
            ``normalize_production_rate=True`` or fix a rate by other means.
        ``delay_no_product``
            A ``=>`` reaction with no product: the delay changes nothing
            observable, so its ``tauinv`` parameter has no effect at all.
        ``duplicate_reaction``
            Two reactions with the same reactant and products. The steady state
            depends only on their combined effect, so their parameters are not
            separately identifiable.
        ``no_source``
            A species that can never be produced. Its steady state is
            identically zero and the rates of reactions consuming it are
            unconstrained.
        ``no_sink``
            A species that no reaction removes (as a reactant that is not also a
            product). Its count grows without bound and no steady state exists.
        ``autocatalytic``
            A reaction that turns one molecule of a species into two or more of
            the same species. The count grows without bound.

        Returns
        -------
        list of dict
            Empty when the network passes every check.
        """
        issues: List[Dict[str, object]] = []
        rxns = self.reactions

        # -- time units -----------------------------------------------------
        free_rates = list(self.rate_names) + list(self.delay_params)
        if self._norm_rate is None and self.rate_names:
            issues.append(dict(
                code="timescale", params=free_rates,
                message=(
                    "every rate constant is free, so the steady state only depends on their "
                    f"ratios: scaling {', '.join(free_rates)} together leaves the distribution "
                    "unchanged and the fit has an exactly flat direction. Pass "
                    "normalize_production_rate=True (fixes the first production rate to 1) "
                    "or fix one rate yourself."
                ),
            ))

        # -- delayed reaction with nothing to delay ------------------------------
        for rxn in rxns:
            if rxn.is_delayed and not rxn.products:
                issues.append(dict(
                    code="delay_no_product", params=[rxn.tau_name],
                    message=(
                        f"'{self._reaction_str(rxn)}' is delayed but has no product, so the "
                        f"delay is unobservable and {rxn.tau_name} has no effect on the "
                        "steady state. Write it as an immediate reaction (->)."
                    ),
                ))

        # -- duplicate reactions ---------------------------------------------
        seen: Dict[tuple, int] = {}
        for i, rxn in enumerate(rxns):
            key = (
                tuple(sorted(rxn.reactants.items())),
                tuple(sorted(rxn.products.items())),
                rxn.is_delayed,
                rxn.burst_species,
            )
            if key in seen:
                j = seen[key]
                other = rxns[j]
                names = [n for n in (other.rate_name, rxn.rate_name, other.burst_param,
                                     rxn.burst_param, other.tau_name, rxn.tau_name) if n]
                issues.append(dict(
                    code="duplicate_reaction", params=names,
                    message=(
                        f"reactions {j} ('{self._reaction_str(other)}') and {i} "
                        f"('{self._reaction_str(rxn)}') have the same reactant and products; the "
                        f"steady state only depends on their combined effect, so {', '.join(names)} "
                        "are not separately identifiable. Merge them into one reaction."
                    ),
                ))
            else:
                seen[key] = i

        # -- reachability from the empty set --------------------------------
        reachable = set()
        for rxn in rxns:
            if not rxn.reactants:
                reachable.update(rxn.products)
        changed = True
        while changed:
            changed = False
            for rxn in rxns:
                if rxn.reactants and set(rxn.reactants) <= reachable:
                    new = set(rxn.products) - reachable
                    if new:
                        reachable |= new
                        changed = True
        for sp in self.species:
            if sp not in reachable:
                consuming = [r.rate_name for r in rxns if sp in r.reactants]
                issues.append(dict(
                    code="no_source", params=consuming,
                    message=(
                        f"species {sp} is never produced, so its steady state is identically "
                        f"zero and {', '.join(consuming) or 'the reactions consuming it'} "
                        "cannot be inferred."
                    ),
                ))

        # -- removal routes ------------------------------------------------------
        for sp in self.species:
            removed = any(sp in r.reactants and sp not in r.products for r in rxns)
            if not removed:
                issues.append(dict(
                    code="no_sink", params=[],
                    message=(
                        f"species {sp} is never removed (no reaction has it as a reactant "
                        "without also producing it), so its count grows without bound and "
                        "the network has no steady state."
                    ),
                ))
        for rxn in rxns:
            for sp in rxn.reactants:
                if rxn.products.get(sp, 0) >= 2:
                    issues.append(dict(
                        code="autocatalytic", params=[rxn.rate_name],
                        message=(
                            f"'{self._reaction_str(rxn)}' turns one {sp} into "
                            f"{rxn.products[sp]}, so the count grows without bound and the "
                            "network has no steady state."
                        ),
                    ))
        return issues

    def validate(self, for_inference: bool = True, warn: bool = True, stacklevel: int = 2
                 ) -> List[Dict[str, object]]:
        """Run :meth:`structural_issues` and warn about what it finds.

        Parameters
        ----------
        for_inference : bool, optional
            Include the ``timescale`` issue. It is harmless when a network is
            only evaluated at given parameters, so the constructor leaves it
            out and the inference entry points include it.
        warn : bool, optional
            Emit one :class:`ReactionNetworkWarning` per issue.
        stacklevel : int, optional
            Passed to :func:`warnings.warn` so the warning points at the caller.

        Returns
        -------
        list of dict
            The issues found (see :meth:`structural_issues`).
        """
        issues = self.structural_issues()
        if not for_inference:
            issues = [it for it in issues if it["code"] != "timescale"]
        if warn:
            for it in issues:
                warnings.warn(
                    f"ReactionNetwork({self.network_str!r}): {it['message']}",
                    ReactionNetworkWarning, stacklevel=stacklevel,
                )
        return issues

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
    ) -> Tuple[Deque, Dict[str, Tuple[int, float]]]:
        """Allocate the history ring and the interpolation weights per delay.

        For each delay τ = 1/tauinv, ``τ/dt = j + α`` with integer ``j`` and
        ``0 <= α < 1``; the delayed state is ``(1 − α)·u_{n−j} + α·u_{n−j−1}``,
        with ``u_m = 0`` for ``m < 0``: a product that would appear after the
        observation time contributes nothing.  Mirrors ``monod_core``'s
        ``DelaySpec``.

        Returns
        -------
        u_hist : deque of List[np.ndarray]
            Ring of the last ``max_j + 2`` states, newest last; zeros for the
            pre-history and ``u_init`` in the last slot.
        specs : dict  tau_name → (j, α)
        """
        specs: Dict[str, Tuple[int, float]] = {}
        for rxn in self.reactions:
            if rxn.is_delayed and rxn.tau_name not in specs:
                s = (1.0 / params[rxn.tau_name]) / dt
                j = int(np.floor(s))
                specs[rxn.tau_name] = (j, float(s - j))

        max_j = max((j for j, _ in specs.values()), default=0)
        u_hist: Deque = deque(
            [[np.zeros_like(arr) for arr in u_init] for _ in range(max_j + 1)]
            + [[arr.copy() for arr in u_init]],
            maxlen=max_j + 2,
        )
        return u_hist, specs

    @staticmethod
    def _delayed_state(
        u_hist: Deque,
        specs: Dict[str, Tuple[int, float]],
    ) -> Dict[str, List[np.ndarray]]:
        """Interpolated delayed state for each delay parameter.

        ``u_hist[-1]`` is the current state ``u_n`` and ``u_hist[-1-m]`` is
        ``u_{n-m}``.
        """
        result: Dict[str, List[np.ndarray]] = {}
        for tau_name, (j, alpha) in specs.items():
            a = u_hist[-1 - j]
            b = u_hist[-2 - j]
            result[tau_name] = [x * (1.0 - alpha) + y * alpha for x, y in zip(a, b)]
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
        size; the delayed state is interpolated between stored steps.

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

        # phi starts at 0 and is integrated inside each RK stage; no extra
        # end-point corrections are needed (they would double count).

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

        return phi

    def _eval_pgf_rk4_fixed(
        self,
        p_log: np.ndarray,
        g: List[np.ndarray],
        T: float,
        n_steps: int,
    ) -> np.ndarray:
        """Fixed-step RK4 integrator (used for delayed networks).

        Maintains a history ring for the ``=>`` DDE reactions; the delayed
        state is linearly interpolated between stored steps (see
        ``_build_history``).  The step size is fixed throughout so the ring
        index arithmetic stays exact.  Mirrors ``monod_core``'s
        ``custom_pgf_delayed_parallel``.
        """
        params = self._param_map(p_log)
        sp_idx = {s: i for i, s in enumerate(self.species)}

        u   = [np.array(g[i], dtype=np.complex128) for i in range(len(self.species))]
        phi = np.zeros_like(u[0])
        dt  = T / n_steps if n_steps > 0 else T

        u_hist, specs = self._build_history(u, params, dt)

        def _step(u, phi):
            u_del = self._delayed_state(u_hist, specs)
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

        return phi

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def rust_topology(self) -> Dict[str, list]:
        """Flat encoding of the network for the ``monod_core`` kernels.

        Returns a dict with ``rxn_kinds`` (0 geometric burst, 1 deterministic
        production, 2 first-order, 3 delayed first-order), ``rxn_rate_idxs``,
        ``rxn_extra1``, ``rxn_extra2``, the CSR product arrays ``prod_sp``,
        ``prod_st``, ``prod_off``, and ``param_names``: the parameter order the
        indices refer to (``all_params`` plus the normalised rate, if any).
        """
        param_names = list(self.all_params) + ([self._norm_rate] if self._norm_rate else [])
        name_to_idx = {name: i for i, name in enumerate(param_names)}
        sp_idx = {s: i for i, s in enumerate(self.species)}
        kinds, rate_idxs, extra1, extra2 = [], [], [], []
        prod_sp, prod_st, prod_off = [], [], [0]
        for rxn in self.reactions:
            rate_idx = name_to_idx[rxn.rate_name]
            if not rxn.reactants and rxn.burst_param is not None:
                kinds.append(0); extra1.append(name_to_idx[rxn.burst_param]); extra2.append(sp_idx[rxn.burst_species])
            elif not rxn.reactants:
                kinds.append(1); extra1.append(-1); extra2.append(-1)
            else:
                # The parser guarantees a single first-order reactant; guard so a
                # future parser change cannot silently mis-encode a reaction.
                if len(rxn.reactants) != 1 or next(iter(rxn.reactants.values())) != 1:
                    raise ValueError(f"Rust custom-network path requires first-order reactions; got {rxn}")
                src = next(iter(rxn.reactants))
                if rxn.is_delayed:
                    kinds.append(3); extra1.append(sp_idx[src]); extra2.append(name_to_idx[rxn.tau_name])
                else:
                    kinds.append(2); extra1.append(sp_idx[src]); extra2.append(-1)
            rate_idxs.append(rate_idx)
            for sp, st in rxn.products.items():
                prod_sp.append(sp_idx[sp]); prod_st.append(int(st))
            prod_off.append(len(prod_sp))
        return dict(rxn_kinds=kinds, rxn_rate_idxs=rate_idxs, rxn_extra1=extra1, rxn_extra2=extra2,
                    prod_sp=prod_sp, prod_st=prod_st, prod_off=prod_off, param_names=param_names)

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
