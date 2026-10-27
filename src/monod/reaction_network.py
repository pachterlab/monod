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

        Integrated by the exponential integrator of ``eval_pgf`` (method of
        steps) with the delayed state read from the accepted history by
        linear interpolation, so the result depends smoothly on τ.  Before
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

With ``burst_mean_rate=True`` the rate constant of a geometric-burst
production reaction is the **mean production rate** (bursts of mean size B
arrive at frequency k/B) instead of the burst frequency:

  ``dφ/dt += (k/B) · B·u_A / (1 − B·u_A)``

The two forms describe the same networks; the mean-rate form makes the
single-molecule network ``0 -> U`` the small-B limit of ``0 -> B x U`` at the
same ``k``, so a constitutive model is nested in its bursty counterpart at
the burst-size floor and the two can be compared with
:func:`monod.analysis.compare_nested_models`.  In the frequency form the
small-B limit is no production at all.

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

import bisect
import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Exponential-integrator controls (shared with monod_core::custom_pgf_sched).
_CUSTOM_ODE_TOL = 1e-6        # per-step error target on the generating-function scale
_CUSTOM_ODE_C0 = 0.05         # first step as a fraction of the fastest time scale
_CUSTOM_ODE_U_END = 1e-8      # stop once exp(Re φ)·|u| is below this everywhere (for a delay's length, if any)
_CUSTOM_ODE_MAX_STEPS = 20_000
_CUSTOM_ODE_H_MAX = 1e4       # step cap in units of the slowest time scale


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
    burst_mean_rate : bool, optional
        If ``True``, the rate constant of every geometric-burst production
        reaction is the mean production rate rather than the burst frequency
        (see the module docstring); the parameter list is unchanged.

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
        burst_mean_rate: bool = False,
    ) -> None:
        self.network_str = network_str
        self.normalize_production_rate = normalize_production_rate
        self.burst_mean_rate = bool(burst_mean_rate)
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
        if self.burst_mean_rate:
            # mean-rate bursts: the kernels take the burst frequency k/B
            for rxn in self.reactions:
                if not rxn.reactants and rxn.burst_param is not None:
                    m[rxn.rate_name] = m[rxn.rate_name] / m[rxn.burst_param]
        return m

    # ------------------------------------------------------------------
    # Characteristic equations: linear part, remainder and source
    # ------------------------------------------------------------------

    def _linear_system(
        self, params: Dict[str, float], sp_idx: Dict[str, int],
    ) -> Tuple[np.ndarray, List[bool], List[float], float, float]:
        """Split the characteristic equations into ``du/dt = A u + N(u)``.

        Returns ``(A, nonlin, taus, lam_fast, lam_slow)``: the real first-order
        rate matrix, one flag per reaction marking those that contribute to the
        remainder ``N`` (several products from one reaction, as in
        ``S -> S + P``, or a delayed product term), the delay lengths in
        ``delay_params`` order, and the fastest and slowest first-order time
        scales.  Mirrors ``monod_core::custom_system``.
        """
        n = len(self.species)
        A = np.zeros((n, n))
        nonlin: List[bool] = []
        rates: List[float] = []
        for rxn in self.reactions:
            if not rxn.reactants:
                nonlin.append(False)
                continue
            if len(rxn.reactants) > 1:
                raise NotImplementedError(
                    "Bimolecular reactions (two distinct reactants) are not yet "
                    "supported in the characteristic-ODE framework.  "
                    f"Offending reaction: {rxn}"
                )
            (src_sp, src_n), = rxn.reactants.items()
            if src_n != 1:
                raise NotImplementedError(
                    f"Reactant stoichiometry > 1 not yet supported: {rxn}"
                )
            k = params[rxn.rate_name]
            i = sp_idx[src_sp]
            A[i, i] -= k
            rates.append(k)
            if rxn.is_delayed:
                nonlin.append(True)
            else:
                for sp, st in rxn.products.items():
                    A[i, sp_idx[sp]] += k * st
                prods = list(rxn.products.values())
                nonlin.append(not (len(prods) == 0 or (len(prods) == 1 and prods[0] == 1)))
        taus = [1.0 / params[name] for name in self.delay_params]
        if not rates:
            rates = [params[r.rate_name] for r in self.reactions]
        rates = [k for k in rates if k > 0]
        diag = [-A[i, i] for i in range(n) if -A[i, i] > 0]
        lam_fast = max(max(rates, default=0.0), 1e-300)
        lam_slow = min(diag) if diag else (min(rates) if rates else lam_fast)
        return A, nonlin, taus, lam_fast, lam_slow

    def _nonlin(
        self,
        u: np.ndarray,
        udel: List[np.ndarray],
        params: Dict[str, float],
        sp_idx: Dict[str, int],
        nonlin: List[bool],
    ) -> np.ndarray:
        """Remainder ``N(u)``: ``k (Z(u) − 1 − Σ s_j u_j)`` for multi-product
        first-order reactions and ``k (Z(u_del) − 1)`` for delayed ones.

        ``u`` has shape ``(n_species, n_grid)``; ``udel[d]`` is the delayed
        state for the ``d``-th entry of ``delay_params``.
        """
        shape = u[0].shape
        out = np.zeros(u.shape, dtype=np.complex128)
        for rxn, flag in zip(self.reactions, nonlin):
            if not flag:
                continue
            k = params[rxn.rate_name]
            i = sp_idx[next(iter(rxn.reactants))]
            if rxn.is_delayed:
                ud = udel[self.delay_params.index(rxn.tau_name)]
                z = _product_z(ud, rxn.products, sp_idx, shape, np.complex128)
                out[i] += k * (z - 1.0)
            else:
                z = _product_z(u, rxn.products, sp_idx, shape, np.complex128)
                lin = np.zeros(shape, dtype=np.complex128)
                for sp, st in rxn.products.items():
                    lin += u[sp_idx[sp]] * float(st)
                out[i] += k * (z - 1.0 - lin)
        return out

    def _source(
        self, u: np.ndarray, params: Dict[str, float], sp_idx: Dict[str, int],
    ) -> np.ndarray:
        """Source ``S(u) = dφ/dt`` from the production reactions."""
        shape = u[0].shape
        s = np.zeros(shape, dtype=np.complex128)
        for rxn in self.reactions:
            if rxn.reactants:
                continue
            k = params[rxn.rate_name]
            if rxn.burst_param is not None:
                b = params[rxn.burst_param]
                ui = u[sp_idx[rxn.burst_species]]
                s += ui * (k * b) / (1.0 - ui * b)
            else:
                z = _product_z(u, rxn.products, sp_idx, shape, np.complex128)
                s += k * (z - 1.0)
        return s

    # ------------------------------------------------------------------
    # Exponential integrator
    # ------------------------------------------------------------------

    @staticmethod
    def _expm(m: np.ndarray) -> np.ndarray:
        """exp(m) by scaling and squaring around a degree-18 Taylor polynomial
        (the scaled 1-norm is at most 1/2).  Mirrors ``SmallMat::expm``."""
        nrm = float(np.abs(m).sum(axis=0).max())
        s = int(np.ceil(np.log2(nrm / 0.5))) if nrm > 0.5 else 0
        x = m * (0.5 ** s)
        n = m.shape[0]
        eye = np.eye(n)
        r = eye.copy()
        for k in range(18, 0, -1):
            r = eye + (x @ r) * (1.0 / k)
        for _ in range(s):
            r = r @ r
        return r

    @classmethod
    def _step_ops(cls, A: np.ndarray, h: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(E, P1, P2) = (exp(A h), h φ₁(A h), h φ₂(A h))`` read off the
        exponential of the block matrix ``[[A h, I, 0], [0, 0, I], [0, 0, 0]]``."""
        n = A.shape[0]
        m = np.zeros((3 * n, 3 * n))
        m[:n, :n] = A * h
        m[:n, n:2 * n] = np.eye(n)
        m[n:2 * n, 2 * n:] = np.eye(n)
        x = cls._expm(m)
        return x[:n, :n], h * x[:n, n:2 * n], h * x[:n, 2 * n:]

    def _etd_step(
        self,
        u: np.ndarray,
        h: float,
        ops: Tuple[np.ndarray, np.ndarray, np.ndarray],
        del_start: List[np.ndarray],
        del_end: List[Optional[np.ndarray]],
        ctx: tuple,
    ) -> np.ndarray:
        """One ETD2RK step of size ``h``: ``a = E u + P1 N(u)``,
        ``u' = a + P2 (N(a) − N(u))``.  A delayed state at the end of the step
        that falls inside the step (delay shorter than ``h``) is read off the
        line through ``(t, u)`` and ``(t + h, a)``."""
        params, sp_idx, nonlin, taus, linear = ctx
        E, P1, P2 = ops
        if linear:
            return E @ u
        n0 = self._nonlin(u, del_start, params, sp_idx, nonlin)
        a = E @ u + P1 @ n0
        del_a = [
            d if d is not None else u + (a - u) * ((h - tau) / h)
            for d, tau in zip(del_end, taus)
        ]
        n1 = self._nonlin(a, del_a, params, sp_idx, nonlin)
        return a + P2 @ (n1 - n0)

    def _trial(
        self,
        u: np.ndarray,
        phi: np.ndarray,
        h: float,
        ops_h: tuple,
        ops_h2: tuple,
        dl: list,
        estimate: bool,
        ctx: tuple,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Two half-steps (Simpson's rule for φ) and, when ``estimate``, one
        full step for the error estimate.  ``dl[d] = (state at t − τ,
        at t + h/2 − τ, at t + h − τ)`` from the accepted history, the latter
        two ``None`` when they fall inside the current step."""
        params, sp_idx, nonlin, taus, linear = ctx
        hh = 0.5 * h
        d0 = [d[0] for d in dl]
        dm = [d[1] for d in dl]
        um = self._etd_step(u, hh, ops_h2, d0, dm, ctx)
        dm_from_um = [
            d[1] if d[1] is not None else u + (um - u) * ((hh - tau) / hh)
            for d, tau in zip(dl, taus)
        ]
        d1_second: List[Optional[np.ndarray]] = []
        for d, tau in zip(dl, taus):
            if d[2] is not None:
                d1_second.append(d[2])
            elif tau >= hh:
                d1_second.append(u + (um - u) * ((h - tau) / hh))
            else:
                d1_second.append(None)
        u2 = self._etd_step(um, hh, ops_h2, dm_from_um, d1_second, ctx)
        s0 = self._source(u, params, sp_idx)
        sm = self._source(um, params, sp_idx)
        s2 = self._source(u2, params, sp_idx)
        dphi = (s0 + sm * 4.0 + s2) * (h / 6.0)
        du = 0.0
        if not linear:
            # one full step: its difference from the two half-steps estimates
            # the error, and Richardson extrapolation removes the leading term
            d1 = [d[2] for d in dl]
            u1 = self._etd_step(u, h, ops_h, d0, d1, ctx)
            du = np.abs(u2 - u1).max(axis=0)
            u2 = u2 + (u2 - u1) / 3.0
        err = 0.0
        if estimate:
            trap = (s0 + sm * 2.0 + s2) * (h / 4.0)
            err = float(np.max(np.exp(phi.real) * (np.abs(dphi - trap) + du)))
        return um, u2, dphi, err

    # ------------------------------------------------------------------
    # Public PGF evaluator
    # ------------------------------------------------------------------

    def eval_pgf(
        self,
        p_log: np.ndarray,
        g: List[np.ndarray],
        fixed_schedule: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Evaluate the steady-state log-PGF φ at complex grid points.

        The characteristic equations ``du/dt = A u + N(u)``, ``dφ/dt = S(u)``
        are integrated with an exponential integrator: the first-order rate
        matrix ``A`` is propagated exactly through ``exp(A h)``, so the stiff
        rates never limit the step, and the remainder ``N`` (several products
        from one reaction, delayed product terms) is handled by the
        second-order exponential Runge-Kutta scheme ETD2RK.  φ is accumulated
        with Simpson's rule on two half-steps.  The step size is controlled by
        comparing one full step with the two half-steps (whose difference also
        feeds a Richardson extrapolation of ``u``), taking the maximum error
        over all grid points so the whole grid shares one schedule.  The
        accepted steps are recorded in ``last_schedule``; ``fixed_schedule``
        replays a given sequence (finite-difference stencils use the base
        point's, warped onto their own delays).  Delayed states come from the
        accepted history by cubic
        Hermite interpolation (every knot stores its derivative); a delay
        shorter than the step is read off the step's own predictor.  Mirrors
        ``monod_core::custom_pgf_sched``.

        Parameters
        ----------
        p_log : np.ndarray, shape (n_params,)
            Log₁₀ parameters in the order ``all_params``.
        g : list of np.ndarray
            Complex initial conditions, one per species, all the same shape.
        fixed_schedule : StepSchedule or array-like of float, optional
            Step sizes to replay instead of adapting; a ``StepSchedule`` is
            warped onto the current delays (see ``_warp_schedule``).

        Returns
        -------
        phi : np.ndarray (complex128)
            Log-PGF values at every grid point, in the shape of ``g[0]``.
        """
        if len(g) != len(self.species):
            raise ValueError(
                f"Expected {len(self.species)} grid arrays (species {self.species}), "
                f"got {len(g)}"
            )
        params = self._param_map(p_log)
        sp_idx = {s: i for i, s in enumerate(self.species)}
        A, nonlin, taus, lam_fast, lam_slow = self._linear_system(params, sp_idx)
        linear = not any(nonlin)
        ctx = (params, sp_idx, nonlin, taus, linear)

        shape = np.shape(g[0])
        u = np.stack([np.asarray(gi, dtype=np.complex128).ravel() for gi in g])
        phi = np.zeros(u.shape[1], dtype=np.complex128)
        tau_max = max(taus, default=0.0)

        def knot_deriv(hist, tk, uk):
            udel = [hist.at(tk - tau) for tau in taus]
            return A @ uk + self._nonlin(uk, udel, params, sp_idx, nonlin)

        h0 = _CUSTOM_ODE_C0 / lam_fast
        hist = None
        if taus:
            hist = _History(0.0, u, h0)
            hist.f[0] = knot_deriv(hist, 0.0, u)

        adaptive = fixed_schedule is None
        if not adaptive:
            # warp the recorded schedule onto the current delays
            fixed_steps = _warp_schedule(
                np.asarray(fixed_schedule, dtype=float),
                getattr(fixed_schedule, "breakpoints", ()), taus,
            )
        steps: List[float] = []
        t = 0.0
        h = h0
        # the controller's proposal; ``h`` is what is actually taken after the
        # history cap and the clipping to a delay breakpoint
        h_prop = h0
        clipped = False
        h_min = 1e-12 / lam_fast
        h_max = _CUSTOM_ODE_H_MAX / lam_slow
        # A delayed network is finished only once the state has been negligible
        # for a full delay: the history still feeds the system until then.
        quiet_since: Optional[float] = None
        while True:
            if not adaptive:
                if len(steps) >= len(fixed_steps):
                    break
                h = float(fixed_steps[len(steps)])
            else:
                if len(steps) >= _CUSTOM_ODE_MAX_STEPS:
                    break
                h = min(h_prop, h_max)
                if hist is not None:
                    for tau in taus:
                        h = hist.max_step(t - tau, h)
                # land exactly on the time a delayed product first switches on
                clipped = False
                for tau in taus:
                    if t < tau and t + h > tau and tau - t > 1e-6 * h:
                        h = tau - t
                        clipped = True
            t_mid = t + 0.5 * h
            t_end = t + h
            ops_h2 = self._step_ops(A, 0.5 * h)
            ops_h = ops_h2 if linear else self._step_ops(A, h)
            dl = []
            if hist is not None:
                for tau in taus:
                    d0 = hist.at(t - tau)
                    dm = hist.at(t_mid - tau) if tau >= 0.5 * h else None
                    d1 = hist.at(t_end - tau, left=True) if tau >= h else None
                    dl.append((d0, dm, d1))
            um, u2, dphi, err = self._trial(u, phi, h, ops_h, ops_h2, dl, adaptive, ctx)
            fac = 1.0
            if adaptive:
                if err > _CUSTOM_ODE_TOL and h > h_min:
                    h_prop = h * max(0.2, 0.9 * (_CUSTOM_ODE_TOL / err) ** (1.0 / 3.0))
                    continue
                fac = min(2.0, max(0.2, 0.9 * (_CUSTOM_ODE_TOL / max(err, 1e-300)) ** (1.0 / 3.0)))
            u = u2
            phi = phi + dphi
            t = t_end
            steps.append(h)
            remaining = float(np.max(np.exp(phi.real) * np.abs(u).max(axis=0))) if adaptive else 0.0
            if hist is not None:
                h_ctrl = h_prop if clipped else h
                hist.push(t_mid, um, h_ctrl, remaining)
                hist.push(t_end, u, h_ctrl, remaining)
                hist.f[-2] = knot_deriv(hist, t_mid, um)
                hist.f[-1] = knot_deriv(hist, t_end, u)
                hist.prune(t - tau_max)
            if adaptive:
                if remaining < _CUSTOM_ODE_U_END:
                    if not taus:
                        break
                    if quiet_since is None:
                        quiet_since = t
                    if t - quiet_since >= tau_max:
                        break
                else:
                    quiet_since = None
                # after landing on a breakpoint resume from the unclipped proposal
                h_prop = (h_prop if clipped else h) * fac

        self.last_schedule = StepSchedule(steps, taus)
        return phi.reshape(shape)
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
            burst = f"  [geometric {rxn.burst_param}{', mean rate' if self.burst_mean_rate else ''}]" if rxn.burst_param else ""
            if rxn.is_delayed:
                arrow = f"⇒({rxn.rate_name},{rxn.tau_name})"
            else:
                arrow = f"→({rxn.rate_name})"
            lines.append(f"    {src} {arrow} {tgt}{burst}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level helper (used by the ODE RHS)
# ---------------------------------------------------------------------------

class StepSchedule:
    """Accepted step sizes of one custom-network integration, with the delay
    lengths (breakpoints) they were recorded at.  Behaves as the 1-D array of
    steps; replaying it at other delays warps the time axis so every knot keeps
    its position relative to the delays (see ``_warp_schedule``)."""

    __slots__ = ("steps", "breakpoints")

    def __init__(self, steps, breakpoints=()) -> None:
        self.steps = np.asarray(steps, dtype=float)
        self.breakpoints = tuple(float(b) for b in breakpoints)

    @property
    def size(self) -> int:
        return int(self.steps.size)

    def __len__(self) -> int:
        return int(self.steps.size)

    def __iter__(self):
        return iter(self.steps)

    def __getitem__(self, i):
        return self.steps[i]

    def __array__(self, dtype=None, copy=None):
        return self.steps if dtype is None else self.steps.astype(dtype)

    def copy(self) -> "StepSchedule":
        return StepSchedule(self.steps.copy(), self.breakpoints)

    @property
    def times(self) -> np.ndarray:
        """Knot times (end of every step)."""
        return np.cumsum(self.steps)

    def __repr__(self) -> str:
        return f"StepSchedule({self.size} steps, breakpoints={self.breakpoints})"


def _warp_schedule(steps: np.ndarray, base, new) -> np.ndarray:
    """Map a recorded schedule onto new delay breakpoints: the time axis is
    stretched piecewise-linearly so every knot keeps its position relative to
    the delays, which makes a replayed result a smooth function of the delays.
    Identity when there are no delays, when they are unchanged, or when their
    order would change.  Mirrors ``monod_core::warp_schedule``."""
    steps = np.asarray(steps, dtype=float)
    base, new = list(base), list(new)
    if not base or len(base) != len(new) or base == new:
        return steps
    order = sorted(range(len(base)), key=lambda i: base[i])
    bx, by = [0.0], [0.0]
    for i in order:
        b, nb = base[i], new[i]
        if b <= bx[-1]:
            if nb != by[-1]:
                return steps
            continue
        if nb <= by[-1]:
            return steps
        bx.append(b)
        by.append(nb)

    def f(t):
        k = bisect.bisect_right(bx, t)
        if k >= len(bx):
            return by[-1] + (t - bx[-1])
        return by[k - 1] + (t - bx[k - 1]) * ((by[k] - by[k - 1]) / (bx[k] - bx[k - 1]))

    out = []
    t = 0.0
    for h in steps:
        t1 = t + h
        out.append(f(t1) - f(t))
        t = t1
    return np.asarray(out, dtype=float)


class _History:
    """Accepted knots of a delayed network: time, state ``(n_species, n_grid)``,
    its derivative once known, and the step size the error controller chose
    for the step that produced the knot (before clipping to a breakpoint)."""

    def __init__(self, t0: float, u0: np.ndarray, h0: float) -> None:
        self.t: List[float] = [t0]
        self.u: List[np.ndarray] = [u0.copy()]
        self.f: List[Optional[np.ndarray]] = [None]
        self.h_ctrl: List[float] = [h0]
        self.amp: List[float] = [1.0]

    def at(self, s: float, left: bool = False) -> np.ndarray:
        """State at time ``s``: zero before the observation time; cubic
        Hermite interpolation between knots when both derivatives are known,
        linear otherwise.  The state jumps from zero to the initial condition
        at the observation time: a stage at the end of a step reads the left
        limit (``left``), one at the start the right limit, so a step that
        lands on a delay sees no forcing and the next one sees it from its
        first stage."""
        # a landing on the onset is exact only up to rounding
        t0 = self.t[0]
        if abs(s - t0) <= 1e-10 * max(self.t[-1], 1.0):
            s = t0
        if s < t0 or (left and s <= t0):
            return np.zeros_like(self.u[0])
        lo = bisect.bisect_right(self.t, s)
        if lo >= len(self.t):
            return self.u[-1]
        ta, ua, fa = self.t[lo - 1], self.u[lo - 1], self.f[lo - 1]
        tb, ub, fb = self.t[lo], self.u[lo], self.f[lo]
        d = tb - ta
        th = (s - ta) / d
        if fa is not None and fb is not None:
            h00 = (2.0 * th - 3.0) * th * th + 1.0
            h10 = ((th - 2.0) * th + 1.0) * th
            h01 = (3.0 - 2.0 * th) * th * th
            h11 = (th - 1.0) * th * th
            return ua * h00 + fa * (h10 * d) + ub * h01 + fb * (h11 * d)
        return ua + (ub - ua) * th

    def push(self, t: float, u: np.ndarray, h_ctrl: float, amp: float) -> None:
        self.t.append(t)
        self.u.append(u)
        self.f.append(None)
        self.h_ctrl.append(h_ctrl)
        self.amp.append(amp)

    def max_step(self, s0: float, h: float) -> float:
        """Largest step from ``s0`` that does not out-run the resolution of
        the history it reads: within ``[s0, s0 + h]`` the step is capped at
        twice the step the controller chose there, wherever the history is
        large enough to matter (``amp`` at least the tolerance).  Without this
        a step that has grown long during a quiet stretch reads a delayed pulse
        narrower than itself only at its stage times, both the full step and
        the half-steps miss it, and the pulse is skipped."""
        m = len(self.t)
        k = max(bisect.bisect_right(self.t, s0), 1)
        while k < m:
            if self.t[k - 1] >= s0 + h:
                break
            if self.amp[k] >= _CUSTOM_ODE_TOL:
                h = min(h, 2.0 * self.h_ctrl[k])
            k += 1
        return h

    def prune(self, t_oldest_needed: float) -> None:
        while len(self.t) >= 2 and self.t[1] <= t_oldest_needed:
            self.t.pop(0)
            self.u.pop(0)
            self.f.pop(0)
            self.h_ctrl.pop(0)
            self.amp.pop(0)


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
