"""Draw cells from a model's generative process, independently of its kernel.

The kernels of :class:`monod.cme_toolbox.CMEModel` solve for the steady-state
distribution of counts; this module simulates the process itself, molecule by
molecule or reaction by reaction, and then applies the observation layers
(sequencing capture, ambiguous assignment, ATAC dropout) to each simulated
cell.  Nothing here reads a generating function, so agreement between a
simulated sample and the kernel's distribution is a check that the kernel
solves the process the model claims to describe.

Two engines cover the families:

- **Molecule lifelines** for the linear models, where every molecule evolves
  independently once born: ``Constitutive``, ``Bursty``, ``Extrinsic``,
  ``Delay``, ``DelayedSplicing`` and ``ProteinBursty``.  Births over a long
  window before the observation time, exponential or fixed lifetimes per
  stage, and for the protein model a Poisson number of proteins per mature
  molecule's lifetime; exact and vectorised.
- **Stochastic simulation (Gillespie)** for the Markov-modulated and network
  models: ``Chromatin`` (vectorised over cells) and ``Custom`` networks
  (vectorised without delays, one cell at a time with ``=>`` reactions), run
  to a burn-in horizon set by the slowest rate.

``CIR`` has no simulator: its kernel is the limiting law of an inverse-Gaussian
driven production process, and the finite-rate process behind it is not part
of the model's parameters.
"""
import heapq

import numpy as np

from . import chromatin as _ch

_WINDOW_FOLDS = 30.0          # exponential tails beyond this many lifetimes are below 1e-13
_CHUNK = 2000                 # cells per vectorised lifeline chunk


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def simulate_counts(model, theta, n_cells, samp=None, seed=0):
    """Counts of ``n_cells`` cells drawn from ``model``'s process at log10
    parameters ``theta`` (biological parameters, then the ambiguous
    probabilities when the model has them) and sampling vector ``samp``.

    Returns a dict ``{modality: int array of length n_cells}`` in the model's
    modality order, with the observation layers applied: Poisson or Bernoulli
    capture on the sequenced channels, the ambiguous split on the source
    species (per read under Poisson capture, per molecule otherwise), ATAC
    dropout on the configuration.
    """
    rng = np.random.default_rng(seed)
    theta = np.asarray(theta, dtype=float)
    n_amb = model._n_amb()
    theta_bio = theta[:theta.size - n_amb] if n_amb else theta
    latent = _latent_counts(model, theta_bio, int(n_cells), rng)
    return _observe(model, latent, samp, theta[theta.size - n_amb:] if n_amb else None, rng)


def simulate_adata(model, thetas, n_cells, samp=None, seed=0, gene_names=None):
    """An AnnData of simulated cells, one gene per row of ``thetas``, with one
    layer per modality (named after the modalities, so a model built by
    :meth:`CMEModel.compose` or with layer-named species fits it without a
    ``modality_name_dict``) and ``X`` the sum of the sequenced channels."""
    import anndata as ad
    thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
    rng = np.random.default_rng(seed)
    per_gene = [simulate_counts(model, t, n_cells, samp, seed=int(rng.integers(2 ** 31))) for t in thetas]
    layers = {m: np.column_stack([g[m] for g in per_gene]).astype(np.int64) for m in model.model_modalities}
    seq = [m for m in model.model_modalities if m not in ("configuration",)]
    a = ad.AnnData(X=sum(layers[m] for m in seq), layers=layers)
    a.var_names = list(gene_names) if gene_names is not None else [f"g{i}" for i in range(thetas.shape[0])]
    a.obs_names = [f"c{i}" for i in range(int(n_cells))]
    return a


# ---------------------------------------------------------------------------
# Latent process
# ---------------------------------------------------------------------------

def _latent_counts(model, theta, n_cells, rng):
    bio = model.bio_model
    lin = 10.0 ** theta
    if bio == "Constitutive":
        beta, gamma = lin
        return _lifelines(n_cells, rng, np.ones(n_cells), None, ("exp", beta), ("exp", gamma))
    if bio == "Bursty":
        b, beta, gamma = lin
        return _lifelines(n_cells, rng, np.ones(n_cells), b, ("exp", beta), ("exp", gamma))
    if bio == "Extrinsic":
        alpha, beta, gamma = lin
        return _lifelines(n_cells, rng, rng.gamma(alpha, 1.0, n_cells), None, ("exp", beta), ("exp", gamma))
    if bio == "Delay":
        b, beta, tauinv = lin
        return _lifelines(n_cells, rng, np.ones(n_cells), b, ("exp", beta), ("fixed", 1.0 / tauinv))
    if bio == "DelayedSplicing":
        b, tauinv, gamma = lin
        return _lifelines(n_cells, rng, np.ones(n_cells), b, ("fixed", 1.0 / tauinv), ("exp", gamma))
    if bio == "ProteinBursty":
        b, beta, gamma, k_p, gamma_p = lin
        return _lifelines(n_cells, rng, np.ones(n_cells), b, ("exp", beta), ("exp", gamma), protein=(k_p, gamma_p))
    if bio == "Chromatin":
        if getattr(model, "protein", False):
            raise NotImplementedError("the chromatin model's protein layer is not in the simulator yet "
                                      "(docs/chromatin-protein-design.md, stage 2)")
        return _chromatin_ssa(model, theta, n_cells, rng)
    if bio == "Custom":
        return _network_ssa(model, theta, n_cells, rng)
    if bio == "CIR":
        raise NotImplementedError("the CIR kernel is the limiting law of an inverse-Gaussian driven production "
                                  "process; the finite-rate process is not part of the model's parameters, so "
                                  "there is nothing to simulate")
    raise NotImplementedError(f"no simulator for bio_model {bio!r}")


def _geometric(rng, mean, size):
    """Geometric burst sizes on 0, 1, 2, ... with the given mean (the kernels' convention)."""
    return rng.geometric(1.0 / (1.0 + mean), size) - 1


def _lifelines(n_cells, rng, event_rate, burst_mean, nascent, mature, protein=None):
    """Counts at time 0 of a linear two-stage process observed in its steady
    state: birth events at ``event_rate`` per cell (single molecules, or
    geometric bursts of ``burst_mean``), each molecule nascent for an
    exponential or fixed time (``("exp", rate)`` / ``("fixed", tau)``), then
    mature likewise, and with ``protein=(k_p, gamma_p)`` a Poisson number of
    proteins made during the mature stage and still alive at 0."""
    scales = [1.0 / v if kind == "exp" else v for kind, v in (nascent, mature)]
    if protein is not None:
        scales.append(1.0 / protein[1])
    window = _WINDOW_FOLDS * max(scales) + sum(v for kind, v in (nascent, mature) if kind == "fixed")
    U = np.zeros(n_cells, dtype=np.int64)
    S = np.zeros(n_cells, dtype=np.int64)
    P = np.zeros(n_cells, dtype=np.int64)
    for c0 in range(0, n_cells, _CHUNK):
        idx = np.arange(c0, min(c0 + _CHUNK, n_cells))
        n_events = rng.poisson(np.asarray(event_rate)[idx] * window)
        cell = np.repeat(idx, n_events)
        if cell.size == 0:
            continue
        sizes = np.ones(cell.size, dtype=np.int64) if burst_mean is None else _geometric(rng, burst_mean, cell.size)
        t_birth = np.repeat(-rng.uniform(0.0, window, cell.size), sizes)
        mol = np.repeat(cell, sizes)
        if mol.size == 0:
            continue
        life_n = rng.exponential(1.0 / nascent[1], mol.size) if nascent[0] == "exp" else np.full(mol.size, nascent[1])
        t_spliced = t_birth + life_n
        life_m = rng.exponential(1.0 / mature[1], mol.size) if mature[0] == "exp" else np.full(mol.size, mature[1])
        t_gone = t_spliced + life_m
        U += np.bincount(mol[t_spliced > 0], minlength=n_cells)
        S += np.bincount(mol[(t_spliced <= 0) & (t_gone > 0)], minlength=n_cells)
        if protein is not None:
            k_p, g_p = protein
            start = t_spliced
            end = np.minimum(t_gone, 0.0)
            ok = start < 0
            mean = k_p * (np.exp(g_p * end[ok]) - np.exp(g_p * start[ok])) / g_p
            P += np.bincount(mol[ok], weights=rng.poisson(mean), minlength=n_cells).astype(np.int64)
    out = {"unspliced": U, "spliced": S}
    if protein is not None:
        out["protein"] = P
    return out


def _chromatin_ssa(model, theta, n_cells, rng):
    """Gillespie simulation of the locus chain with nascent and mature RNA,
    vectorised over cells, from the stationary chain and no RNA to a burn-in
    horizon of thirty times the slowest timescale."""
    n = model.n_sites
    k_on, eps, rates, beta, gamma = _ch.split_theta(theta, model.rule, n)
    burst = _ch.burst_size(theta, model.rule, n)
    H, A, _ = _ch.nascent_mature_system(k_on, 1.0, eps, model.rule, beta, gamma, periodic=model.periodic, rule_params=rates)
    N = H.shape[0]
    w = _ch.stationary_distribution(H)
    Hoff = H.copy()
    np.fill_diagonal(Hoff, 0.0)
    slowest = min(beta, gamma, float(np.min(k_on)), 1.0)
    t_end = _WINDOW_FOLDS / slowest
    state = rng.choice(N, size=n_cells, p=w)
    U = np.zeros(n_cells, dtype=np.int64)
    S = np.zeros(n_cells, dtype=np.int64)
    t = np.zeros(n_cells)
    while True:
        act = np.flatnonzero(t < t_end)
        if act.size == 0:
            break
        s = state[act]
        prod = A[s, 0] if burst is None else A[s, 0] / burst
        props = np.column_stack([Hoff[s], prod, beta * U[act], gamma * S[act]])     # (n_act, N + 3)
        total = props.sum(axis=1)
        t_new = t[act] + rng.exponential(1.0 / total)
        crossed = t_new > t_end                     # the state at t_end is the one before this event
        t[act] = np.where(crossed, t_end, t_new)
        act, props, total = act[~crossed], props[~crossed], total[~crossed]
        if act.size == 0:
            continue
        r = rng.uniform(0.0, total)
        which = (np.cumsum(props, axis=1) < r[:, None]).sum(axis=1)
        chain = which < N
        state[act[chain]] = which[chain]
        is_prod = which == N
        if burst is None:
            U[act[is_prod]] += 1
        else:
            U[act[is_prod]] += _geometric(rng, burst, int(is_prod.sum()))
        is_spl = which == N + 1
        U[act[is_spl]] -= 1
        S[act[is_spl]] += 1
        S[act[which == N + 2]] -= 1
    return {"unspliced": U, "spliced": S, "configuration": state}


def _network_ssa(model, theta, n_cells, rng):
    """Gillespie simulation of a reaction network: mass-action propensities,
    geometric bursts, and ``=>`` reactions whose products appear a delay after
    the commit; vectorised over cells unless the network has delays."""
    net = model.network
    species = list(net.species)
    k = len(species)
    params = net._param_map(np.asarray(theta, dtype=float))
    rxns = []
    for r in net.reactions:
        rxns.append(dict(
            rate=float(params[r.rate_name]),
            reactants=[(species.index(s), int(c)) for s, c in r.reactants.items()],
            products=[(species.index(s), int(c)) for s, c in r.products.items()],
            burst=(species.index(r.burst_species), float(params[r.burst_param])) if r.burst_param else None,
            delay=(1.0 / float(params[r.tau_name])) if r.is_delayed else None,
        ))
    consuming = [r["rate"] for r in rxns if r["reactants"]]
    slowest = min(consuming) if consuming else 1.0
    t_end = _WINDOW_FOLDS / min(slowest, 1.0) + sum(r["delay"] or 0.0 for r in rxns)
    if any(r["delay"] is not None for r in rxns):
        counts = np.array([_network_ssa_one(rxns, k, t_end, rng) for _ in range(n_cells)], dtype=np.int64)
    else:
        counts = _network_ssa_vectorised(rxns, k, n_cells, t_end, rng)
    return {s: counts[:, j] for j, s in enumerate(species)}


def _propensity(r, x):
    """Mass-action propensity for state ``x`` (``(n_cells, k)``): rate times
    the falling factorials of the reactant counts."""
    a = np.full(x.shape[0], r["rate"])
    for j, c in r["reactants"]:
        for i in range(c):
            a = a * np.maximum(x[:, j] - i, 0)
    return a


def _network_ssa_vectorised(rxns, k, n_cells, t_end, rng):
    x = np.zeros((n_cells, k), dtype=np.int64)
    t = np.zeros(n_cells)
    while True:
        act = np.flatnonzero(t < t_end)
        if act.size == 0:
            break
        xa = x[act]
        props = np.column_stack([_propensity(r, xa) for r in rxns])
        total = props.sum(axis=1)
        alive = total > 0
        t[act[~alive]] = t_end                                  # absorbing state: nothing more happens
        act, xa, props, total = act[alive], xa[alive], props[alive], total[alive]
        if act.size == 0:
            break
        t_new = t[act] + rng.exponential(1.0 / total)
        crossed = t_new > t_end                     # the state at t_end is the one before this event
        t[act] = np.where(crossed, t_end, t_new)
        act, props, total = act[~crossed], props[~crossed], total[~crossed]
        if act.size == 0:
            continue
        which = (np.cumsum(props, axis=1) < rng.uniform(0.0, total)[:, None]).sum(axis=1)
        for ri, r in enumerate(rxns):
            cells = act[which == ri]
            if cells.size == 0:
                continue
            for j, c in r["reactants"]:
                x[cells, j] -= c
            for j, c in r["products"]:
                if r["burst"] is not None and j == r["burst"][0]:
                    x[cells, j] += _geometric(rng, r["burst"][1], cells.size)
                else:
                    x[cells, j] += c
    return x


def _network_ssa_one(rxns, k, t_end, rng):
    x = np.zeros(k, dtype=np.int64)
    t = 0.0
    pending = []                                                 # (time, reaction index) of delayed products
    counter = 0
    while True:
        props = np.array([_propensity(r, x[None, :])[0] for r in rxns])
        total = props.sum()
        t_next = t + (rng.exponential(1.0 / total) if total > 0 else np.inf)
        if pending and pending[0][0] <= t_next:
            t, _, ri = heapq.heappop(pending)
            if t > t_end:
                break
            for j, c in rxns[ri]["products"]:
                x[j] += c
            continue
        if t_next > t_end:
            break
        t = t_next
        ri = int((np.cumsum(props) < rng.uniform(0.0, total)).sum())
        r = rxns[ri]
        for j, c in r["reactants"]:
            x[j] -= c
        if r["delay"] is not None:
            counter += 1
            heapq.heappush(pending, (t + r["delay"], counter, ri))
            continue
        for j, c in r["products"]:
            if r["burst"] is not None and j == r["burst"][0]:
                x[j] += _geometric(rng, r["burst"][1], 1)[0]
            else:
                x[j] += c
    return x


# ---------------------------------------------------------------------------
# Observation layers
# ---------------------------------------------------------------------------

def _observe(model, latent, samp, theta_amb, rng):
    mods = list(model.model_modalities)
    counts = {m: np.asarray(latent[m]).copy() for m in mods if m != "ambiguous"}
    samp = None if samp is None else np.asarray(samp, dtype=float).ravel()
    # sequencing: the first len(samp) sequenced channels, as _apply_seq_transform does
    if model.seq_model in ("Poisson", "Bernoulli"):
        if samp is None:
            raise ValueError(f"seq_model={model.seq_model!r} needs samp")
        sequenced = [m for m in mods if m not in ("configuration", "ambiguous")]
        for m, s in zip(sequenced, samp):
            lam = 10.0 ** s
            if model.seq_model == "Poisson":
                counts[m] = rng.poisson(lam * counts[m])
            else:
                counts[m] = rng.binomial(counts[m], min(lam, 1.0))
    # ambiguous assignment: per read (Poisson) or molecule, independently, after capture
    if model.amb_model != "None":
        probs = 10.0 ** np.asarray(theta_amb, dtype=float)
        if model.amb_model == "Equal":
            probs = np.repeat(probs, len(model.amb_species))
        amb = np.zeros_like(next(iter(counts.values())))
        for sp, p in zip(model.amb_species, probs):
            moved = rng.binomial(counts[sp], min(float(p), 1.0))
            counts[sp] = counts[sp] - moved
            amb = amb + moved
        counts["ambiguous"] = amb
    # ATAC dropout: each open site read as closed, independently
    if model.bio_model == "Chromatin":
        p_drop = model.chromatin_dropout(samp)
        if p_drop is not None:
            cfg = _ch.configurations(model.n_sites)[counts["configuration"]]        # (n_cells, n) 0/1
            keep = rng.random(cfg.shape) >= p_drop
            observed = cfg & keep
            counts["configuration"] = observed @ (2 ** np.arange(model.n_sites))
    return {m: np.asarray(counts[m], dtype=np.int64) for m in mods}
