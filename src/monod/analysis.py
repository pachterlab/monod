from .cme_toolbox import CMEModel
from .inference import SearchResults, plot_hist_and_fit, _resolve_marg, _genes_to_plot, searchdata_from_adata, _read_h5, search_result_from_uns, search_result_list_from_uns, aic_values, split_info
from .extract_data import knee_plot
from ._core import log
from .plot_aesthetics import aesthetics
import pickle
import scipy
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


########################
## Helper functions
########################


def nesting_embedding(model_small, model_big):
    """The map from ``model_small``'s log10 parameters into ``model_big``'s at
    which the two models coincide, for the nestings monod knows:

    - Chromatin models of the same locus: a ``switch`` or ``additive`` rule
      inside the ``table`` rule (:func:`monod.chromatin_fit.embed_in_table`),
      and a constitutive emission inside a bursty one (burst size at its
      floor); both at once as well.
    - Custom reaction networks: every reaction of the small network matched
      in the big one by reactants and products; a reaction the big network
      adds gets its rate at the floor of its box, a geometric burst the big
      network adds gets its burst size at the floor (which needs
      ``burst_mean_rate=True``, since in the burst-frequency form the small-size
      limit is no production), and a delay the big network adds gets its
      inverse delay at the ceiling (a delay of about 3e-4, so this one is
      approximate).
    - The same model family twice: the identity.

    Anything else raises ``ValueError`` saying why; for pairs with no nesting
    use :func:`compare_models`.  The built-in ``Bursty`` model is written in
    units of the burst frequency and ``Constitutive`` in units of the
    production rate, so the one is not a point of the other's box: write both
    as reaction networks with ``burst_mean_rate=True`` instead.
    """
    ms, mb = model_small, model_big
    if ms.seq_model != mb.seq_model or ms.amb_model != mb.amb_model:
        raise ValueError("the two models must share the sequencing and ambiguity models")
    if ms.bio_model == "Chromatin" and mb.bio_model == "Chromatin":
        from . import chromatin_fit as cf
        if ms.n_sites != mb.n_sites or bool(ms.periodic) != bool(mb.periodic):
            raise ValueError("Chromatin models must describe the same locus (n_sites, periodic)")
        if ms.rule != mb.rule and mb.rule != "table":
            raise ValueError(f"the {mb.rule!r} rule does not nest the {ms.rule!r} rule; only 'table' nests the others")
        if ms.bursty and not mb.bursty:
            raise ValueError("a constitutive emission does not nest a bursty one")
        n, rule_small, same_rule = ms.n_sites, ms.rule, ms.rule == mb.rule
        add_burst = bool(mb.bursty) and not ms.bursty
        b_floor = float(mb.bio_bounds["phys_lb"][-1])

        def embed(x):
            x = np.asarray(x, dtype=float)
            if add_burst:
                x = np.append(x, b_floor)                   # the burst size rides last and is carried by embed_in_table
            return x if same_rule else cf.embed_in_table(x, rule_small, n)
        return embed
    if ms.bio_model == "Custom" and mb.bio_model == "Custom":
        return _network_embedding(ms, mb)
    if ms.bio_model == mb.bio_model and ms.get_num_params() == mb.get_num_params():
        return lambda x: np.asarray(x, dtype=float).copy()
    if {ms.bio_model, mb.bio_model} == {"Constitutive", "Bursty"}:
        raise ValueError("monod's Bursty model is in units of the burst frequency and Constitutive in units of the "
                         "production rate, so Constitutive is not a point of Bursty's parameter box. Write both as "
                         "reaction networks with burst_mean_rate=True ('0 -> U -> S -> 0' and '0 -> B x U -> S -> 0', "
                         "normalize_production_rate=True) and the constitutive one is nested at the burst-size floor; "
                         "or use compare_models for a non-nested comparison.")
    raise ValueError(f"no known nesting of {ms.bio_model} in {mb.bio_model}; pass embed= or use compare_models")


def _network_embedding(ms, mb):
    ns, nb = ms.network, mb.network
    if ns.normalize_production_rate != nb.normalize_production_rate:
        raise ValueError("both networks must use the same normalize_production_rate")
    if ns.named_params and ns.burst_mean_rate != nb.burst_mean_rate:
        raise ValueError("both networks must use the same burst_mean_rate")

    def key(r):
        return (tuple(sorted(r.reactants.items())), tuple(sorted(r.products.items())))

    unmatched = list(nb.reactions)
    mapping = {}                                            # big parameter -> ("copy", small name) | ("lb",) | ("ub",)
    for r in ns.reactions:
        cands = [c for c in unmatched if key(c) == key(r) and (r.burst_param is None or c.burst_param is not None)]
        if not cands:
            raise ValueError(f"the big network has no counterpart of the small network's reaction "
                             f"{r.reactants or '0'} -> {r.products}")
        c = cands[0]
        unmatched.remove(c)
        if (r.rate_name == ns._norm_rate) != (c.rate_name == nb._norm_rate):
            raise ValueError("the normalised production reaction must be the same reaction in both networks")
        if r.rate_name != ns._norm_rate:
            mapping[c.rate_name] = ("copy", r.rate_name)
        if c.burst_param is not None:
            if r.burst_param is not None:
                mapping[c.burst_param] = ("copy", r.burst_param)
            elif not nb.burst_mean_rate:
                raise ValueError("the big network adds a geometric burst in the burst-frequency form, whose small-size "
                                 "limit is no production; build it with burst_mean_rate=True")
            else:
                mapping.setdefault(c.burst_param, ("lb",))
        if r.is_delayed and not c.is_delayed:
            raise ValueError("an immediate reaction does not nest a delayed one")
        if c.is_delayed:
            mapping[c.tau_name] = ("copy", r.tau_name) if r.is_delayed else ("ub",)
    for c in unmatched:
        if c.rate_name == nb._norm_rate:
            raise ValueError("the big network's normalised production reaction has no counterpart in the small one")
        mapping[c.rate_name] = ("lb",)
        if c.burst_param is not None and c.burst_param not in mapping:
            if not nb.burst_mean_rate:
                raise ValueError("the big network adds a geometric burst in the burst-frequency form; build it with "
                                 "burst_mean_rate=True so that the small-size limit is the single-molecule reaction")
            mapping[c.burst_param] = ("lb",)
        if c.is_delayed:
            mapping[c.tau_name] = ("ub",)
    missing = [n for n in nb.all_params if n not in mapping]
    if missing:
        raise ValueError(f"no embedding for the big network's parameters {missing}")
    lb = np.asarray(mb.bio_bounds["phys_lb"], float)
    ub = np.asarray(mb.bio_bounds["phys_ub"], float)
    idx = {n: i for i, n in enumerate(ns.all_params)}
    plan = [(mapping[n][0], idx.get(mapping[n][1]) if mapping[n][0] == "copy" else j) for j, n in enumerate(nb.all_params)]

    def embed(x):
        x = np.asarray(x, dtype=float)
        return np.array([x[i] if how == "copy" else (lb[i] if how == "lb" else ub[i]) for how, i in plan])
    return embed


def compare_models(model_a, model_b, x_a, x_b, limits, samp, data, n_cells, EPS=1e-15):
    """Compare two fitted models on one gene without assuming either nests the
    other: divergences, AIC and BIC in the deviance form (``2k + 2 n_cells KLD``
    and ``k log n_cells + 2 n_cells KLD``; these differ from ``adata.var['AIC']``
    and :func:`monod.inference.aic_values` by ``2 n_cells`` times the entropy of
    the empirical distribution, a per-gene constant that cancels in any
    difference between models) and Vuong's test, the per-cell log-likelihood ratios of the two
    fits read off the histogram.  ``vuong_z`` is the raw statistic (positive
    favours ``model_a``), ``vuong_z_bic`` the Schwarz-corrected one used for
    the two-sided p-value ``p_vuong``, and ``prefer`` is ``"a"``, ``"b"`` or
    ``None`` at the 5 % level.  For nested pairs use
    :func:`compare_nested_models`, whose likelihood ratio is the sharper test.
    """
    from scipy.stats import norm
    coords, freqs = data
    coords = np.asarray(coords)
    freqs = np.asarray(freqs, dtype=float)
    freqs = freqs / freqs.sum()

    def logp(m, x):
        pss = np.asarray(m.eval_model_pss(np.asarray(x, dtype=float), limits, samp), dtype=float)
        pss = np.where(pss < EPS, EPS, pss)
        return np.log(pss[tuple(coords.T)])
    la, lb_ = logp(model_a, x_a), logp(model_b, x_b)
    lf = np.log(freqs)
    kld_a, kld_b = float(freqs @ (lf - la)), float(freqs @ (lf - lb_))
    ka, kb = model_a.get_num_params(), model_b.get_num_params()
    n = float(n_cells)
    d = la - lb_
    m = float(freqs @ d)
    v = float(freqs @ (d - m) ** 2)
    if v > 0:
        z = np.sqrt(n) * m / np.sqrt(v)
        z_bic = (n * m - 0.5 * (ka - kb) * np.log(n)) / np.sqrt(n * v)
    else:
        z = z_bic = 0.0
    p = float(2.0 * norm.sf(abs(z_bic)))
    return dict(kld_a=kld_a, kld_b=kld_b, k_a=ka, k_b=kb,
                aic_a=2 * ka + 2 * n * kld_a, aic_b=2 * kb + 2 * n * kld_b,
                bic_a=ka * np.log(n) + 2 * n * kld_a, bic_b=kb * np.log(n) + 2 * n * kld_b,
                vuong_z=float(z), vuong_z_bic=float(z_bic), p_vuong=p,
                prefer=None if p >= 0.05 else ("a" if z_bic > 0 else "b"))


def compare_nested_models(model_small, model_big, embed=None, x_small=None, limits=None, samp=None, data=None, n_cells=None,
                          alpha=0.05, table="score", hess_step=1e-2, maxiter=200, bounds=None, gtol=1e-2,
                          score=None, n_jobs=1, nudge=True, extra_starts=None):
    """Test a fitted model against one that nests it, for one gene.

    ``embed`` maps the small model's log10 parameters into the big model's
    space; ``None`` takes the map :func:`nesting_embedding` knows for the pair
    (Chromatin rules and emissions, reaction networks that add reactions,
    bursts or delays), and raises when there is none.  Both models see the
    same ``data`` ``(coords, freqs)`` on ``limits`` with sampling vector
    ``samp``.

    With ``table="score"`` a Rao score test at the embedded optimum decides
    whether the big model is fitted at all: the gradient and observed
    information of the divergence (times ``n_cells``, so in nats) are taken
    on the embedded point's pinned step schedule, coordinates pinned at a
    bound are held fixed, and the statistic is compared with chi-square on
    the remaining extra parameters.  ``"always"`` fits the big model
    regardless; ``"never"`` only reports the score test.  The big model is
    warm-started from the embedding and can therefore never end worse than
    the small one (the nested-model check); the likelihood ratio
    ``2 n_cells (KLD_small - KLD_big)`` and its p-value are reported.  The
    big model is optimised in nats; ``gtol`` is the L-BFGS-B projected
    gradient tolerance on that scale.  The score test needs the observed
    information, ``2 d`` gradient evaluations of the big model; with
    ``table="always"`` it decides nothing and is skipped unless
    ``score=True`` (``score`` is then ``(nan, df, nan)``).  ``n_jobs`` is
    the thread count of the finite-difference stencil behind every gradient
    (:meth:`CMEModel.eval_model_kld_and_grad`); the Hessian columns of the
    score test are taken in the same pool.

    Returns a dict with ``x_small``, ``kld_small``, ``score`` (statistic, df,
    p), ``fitted`` (bool), ``x_big``, ``kld_big``, ``lr``, ``df``, ``p_lr``,
    ``prefer_big`` (the likelihood ratio rejects the small model at
    ``alpha``) and the ``aic``/``bic`` of both fits as in
    :func:`compare_models`.  When the extra parameters sit at a bound in the
    embedding (a burst size or a rate at its floor) the chi-square p-value is
    conservative and the score test cannot see them, so use
    ``table="always"`` there.  A parameter pinned at a bound is also a poor
    place to start from when the data want it far inside the box (the
    divergence is flat in a burst size near zero, and a gradient optimiser
    does not leave), so with ``nudge=True`` the big model is fitted once
    from the embedding and once more per pinned coordinate with that
    coordinate moved to the middle of its box, and the best fit is kept;
    ``extra_starts`` adds starts of your own (for instance the big model's
    own moments fit).  The nesting guarantee holds for every start.
    """
    import scipy.optimize
    from scipy.stats import chi2

    if x_small is None or limits is None or data is None or n_cells is None:
        raise TypeError("compare_nested_models needs x_small, limits, data and n_cells")
    if embed is None:
        embed = nesting_embedding(model_small, model_big)
    x_small = np.asarray(x_small, dtype=float)
    kld_small = float(model_small.eval_model_kld(x_small, limits, samp, data))
    x0 = np.asarray(embed(x_small), dtype=float)
    lb = np.asarray(bounds[0] if bounds is not None else model_big.bio_bounds["phys_lb"], float)
    ub = np.asarray(bounds[1] if bounds is not None else model_big.bio_bounds["phys_ub"], float)
    x0 = np.clip(x0, lb, ub)
    d = x0.size
    df_total = d - x_small.size

    # score test at the embedded optimum
    free = np.where((x0 > lb + 1e-9) & (x0 < ub - 1e-9))[0]
    df = max(df_total - (d - free.size), 1)
    if score is None:
        score = table != "always"
    if score:
        with model_big.pinned_schedule(x0, limits, samp):
            kld0, g = model_big.eval_model_kld_and_grad(x0, limits, samp, data, n_jobs=n_jobs)
            state = model_big.thread_state()                 # the pin, for the pool's worker threads

            def column(j):
                e = np.zeros(d); e[j] = hess_step
                with model_big.carry_thread_state(state):
                    _, gp = model_big.eval_model_kld_and_grad(x0 + e, limits, samp, data)
                    _, gm = model_big.eval_model_kld_and_grad(x0 - e, limits, samp, data)
                return (np.asarray(gp) - np.asarray(gm)) / (2 * hess_step)
            if n_jobs == 1 or d == 1:
                cols = [column(j) for j in range(d)]
            else:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=d if n_jobs < 0 else min(d, n_jobs)) as ex:
                    cols = list(ex.map(column, range(d)))
        I = 0.5 * (np.column_stack(cols) + np.column_stack(cols).T) * n_cells
        g = np.asarray(g) * n_cells
        gf, If = g[free], I[np.ix_(free, free)]
        try:
            y = np.linalg.solve(np.linalg.cholesky(If), gf)
            stat = float(y @ y)
        except np.linalg.LinAlgError:
            vals, vecs = np.linalg.eigh(If)
            vals = np.maximum(vals, 1e-8 * max(vals.max(), 1e-300))
            stat = float((vecs.T @ gf) ** 2 @ (1.0 / vals))
        p_score = float(chi2.sf(stat, df))
    else:
        kld0 = float(model_big.eval_model_kld(x0, limits, samp, data))
        stat, p_score = float("nan"), float("nan")
    out = dict(x_small=x_small, kld_small=kld_small, score=(stat, df, p_score), fitted=False,
               x_big=x0, kld_big=float(kld0), lr=0.0, df=df_total, p_lr=1.0, prefer_big=False)

    run = table == "always" or (table == "score" and p_score < alpha)
    if run:
        # Optimise in nats (n_cells times the divergence) so the gradient
        # tolerance has a fixed meaning: the per-cell divergence gradient of
        # a well-fitted gene on real data is far below 1e-2 already at the
        # embedded start, and the optimizer would not move at all.
        def fg(x):
            k, g = model_big.eval_model_kld_and_grad(x, limits, samp, data, n_jobs=n_jobs)
            return n_cells * k, n_cells * np.asarray(g)
        starts = [x0]
        if nudge:
            pinned = np.where((x0 <= lb + 1e-9) | (x0 >= ub - 1e-9))[0]
            for j in pinned:
                x_j = x0.copy()
                x_j[j] = 0.5 * (lb[j] + ub[j])
                starts.append(x_j)
        for x_extra in (extra_starts or []):
            starts.append(np.clip(np.asarray(x_extra, dtype=float), lb, ub))
        x_big, kld_big = x0, float(kld0)              # a nesting model never fits worse than its start
        with model_big.search_tolerance():
            for x_start in starts:
                res = scipy.optimize.minimize(fg, x_start, jac=True, method="L-BFGS-B",
                                              bounds=scipy.optimize.Bounds(lb, ub),
                                              options=dict(maxiter=maxiter, ftol=1e-9, gtol=gtol))
                k_res = float(model_big.eval_model_kld(res.x, limits, samp, data))
                if k_res < kld_big:
                    x_big, kld_big = res.x, k_res
        out.update(fitted=True, x_big=x_big, kld_big=kld_big, n_starts=len(starts))
    lr = max(2.0 * n_cells * (kld_small - out["kld_big"]), 0.0)
    out.update(lr=lr, p_lr=float(chi2.sf(lr, df_total)), prefer_big=bool(out["fitted"] and chi2.sf(lr, df_total) < alpha))
    k_s, k_b, n = model_small.get_num_params(), model_big.get_num_params(), float(n_cells)
    out.update(aic_small=2 * k_s + 2 * n * kld_small, aic_big=2 * k_b + 2 * n * out["kld_big"],
               bic_small=k_s * np.log(n) + 2 * n * kld_small, bic_big=k_b * np.log(n) + 2 * n * out["kld_big"])
    return out


def cluster_confusion(labels_a, labels_b):
    """Contingency table of two label vectors over the same cells.

    Returns ``(table, values_a, values_b)``: ``table[i, j]`` is the number of
    cells with ``labels_a == values_a[i]`` and ``labels_b == values_b[j]``, the
    distinct values of each vector sorted.
    """
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    if a.shape != b.shape:
        raise ValueError(f"label vectors differ in length: {a.shape} vs {b.shape}")
    values_a, ia = np.unique(a, return_inverse=True)
    values_b, ib = np.unique(b, return_inverse=True)
    table = np.zeros((len(values_a), len(values_b)), dtype=int)
    np.add.at(table, (ia.ravel(), ib.ravel()), 1)
    return table, values_a, values_b


def match_clusters(reference, labels):
    """Relabel integer cluster ``labels`` so that each cluster takes the id of the
    ``reference`` cluster it shares most cells with (a one-to-one assignment on
    the contingency table).

    Cluster ids are arbitrary, so two clusterings of the same cells, such as two
    meK-Means runs, are compared cell by cell after this.  Negative labels
    (unassigned cells) are kept; clusters left without a partner, when
    ``labels`` has more clusters than ``reference``, get fresh ids above the
    reference's largest.
    """
    ref = np.asarray(reference)
    lab = np.asarray(labels)
    keep = (ref >= 0) & (lab >= 0)
    table, ref_values, lab_values = cluster_confusion(ref[keep], lab[keep])
    rows, cols = linear_sum_assignment(-table)
    mapping = {int(lab_values[j]): int(ref_values[i]) for i, j in zip(rows, cols)}
    next_id = int(ref_values.max()) + 1 if len(ref_values) else 0
    for v in lab_values:
        if int(v) not in mapping:
            mapping[int(v)] = next_id
            next_id += 1
    out = lab.copy()
    for v, new in mapping.items():
        out[lab == v] = new
    return out


def plot_cluster_confusion(true_labels, assigns, row_names=None, col_names=None, ax=None,
                           align=True, title=None, cmap="Blues", fontsize=9, colorbar=True):
    """Annotated confusion matrix of cluster assignments against reference labels,
    with the adjusted Rand index in the title.

    ``true_labels`` may be strings (cell-type annotations) or integers;
    ``assigns`` are the integer clusters, e.g. from
    :func:`monod.mminference.cell_assignments`.  ``row_names`` and
    ``col_names`` rename the sorted distinct values of each vector (sequences
    aligned with them, or dicts); by default rows show the values and columns
    ``Cluster <k>``.  With ``align`` the columns are ordered so that the best
    one-to-one match with the rows runs down the diagonal; this is cosmetic,
    the ARI does not depend on it.

    Returns a dict with ``ari``, the plotted ``table``, its ``rows`` and
    ``cols`` (values in plotted order) and ``ax``.
    """
    true_labels = np.asarray(true_labels)
    assigns = np.asarray(assigns)
    table, rows, cols = cluster_confusion(true_labels, assigns)
    if align:
        r_ind, c_ind = linear_sum_assignment(-table)
        order = list(c_ind[np.argsort(r_ind)]) + [j for j in range(len(cols)) if j not in set(c_ind)]
        table, cols = table[:, order], cols[order]
    ari = float(adjusted_rand_score(true_labels, assigns))

    def _names(values, given, default):
        if given is None:
            return [default(v) for v in values]
        if isinstance(given, dict):
            return [str(given.get(v, v)) for v in values]
        lookup = dict(zip(np.unique(values), given))
        return [str(lookup[v]) for v in values]

    row_text = _names(rows, row_names, str)
    col_text = _names(cols, col_names, lambda v: f"Cluster {v}")
    if ax is None:
        _, ax = plt.subplots(figsize=(1.2 * len(cols) + 2.5, 1.0 * len(rows) + 1.5))
    im = ax.imshow(table, cmap=cmap)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(col_text, rotation=15, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(row_text)
    thresh = table.max() * 0.5
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{table[i, j]:,}", ha="center", va="center", fontsize=fontsize,
                    color="white" if table[i, j] > thresh else "black")
    ax.set_title(f"{title}\nARI = {ari:.3f}" if title else f"ARI = {ari:.3f}")
    if colorbar:
        ax.figure.colorbar(im, ax=ax, shrink=0.8)
    return {"ari": ari, "table": table, "rows": rows, "cols": cols, "ax": ax}


def plot_pairwise_projections(pss, limits=None, data=None, labels=("Unspliced", "Spliced", "Protein"),
                              axes=None, cmap="Blues", title=None, scatter_scale=150, colorbar=False):
    """The three pairwise marginals of a three-species distribution, each as a heat
    map on a power-law colour scale, with the observed counts scattered on top
    when ``data`` is given.

    ``pss`` is the joint PMF of shape (n0, n1, n2), as ``eval_model_pss``
    returns it for the protein model or a three-species network; ``limits``
    are its grid limits (default: its shape); ``data`` is the ``(coords, freqs)``
    unique histogram of the gene, drawn with marker area proportional to
    frequency; ``labels`` names the three species in axis order.  Draws on
    ``axes`` (three of them) or a new 1 x 3 figure, and returns the axes.
    """
    pss = np.asarray(pss)
    if pss.ndim != 3:
        raise ValueError(f"pss must be three-dimensional, got shape {pss.shape}")
    limits = list(pss.shape) if limits is None else [int(m) for m in limits]
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (i, j) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
        pss2 = pss.sum(axis=3 - i - j)
        im = ax.imshow(pss2.T, origin="lower", aspect="auto", cmap=cmap,
                       norm=mcolors.PowerNorm(gamma=0.4, vmin=0),
                       extent=[-0.5, limits[i] - 0.5, -0.5, limits[j] - 0.5])
        if data is not None:
            coords, freqs = data
            coords = np.asarray(coords)
            ax.scatter(coords[:, i], coords[:, j], s=np.asarray(freqs) * scatter_scale,
                       c="red", alpha=0.35, linewidths=0)
        ax.set_xlabel(f"{labels[i]} count")
        ax.set_ylabel(f"{labels[j]} count")
        ax.set_title(f"{labels[i]} vs {labels[j]}")
        if colorbar:
            ax.figure.colorbar(im, ax=ax, shrink=0.8)
    if title:
        axes[0].figure.suptitle(title, y=1.02, fontsize=12)
    return axes


def _group_index(adata, group):
    """Position of ``group`` in ``adata.uns['search_result_list']``: a group name
    or table key of a split fit, a ``(cell type, experiment)`` pair when the fit
    used ``batch_by``, or a meK-Means cluster index."""
    split = split_info(adata)
    if split is None:
        return int(group)
    if isinstance(group, (tuple, list)) and len(group) == 2:
        want = tuple(None if v is None else str(v) for v in group)
        for i, pair in enumerate(zip(split['split'], split['batch'])):
            if pair == want:
                return i
        raise KeyError(f"{group!r} is not one of the (cell type, experiment) pairs "
                       f"{list(zip(split['split'], split['batch']))}")
    g = str(group)
    if g in split['groups']:
        return split['groups'].index(g)
    if g in split['keys']:
        return split['keys'].index(g)
    raise KeyError(f"{group!r} is not one of the groups {split['groups']}")


def group_result(adata, group):
    """The SearchResults of one group of a split fit (``perform_inference(split_by=...)``),
    given its name, table key or ``(cell type, experiment)`` pair; for a meK-Means
    run, the cluster with that index."""
    return search_result_list_from_uns(adata.uns['search_result_list'])[_group_index(adata, group)]


def group_table(adata, group):
    """The per-gene table of one group of a split fit, ``adata.varm[<key>]``,
    given the group's label or key."""
    return adata.varm[split_info(adata)['keys'][_group_index(adata, group)]]


def group_search_data(adata, group, search_data=None):
    """The SearchData of the cells of one group (or meK-Means cluster), for the
    functions that take a per-dataset SearchData such as ``diffexp_mean``;
    ``search_data`` is rebuilt from the AnnData when not given."""
    sr = group_result(adata, group)
    if search_data is None:
        search_data = searchdata_from_adata(adata)
    return sr._search_data_for(search_data)


def run_qc(monod_adata, mek_means_params=None, cluster=0, discard_rejected=True, marg='joint',
           gene_dist_size=(5, 5), show_axes=False, group=None):
    """Quick look at a finished run: the sampling optimum, the divergence
    landscape and the fitted distributions of a few genes.  ``group`` picks a
    group of a split fit (by label) or a meK-Means cluster (by index); the
    ``mek_means_params``/``cluster`` pair is the older spelling of the latter."""
    # The SearchData is never stored in the AnnData; rebuild it in memory.
    sd = searchdata_from_adata(monod_adata)
    if group is not None:
        sr = group_result(monod_adata, group)
    elif 'search_result' in monod_adata.uns and not mek_means_params:
        sr = search_result_from_uns(monod_adata.uns['search_result'])
    else:
        sr = search_result_list_from_uns(monod_adata.uns['search_result_list'])[cluster]
    sr.find_sampling_optimum()
    _, ax1 = plt.subplots(1, 1)
    sr.plot_KL(ax1, discard_rejected=discard_rejected)
    sr.plot_gene_distributions(sd, marg=marg, sz=gene_dist_size, show_axes=show_axes)


def load_search_results(full_result_string):
    """Load search results from disk.

    Accepts the ``monod_adata.h5ad`` written by ``perform_inference(save=True)``
    (returns the SearchResults, or the per-cluster list for a meK-Means run), a
    ``grid_scan_results.h5`` written by ``SearchResults.store_on_disk`` or
    ``run_mek_means(save=True)``, or a legacy pickled ``.res`` file.

    Parameters
    ----------
    full_result_string: str
        file location.

    Returns
    -------
    search_results: SearchResults or list of SearchResults
    """
    try:
        try:
            if full_result_string.endswith(".h5ad"):
                import h5py
                from anndata.io import read_elem
                with h5py.File(full_result_string, "r") as f:
                    uns = read_elem(f["uns"])
                if "search_result_list" in uns:
                    return search_result_list_from_uns(uns["search_result_list"])
                return search_result_from_uns(uns["search_result"])
            d = _read_h5(full_result_string)
            if str(d.get("kind", "")) == "SearchResultsList":
                return search_result_list_from_uns(d)
            search_results = SearchResults.from_dict(d)
        except OSError:
            # Legacy pickled .res file written by older monod versions.
            log.warning("%s is not an HDF5 results file; falling back to the legacy pickle loader.", full_result_string)
            with open(full_result_string, "rb") as srfs:
                search_results = pickle.load(srfs)
            if not hasattr(search_results.model, "amb_model"):  # bck
                search_results.model.amb_model = "None"
        log.info("Grid scan results loaded from {}.".format(full_result_string))
        return search_results
    except Exception as ex:
        log.error(
            "Grid scan results could not be loaded from {}.".format(full_result_string)
        )
        raise ex


def plot_params_for_pair(
    sr1,
    sr2,
    gene_filter_=None,
    plot_errorbars=False,
    figsize=None,
    c=2.576,
    axis_search_bounds=True,
    distinguish_rej=True,
    plot_identity=True,
    meta="12",
    xlabel="dataset 1",
    ylabel="dataset 2",
    savefig=True,
):
    """Plot the inferred physical parameters at the sampling parameter optimum for a matched pair of datasets.

    This is typically useful to show similarities between controls, or show differences between different
    tissues, for a common set of genes.

    Parameters
    -------
    sr1: monod.inference.SearchResults
        search result with parameters to plot on abscissa.
    sr2: monod.inference.SearchResults
        search result with parameters to plot on ordinate.
    gene_filter_: None or np.ndarray, optional
        If None, plot all genes.
        If a boolean or integer filter, plot the filtered gene subset.
    plot_errorbars: bool, optional
        whether to use inferred standard error of maximum likelihood estimates to plot error bars.
    figsize: None or tuple of floats, optional
        figure dimensions.
    c: float
        if plotting the errorbars, the number of standard deviations to display.
        this can be used for Gaussian approximations to the MLE confidence intervals.
    axis_search_bounds: bool, optional
        whether to place the x-limits of the plots at the parameter search bounds.
    distinguish_rej: bool, optional
         whether to distinguish the genes in either of the SearchResults objects'
         rejected_genes attributes by color.
    plot_identity: bool, optional
        whether to plot the identity line. Useful for displaying biases.
    meta: str, optional
        figure filestring suffix.
    xlabel: str, optional
        figure x-label, name of the first dataset.
    ylabel: str, optional
        figure y-label, name of the second dataset.
    savefig: bool, optional
        save to this file path when a string is given
        with make_batch_analysis_dir; otherwise the figure is only returned.

    Returns
    -------
    fig1: matplotlib.figure.Figure
    """
    num_params = sr1.sp.n_phys_pars
    figsize = figsize or (4 * num_params, 4)
    fig1, ax1 = plt.subplots(nrows=1, ncols=num_params, figsize=figsize)
    ax1 = np.atleast_1d(ax1)
    if plot_errorbars and not (hasattr(sr1, "sigma") and hasattr(sr2, "sigma")):
        log.warning("plot_errorbars requested but sigma is missing; run compute_sigma first. Plotting without error bars.")
        plot_errorbars = False

    gene_filter = sr1.get_bool_filt(gene_filter_, discard_rejected=False)
    # gene_filter2 = sr2.get_bool_filt(gene_filter_,discard_rejected=False)
    gene_filter_rej = np.zeros(sr1.n_genes, dtype=bool)
    # gene_filter_rej2 = np.zeros(sr2.n_genes,dtype=bool)

    if distinguish_rej:  # default
        filt_rej1 = sr1.get_bool_filt(gene_filter_, discard_rejected=True)
        filt_rej2 = sr2.get_bool_filt(gene_filter_, discard_rejected=True)
        filt_rej = np.logical_and(filt_rej1, filt_rej2)

        gene_filter_rej = np.logical_and(gene_filter, np.logical_not(filt_rej))
        gene_filter = np.logical_and(gene_filter, filt_rej)
        acc_point_aesth = (
            "accepted_gene_color",
            "accepted_gene_alpha",
            "accepted_gene_ms",
        )
        rej_point_aesth = (
            "rejected_gene_color",
            "rejected_gene_alpha",
            "rejected_gene_ms",
        )
    else:  # don't distinguish
        acc_point_aesth = (
            "generic_gene_color",
            "generic_gene_alpha",
            "generic_gene_ms",
        )
        log.info("Falling back on generic marker properties.")

    for i in range(num_params):
        if plot_errorbars:
            ax1[i].errorbar(
                sr1.phys_optimum[gene_filter, i],
                sr2.phys_optimum[gene_filter, i],
                sr1.sigma[gene_filter, i] * c,
                sr2.sigma[gene_filter, i] * c,
                c=aesthetics["errorbar_gene_color"],
                alpha=aesthetics["errorbar_gene_alpha"],
                linestyle="None",
                linewidth=aesthetics["errorbar_lw"],
            )
        ax1[i].scatter(
            sr1.phys_optimum[gene_filter, i],
            sr2.phys_optimum[gene_filter, i],
            c=aesthetics[acc_point_aesth[0]],
            alpha=aesthetics[acc_point_aesth[1]],
            s=aesthetics[acc_point_aesth[2]],
        )
        if np.any(gene_filter_rej):
            ax1[i].scatter(
                sr1.phys_optimum[gene_filter_rej, i],
                sr2.phys_optimum[gene_filter_rej, i],
                c=aesthetics[rej_point_aesth[0]],
                alpha=aesthetics[rej_point_aesth[1]],
                s=aesthetics[rej_point_aesth[2]],
            )

        ax1[i].set_xlabel(xlabel)
        ax1[i].set_ylabel(ylabel)
        ax1[i].set_title(sr1.model.get_log_name_str()[i])
        if axis_search_bounds:
            ax1[i].set_xlim([sr1.sp.phys_lb[i], sr1.sp.phys_ub[i]])
            ax1[i].set_ylim([sr1.sp.phys_lb[i], sr1.sp.phys_ub[i]])
        if plot_identity:
            xl = ax1[i].get_xlim()
            ax1[i].plot(xl, xl, "r--", linewidth=2)
    fig1.tight_layout()

    if isinstance(savefig, str):
        plt.savefig(savefig, dpi=450)
        log.info("Figure stored to {}.".format(savefig))
    elif savefig:
        log.info("Pass a file path as savefig to store the figure; nothing saved.")
    return fig1


def find_most_concordant_samp(sr1, sr2):
    """Find sampling parameter optimum by comparing two matched datasets.

    This function attempts to find a sampling parameter optimum by comparing two matched (control) datasets
    and finding the point at which their parameter values are most concordant, according to l2 distance over
    all genes and parameters.

    Conceptually, if the datasets are biologically similar (e.g., the same cell type), we expect their
    biophysical parameters to match. We can come up with an ad hoc estimate for the sampling parameters
    by _assuming_ they are identical for both datasets and finding the grid point that makes the
    biological parameters most similar.

    This typically works poorly: for two cell types from one experiment, prefer
    ``inference.find_shared_sampling_optimum([sr1, sr2])``, which minimises the
    summed (cell-weighted) divergence instead of assuming the biology is the same.

    Parameters
    ---------
    sr1: monod.inference.SearchResults
        first search result object.
    sr2: monod.inference.SearchResults
        second search result object.

    Returns
    -------
    samp_optimum: list of floats
        value of the technical noise parameters at the grid point.
    """
    discordance = ((sr1.param_estimates - sr2.param_estimates) ** 2).sum(2).sum(1)
    srt = np.argsort(discordance)
    samp_concordant_ind = srt[0]
    sr1.set_sampling_optimum(samp_concordant_ind)
    sr2.set_sampling_optimum(samp_concordant_ind)

    log.info(f"Optimum set to {sr1.samp_optimum[0]:.2f}, {sr1.samp_optimum[1]:.2f}.")
    return sr1.samp_optimum


def get_AIC_weights(sr_arr, sd):
    """Compute Akaike weights for several proposed models.

    This method computes the Akaike Information Criterion (AIC) weights according to the optimal
    physical and sampling parameters obtained for a single dataset (sd) under several models whose
    results are stored in sr_arr.

    For more reference, see Burnham and Anderson (2002).

    Parameters
    ---------
    sr_arr: list of monod.inference.SearchResults
        search result array to process.
    sd: monod.extract_data.SearchData
        the SearchData object.

    Returns
    -------
    w: np.ndarray
        Akaike weights corresponding to each model, an n_models x n_genes array.
    """

    AIC = np.asarray([aic_values(sr, sd) for sr in sr_arr])          # the convention of adata.var['AIC']
    min_AIC = AIC.min(0)
    normalization = np.exp(-(AIC - min_AIC) / 2).sum(0)
    w = np.exp(-(AIC - min_AIC) / 2) / normalization
    return w


def plot_AIC_weights(
    sr_arr,
    sd,
    models,
    ax1=None,
    meta=None,
    figsize=None,
    facecolor=aesthetics["hist_face_color"],
    facealpha=aesthetics["hist_face_alpha"],
    nbin=20,
    savefig=True,
):
    """Compute Akaike weights and plot their distributions.

    This function wraps get_AIC_weights and plots the distribution of weights for
    each model.

    Parameters
    ---------
    sr_arr: list of monod.inference.SearchResults
        search result array to process.
    sd: monod.extract_data.SearchData
        the SearchData object.
    models: list of str
        model names.
    ax1: list of matplotlib.axes.Axes or None, optional
        axes to plot into.
        if None, creates a new figure.
    meta: str or None, optional
        metadata suffix for figure file.
    figsize: None or tuple of floats, optional
        figure dimensions.
    facecolor: str or tuple, optional
        histogram face color in a matplotlib-compatible format.
    facealpha: float, optional
        histogram face alpha.
    nbin: int, optional
        number of bins used to construct the histogram.
    savefig: bool, optional
        whether to save the figure to disk.

    Returns
    -------
    w: np.ndarray
        Akaike weights corresponding to each model, an n_models x n_genes array.

    """
    w = get_AIC_weights(sr_arr, sd)

    if meta is None:
        meta = ""
    else:
        meta = "_" + meta

    n_models = w.shape[0]
    figsize = figsize or (4 * n_models, 4)

    if ax1 is None:
        fig1, ax1 = plt.subplots(nrows=1, ncols=n_models, figsize=figsize)
    else:
        fig1 = plt.gcf()
    ax1 = np.atleast_1d(ax1)

    for i in range(n_models):
        ax1[i].hist(w[i], bins=nbin, density=False, color=facecolor, alpha=facealpha)
        ax1[i].set_xlabel("AIC weight at MLE")
        ax1[i].set_ylabel("# genes")
        ax1[i].set_title(models[i])

    fig1.tight_layout()
    if isinstance(savefig, str):
        plt.savefig(savefig, dpi=450)
        log.info("Figure stored to {}.".format(savefig))
    elif savefig:
        log.info("Pass a file path as savefig to store the figure; nothing saved.")
    return w


def _param_labels(sr):
    """Parameter column names: the model's TeX labels minus the log prefix."""
    return [p.replace(r"\log_{10} ", "") for p in sr.model.get_log_name_str()]


def _write_fold_changes(table, labels, offs, resid, sign=1, suffix="", adjusted=False):
    """Record per-parameter fold changes ``offs + resid`` in ``table`` (an
    ``adata.var`` or a group table from ``adata.varm``).

    ``FC_<param><suffix>`` holds the fold change; with ``adjusted`` the
    offset-free residual goes to ``adj_FC_<param><suffix>`` as well.  ``sign``
    is -1 for the first dataset of a pair, since fold changes are second minus
    first.  Returns the last ``FC_`` column name written.
    """
    for pi, name in enumerate(labels):
        label = f"FC_{name}{suffix}"
        table[label] = sign * (offs[pi][0] + resid[pi])
        if adjusted:
            table[f"adj_FC_{name}{suffix}"] = sign * resid[pi]
    return label


def DE_parameters(adata1, adata2=None, modeltype="id",
    gene_filter_=None,
    discard_rejected=True,
    use_sigma=True,
    figsize=None,
    meta="12",
    viz=True,
    pval_thr=0.001,
    nit=10, save_cluster_FCs=True, pairs=None):
    """Differential expression on the fitted parameters.

    With two AnnDatas, compares their ``uns['search_result']`` fits
    (:func:`diffexp_pars`) and writes ``FC_<param>``, ``adj_FC_<param>`` to
    both ``.var`` tables and ``adj_<param>`` to the second.  With one AnnData
    holding a meK-Means ``uns['search_result_list']``, compares every pair of
    clusters and writes ``FC_<param>_(i, j)`` to its ``.var``.  With one
    AnnData from ``perform_inference(split_by=...)``, compares every pair of
    groups and writes ``FC_<param>_vs_<other>`` and ``adj_FC_<param>_vs_<other>``
    into both groups' tables in ``.varm`` (second group minus first, in the
    order the groups were fitted).  With several experiments (``batch_by``) the
    default pairs are the cell types within each experiment and each cell type
    across experiments, not the mixed pairs; ``pairs`` (a list of
    ``(group, group)`` names, keys or ``(cell type, experiment)`` tuples)
    chooses the comparisons explicitly.

    Returns the :func:`diffexp_pars` outputs, keyed by cluster-index pair or by
    group-name pair in the single-AnnData cases.
    """
    if modeltype != 'id':
        raise ValueError('Only modeltype="id" is implemented for DE_parameters.')
    fpi_kwargs = dict(modeltype=modeltype, gene_filter_=gene_filter_, discard_rejected=discard_rejected,
                      use_sigma=use_sigma, figsize=figsize, meta=meta, viz=viz, pval_thr=pval_thr, nit=nit)

    if adata2:
        if not (adata1.var_names == adata2.var_names).all():
            raise KeyError('Genes are not the same between datasets')
        sr1 = search_result_from_uns(adata1.uns['search_result'])
        sr2 = search_result_from_uns(adata2.uns['search_result'])
        gn, gf, offs, resid = diffexp_pars(sr1, sr2, **fpi_kwargs)

        labels = _param_labels(sr1)
        FC_label = _write_fold_changes(adata2.var, labels, offs, resid, adjusted=True)
        _write_fold_changes(adata1.var, labels, offs, resid, sign=-1, adjusted=True)
        for pi, name in enumerate(labels):
            adata2.var[f"adj_{name}"] = adata2.var[name] - offs[pi][0]

        print('Fold changes saved to each dataframe under .var, e.g. ' + FC_label)
        print('Adjusted parameter values (subtracting the average fold change) saved to the second dataframe under .var, e.g. adj_' + labels[-1])
        print('Adjusted fold changes (subtracting the average fold change) saved to each dataframe under .var, e.g. adj_' + FC_label)
        return gn, gf, offs, resid

    sr_list = search_result_list_from_uns(adata1.uns['search_result_list'])
    split = split_info(adata1)
    if pairs is not None:
        if split is None:
            raise ValueError("pairs are given by group name, which needs a split fit")
        pair_idx = [(_group_index(adata1, a), _group_index(adata1, b)) for a, b in pairs]
    else:
        pair_idx = _default_pairs(split, len(sr_list))
    gn, gf, offs, resid = {}, {}, {}, {}
    for i, j in pair_idx:
        sr1, sr2 = sr_list[i], sr_list[j]
        dict_key = (split['groups'][i], split['groups'][j]) if split is not None else (sr1.assigns, sr2.assigns)
        print('Groups:' if split is not None else 'Clusters:', dict_key)
        gn[dict_key], gf[dict_key], offs[dict_key], resid[dict_key] = diffexp_pars(sr1, sr2, **fpi_kwargs)
        plt.show()
        if not save_cluster_FCs:
            continue
        labels = _param_labels(sr1)
        if split is not None:
            key_i, key_j = split['keys'][i], split['keys'][j]
            tab_j = adata1.varm[key_j]
            FC_label = _write_fold_changes(tab_j, labels, offs[dict_key], resid[dict_key],
                                           suffix=f"_vs_{key_i}", adjusted=True)
            adata1.varm[key_j] = tab_j
            tab_i = adata1.varm[key_i]
            _write_fold_changes(tab_i, labels, offs[dict_key], resid[dict_key],
                                sign=-1, suffix=f"_vs_{key_j}", adjusted=True)
            adata1.varm[key_i] = tab_i
            print(f'Fold changes recorded in .varm[{key_j!r}] and .varm[{key_i!r}], e.g. {FC_label} '
                  f'(and adj_{FC_label} without the common offset); second group minus first.')
        else:
            FC_label = _write_fold_changes(adata1.var, labels, offs[dict_key], resid[dict_key],
                                           suffix=f"_{dict_key}")
            print('Fold changes between two clusters recorded in .var in form: ' + FC_label
                  + ', where the tuple (i,j) represents the fold change between cluster i and j.')
    return gn, gf, offs, resid


def _default_pairs(split, n):
    """Pairs of result indices to compare: every pair when the groups come from
    one experiment; with several experiments, the cell types within each
    experiment and each cell type across experiments (mixed pairs would
    confound the two)."""
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if split is None or split['batch_column'] is None or split['column'] is None:
        return all_pairs
    return [(i, j) for i, j in all_pairs
            if split['batch'][i] == split['batch'][j] or split['split'][i] == split['split'][j]]


def diffexp_pars(
    sr1,
    sr2,
    modeltype="id",
    gene_filter_=None,
    discard_rejected=True,
    use_sigma=True,
    figsize=None,
    meta="12",
    viz=True,
    pval_thr=0.001,
    nit=10,
):
    """Use fixed-point iteration to find differentially expressed genes without replicates.

    This function uses the optimal physical and sampling parameters obtained for a pair of datasets
    to attempt to identify sources of differential expression under a model of transcription,
    i.e., calling patterns of DE-theta, where theta is a biophysical parameter identifiable at steady state.
    Specifically, it uses a fixed-point iteration procedure to distinguish a set of genes with
    Gaussian aleatory variation from a set with systematic, high-magnitude deviations between the datasets.

    The outputs of plot_params_for_pair typically show similar parameter values for paired datasets, with
    the following structure: many genes lie around the identity line (potentially containing a small bias),
    a few genes are quite far from the identity line. This function attempts to find the divergent genes,
    which show meaningful deviations in parameter values, using a Gaussian model for non-systematic variation
    between datasets.

    By fitting a Gaussian model, we can identify genes that diverge from identity and tentatively propose that
    they are differentially expressed. Excluding them from changes the dataset used to fit the Gaussian, so we need to
    repeat the procedure several times until convergence.

    Parameters
    ---------
    sr1: monod.inference.SearchResults
        first search result object.
    sr2: monod.inference.SearchResults
        second search result object.
    modeltype: str, optional
        If 'id', the relationship between datasets' parameters is assumed to be x + b = y; n_model_pars = 1.
        If 'lin', the relationship between datasets' parameters is assumed to be ax + b = y; n_model_pars = 2.
    gene_filter_: None or np.ndarray, optional
        If None, use all genes.
        If a boolean or integer filter, use the filtered gene subset.
    discard_rejected: bool, optional
         whether to omit genes in the rejected_genes attribute from the analysis.
    use_sigma: bool, optional
        whether to use the inferred standard error of MLEs to fit the aleatory variation model.
    figsize: None or tuple of floats, optional
        figure dimensions.
    meta: str, optional
        metadata suffix for figure file.
    viz: bool, optional
        whether to visualize the histograms of residuals.
    pval_thr: float, optional
        the p-value threshold to use for the Z-test. If the Z-value of the gene's parameter deviation
        corresponds to a p-value lower than this threshold, the gene is considered differentially expressed
        with respect to this parameter.
    nit: int, optional
        number of rounds of fixed-point iteration to perform.

    Returns
    -------
    gn: list of lists of str
        list of size n_phys_pars. each entry contains names of genes identified as DE with respect
        to each parameter.
    gf: bool np.ndarray
        an n_phys_pars x n_genes array. if True, the gene has been identified as DE with respect to
        the parameter.
    offs: float np.ndarray
        an n_phys_pars x n_model_pars array. contains (b) or (b,a) for each physical parameter.
        b should typically be fairly small (under 0.5).
    resid: float np.ndarray
        an n_phys_pars x n_genes array. contains residuals for each parameter under the final
        fit statistical model.
    """
    num_params = sr1.sp.n_phys_pars
    if viz:
        figsize = figsize or (4 * num_params, 4)
        fig1, ax1 = plt.subplots(nrows=1, ncols=num_params, figsize=figsize)
        ax1 = np.atleast_1d(ax1)
    if use_sigma and not (hasattr(sr1, "sigma") and hasattr(sr2, "sigma")):
        log.warning("use_sigma=True but sigma has not been computed (run compute_sigma); falling back to use_sigma=False.")
        use_sigma = False

    gene_filter = sr1.get_bool_filt(gene_filter_, discard_rejected=False)

    if discard_rejected:  # default
        filt_rej1 = sr1.get_bool_filt(gene_filter_, discard_rejected=True)
        filt_rej2 = sr2.get_bool_filt(gene_filter_, discard_rejected=True)
        filt_rej = np.logical_and(filt_rej1, filt_rej2)
        gene_filter = np.logical_and(gene_filter, filt_rej)

    gn = []
    gf = []
    offs = []
    resid = []

    parnames = sr1.model.get_log_name_str()
    for i in range(num_params):

        if viz:
            ax = ax1[i]
        else:
            ax = None

        sig1 = sr1.sigma[gene_filter, i] if use_sigma else None
        sig2 = sr2.sigma[gene_filter, i] if use_sigma else None
        gf_, offs_, resid_, pval_ = diffexp_fpi(
            sr1.phys_optimum[gene_filter, i],
            sr2.phys_optimum[gene_filter, i],
            parnames[i],
            modeltype=modeltype,
            ax1=ax,
            s1=sig1,
            s2=sig2,
            nit=nit,
            viz=viz,
            pval_thr=pval_thr,
        )
        resid_arr = np.full(sr1.n_genes, np.nan)
        resid_arr[gene_filter] = resid_

        gf__ = np.zeros(sr1.n_genes, dtype=bool)
        gf__[np.flatnonzero(gene_filter)[gf_]] = True
        gn_ = np.asarray(sr1.gene_names)[gf__]   # gene_names may be a list (Rust SearchData)
        gn.append(gn_)
        gf.append(gf__)
        offs.append(offs_)
        resid.append(resid_arr)

        if i == 0 and viz:
            ax.legend()
    if viz:
        fig1.tight_layout()
        # plt.savefig(fig_string, dpi=450)
        # log.info("Figure stored to {}.".format(fig_string))

    return gn, np.asarray(gf), np.asarray(offs), np.asarray(resid)

# def diffexp_pars_plot(gn, gf, offs, resid)


def _fit_line_eiv(x, y, sx=None, sy=None, fit_slope=True, b0=1.0, tol=1e-12, maxiter=500):
    """Straight-line fit with errors in both coordinates (errors-in-variables).

    Minimises sum_i [(y_i - a - b x_i)^2 / (sy_i^2 + b^2 sx_i^2)], the weighted
    orthogonal-distance objective for uncorrelated per-point errors. With
    ``fit_slope=False`` the slope is fixed at 1 and the offset has the closed form
    of an inverse-variance weighted mean of ``y - x``; otherwise the slope is found
    by York's fixed-point iteration (York et al. 2004), which for unit errors reduces
    to total least squares. Absent errors are taken as unit (``None`` for both) or
    zero (``None`` for one of them); zero errors are floored at 1e-12. With fewer
    than two points, or when all ``x`` coincide, the slope is not determined and
    stays at ``b0``.

    Returns ``[a]`` or ``[a, b]``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if sx is None and sy is None:
        vx = np.ones_like(x)
        vy = np.ones_like(y)
    else:
        tiny = 1e-12
        vx = np.maximum(np.zeros_like(x) if sx is None else np.asarray(sx, dtype=float), tiny) ** 2
        vy = np.maximum(np.zeros_like(y) if sy is None else np.asarray(sy, dtype=float), tiny) ** 2
    if not fit_slope:
        w = 1.0 / (vx + vy)
        return np.array([np.sum(w * (y - x)) / np.sum(w)])
    b = float(b0)
    for _ in range(maxiter):
        W = 1.0 / (vy + b * b * vx)
        xbar = np.sum(W * x) / np.sum(W)
        ybar = np.sum(W * y) / np.sum(W)
        U = x - xbar
        V = y - ybar
        beta = W * (U * vy + b * V * vx)
        denom = np.sum(W * beta * U)
        if not (np.isfinite(denom) and denom != 0.0):
            break                      # slope undetermined (fewer than two distinct x)
        b_new = np.sum(W * beta * V) / denom
        done = abs(b_new - b) <= tol * max(1.0, abs(b))
        b = b_new
        if done:
            break
    W = 1.0 / (vy + b * b * vx)
    a = np.sum(W * y) / np.sum(W) - b * np.sum(W * x) / np.sum(W)
    return np.array([a, b])


def diffexp_fpi(
    m1,
    m2,
    parname=None,
    modeltype="id",
    ax1=None,
    s1=None,
    s2=None,
    nit=10,
    pval_thr=0.001,
    viz=True,
):
    """Use fixed-point interation on a set of generic parameter values to debias them and call outliers.

    This function uses the optimal physical and sampling parameters obtained for a pair of datasets
    to attempt to identify differentially expressed genes under a model of transcription, for a single stest statistic.
    
    Specifically, it uses a fixed-point iteration (FPI) procedure to distinguish a set of genes with
    Gaussian aleatory variation from a set with systematic, high-magnitude deviations between the datasets.

    This is a helper function that has no access to the raw data, only generic arrays of floats, and can
    be used for debiasing more generally.

    The fit uses orthogonal distance regression to impose symmetry between datasets.
    If s1 and s2 are provided, the fit uses these error estimates to penalize higher-uncertainty data points.

    Parameters
    ----------
    m1: float np.ndarray
        parameter estimates from the first search result object.
    m2: float np.ndarray
        parameter estimates from the second search result object.
    parname: float or None, optional
        if float, the name of the parameter being considered. Used only for visualization.
    modeltype: str, optional
        If 'id', the relationship between datasets' parameters is assumed to be x + b = y; n_model_pars = 1.
        If 'lin', the relationship between datasets' parameters is assumed to be ax + b = y; n_model_pars = 2.
    ax1: matplotlib.axes.Axes or None, optional
        axes to plot into.
    s1: float np.ndarray or None, optional
        standard errors associated with values in m1.
    s2: float np.ndarray or None, optional
        standard errors associated with values in m2.
    nit: int, optional
        number of rounds of fixed-point iteration to perform.
    pval_thr: float, optional
        the p-value threshold to use for the Z-test. If the Z-value of the gene's parameter deviation
        corresponds to a p-value lower than this threshold, the gene is considered differentially expressed
        with respect to this parameter.
    viz: bool, optional
        whether to visualize the histogram of residuals.

    Returns
    -------
    gf: bool np.ndarray
        if True, the gene has been identified as DE with respect to the current parameter.
    beta: float np.ndarray
        statistical model parameters: the offset (a), followed by the slope (b) when
        modeltype is "lin". The offset should typically be fairly small (under 0.5).
    resid: float np.ndarray
        residuals for each gene under the final fit statistical model.
    pval: float np.ndarray
        residuals' p-values under the Z-test.
    """
    fs = 12
    gf = np.ones(len(m1), dtype=bool)
    # x = np.linspace(xl[0], xl[1], 100)
    fitlaw = scipy.stats.norm

    if modeltype not in ("id", "lin"):
        raise ValueError(f"modeltype must be 'id' or 'lin', got {modeltype!r}.")

    for j in range(nit):
        beta = _fit_line_eiv(m1[gf], m2[gf],
                             None if s1 is None else s1[gf],
                             None if s2 is None else s2[gf],
                             fit_slope=(modeltype == "lin"))
        if modeltype == "id":
            resid = m2 - m1 - beta[0]
        else:
            resid = m2 - m1 * beta[1] - beta[0]

        fitparams = fitlaw.fit(resid[gf])

        x = np.linspace(resid.min(), resid.max(), 100)
        p = fitlaw.pdf(x, *fitparams)
        if j == 0 and viz:
            ax1.plot(
                x,
                p,
                linestyle=aesthetics["init_fit_linestyle"],
                linewidth=aesthetics["init_fit_linewidth"],
                color=aesthetics["hist_fit_color_2"],
                label="Initial fit",
            )

        z = (resid - fitparams[0]) / fitparams[1]
        pval = scipy.stats.norm.sf(np.abs(z)) * 2
        gf = np.logical_not(pval < pval_thr)
        if j == (nit - 1) and viz:
            n, bins = np.histogram(resid, 200, density=True)
            binc = np.diff(bins) / 2 + bins[:-1]
            ax1.bar(
                binc,
                n,
                width=np.diff(bins),
                color=aesthetics["hist_face_color"],
                align="center",
                label="Putative static genes",
            )
            ax1.plot(
                x,
                p,
                linestyle=aesthetics["linestyle"],
                linewidth=2,
                color=aesthetics["hist_fit_color"],
                label="FPI fit",
            )
            y_, _ = np.histogram(resid[~gf], bins=bins, density=True)
            y_ *= (~gf).mean()
            ax1.bar(
                binc,
                y_,
                width=np.diff(bins),
                color=aesthetics["diffreg_gene_color"],
                align="center",
                label="Putative DE genes",
            )
    if viz:
        ax1.set_xlabel(parname + " residual", fontsize=fs)
        ax1.set_ylabel("Density", fontsize=fs)
    gf = np.logical_not(gf)
    return gf, beta, resid, pval


def compare_gene_distributions(
    sr_arr,
    sd_arr,
    sz=(5, 5),
    figsize=(10, 10),
    marg="mature",
    logscale=False,
    title=True,
    genes_to_plot=None,
):
    """Plot marginal histograms for multiple datasets.

    This function is analogous to the SearchResults method plot_gene_distributions, but it
    plots the marginal histograms for multiple datasets at a time (2 supported at this time).
    The intended use case is to identify a set of DE genes of interest, pass them in as
    genes_to_plot, and inspect the difference between the distributions.

    By default, dataset 1 is plotted in blue and dataset 2 in red (``hist_*_color_1`` and
    ``hist_*_color_2`` in monod.plot_aesthetics.aesthetics).

    Parameters
    ----------
    sr_arr: list of monod.inference.SearchResults
        search results to visualize as lines..
    sd_arr: list of monod.extract_data.SearchData
        search data to visualize as histograms.
    sz: tuple of ints, optional
        dimensions of the figure subplot grid.
    figsize: tuple of floats, optional
        figure dimensions.
    marg: str, optional
        the modality to plot; joint distributions are not supported here.
    logscale: bool, optional
        whether to plot probabilities or log-probabilities.
    title: bool, optional
        whether to report the gene name in each subplot title.
    genes_to_plot: bool or int np.ndarray or None, optional
        if array, which genes to plot.
        if None, plot by internal order.

    """
    if marg == "joint":
        raise ValueError("compare_gene_distributions plots marginals; joint comparison is not supported.")
    modalities = list(sr_arr[0].model.model_modalities)
    lind = _resolve_marg(marg, modalities)

    fig1, ax1 = plt.subplots(nrows=sz[0], ncols=sz[1], figsize=figsize)
    axes = np.atleast_1d(ax1).ravel()
    genes_to_plot = _genes_to_plot(genes_to_plot, sr_arr[0].n_genes, np.prod(sz))

    for ax, i_ in zip(axes, genes_to_plot):
        for j, (sr, sd) in enumerate(zip(sr_arr, sd_arr)):
            plot_hist_and_fit(
                ax, sd, i_, sr.gene_pss(sd, i_, marg), marg,
                modalities=modalities,
                facealpha=aesthetics["hist_face_alpha_" + str(j + 1)],
                facecolor=aesthetics["hist_face_color_" + str(j + 1)],
                fitcolor=aesthetics["hist_fit_color_" + str(j + 1)],
            )
        xlim = max(int(sd.M[lind, i_]) for sd in sd_arr)
        ax.set_xlim([-0.5, max(xlim - 8, 1)])
        if logscale:
            ax.set_yscale("log")
        if title:
            ax.set_title(sr_arr[0].gene_names[i_], fontdict={"fontsize": 9})
        ax.set_xticks([])
        ax.set_yticks([])
    fig1.tight_layout(pad=0.02)
    return fig1


def _normalize_count_matrix(X, sizefactor="pf", lognormalize=True, pcount=0, logbase=2):
    """Size-normalise (and optionally log-transform) a genes x cells count matrix.

    sizefactor: 'pf' scales every cell to the mean total count (proportional
    fitting); a float scales to that total (e.g. 1e4 for cp10k); None skips
    depth normalisation.
    """
    X = np.asarray(X, dtype=float)
    if sizefactor is not None:
        totals = X.sum(0)
        target = totals.mean() if sizefactor == "pf" else float(sizefactor)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(totals > 0, target / totals, 0.0)
        X = X * scale[None, :]
    if lognormalize:
        X = np.log(X + pcount + 1) / np.log(logbase)
    return X


def _volcano(ax, fc, p, gf):
    """Volcano plot: log2 fold change against -log10 p, DE genes in red."""
    pv = -np.log10(p)
    ax.scatter(fc[gf], pv[gf], 5, "r")
    ax.scatter(fc[~gf], pv[~gf], 3, "darkgray")
    ax.set_xlabel(r"Fold change ($\log_2$)")
    ax.set_ylabel(r"$-\log_{10} p$")


def diffexp_mean(
    sd1,
    sd2,
    sizefactor="pf",
    lognormalize=True,
    pcount=0,
    pval_thr=0.001,
    method="ttest",
    bonferroni=True,
    modeltype="lin",
    viz=True,
    knee_thr=None,
    fc_thr=1,
    ax1=None,
    viz_resid=False,
):
    """Identify DE genes through the means.

    This function attempts to identify differentially expressed genes using a simple comparison of
    the means of gene-specific count distributions, i.e., DE-theta using the normalized mean as a test statistic.

    Two methods are implemented.
    The first, 'ttest', simply applies a t-test to each gene in a normalized/transformed count matrix.
    The second, 'logmeanfpi' or 'meanlogfpi', applies the fixed-point iteration to fit a model of Gaussian
        aleatory variation and call outliers.

    Parameters
    ---------
    sd1: monod.extract_data.SearchData
        first search data object.
    sd2: monod.extract_data.SearchData
        second search data object.
    sizefactor: str or float or None, optional
        size factor to use in normalization.
        if 'pf': Proportional fitting; set the size of each cell to the mean size.
        if a float: use this value (e.g., 1e4 for cp10k).
        if None: do not perform size/depth normalization.
    lognormalize: bool, optional
        whether to apply a log-transformation to the data.
    pcount: float, optional
        pseudocount to use for size-normalization.
    pval_thr: float, optional
        the p-value threshold to use for the Z-test or t-test. If the p-value is lower than
        this threshold, the gene is considered differentially expressed with respect to the mean.
    method: str, optional
        if 'ttest', use the t-test on transformed data.
        If 'logmeanfpi', use the fixed-point iteration procedure on the distribution of log-means.
        If 'meanlogfpi', use the FPI procedure on the distribution of means of log+1 counts.
    bonferroni: bool, optional
        whether to use a multiple comparison correction for the t-test.
    modeltype: str, optional
        statistical model for FPI
        If 'id', the relationship between datasets' means is assumed to be x + b = y; n_model_pars = 1.
        If 'lin', the relationship between datasets' means is assumed to be ax + b = y; n_model_pars = 2.
    viz: bool, optional
        whether to visualize t-test or FPI volcano plots.
    knee_thr: float or None, optional
        if not None, use a subset of cells with more than knee_thr total counts.
        this threshold is compared to the total number of molecules of the genes stored in sd1 and sd2.
    fc_thr: float, optional
        log2 fold change threshold for the t-test.
    ax1: matplotlib.axes.Axes or None, optional
        axes to plot into.
    viz_resid
        whether to visualize FPI residual plots.

    Returns
    -------
    gf: bool np.ndarray
        if True, the gene has been identified as differentially expressed.
    fc: float np.ndarray
        effect sizes for all genes.

    """

    # layers are stored n_layers x n_cells x n_genes; work in genes x cells here.
    s1 = np.asarray(sd1.layers[1]).T.copy()
    s2 = np.asarray(sd2.layers[1]).T.copy()

    if (viz or viz_resid) and ax1 is None:
        fig1, ax1 = plt.subplots(1, 1)

    if method == "ttest":
        if knee_thr is not None:
            s1 = s1[:, knee_plot(s1, thr=knee_thr[0])]
            s2 = s2[:, knee_plot(s2, thr=knee_thr[1])]
        if bonferroni:
            pval_thr /= sd1.n_genes
        s1 = _normalize_count_matrix(s1, sizefactor=sizefactor, lognormalize=lognormalize, pcount=pcount, logbase=2)
        s2 = _normalize_count_matrix(s2, sizefactor=sizefactor, lognormalize=lognormalize, pcount=pcount, logbase=2)
        p = np.asarray(scipy.stats.ttest_ind(s1, s2, axis=1, equal_var=False).pvalue)
        if lognormalize:
            fc = s2.mean(1) - s1.mean(1)
        else:
            fc = np.log2(s2 + 1).mean(1) - np.log2(s1 + 1).mean(1)
        gf = (p < pval_thr) & (np.abs(fc) > fc_thr)
    elif method in ("logmeanfpi", "meanlogfpi"):
        if method == "logmeanfpi":
            m1, m2 = np.log2(s1.mean(1)), np.log2(s2.mean(1))
        else:
            m1, m2 = np.log2(s1 + 1).mean(1), np.log2(s2 + 1).mean(1)
        gf, _, _, p = diffexp_fpi(m1, m2, "Spliced mean", modeltype=modeltype, ax1=ax1,
                                  nit=30, viz=viz_resid, pval_thr=pval_thr)
        fc = m2 - m1
    else:
        raise ValueError(f"method must be 'ttest', 'logmeanfpi' or 'meanlogfpi', got {method!r}.")
    if viz:
        _volcano(ax1, fc, p, gf)
    return gf, fc
