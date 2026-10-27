"""From ATAC peaks to the Chromatin model's ``configuration`` layer.

The Chromatin model (:class:`monod.cme_toolbox.CMEModel` with
``bio_model="Chromatin"``) reads three layers per gene: ``unspliced``,
``spliced`` and ``configuration``, the last an integer per cell encoding which
of the gene's ``n_sites`` chromatin sites are open.  This module builds that
layer from a peak-by-cell ATAC matrix and a gene annotation:

1. **Assign peaks to genes** (:func:`assign_peaks`) by one of three rules,
   each a hypothesis about which accessible sites drive a gene:

   - ``"tss_window"``: peaks overlapping a window around the transcription
     start site;
   - ``"gene_body"``: peaks overlapping the gene from its promoter to its end;
   - ``"loci_promoter"``: the loci of Felce, Gorin and Pachter (2024), runs of
     peaks closer than ``locus_gap`` to each other, taken whole when a locus
     overlaps the promoter.

   A gene keeps the ``n_sites`` peaks nearest its start site, ordered along
   the direction of transcription (site 0 is the most upstream, the least
   significant bit of the configuration); genes with fewer peaks are left
   out.
2. **Binarise** each site at ``threshold`` fragments (one, by default: the
   paper's choice, and the right one at multiome depth).
3. **Encode** the configuration ``sum_j open_j 2**j``
   (:func:`configuration_layer`).

:func:`build_multiome_adata` does all three for an RNA AnnData with
``unspliced`` and ``spliced`` layers and an ATAC AnnData of fragment counts
per peak, and returns the AnnData that :func:`monod.extract_data.extract_data`
and :func:`monod.inference.perform_inference` take.  The assignment table is
kept in ``uns["chromatin_sites"]`` so a fitted ``k_on`` can be read back to a
peak.
"""
from __future__ import annotations

import gzip
import re
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

ASSIGNMENT_RULES = ("tss_window", "gene_body", "loci_promoter")
_PEAK_NAME = re.compile(r"^(?P<chrom>[^:\s]+?)[:\-](?P<start>\d+)[\-:](?P<end>\d+)$")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def peaks_from_var(var: pd.DataFrame) -> pd.DataFrame:
    """Peak coordinates from an ATAC AnnData's ``var``: columns ``chrom``,
    ``start``, ``end`` (or ``Chromosome``/``Start``/``End``, ``seqnames``),
    else parsed from names such as ``chr1:9992-10688`` or ``chr1-9992-10688``.
    Returns a frame indexed like ``var`` with ``chrom``, ``start``, ``end``."""
    cols = {c.lower(): c for c in var.columns}
    chrom = next((cols[k] for k in ("chrom", "chromosome", "seqnames", "chr") if k in cols), None)
    start = next((cols[k] for k in ("start",) if k in cols), None)
    end = next((cols[k] for k in ("end",) if k in cols), None)
    if chrom is not None and start is not None and end is not None:
        out = pd.DataFrame({"chrom": var[chrom].astype(str).values,
                            "start": var[start].astype(np.int64).values,
                            "end": var[end].astype(np.int64).values}, index=var.index)
    else:
        rows = []
        for name in var.index:
            m = _PEAK_NAME.match(str(name))
            if m is None:
                raise ValueError(f"cannot parse peak coordinates from {name!r}; give chrom/start/end columns")
            rows.append((m["chrom"], int(m["start"]), int(m["end"])))
        out = pd.DataFrame(rows, columns=["chrom", "start", "end"], index=var.index)
    if (out["end"] <= out["start"]).any():
        raise ValueError("every peak needs end > start")
    return out


def read_gtf_genes(path: str, feature: str = "gene", name_attrs: Sequence[str] = ("gene_name", "gene_id")) -> pd.DataFrame:
    """Gene coordinates from a GTF (plain or gzipped): one row per ``feature``
    record with ``gene``, ``chrom``, ``start``, ``end``, ``strand``; the name
    is the first of ``name_attrs`` present."""
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != feature:
                continue
            attrs = dict(re.findall(r'(\S+)\s+"([^"]*)"', f[8]))
            name = next((attrs[a] for a in name_attrs if a in attrs), None)
            if name is None:
                continue
            rows.append((name, f[0], int(f[3]) - 1, int(f[4]), f[6] if f[6] in "+-" else "+"))   # GTF is 1-based, closed
    return pd.DataFrame(rows, columns=["gene", "chrom", "start", "end", "strand"])


def genes_table(genes) -> pd.DataFrame:
    """Normalise a gene annotation (a frame with ``gene``/``name``, ``chrom``,
    ``start``, ``end`` and optional ``strand``, or a GTF path) to the columns
    :func:`assign_peaks` uses, adding ``tss``."""
    if isinstance(genes, (str, bytes)):
        genes = read_gtf_genes(genes)
    g = pd.DataFrame(genes).copy()
    cols = {c.lower(): c for c in g.columns}
    if "gene" not in cols:
        name = next((cols[k] for k in ("name", "gene_name", "gene_id") if k in cols), None)
        if name is None:
            if g.index.name or not isinstance(g.index, pd.RangeIndex):
                g["gene"] = g.index.astype(str)
            else:
                raise ValueError("gene annotation needs a 'gene' column or a named index")
        else:
            g["gene"] = g[name].astype(str)
    g = g.rename(columns={cols.get("chromosome", "chrom"): "chrom", cols.get("seqnames", "chrom"): "chrom"})
    g["chrom"] = g["chrom"].astype(str)
    g["start"] = g["start"].astype(np.int64)
    g["end"] = g["end"].astype(np.int64)
    if "strand" not in g.columns:
        g["strand"] = "+"
    g["strand"] = g["strand"].where(g["strand"].isin(["+", "-"]), "+")
    g["tss"] = np.where(g["strand"] == "+", g["start"], g["end"])
    return g[["gene", "chrom", "start", "end", "strand", "tss"]].drop_duplicates("gene").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------

def _overlapping(starts: np.ndarray, ends: np.ndarray, a: int, b: int) -> np.ndarray:
    """Indices (into the sorted arrays) of intervals overlapping [a, b)."""
    return np.where((ends > a) & (starts < b))[0]


def _loci(starts: np.ndarray, ends: np.ndarray, gap: int) -> np.ndarray:
    """Locus label per peak (sorted by start): a new locus starts where the
    gap to the running end of the previous peaks exceeds ``gap``."""
    labels = np.zeros(starts.size, dtype=np.int64)
    run_end = -np.inf
    k = -1
    for i in range(starts.size):
        if starts[i] - run_end > gap:
            k += 1
        labels[i] = k
        run_end = max(run_end, ends[i])
    return labels


def assign_peaks(peaks: pd.DataFrame, genes, rule: str = "tss_window", n_sites: Optional[int] = None,
                 window: int = 5000, upstream: int = 2000, downstream: int = 500, locus_gap: int = 1000,
                 max_sites: int = 6) -> pd.DataFrame:
    """Which peaks are a gene's chromatin sites, by ``rule`` (see the module
    docstring).  ``window`` is the half-width around the start site for
    ``"tss_window"``; ``upstream``/``downstream`` define the promoter for
    ``"gene_body"`` (where it extends the gene) and ``"loci_promoter"``;
    ``locus_gap`` the paper's peak spacing.  Of the candidates, the
    ``n_sites`` (else at most ``max_sites``) nearest the start site are kept;
    with ``n_sites`` given, genes with fewer candidates are dropped.

    Returns one row per (gene, site): ``gene``, ``site`` (0 = most upstream
    along transcription), ``peak`` (the peak's index label), ``chrom``,
    ``start``, ``end``, ``distance`` (to the start site, 0 if it lies inside).
    """
    if rule not in ASSIGNMENT_RULES:
        raise ValueError(f"rule must be one of {ASSIGNMENT_RULES}, got {rule!r}")
    peaks = peaks_from_var(peaks) if not {"chrom", "start", "end"} <= set(peaks.columns) else peaks
    g = genes_table(genes)
    keep = n_sites if n_sites is not None else max_sites
    rows = []
    for chrom, gc in g.groupby("chrom", sort=False):
        pc = peaks[peaks["chrom"] == chrom]
        if pc.empty:
            continue
        order = np.argsort(pc["start"].values, kind="stable")
        pc = pc.iloc[order]
        starts, ends = pc["start"].values, pc["end"].values
        labels = _loci(starts, ends, locus_gap) if rule == "loci_promoter" else None
        for gene_row in gc.itertuples(index=False):
            tss, plus = int(gene_row.tss), gene_row.strand == "+"
            if rule == "tss_window":
                a, b = tss - window, tss + window
                idx = _overlapping(starts, ends, a, b)
            elif rule == "gene_body":
                if plus:
                    a, b = gene_row.start - upstream, gene_row.end
                else:
                    a, b = gene_row.start, gene_row.end + upstream
                idx = _overlapping(starts, ends, a, b)
            else:
                a, b = (tss - upstream, tss + downstream) if plus else (tss - downstream, tss + upstream)
                hit = _overlapping(starts, ends, a, b)
                idx = np.where(np.isin(labels, labels[hit]))[0] if hit.size else hit
            if idx.size == 0 or (n_sites is not None and idx.size < n_sites):
                continue
            dist = np.where((starts[idx] <= tss) & (ends[idx] > tss), 0,
                            np.minimum(np.abs(starts[idx] - tss), np.abs(ends[idx] - 1 - tss)))
            if idx.size > keep:
                nearest = np.argsort(dist, kind="stable")[:keep]
                idx, dist = idx[nearest], dist[nearest]
            along = np.argsort(starts[idx], kind="stable")
            if not plus:
                along = along[::-1]
            for site, k in enumerate(along):
                rows.append((gene_row.gene, site, pc.index[idx[k]], chrom, int(starts[idx[k]]), int(ends[idx[k]]), int(dist[k])))
    out = pd.DataFrame(rows, columns=["gene", "site", "peak", "chrom", "start", "end", "distance"])
    return out.sort_values(["gene", "site"], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# The configuration layer
# ---------------------------------------------------------------------------

def configuration_layer(atac, assignments: pd.DataFrame, n_sites: int, threshold: int = 1):
    """The configuration of every cell for every gene with exactly ``n_sites``
    assigned sites.  ``atac`` is a cells-by-peaks AnnData (or a ``(X, peak
    names)`` pair) of fragment counts; a site is open at ``threshold`` or more
    fragments.  Returns ``(genes, config)``: the gene names and an
    ``(n_cells, n_genes)`` int64 array with site ``j`` as bit ``j``."""
    if hasattr(atac, "var_names"):
        X, names = atac.X, list(atac.var_names)
    else:
        X, names = atac
        names = list(names)
    pos = {n: i for i, n in enumerate(names)}
    counts = assignments.groupby("gene")["site"].count()
    genes = [gname for gname, c in counts.items() if c == n_sites]
    if not genes:
        return [], np.zeros((X.shape[0], 0), dtype=np.int64)
    cols = []
    for gname in genes:
        rows = assignments[assignments["gene"] == gname].sort_values("site")
        cols.extend(pos[p] for p in rows["peak"])
    sub = X[:, cols]
    sub = sub.toarray() if hasattr(sub, "toarray") else np.asarray(sub)
    open_ = (sub >= threshold).astype(np.int64).reshape(X.shape[0], len(genes), n_sites)
    weights = (1 << np.arange(n_sites)).astype(np.int64)
    return genes, open_ @ weights


def build_multiome_adata(rna, atac, genes, rule: str = "tss_window", n_sites: int = 2, threshold: int = 1,
                         layers: Sequence[str] = ("unspliced", "spliced"), min_open_fraction: float = 0.0,
                         **rule_kwargs):
    """The AnnData the Chromatin model fits, from an RNA AnnData (cells by
    genes, with ``layers``) and an ATAC AnnData (cells by peaks, fragment
    counts, coordinates in ``var`` or in the peak names).  Cells are the
    intersection of the two, in the RNA's order; genes are those present in
    the RNA data with exactly ``n_sites`` assigned peaks under ``rule`` (see
    :func:`assign_peaks` for ``rule_kwargs``) and, if ``min_open_fraction``
    is set, every site open in at least that fraction of cells.  The result
    has the RNA layers, an integer ``configuration`` layer, ``X`` equal to the
    spliced layer, the assignment table in ``uns["chromatin_sites"]`` and
    the settings in ``uns["chromatin_data"]``."""
    import anndata as ad
    import scipy.sparse as sp
    cells = [c for c in rna.obs_names if c in set(atac.obs_names)]
    if not cells:
        raise ValueError("the RNA and ATAC data share no cell names")
    rna_c = rna[cells]
    atac_c = atac[cells]
    assignments = assign_peaks(peaks_from_var(atac.var), genes, rule=rule, n_sites=n_sites, **rule_kwargs)
    assignments = assignments[assignments["gene"].isin(set(rna.var_names))]
    gene_names, config = configuration_layer(atac_c, assignments, n_sites, threshold)
    if min_open_fraction > 0 and gene_names:
        bits = (config[:, :, None] >> np.arange(n_sites)) & 1                      # (cells, genes, sites)
        ok = (bits.mean(axis=0) >= min_open_fraction).all(axis=1)
        gene_names = [g for g, k in zip(gene_names, ok) if k]
        config = config[:, ok]
    if not gene_names:
        raise ValueError(f"no gene has {n_sites} assigned peaks under rule {rule!r}")
    rna_g = rna_c[:, gene_names]
    new_layers = {}
    for name in layers:
        if name not in rna.layers:
            raise ValueError(f"the RNA data has no layer {name!r}")
        L = rna_g.layers[name]
        new_layers[name] = L.copy() if sp.issparse(L) else np.asarray(L).copy()
    new_layers["configuration"] = config.astype(np.int64)
    X = new_layers[layers[-1]]
    out = ad.AnnData(X=X.copy() if sp.issparse(X) else np.asarray(X, dtype=float).copy(), layers=new_layers,
                     obs=rna_g.obs.copy(), var=rna_g.var.copy())
    out.uns["chromatin_sites"] = assignments[assignments["gene"].isin(gene_names)].reset_index(drop=True)
    out.uns["chromatin_data"] = dict(rule=rule, n_sites=int(n_sites), threshold=int(threshold),
                                     **{k: (int(v) if isinstance(v, (int, np.integer)) else v) for k, v in rule_kwargs.items()})
    return out


def site_summary(adata) -> pd.DataFrame:
    """Per gene and site: the peak and the fraction of cells in which it is open."""
    sites = adata.uns["chromatin_sites"]
    n_sites = int(adata.uns["chromatin_data"]["n_sites"])
    config = np.asarray(adata.layers["configuration"])
    rows = []
    for j, gname in enumerate(adata.var_names):
        bits = (config[:, j][:, None] >> np.arange(n_sites)) & 1
        for s in range(n_sites):
            row = sites[(sites["gene"] == gname) & (sites["site"] == s)].iloc[0]
            rows.append((gname, s, row["peak"], row["chrom"], int(row["start"]), int(row["end"]), float(bits[:, s].mean())))
    return pd.DataFrame(rows, columns=["gene", "site", "peak", "chrom", "start", "end", "open_fraction"])
