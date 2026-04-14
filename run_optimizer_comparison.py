"""
Optimizer concordance: scipy L-BFGS-B vs Rust L-BFGS-B.

Fixed data path (Rust searchdata_from_h5ad), vary only the optimizer.
Compares results at EACH grid point (not just the phys_optimum) to get
full picture of optimizer landscape.

Key question: of gene-gp pairs where params differ >0.1 log10, how many
have KLD also differing (genuine failure) vs similar KLD (flat landscape)?
"""
import gc, sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'monod'))

import monod_core as _mc
from cme_toolbox import CMEModel
from inference import InferenceParameters

H5AD_PATH = 'example_h5ad/processed_pbmc_10k_raw.h5ad'
MODEL     = CMEModel('Bursty', 'Poisson')
GRIDSIZE  = [3, 4]   # 12 grid points
N_GENES   = 50
N_ITER    = 200

import anndata as ad, scipy.sparse as _ssp
_adata = ad.read_h5ad(H5AD_PATH)
_s_sp  = _adata.layers['spliced']
_mask  = np.asarray(_s_sp.getnnz(axis=0)) >= 10 if _ssp.issparse(_s_sp) else (_s_sp > 0).sum(0) >= 10
EXPR_GENES = list(_adata.var_names[_mask])
del _adata, _s_sp, _mask; gc.collect()

genes = EXPR_GENES[:N_GENES]

print(f'Building SearchData for {N_GENES} genes...')
sd = _mc.searchdata_from_h5ad(H5AD_PATH, ['unspliced', 'spliced'], gene_names=genes)

def _run(use_rust):
    gp = {
        'max_iterations': N_ITER,
        'init_pattern': 'moments',
        'num_restarts': 1,
        'num_gene_cores': -1,
        'use_rust_lbfgsb': use_rust,
    }
    ip = InferenceParameters('bench', MODEL, use_lengths=False,
                             gradient_params=gp, gridsize=GRIDSIZE, save=False)
    t0 = time.perf_counter()
    result = ip.fit_all_grid_points(sd, num_cores=1, save=False)
    elapsed = time.perf_counter() - t0
    # param_estimates: (n_gp, n_genes, n_params)
    # klds:            (n_gp, n_genes)
    return result.param_estimates.copy(), result.klds.copy(), elapsed

print('Running scipy L-BFGS-B...')
p_scipy, k_scipy, t_scipy = _run(use_rust=False)
print(f'  done in {t_scipy:.1f}s')

print('Running Rust L-BFGS-B...')
p_rust, k_rust, t_rust = _run(use_rust=True)
print(f'  done in {t_rust:.1f}s')

print(f'\nSpeedup: {t_scipy/t_rust:.1f}x')

# ── Analysis ──────────────────────────────────────────────────────────────────
n_gp, n_genes, n_params = p_scipy.shape
assert p_rust.shape == (n_gp, n_genes, n_params)

param_diff = np.abs(p_scipy - p_rust)          # (n_gp, n_genes, n_params)
kld_diff   = k_rust - k_scipy                  # positive = Rust worse
max_pdiff  = param_diff.max(axis=2)            # (n_gp, n_genes): max diff over params

PDIFF_THRESH = 0.1   # log10 units
KLD_THRESH   = 0.01  # KLD units

large_param = max_pdiff > PDIFF_THRESH         # (n_gp, n_genes)
rust_worse  = kld_diff  > KLD_THRESH           # (n_gp, n_genes)
rust_better = kld_diff  < -KLD_THRESH          # (n_gp, n_genes)

n_total = n_gp * n_genes
print(f'\n{"="*60}')
print(f'Total gene-gp pairs:  {n_total}  ({n_gp} gp × {n_genes} genes)')
print(f'')
print(f'Large param diff (>{PDIFF_THRESH} log10):  {large_param.sum()}  ({100*large_param.mean():.1f}%)')
print(f'Rust worse KLD (>{KLD_THRESH}):            {rust_worse.sum()}  ({100*rust_worse.mean():.1f}%)')
print(f'Rust better KLD (>{KLD_THRESH}):           {rust_better.sum()}  ({100*rust_better.mean():.1f}%)')
print(f'')
print(f'KLD diff stats:  mean={kld_diff.mean():.4f}  std={kld_diff.std():.4f}')
print(f'                 min={kld_diff.min():.4f}  max={kld_diff.max():.4f}')

# Contingency table: large param diff × Rust worse
both     = (large_param & rust_worse).sum()
pdiff_only = (large_param & ~rust_worse).sum()
worse_only = (~large_param & rust_worse).sum()
neither  = (~large_param & ~rust_worse).sum()

print(f'\nContingency (Rust worse by >{KLD_THRESH}):')
print(f'  param diff >{PDIFF_THRESH}  AND Rust worse:  {both}')
print(f'  param diff >{PDIFF_THRESH}  but KLD ~same:   {pdiff_only}  ← flat landscape')
print(f'  param diff OK  but Rust worse:          {worse_only}')
print(f'  param diff OK  and KLD ~same:            {neither}')

# Distribution of KLD diffs given large param diff
print(f'\nKLD diff distribution where param diff >{PDIFF_THRESH}:')
flat_kld = kld_diff[large_param]
print(f'  n={len(flat_kld)}  mean={flat_kld.mean():.4f}  '
      f'Rust better: {(flat_kld < -KLD_THRESH).sum()}  '
      f'same: {(np.abs(flat_kld) <= KLD_THRESH).sum()}  '
      f'Rust worse: {(flat_kld > KLD_THRESH).sum()}')

# Per-gene summary: how many grid points is Rust worse?
print(f'\nPer-gene: # gp where Rust worse by >{KLD_THRESH}:')
gp_worse = rust_worse.sum(axis=0)   # (n_genes,)
gp_pdiff = large_param.sum(axis=0)  # (n_genes,)
for thr in [0, 1, 3, 6, 9, 12]:
    n = (gp_worse > thr).sum()
    print(f'  > {thr:2d} gp:  {n:3d} genes ({100*n/n_genes:.0f}%)')

# Show worst genes
worst_gene_idx = np.argsort(k_rust.min(axis=0) - k_scipy.min(axis=0))[::-1]
print(f'\nTop 10 genes where Rust finds worse min-KLD:')
print(f'  {"Gene":<30}  {"scipy KLD":>10}  {"Rust KLD":>10}  {"diff":>8}  {"gp_worse":>8}')
for gi in worst_gene_idx[:10]:
    sc_kld = k_scipy[:, gi].min()
    ru_kld = k_rust[:,  gi].min()
    print(f'  {genes[gi]:<30}  {sc_kld:10.4f}  {ru_kld:10.4f}  {ru_kld-sc_kld:8.4f}  {gp_worse[gi]:8d}/{n_gp}')

print(f'\nTop 10 genes where Rust finds BETTER min-KLD:')
best_gene_idx = np.argsort(k_rust.min(axis=0) - k_scipy.min(axis=0))
print(f'  {"Gene":<30}  {"scipy KLD":>10}  {"Rust KLD":>10}  {"diff":>8}')
for gi in best_gene_idx[:10]:
    sc_kld = k_scipy[:, gi].min()
    ru_kld = k_rust[:,  gi].min()
    print(f'  {genes[gi]:<30}  {sc_kld:10.4f}  {ru_kld:10.4f}  {ru_kld-sc_kld:8.4f}')

print(f'\nDone.')
