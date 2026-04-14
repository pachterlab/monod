"""
Concordance check: Python data path vs Rust data path.

Each row holds the optimizer fixed and varies only the data pipeline:
  x-axis: Python extract_data (main branch equivalent)
  y-axis: Rust searchdata_from_h5ad (rusty branch)

Row 1: scipy L-BFGS-B optimizer
Row 2: Rust L-BFGS-B optimizer

With num_restarts=1 (MoM init, deterministic from histogram data), both rows
should show near-perfect concordance since the data is identical and the
optimizer starts from the same initial point.
"""
import gc, sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
os.makedirs('figures', exist_ok=True)

sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'monod'))

import cme_toolbox
import extract_data as _ed_bare
from cme_toolbox import CMEModel, _HAS_RUST
from extract_data import extract_data
from inference import InferenceParameters, searchdata_from_adata
import monod_core as _mc
import anndata as ad
import scipy.sparse as _ssp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H5AD_PATH    = 'example_h5ad/processed_pbmc_10k_raw.h5ad'
MODEL        = CMEModel('Bursty', 'Poisson')
GRIDSIZE     = [3, 4]
N_ITER_CONC  = 200
N_ITER_RUST  = 200
N_CONC_GENES = 200

_adata_ref = ad.read_h5ad(H5AD_PATH)
_s_sp = _adata_ref.layers['spliced']
_mask = np.asarray(_s_sp.getnnz(axis=0)) >= 10 if _ssp.issparse(_s_sp) else (_s_sp > 0).sum(0) >= 10
EXPR_GENES = list(_adata_ref.var_names[_mask])
del _adata_ref, _s_sp, _mask
gc.collect()
print(f'Expressed genes: {len(EXPR_GENES)}')

GENES_CONC = EXPR_GENES[:N_CONC_GENES]


def _run_conc(genes_n, use_rust_data, use_rust_optimizer=False):
    """
    Returns (params [n_genes x 3], klds [n_genes]).
    klds = best KLD per gene across all grid points (min over grid).
    """
    gp = {
        'max_iterations': N_ITER_RUST if use_rust_optimizer else N_ITER_CONC,
        'init_pattern': 'moments',
        'num_restarts': 1,          # deterministic: MoM x0 only
        'num_gene_cores': -1,
        'use_rust_lbfgsb': use_rust_optimizer,
    }
    ip = InferenceParameters('bench', MODEL, use_lengths=False,
                             gradient_params=gp, gridsize=GRIDSIZE, save=False)
    if use_rust_data:
        sd = _mc.searchdata_from_h5ad(H5AD_PATH, ['unspliced', 'spliced'],
                                      gene_names=genes_n)
    else:
        cme_toolbox._HAS_RUST = False
        _ed_bare._HAS_RUST    = False
        adata_ex = extract_data(
            H5AD_PATH, MODEL, dataset_name='bench',
            modality_name_dict={'unspliced': 'unspliced', 'spliced': 'spliced'},
            n_genes=len(genes_n), genes_to_fit=genes_n,
            hist_type='unique', viz=False,
        )
        sd = searchdata_from_adata(adata_ex)
        del adata_ex
        cme_toolbox._HAS_RUST = _HAS_RUST
        _ed_bare._HAS_RUST    = _HAS_RUST
    result = ip.fit_all_grid_points(sd, num_cores=1, save=False)
    result.find_sampling_optimum()
    params = result.phys_optimum.copy()
    klds   = result.klds.min(axis=0).copy()  # best KLD per gene across all grid points
    del sd, result; gc.collect()
    return params, klds


print(f'\nConcordance: n={N_CONC_GENES}, 3x4 grid, num_restarts=1 (MoM init)')
print('  [scipy] Running: Python data (baseline)...')
t0 = time.perf_counter()
params_scipy_py, klds_scipy_py = _run_conc(GENES_CONC, use_rust_data=False, use_rust_optimizer=False)
print(f'  done in {time.perf_counter()-t0:.0f}s')

print('  [scipy] Running: Rust data...')
t0 = time.perf_counter()
params_scipy_rust, klds_scipy_rust = _run_conc(GENES_CONC, use_rust_data=True,  use_rust_optimizer=False)
print(f'  done in {time.perf_counter()-t0:.0f}s')

print(f'  [Rust L-BFGS-B] Running: Python data...')
t0 = time.perf_counter()
params_rust_py, klds_rust_py = _run_conc(GENES_CONC, use_rust_data=False, use_rust_optimizer=True)
print(f'  done in {time.perf_counter()-t0:.0f}s')

print(f'  [Rust L-BFGS-B] Running: Rust data...')
t0 = time.perf_counter()
params_rust_rust, klds_rust_rust = _run_conc(GENES_CONC, use_rust_data=True,  use_rust_optimizer=True)
print(f'  done in {time.perf_counter()-t0:.0f}s')

# ── Figure ────────────────────────────────────────────────────────────────────
param_names  = ['log\u2081\u2080 b\u2003(burst)', 'log\u2081\u2080 \u03b2\u2003(splicing)', 'log\u2081\u2080 \u03b3\u2003(degr.)']
param_colors = ['#2166ac', '#4dac26', '#d01c8b']


def _scatter_row(axes, x_params, y_params, x_klds, y_klds):
    for pi, (pname, color) in enumerate(zip(param_names, param_colors)):
        ax = axes[pi]
        xv, yv = x_params[:, pi], y_params[:, pi]
        ax.scatter(xv, yv, s=5, alpha=0.4, color=color, rasterized=True)
        lo = min(xv.min(), yv.min()) - 0.1
        hi = max(xv.max(), yv.max()) + 0.1
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8)
        r2 = float(np.corrcoef(xv, yv)[0, 1] ** 2)
        ax.set_xlabel(f'Python data  {pname}', fontsize=8)
        ax.set_ylabel(f'Rust data  {pname}', fontsize=8)
        ax.set_title(f'{pname}\nR\u00b2={r2:.5f}', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', 'box')
    ax = axes[3]
    ax.scatter(x_klds, y_klds, s=5, alpha=0.4, color='#7b2d8b', rasterized=True)
    lo = min(x_klds.min(), y_klds.min()) * 0.95
    hi = max(x_klds.max(), y_klds.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8)
    r2_kld = float(np.corrcoef(x_klds, y_klds)[0, 1] ** 2)
    ax.set_xlabel('Python data  KLD (min)', fontsize=8)
    ax.set_ylabel('Rust data  KLD (min)', fontsize=8)
    ax.set_title(f'Best KLD per gene\nR\u00b2={r2_kld:.5f}', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', 'box')


fig, axes = plt.subplots(2, 4, figsize=(16, 8))
_scatter_row(axes[0], params_scipy_py, params_scipy_rust, klds_scipy_py, klds_scipy_rust)
_scatter_row(axes[1], params_rust_py,  params_rust_rust,  klds_rust_py,  klds_rust_rust)

for row, label in enumerate([
    f'scipy L-BFGS-B ({N_ITER_CONC} iter)',
    f'Rust L-BFGS-B ({N_ITER_RUST} iter)',
]):
    axes[row][0].annotate(label, xy=(-0.3, 0.5), xycoords='axes fraction',
                          fontsize=10, fontweight='bold', va='center', rotation=90)

plt.suptitle(
    f'Data path concordance: Python extract_data (x) vs Rust searchdata_from_h5ad (y)\n'
    f'n={N_CONC_GENES} genes, PBMC 10k, 3\u00d74 grid, MoM init — optimizer held fixed per row',
    fontsize=10, y=1.01,
)
plt.tight_layout()
plt.savefig('figures/rusty_vs_main_concordance.png', dpi=150, bbox_inches='tight')
print('\nSaved figures/rusty_vs_main_concordance.png')

# ── Summary ───────────────────────────────────────────────────────────────────
for label, xp, yp, xk, yk in [
    (f'scipy: Python vs Rust data', params_scipy_py, params_scipy_rust, klds_scipy_py, klds_scipy_rust),
    (f'Rust opt: Python vs Rust data', params_rust_py, params_rust_rust, klds_rust_py, klds_rust_rust),
    (f'scipy vs Rust opt: same (Python) data', params_scipy_py, params_rust_py, klds_scipy_py, klds_rust_py),
    (f'scipy vs Rust opt: same (Rust) data',   params_scipy_rust, params_rust_rust, klds_scipy_rust, klds_rust_rust),
]:
    pd = np.abs(xp - yp)
    kd = np.abs(xk - yk)
    print(f'\n{label}:')
    print(f'  param max diff (log10): {pd.max():.4f}  mean: {pd.mean():.4f}')
    print(f'  KLD  max diff:          {kd.max():.6f}  mean: {kd.mean():.6f}')
    print(f'  genes param diff > 0.1: {(pd.max(1) > 0.1).sum()}')
