"""
Standalone timing sweep for GRIDSIZE=[6,7] (42 sampling points).
Results saved to /tmp/grid67_results.pkl for use in demo_rusty_vs_main.ipynb.
"""
import gc, sys, os, time, pickle, warnings
import numpy as np
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'monod'))

import cme_toolbox
import extract_data as _ed_bare
from cme_toolbox import CMEModel, _HAS_RUST
from extract_data import extract_data
from inference import InferenceParameters, searchdata_from_adata
import monod_core as _mc
import anndata as ad
import scipy.sparse as _ssp

H5AD_PATH = 'example_h5ad/processed_pbmc_10k_raw.h5ad'
MODEL     = CMEModel('Bursty', 'Poisson')
N_ITER      = 50   # main-serial (scipy)
N_ITER_RUST = 200  # Rust optimizer (matches notebook default)
GRIDSIZE  = [6, 7]   # 42-point grid (default / biologically standard)
N_SWEEP   = [25, 50, 100, 200]

_adata_ref = ad.read_h5ad(H5AD_PATH)
_s_sp = _adata_ref.layers['spliced']
if _ssp.issparse(_s_sp):
    mask = np.asarray(_s_sp.getnnz(axis=0)) >= 10
else:
    mask = (_s_sp > 0).sum(0) >= 10
EXPR_GENES = list(_adata_ref.var_names[mask])
del _adata_ref, _s_sp, mask
gc.collect()
print(f'Expressed genes: {len(EXPR_GENES)}')

n_cpu = os.cpu_count() or 4


def _make_ip(n_gene_cores, use_rust_lbfgsb=False):
    gp = {
        'max_iterations': N_ITER_RUST if use_rust_lbfgsb else N_ITER,
        'init_pattern': 'moments',
        'num_restarts': 1,
        'num_gene_cores': n_gene_cores,
        'use_rust_lbfgsb': use_rust_lbfgsb,
    }
    return InferenceParameters(
        'bench', MODEL, use_lengths=False,
        gradient_params=gp, gridsize=GRIDSIZE, save=False,
    )


def time_main_serial(genes_n):
    cme_toolbox._HAS_RUST = False
    _ed_bare._HAS_RUST    = False
    t0 = time.perf_counter()
    adata_ex = extract_data(
        H5AD_PATH, MODEL, dataset_name='bench',
        modality_name_dict={'unspliced': 'unspliced', 'spliced': 'spliced'},
        n_genes=len(genes_n), genes_to_fit=genes_n, hist_type='unique', viz=False,
    )
    sd = searchdata_from_adata(adata_ex)
    t_load = (time.perf_counter() - t0) * 1e3
    del adata_ex; gc.collect()
    ip = _make_ip(1, use_rust_lbfgsb=False)
    t0 = time.perf_counter()
    ip.fit_all_grid_points(sd, num_cores=1, save=False)
    t_infer = (time.perf_counter() - t0) * 1e3
    del sd; gc.collect()
    cme_toolbox._HAS_RUST = _HAS_RUST
    _ed_bare._HAS_RUST    = _HAS_RUST
    return t_load, t_infer


def time_rusty_full(genes_n):
    t0 = time.perf_counter()
    sd = _mc.searchdata_from_h5ad(H5AD_PATH, ['unspliced', 'spliced'], gene_names=genes_n)
    t_load = (time.perf_counter() - t0) * 1e3
    ip = _make_ip(-1, use_rust_lbfgsb=True)
    t0 = time.perf_counter()
    ip.fit_all_grid_points(sd, num_cores=1, save=False)
    t_infer = (time.perf_counter() - t0) * 1e3
    del sd; gc.collect()
    return t_load, t_infer


results_67 = {}
print(f'\n{"n":>5}  {"Config":<18}  {"Load (ms)":>10}  {"Infer (ms)":>11}  {"Total (ms)":>11}')
print('-' * 62)
for n in N_SWEEP:
    genes_n = EXPR_GENES[:n]
    for cfg_name, fn in [('main_serial', time_main_serial), ('rusty_full', time_rusty_full)]:
        t_load, t_infer = fn(genes_n)
        results_67[(cfg_name, n)] = (t_load, t_infer)
        print(f'{n:5d}  {cfg_name:<18}  {t_load:10.0f}  {t_infer:11.0f}  {t_load+t_infer:11.0f}')
    t_ms = sum(results_67[('main_serial', n)])
    t_ru = sum(results_67[('rusty_full',  n)])
    print(f'       rusty speedup: {t_ms/t_ru:.1f}x')
    print()

with open('/tmp/grid67_results.pkl', 'wb') as f:
    pickle.dump({'results': results_67, 'N_SWEEP': N_SWEEP, 'n_cpu': n_cpu}, f)
print('Saved /tmp/grid67_results.pkl')
