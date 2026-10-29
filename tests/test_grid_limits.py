"""Grid limits live in var (one column per modality) so AnnData gene subsets keep
them, and select_expressed_genes gives a deterministic gene list (demo cleanup
N5, 2026-10-29)."""
import warnings

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from monod.cme_toolbox import CMEModel
from monod.extract_data import extract_data, grid_limits, select_expressed_genes
from monod.inference import searchdata_from_adata

PERMISSIVE = {"min_means": [0, 0], "max_maxes": [10_000, 10_000], "min_maxes": [0, 0]}


def _synthetic(n_cells=300, n_genes=4, seed=0, sparse=False):
    rng = np.random.default_rng(seed)
    means = np.linspace(0.5, 6.0, n_genes)
    U = rng.poisson(means, (n_cells, n_genes))
    S = rng.poisson(2 * means, (n_cells, n_genes))
    wrap = sp.csr_matrix if sparse else np.asarray
    a = ad.AnnData(X=wrap(U + S), layers={"unspliced": wrap(U), "spliced": wrap(S)})
    a.var_names = [f"g{i}" for i in range(n_genes)]
    a.obs_names = [f"c{i}" for i in range(n_cells)]
    return a


@pytest.fixture(scope="module")
def processed():
    warnings.filterwarnings("ignore")
    a = _synthetic()
    return extract_data(a, CMEModel("Bursty", "None"), dataset_name="gl", n_genes=4,
                        filt_param=PERMISSIVE, viz=False)


class TestGridLimitsInVar:
    def test_columns_hold_max_plus_padding(self, processed):
        assert list(processed.var.columns[processed.var.columns.str.startswith("M_")]) == ["M_unspliced", "M_spliced"]
        assert "M" not in processed.uns
        M = grid_limits(processed)
        assert M.shape == (2, processed.n_vars) and M.dtype.kind == "i"
        np.testing.assert_array_equal(M[0], np.asarray(processed.layers["unspliced"]).max(axis=0) + 10)
        np.testing.assert_array_equal(M[1], np.asarray(processed.layers["spliced"]).max(axis=0) + 10)

    def test_gene_subset_keeps_its_limits(self, processed):
        names = list(processed.var_names)
        pick = [names[2], names[0]]
        sd = searchdata_from_adata(processed[:, pick].copy())
        assert list(sd.gene_names) == pick
        np.testing.assert_array_equal(np.asarray(sd.M), grid_limits(processed)[:, [2, 0]])

    def test_full_searchdata_matches(self, processed):
        sd = searchdata_from_adata(processed)
        np.testing.assert_array_equal(np.asarray(sd.M), grid_limits(processed))

    def test_fallback_to_uns_for_old_files(self, processed):
        old = processed.copy()
        M = grid_limits(processed)
        old.var = old.var.drop(columns=["M_unspliced", "M_spliced"])
        with pytest.raises(KeyError, match="M_unspliced"):
            grid_limits(old)
        old.uns["M"] = M
        np.testing.assert_array_equal(grid_limits(old), M)


class TestSelectExpressedGenes:
    def test_dense_and_sparse_agree_in_file_order(self):
        dense, sparse = _synthetic(seed=3), _synthetic(seed=3, sparse=True)
        n_expr = np.count_nonzero(np.asarray(dense.layers["spliced"]), axis=0)
        expected = [g for g, n in zip(dense.var_names, n_expr) if n >= 200]
        assert select_expressed_genes(dense, min_cells=200) == expected
        assert select_expressed_genes(sparse, min_cells=200) == expected
        assert 0 < len(expected) < dense.n_vars

    def test_n_genes_and_layer(self):
        a = _synthetic(seed=4)
        assert select_expressed_genes(a, min_cells=1, n_genes=2) == ["g0", "g1"]
        assert select_expressed_genes(a, min_cells=0, layer=None) == list(a.var_names)
        assert select_expressed_genes(a, min_cells=a.n_obs + 1) == []

    def test_feeds_extract_data(self):
        warnings.filterwarnings("ignore")
        a = _synthetic(seed=5)
        genes = select_expressed_genes(a, min_cells=1, n_genes=2)
        out = extract_data(a, CMEModel("Bursty", "None"), dataset_name="gl2", n_genes=len(genes),
                           genes_to_fit=genes, filt_param=PERMISSIVE, viz=False)
        assert list(out.var_names) == genes


class TestRenamedLayers:
    """Moments are keyed by modality, so renamed layers still feed the moments start."""

    def test_moments_follow_the_modalities(self, tmp_path, monkeypatch):
        warnings.filterwarnings("ignore")
        monkeypatch.chdir(tmp_path)
        a = _synthetic(seed=7)
        a.layers["nascent"] = a.layers.pop("unspliced")
        a.layers["mature"] = a.layers.pop("spliced")
        names = {"unspliced": "nascent", "spliced": "mature"}
        out = extract_data(a, CMEModel("Bursty", "None"), dataset_name="rn", n_genes=4, filt_param=PERMISSIVE,
                           modality_name_dict=names, viz=False)
        assert "MOM_unspliced_mean" in out.var.columns and "MOM_nascent_mean" not in out.var.columns
        assert "M_nascent" in out.var.columns                         # grid limits keep the layer names
        sd = searchdata_from_adata(out)
        assert list(sd.layer_names) == ["unspliced", "spliced"]
        assert "MOM_unspliced_mean" in sd.moments[0]
        from monod.inference import perform_inference
        fit = perform_inference(a, CMEModel("Bursty", "None"), n_genes=4, filt_param=PERMISSIVE, modality_name_dict=names,
                                gradient_params={"max_iterations": 10, "init_pattern": "moments", "num_restarts": 1,
                                                 "num_gene_cores": 1}, viz=False, dataset_string="rn2")
        assert np.all(np.isfinite(fit.var["$b$"]))
