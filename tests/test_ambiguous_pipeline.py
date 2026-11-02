"""The ambiguous-assignment models through the pipeline: the ambiguous layer is a
third modality, its probabilities per-gene parameters (2026-10-29)."""
import os
import warnings

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from monod.cme_toolbox import CMEModel
from monod.extract_data import extract_data, grid_limits
from monod.inference import perform_inference, reject_genes, search_result_from_uns, searchdata_from_adata

GP = {"max_iterations": 60, "init_pattern": "moments", "num_restarts": 1, "num_gene_cores": 1}
PERMISSIVE3 = {"min_means": [0, 0, 0], "max_maxes": [10_000] * 3, "min_maxes": [0, 0, 0]}
NAMES = lambda m: [p.replace(r"\log_{10} ", "") for p in m.param_str]


def _sample_cells(model, theta_log, limits, n_cells, rng):
    """Draw cells from the model's own three-channel distribution."""
    pss = model.eval_model_pss(np.asarray(theta_log), list(limits))
    flat = pss.ravel() / pss.sum()
    idx = rng.choice(flat.size, size=n_cells, p=flat)
    return np.column_stack(np.unravel_index(idx, pss.shape))


def _simulated(model, thetas, n_cells=3000, limits=(40, 60, 40), seed=0):
    rng = np.random.default_rng(seed)
    cells = [_sample_cells(model, t, limits, n_cells, rng) for t in thetas]
    layers = {name: np.column_stack([c[:, k] for c in cells]).astype(np.int64)
              for k, name in enumerate(["unspliced", "spliced", "ambiguous"])}
    a = ad.AnnData(X=layers["unspliced"] + layers["spliced"] + layers["ambiguous"], layers=layers)
    a.var_names = [f"g{i}" for i in range(len(thetas))]
    a.obs_names = [f"c{i}" for i in range(n_cells)]
    return a


class TestModel:
    def test_modalities_bounds_and_evaluation(self):
        m = CMEModel("Bursty", "None", amb_model="Unequal")
        assert m.model_modalities == ["unspliced", "spliced", "ambiguous"]
        assert m.get_num_params() == 5 and len(m.bio_bounds["phys_lb"]) == 5 and len(m.filter_bounds["min_means"]) == 3
        assert NAMES(m)[-2:] == ["$p_N$", "$p_M$"]
        pss = m.eval_model_pss(np.log10([5, 0.5, 0.3, 0.1, 0.4]), [20, 30, 15])
        assert pss.shape == (20, 30, 15) and pss.sum() == pytest.approx(1.0)
        with pytest.raises(ValueError, match="one grid size per species"):
            m.eval_model_pss(np.log10([5, 0.5, 0.3, 0.1, 0.4]), [20, 30])
        e = CMEModel("Bursty", "None", amb_model="Equal")
        assert e.model_modalities[-1] == "ambiguous" and e.get_num_params() == 4
        assert CMEModel("ProteinBursty", "None", amb_model="Equal").model_modalities == ["unspliced", "spliced", "protein", "ambiguous"]

    def test_moments_start_uses_the_ambiguous_share(self):
        m = CMEModel("Bursty", "None", amb_model="Equal")
        moments = {"MOM_unspliced_mean": 3.0, "MOM_unspliced_var": 12.0, "MOM_spliced_mean": 6.0, "MOM_spliced_var": 20.0,
                   "MOM_ambiguous_mean": 1.0, "MOM_ambiguous_var": 2.0}
        lb, ub = np.array(m.bio_bounds["phys_lb"]), np.array(m.bio_bounds["phys_ub"])
        x0 = m.get_MoM(moments, lb, ub, samp=np.zeros(2))
        assert 10 ** x0[-1] == pytest.approx(0.1)                        # 1 / (3 + 6 + 1)
        # the thinned moments are corrected before the biological start is taken
        plain = CMEModel("Bursty", "None").get_MoM({k: v for k, v in moments.items() if "ambiguous" not in k},
                                                   lb[:3], ub[:3], samp=np.zeros(2))
        assert not np.allclose(x0[:3], plain)


@pytest.fixture(scope="module")
def equal_fit(tmp_path_factory):
    warnings.filterwarnings("ignore")
    cwd = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("amb"))
    try:
        m = CMEModel("Bursty", "None", amb_model="Equal")
        truth = [np.log10([b, 0.5, 0.3, 0.25]) for b in (3.0, 5.0, 8.0)]
        a = _simulated(m, truth)
        out = perform_inference(a, m, n_genes=3, filt_param=PERMISSIVE3, gradient_params=GP, viz=False,
                                exclude_sigma=False, dataset_string="amb_equal")
        yield a, out, truth
    finally:
        os.chdir(cwd)


class TestPipeline:
    def test_layers_limits_and_columns(self, equal_fit):
        a, out, truth = equal_fit
        assert set(out.layers) >= {"unspliced", "spliced", "ambiguous"}
        assert grid_limits(out).shape == (3, 3)
        assert "M_ambiguous" in out.var.columns and "MOM_ambiguous_mean" in out.var.columns
        m = CMEModel("Bursty", "None", amb_model="Equal")
        for name in NAMES(m):
            assert name in out.var.columns and f"error_{name}" in out.var.columns
        sd = searchdata_from_adata(out)
        assert np.asarray(sd.M).shape == (3, 3) and len(sd.layer_names) == 3

    def test_recovers_the_ambiguous_probability(self, equal_fit):
        a, out, truth = equal_fit
        p_hat = 10 ** out.var["$p$"].to_numpy()
        assert np.all(np.abs(p_hat - 0.25) < 0.06), p_hat
        b_hat = 10 ** out.var["$b$"].to_numpy()
        assert np.all((b_hat > 1.5) & (b_hat < 16)), b_hat
        assert np.all(np.isfinite(out.var["AIC"]))

    def test_rejection_plots_and_round_trip(self, equal_fit, tmp_path):
        a, out, truth = equal_fit
        reject_genes(out, viz=False)
        assert "rejected_genes" in out.var.columns and "hellinger" in out.var.columns
        sr = search_result_from_uns(out.uns["search_result"])
        sd = searchdata_from_adata(out)
        sr.plot_gene_distributions(sd, sz=(1, 3), marg="ambiguous", savefig=False)
        sr.plot_gene_distributions(sd, sz=(1, 3), marg="joint", savefig=False)
        plt.close("all")
        path = tmp_path / "amb.h5ad"
        out.write_h5ad(path)
        back = ad.read_h5ad(path)
        assert CMEModel.from_dict(back.uns["model"]).model_modalities[-1] == "ambiguous"
        np.testing.assert_allclose(search_result_from_uns(back.uns["search_result"]).phys_optimum, sr.phys_optimum)

    def test_unequal_tells_the_channels_apart(self, tmp_path, monkeypatch):
        warnings.filterwarnings("ignore")
        monkeypatch.chdir(tmp_path)
        m = CMEModel("Bursty", "None", amb_model="Unequal")
        truth = [np.log10([5.0, 0.5, 0.3, 0.1, 0.4]), np.log10([4.0, 0.6, 0.25, 0.1, 0.4])]
        out = perform_inference(_simulated(m, truth, n_cells=4000, seed=1), m, n_genes=2, filt_param=PERMISSIVE3,
                                gradient_params=GP, viz=False, dataset_string="amb_unequal")
        pN, pM = 10 ** out.var["$p_N$"].to_numpy(), 10 ** out.var["$p_M$"].to_numpy()
        assert np.all(pM > pN + 0.1), (pN, pM)
        assert np.all(np.abs(pM - 0.4) < 0.1) and np.all(np.abs(pN - 0.1) < 0.1), (pN, pM)

    def test_poisson_sampling_runs(self, tmp_path, monkeypatch):
        warnings.filterwarnings("ignore")
        monkeypatch.chdir(tmp_path)
        sim = CMEModel("Bursty", "None", amb_model="Equal")
        a = _simulated(sim, [np.log10([5.0, 0.5, 0.3, 0.25])], n_cells=1500, seed=2)
        m = CMEModel("Bursty", "Poisson", amb_model="Equal")
        out = perform_inference(a, m, n_genes=1, filt_param=PERMISSIVE3, gradient_params=GP, viz=False,
                                samp_lb=[-0.3, -0.3], samp_ub=[0.0, 0.0], gridsize=[2, 2], dataset_string="amb_poisson")
        assert np.isfinite(out.var["$p$"].iloc[0]) and -3 <= out.var["$p$"].iloc[0] <= 0
        sr = search_result_from_uns(out.uns["search_result"])
        assert sr.klds.shape == (4, 1) and np.all(np.isfinite(sr.klds))

    def test_mek_means_refuses(self):
        import monod.mminference as mm
        with pytest.raises(NotImplementedError, match="ambiguous"):
            mm.MEKMeansParameters("x", CMEModel("Bursty", "None", amb_model="Equal"), k=2, epochs=1)

    def test_extract_data_needs_the_layer(self):
        a = ad.AnnData(X=np.ones((10, 2)), layers={"unspliced": np.ones((10, 2)), "spliced": np.ones((10, 2))})
        with pytest.raises(KeyError, match="ambiguous"):
            extract_data(a, CMEModel("Bursty", "None", amb_model="Equal"), n_genes=2, viz=False)


PERMISSIVE2 = {"min_means": [0, 0], "max_maxes": [10_000, 10_000], "min_maxes": [0, 0]}


def _three_layer_adata(seed=0):
    rng = np.random.default_rng(seed)
    U, S, A = (rng.poisson(m, (300, 3)).astype(np.int64) for m in (2.0, 3.0, 1.0))
    a = ad.AnnData(X=U + S + A, layers={"unspliced": U, "spliced": S, "ambiguous": A})
    a.var_names = ["g0", "g1", "g2"]
    a.obs_names = [f"c{i}" for i in range(300)]
    return a


class TestAmbiguousHandling:
    """An ambiguous layer is never ignored silently: fold it, drop it or model it."""

    def test_choice_is_required_when_the_layer_exists(self):
        a = _three_layer_adata()
        with pytest.raises(ValueError, match="ambiguous layer"):
            extract_data(a, CMEModel("Bursty", "None"), n_genes=3, filt_param=PERMISSIVE2, viz=False)
        b = _three_layer_adata(); del b.layers["ambiguous"]
        out = extract_data(b, CMEModel("Bursty", "None"), n_genes=3, filt_param=PERMISSIVE2, viz=False)
        assert "ambiguous" not in out.uns                                # nothing to decide

    def test_fold_and_drop(self):
        a = _three_layer_adata()
        before = {k: np.asarray(v).copy() for k, v in a.layers.items()}
        for side, other in (("spliced", "unspliced"), ("unspliced", "spliced")):
            out = extract_data(a, CMEModel("Bursty", "None"), n_genes=3, filt_param=PERMISSIVE2, viz=False, ambiguous=side)
            np.testing.assert_array_equal(out.layers[side], before[side] + before["ambiguous"])
            np.testing.assert_array_equal(out.layers[other], before[other])
            assert out.uns["ambiguous"] == side and "ambiguous" not in out.layers
        dropped = extract_data(a, CMEModel("Bursty", "None"), n_genes=3, filt_param=PERMISSIVE2, viz=False, ambiguous="drop")
        np.testing.assert_array_equal(dropped.layers["spliced"], before["spliced"])
        assert dropped.uns["ambiguous"] == "drop"
        for k, v in before.items():                                       # the input is untouched
            np.testing.assert_array_equal(np.asarray(a.layers[k]), v)

    def test_renamed_layers_fold_too(self):
        a = _three_layer_adata()
        a.layers["nascent"] = a.layers.pop("unspliced"); a.layers["mature"] = a.layers.pop("spliced")
        a.layers["amb"] = a.layers.pop("ambiguous")
        names = {"unspliced": "nascent", "spliced": "mature", "ambiguous": "amb"}
        out = extract_data(a, CMEModel("Bursty", "None"), n_genes=3, filt_param=PERMISSIVE2, viz=False,
                           modality_name_dict=names, ambiguous="spliced")
        np.testing.assert_array_equal(out.layers["mature"], np.asarray(a.layers["mature"]) + np.asarray(a.layers["amb"]))

    def test_conflicts_and_bad_values(self):
        a = _three_layer_adata()
        plain, amb = CMEModel("Bursty", "None"), CMEModel("Bursty", "None", amb_model="Equal")
        with pytest.raises(ValueError, match="conflicts"):
            extract_data(a, amb, n_genes=3, filt_param=PERMISSIVE3, viz=False, ambiguous="spliced")
        with pytest.raises(ValueError, match="needs a CMEModel"):
            extract_data(a, plain, n_genes=3, filt_param=PERMISSIVE2, viz=False, ambiguous="model")
        with pytest.raises(ValueError, match="must be one of"):
            extract_data(a, plain, n_genes=3, filt_param=PERMISSIVE2, viz=False, ambiguous="sum")
        b = _three_layer_adata(); del b.layers["ambiguous"]
        with pytest.raises(KeyError, match="no 'ambiguous' layer"):
            extract_data(b, plain, n_genes=3, filt_param=PERMISSIVE2, viz=False, ambiguous="spliced")
        modelled = extract_data(a, amb, n_genes=3, filt_param=PERMISSIVE3, viz=False)
        assert modelled.uns["ambiguous"] == "model" and "ambiguous" in modelled.layers

    def test_custom_network_folds_into_a_species(self):
        a = _three_layer_adata()
        net = CMEModel("Custom", "None", network="0 -> B x U -> S -> 0")
        names = {"U": "unspliced", "S": "spliced", "ambiguous": "ambiguous"}
        out = extract_data(a, net, n_genes=3, filt_param=PERMISSIVE2, viz=False, modality_name_dict=names, ambiguous="S")
        np.testing.assert_array_equal(out.layers["spliced"], np.asarray(a.layers["spliced"]) + np.asarray(a.layers["ambiguous"]))
        with pytest.raises(ValueError, match="fold into"):
            extract_data(a, net, n_genes=3, filt_param=PERMISSIVE2, viz=False, modality_name_dict=names, ambiguous="spliced")

    def test_perform_inference_passes_the_choice(self, tmp_path, monkeypatch):
        warnings.filterwarnings("ignore")
        monkeypatch.chdir(tmp_path)
        out = perform_inference(_three_layer_adata(), CMEModel("Bursty", "None"), n_genes=3, filt_param=PERMISSIVE2,
                                gradient_params=GP, viz=False, ambiguous="drop", dataset_string="h")
        assert out.uns["ambiguous"] == "drop" and np.all(np.isfinite(out.var["$b$"]))
        with pytest.raises(ValueError, match="ambiguous layer"):
            perform_inference(_three_layer_adata(), CMEModel("Bursty", "None"), n_genes=3, filt_param=PERMISSIVE2,
                              gradient_params=GP, viz=False, dataset_string="h2")


def _total_count_pmf(pss, axes):
    """Distribution of the sum of the counts on ``axes`` (the other axes summed out)."""
    P = pss.sum(axis=tuple(k for k in range(pss.ndim) if k not in axes))
    P = np.moveaxis(P, [axes.index(a) for a in sorted(axes)], range(len(axes)))
    grids = np.meshgrid(*[np.arange(n) for n in P.shape], indexing="ij")
    return np.bincount(sum(grids).ravel(), weights=P.ravel())


class TestAnyModel:
    """The ambiguous channel on every model family: the generic split equals the
    two-species Rust kernel, and preserves total counts for the others."""

    @pytest.mark.parametrize("seq,samp", [("None", None), ("Poisson", np.log10([0.6, 0.9]))])
    @pytest.mark.parametrize("amb", ["Equal", "Unequal"])
    def test_generic_split_matches_the_kernel(self, amb, seq, samp):
        from monod._core import _mc
        if _mc is None:
            pytest.skip("needs the Rust kernel to compare against")
        m = CMEModel("Bursty", seq, amb_model=amb)
        # a grid wide enough to hold the whole distribution: the kernel evaluates the observed
        # generating function by FFT and wraps mass beyond the grid around, the generic route
        # truncates it, so they agree only where that mass is negligible
        theta = np.log10([2.0, 0.5, 0.3] + ([0.15] if amb == "Equal" else [0.1, 0.25]))
        limits = [30, 36, 26]
        kernel = m.eval_model_pss(theta, limits, samp)
        generic = m._eval_pss_ambiguous(theta, limits, samp)
        assert kernel.shape == generic.shape == (30, 36, 26)
        np.testing.assert_allclose(generic, kernel, atol=1e-6)

    def test_custom_network_matches_the_builtin(self):
        theta = np.log10([3.0, 0.4, 0.25, 0.1, 0.3])
        ref = CMEModel("Bursty", "None", amb_model="Unequal").eval_model_pss(theta, [30, 40, 36])   # wide enough to hold the mass
        net = CMEModel("Custom", "None", network="0 -> B x U -> S -> 0", normalize_production_rate=True,
                       amb_model="Unequal", amb_species=("U", "S"))
        assert net.model_modalities == ["U", "S", "ambiguous"]
        assert [n.replace(r"\log_{10} ", "") for n in net.param_str][-2:] == ["$p_{U}$", "$p_{S}$"]
        pss = net.eval_model_pss(theta, [30, 40, 36])
        np.testing.assert_allclose(pss, ref, atol=5e-6)
        with pytest.raises(ValueError, match="amb_species"):
            CMEModel("Custom", "None", network="0 -> B x U -> S -> 0", amb_model="Equal")

    def test_protein_model(self):
        m = CMEModel("ProteinBursty", "None", protein_limit=8, amb_model="Equal")
        assert m.model_modalities == ["unspliced", "spliced", "protein", "ambiguous"] and m.get_num_params() == 6
        theta = np.log10([1.5, 1.0, 0.5, 1.0, 0.5, 0.2])
        limits = [18, 18, 10, 12]
        pss = m.eval_model_pss(theta, limits)
        assert pss.shape == tuple(limits) and pss.sum() == pytest.approx(1.0)
        # the split moves counts between channels: the total RNA count distribution is the latent one
        latent = m._latent().eval_model_pss(theta[:-1], [29, 29, 10])
        tot_obs = _total_count_pmf(pss, [0, 1, 3])
        tot_lat = _total_count_pmf(latent, [0, 1])
        n = min(len(tot_obs), len(tot_lat), 12)
        np.testing.assert_allclose(tot_obs[:n], tot_lat[:n], atol=5e-4)      # grids wide enough that little mass is cut
        # gradient through the generic route is finite and pins the integration schedule
        coords = np.array([[1, 2, 0, 0], [0, 1, 1, 1], [3, 2, 2, 0]]); freqs = np.array([0.5, 0.3, 0.2])
        kld, grad = m.eval_model_kld_and_grad(theta, limits, None, (coords, freqs))
        assert np.isfinite(kld) and grad.shape == (6,) and np.all(np.isfinite(grad))
        back = CMEModel.from_dict(m.to_dict())
        assert back.model_modalities == m.model_modalities and back.amb_species == ("unspliced", "spliced")

    def test_chromatin_model(self):
        m = CMEModel("Chromatin", "None", n_sites=2, rule="switch", amb_model="Equal")
        assert m.model_modalities == ["unspliced", "spliced", "configuration", "ambiguous"]
        n_lat = m.get_num_params() - 1
        theta = np.concatenate([np.log10([0.5, 0.8, 1.0, 2.0, 0.7, 0.3])[:n_lat], [np.log10(0.25)]])
        limits = [16, 20, 4, 8]
        pss = m.eval_model_pss(theta, limits)
        assert pss.shape == tuple(limits) and pss.sum() == pytest.approx(1.0)
        latent = m._latent().eval_model_pss(theta[:-1], [23, 27, 4])
        np.testing.assert_allclose(pss.sum(axis=(0, 1, 3)), latent.sum(axis=(0, 1)), atol=5e-4)   # configuration untouched (up to grid truncation)

    def test_protein_moments_start_uses_the_corrected_moments(self):
        m = CMEModel("ProteinBursty", "None", amb_model="Unequal")
        moments = {"MOM_unspliced_mean": 3.0, "MOM_unspliced_var": 12.0, "MOM_spliced_mean": 6.0, "MOM_spliced_var": 20.0,
                   "MOM_protein_mean": 10.0, "MOM_protein_var": 50.0, "MOM_cov_unspliced_protein": 2.0,
                   "MOM_cov_unspliced_spliced": 3.0, "MOM_cov_spliced_protein": 4.0,
                   "MOM_ambiguous_mean": 1.0, "MOM_ambiguous_var": 2.0}
        lb, ub = np.array(m.bio_bounds["phys_lb"]), np.array(m.bio_bounds["phys_ub"])
        x0 = m.get_MoM(moments, lb, ub, samp=np.zeros(2))
        assert x0.shape == (7,) and np.all(np.isfinite(x0))
        np.testing.assert_allclose(10 ** x0[-2:], 0.1)

    def test_custom_network_end_to_end(self, tmp_path, monkeypatch):
        warnings.filterwarnings("ignore")
        monkeypatch.chdir(tmp_path)
        sim = CMEModel("Bursty", "None", amb_model="Equal")
        a = _simulated(sim, [np.log10([5.0, 0.5, 0.3, 0.25]), np.log10([3.0, 0.6, 0.4, 0.25])], n_cells=2500, seed=4)
        a.layers["U"] = a.layers.pop("unspliced"); a.layers["S"] = a.layers.pop("spliced")
        net = CMEModel("Custom", "None", network="0 -> B x U -> S -> 0", normalize_production_rate=True,
                       amb_model="Equal", amb_species=("U", "S"))
        out = perform_inference(a, net, n_genes=2, filt_param=PERMISSIVE3, gradient_params=GP, viz=False,
                                dataset_string="amb_custom", modality_name_dict={"U": "U", "S": "S", "ambiguous": "ambiguous"})
        p_hat = 10 ** out.var["$p$"].to_numpy()
        assert np.all(np.abs(p_hat - 0.25) < 0.08), p_hat
        sr = search_result_from_uns(out.uns["search_result"])
        sd = searchdata_from_adata(out)
        sr.plot_gene_distributions(sd, sz=(1, 2), marg="joint", savefig=False)
        sr.plot_gene_distributions(sd, sz=(1, 2), marg="ambiguous", savefig=False)
        plt.close("all")
