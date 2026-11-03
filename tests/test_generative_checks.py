"""Simulation-based check per model family (2026-11-02).

Cells are drawn from the generative process by :mod:`monod.simulate`, which
knows nothing of the kernels, with the observation layers switched on, and
compared with the kernel's distribution: first and second moments within
sampling error, the divergence of the empirical histogram from the kernel at
the level a multinomial sample produces, and for a subset of families the
parameters recovered through the inference pipeline.
"""
import os
import warnings

import numpy as np
import pytest

from monod.cme_toolbox import CMEModel
from monod.inference import perform_inference
from monod.simulate import simulate_adata, simulate_counts

N_CELLS = 20000
Z_MAX = 4.5
GP = {"max_iterations": 50, "init_pattern": "moments", "num_restarts": 1, "num_gene_cores": 1}
slow = pytest.mark.skipif(not os.environ.get("MONOD_SLOW_TESTS"),
                          reason="minutes per gene on the NumPy chromatin-protein kernel; set MONOD_SLOW_TESTS=1")


def _permissive(model):
    k = len(model.model_modalities)
    return {"min_means": [0] * k, "max_maxes": [10_000] * k, "min_maxes": [0] * k}


def _limits(model, counts, pad=10):
    out = []
    for m in model.model_modalities:
        if m == "configuration":
            out.append(2 ** model.n_sites)
        else:
            out.append(int(counts[m].max()) + pad)
    return out


def _pmf_moments(P):
    """Means, variances and pairwise covariances of a PMF on an integer grid."""
    grids = np.meshgrid(*[np.arange(n) for n in P.shape], indexing="ij")
    mu = np.array([(g * P).sum() for g in grids])
    cov = np.array([[((gi - mi) * (gj - mj) * P).sum() for gj, mj in zip(grids, mu)] for gi, mi in zip(grids, mu)])
    return mu, cov


def _sample_moments(X):
    """Sample means, covariance, and standard errors of the mean, the variance and each covariance."""
    n, k = X.shape
    mu = X.mean(axis=0)
    D = X - mu
    cov = D.T @ D / (n - 1)
    se_mean = np.sqrt(np.diag(cov) / n)
    se_cov = np.sqrt(((D[:, :, None] * D[:, None, :]) ** 2).mean(axis=0) - cov ** 2) / np.sqrt(n)
    return mu, cov, se_mean, se_cov


def check_family(model, theta, samp=None, n_cells=N_CELLS, seed=0, kld_fold=1.5):
    counts = simulate_counts(model, theta, n_cells, samp, seed)
    mods = list(model.model_modalities)
    limits = _limits(model, counts)
    P = np.asarray(model.eval_model_pss(np.asarray(theta, float), limits, samp)).reshape(limits)
    assert P.sum() > 0.995, "the grid does not hold the kernel's mass"
    X = np.column_stack([counts[m] for m in mods]).astype(float)
    mu_s, cov_s, se_mean, se_cov = _sample_moments(X)
    mu_k, cov_k = _pmf_moments(P)
    z_mean = (mu_s - mu_k) / np.maximum(se_mean, 1e-12)
    z_cov = (cov_s - cov_k) / np.maximum(se_cov, 1e-12)
    report = "\n".join(f"{m}: mean {mu_s[i]:.4f} vs {mu_k[i]:.4f} (z {z_mean[i]:+.2f}); "
                       f"var {cov_s[i, i]:.4f} vs {cov_k[i, i]:.4f} (z {z_cov[i, i]:+.2f})" for i, m in enumerate(mods))
    assert np.all(np.abs(z_mean) < Z_MAX) and np.all(np.abs(z_cov) < Z_MAX), report
    coords, freq = np.unique(X.astype(int), axis=0, return_counts=True)
    f = freq / freq.sum()
    p = P[tuple(coords.T)]
    assert np.all(p > 0), "the simulation reached a state the kernel gives probability zero"
    kld = float(np.sum(f * np.log(f / p)))
    # the level a sample of this size drawn from the kernel's own law reaches (plug-in
    # divergence is biased upward when most states are seen once, so a chi-square
    # approximation would be far off for the higher-dimensional families)
    boot = np.random.default_rng(seed + 1)
    flat = P.ravel() / P.sum()
    levels = []
    for _ in range(2):
        draw = boot.choice(flat.size, size=n_cells, p=flat)
        _, c = np.unique(draw, return_counts=True)
        fb = c / c.sum()
        levels.append(float(np.sum(fb * np.log(fb / flat[np.unique(draw)]))))
    expected = max(levels)
    assert kld < kld_fold * expected + 2e-3, f"divergence {kld:.4f} against the sampling level {expected:.4f}\n{report}"
    return counts, P


class TestFamilies:
    def test_constitutive(self):
        check_family(CMEModel("Constitutive", "None"), np.log10([0.5, 0.3]))

    def test_bursty_poisson_lengths_and_ambiguous(self):
        m = CMEModel("Bursty", "Poisson", amb_model="Unequal")
        samp = np.log10([0.6, 0.8]) + np.array([np.log10(1.5), 0.0])     # a gene-length offset on the unspliced channel
        check_family(m, np.log10([4.0, 0.5, 0.3, 0.1, 0.3]), samp)

    def test_extrinsic_bernoulli(self):
        check_family(CMEModel("Extrinsic", "Bernoulli"), np.log10([2.0, 0.4, 0.3]), np.log10([0.7, 0.5]))

    def test_delay(self):
        check_family(CMEModel("Delay", "None"), np.log10([3.0, 0.5, 0.4]))

    def test_delayed_splicing_poisson(self):
        check_family(CMEModel("DelayedSplicing", "Poisson"), np.log10([3.0, 0.6, 0.3]), np.log10([0.8, 0.9]))

    def test_protein(self):
        check_family(CMEModel("ProteinBursty", "None"), np.log10([3.0, 0.5, 0.3, 1.0, 0.5]), n_cells=12000)

    def test_protein_with_ambiguous_channel(self):
        m = CMEModel("ProteinBursty", "None", amb_model="Equal")
        check_family(m, np.log10([2.0, 0.6, 0.4, 0.8, 0.5, 0.2]), n_cells=12000)

    def test_network_constitutive_protein(self):
        m = CMEModel.compose("constitutive", protein=True)
        check_family(m, np.log10([0.5, 0.3, 1.0, 0.5]), n_cells=12000)

    def test_network_with_a_delayed_step(self):
        m = CMEModel("Custom", "None", network="0 -> B x unspliced => spliced -> 0", normalize_production_rate=True)
        assert m.network.all_params == ["B", "tauinv0", "k1", "k2"]
        check_family(m, np.log10([3.0, 0.5, 0.8, 0.3]), n_cells=2000, kld_fold=2.0)

    def test_chromatin_bursty_poisson_dropout(self):
        m = CMEModel("Chromatin", "Poisson", n_sites=2, rule="switch", bursty=True)
        theta = np.log10([0.5, 0.8, 1.0, 2.0, 0.7, 0.3, 2.0])            # k_on x2, eps, rate, beta, gamma, b
        check_family(m, theta, np.log10([0.7, 0.9, 0.1]), n_cells=12000)

    def test_chromatin_protein_with_ambiguous_channel(self):
        m = CMEModel("Chromatin", "Poisson", n_sites=2, rule="switch", bursty=True, protein=True, amb_model="Equal")
        theta = np.log10([0.5, 0.8, 1.0, 2.0, 0.7, 0.3, 2.0, 0.5, 0.5, 0.15])
        check_family(m, theta, np.log10([0.7, 0.9, 0.1]), n_cells=12000)

    def test_chromatin_additive_constitutive(self):
        m = CMEModel("Chromatin", "None", n_sites=2, rule="additive")
        theta = np.log10([0.5, 1.5, 0.7, 1.0, 2.0, 0.7, 0.4])            # k_on x2, eps, rates x2, beta, gamma
        check_family(m, theta, None, n_cells=12000)

    def test_chromatin_protein_poisson_dropout(self):
        # capture on the two RNA channels, dropout on the locus, nothing on the protein
        m = CMEModel("Chromatin", "Poisson", n_sites=2, rule="switch", bursty=True, protein=True)
        theta = np.log10([0.5, 0.8, 1.0, 2.0, 0.7, 0.3, 2.0, 0.5, 0.5])  # ..., b, k_p, gamma_p
        check_family(m, theta, np.log10([0.7, 0.9, 0.1]), n_cells=12000)

    def test_chromatin_protein_with_ambiguous_channel(self):
        m = CMEModel("Chromatin", "Poisson", n_sites=2, rule="switch", bursty=True, protein=True, amb_model="Equal")
        theta = np.log10([0.5, 0.8, 1.0, 2.0, 0.7, 0.3, 2.0, 0.5, 0.5, 0.15])
        check_family(m, theta, np.log10([0.7, 0.9, 0.1]), n_cells=12000)

    def test_cir_is_not_a_finite_rate_process(self):
        with pytest.raises(NotImplementedError, match="inverse-Gaussian"):
            simulate_counts(CMEModel("CIR", "None"), np.log10([3.0, 0.5, 0.3]), 10)


def _fit(model, thetas, n_cells, samp=None, tmp_path=None, gp=GP, **kw):
    warnings.filterwarnings("ignore")
    a = simulate_adata(model, thetas, n_cells, samp, seed=1)
    extra = {}
    if model.seq_model != "None":
        extra = dict(samp_lb=list(samp[:2]), samp_ub=list(samp[:2]), gridsize=[1, 1])
    out = perform_inference(a, model, n_genes=len(thetas), filt_param=_permissive(model), gradient_params=gp, viz=False,
                            dataset_string="gen", **extra, **kw)
    names = [p.replace(r"\log_{10} ", "") for p in model.param_str]
    return out, np.column_stack([out.var[n].to_numpy() for n in names])


class TestRecovery:
    """The pipeline recovers the generating parameters from simulated cells."""

    def test_bursty_poisson_ambiguous(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        m = CMEModel("Bursty", "Poisson", amb_model="Unequal")
        thetas = np.log10([[4.0, 0.5, 0.3, 0.1, 0.3], [2.0, 1.0, 0.4, 0.2, 0.2], [6.0, 0.4, 0.6, 0.05, 0.4]])
        _, est = _fit(m, thetas, 5000, samp=np.log10([0.6, 0.8]))
        np.testing.assert_allclose(est, thetas, atol=0.15)

    def test_delay(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        thetas = np.log10([[3.0, 0.5, 0.4], [2.0, 0.8, 0.25]])
        _, est = _fit(CMEModel("Delay", "None"), thetas, 5000)
        np.testing.assert_allclose(est, thetas, atol=0.15)

    def test_protein(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        thetas = np.log10([[3.0, 0.5, 0.3, 1.0, 0.5], [2.0, 0.6, 0.4, 0.5, 0.3]])
        _, est = _fit(CMEModel("ProteinBursty", "None"), thetas, 4000)
        np.testing.assert_allclose(est[:, :3], thetas[:, :3], atol=0.15)
        np.testing.assert_allclose(est[:, 3] - est[:, 4], thetas[:, 3] - thetas[:, 4], atol=0.15)   # k_p / gamma_p

    def test_network_constitutive_protein(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        m = CMEModel.compose("constitutive", protein=True)
        thetas = np.log10([[0.5, 0.3, 1.0, 0.5], [0.8, 0.5, 0.6, 0.4]])
        _, est = _fit(m, thetas, 4000)
        np.testing.assert_allclose(est[:, :2], thetas[:, :2], atol=0.15)
        np.testing.assert_allclose(est[:, 2] - est[:, 3], thetas[:, 2] - thetas[:, 3], atol=0.15)

    def test_chromatin(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        p_drop = np.log10(0.1)
        m = CMEModel("Chromatin", "None", n_sites=2, rule="switch", bursty=True, dropout_bounds=(p_drop, p_drop, 1))
        thetas = np.log10([[0.5, 0.8, 1.0, 2.0, 0.7, 0.3, 2.0], [1.2, 0.4, 1.0, 1.5, 0.5, 0.4, 3.0]])
        _, est = _fit(m, thetas, 4000, samp=np.array([0.0, 0.0, p_drop]))
        np.testing.assert_allclose(est, thetas, atol=0.25)

    @slow
    def test_chromatin_protein(self, tmp_path, monkeypatch):
        # the NumPy kernel costs seconds per evaluation, so one gene and a short run
        monkeypatch.chdir(tmp_path)
        p_drop = np.log10(0.1)
        m = CMEModel("Chromatin", "None", n_sites=2, rule="switch", bursty=True, protein=True, dropout_bounds=(p_drop, p_drop, 1))
        thetas = np.log10([[0.5, 0.8, 1.0, 1.5, 0.7, 0.4, 1.5, 0.4, 0.8]])
        _, est = _fit(m, thetas, 4000, samp=np.array([0.0, 0.0, p_drop]), gp=dict(GP, max_iterations=30))
        np.testing.assert_allclose(est[:, :7], thetas[:, :7], atol=0.25)
        np.testing.assert_allclose(est[:, 7] - est[:, 8], thetas[:, 7] - thetas[:, 8], atol=0.25)   # k_p / gamma_p
