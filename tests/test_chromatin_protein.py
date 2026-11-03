"""The Chromatin model's protein layer (docs/chromatin-protein-design.md, stage 1,
2026-11-02): the chromatin stepper fed by the protein characteristics, checked
at the limits where another kernel gives the answer."""
import numpy as np
import pytest

from monod import chromatin as ch
from monod import chromatin_fit as cf
from monod.cme_toolbox import CMEModel

THETA = cf.join_theta([0.5, 0.8], 0.6, [3.0], 0.8, 0.3, 2.0, (0.5, 0.5))   # two-site switch, bursty, protein
L = [20, 24, 40, 4]
RNA_L = [20, 24, 4]
SMALL = cf.join_theta([0.5, 0.8], 0.6, [1.5], 0.8, 0.4, 1.5, (0.4, 0.8))   # a cheap grid for the stencil checks
L_SMALL = [14, 16, 14, 4]


def _rna(theta):
    return theta[:-2]


def _model(**kw):
    base = dict(n_sites=2, rule="switch", bursty=True, protein=True)
    base.update(kw)
    return CMEModel("Chromatin", base.pop("seq", "None"), **base)


class TestLayout:
    def test_parameter_vector(self):
        assert ch.n_params("switch", 2, bursty=True, protein=True) == 2 + 1 + 1 + 2 + 1 + 2
        assert ch.burst_size(THETA, "switch", 2, protein=True) == pytest.approx(2.0)
        assert ch.protein_rates(THETA) == pytest.approx((0.5, 0.5))
        assert ch.burst_size(_rna(THETA), "switch", 2) == pytest.approx(2.0)
        with pytest.raises(ValueError):
            ch.burst_size(THETA, "switch", 2)
        emb = cf.embed_in_table(THETA, "switch", 2, protein=True)
        assert emb.size == ch.n_params("table", 2, True, True)
        np.testing.assert_allclose(emb[-2:], THETA[-2:])

    def test_model(self):
        m = _model(seq="Poisson", protein_limit=20)
        assert m.model_modalities == ["unspliced", "spliced", "protein", "configuration"]
        assert m.get_num_params() == 9
        assert m.param_str[-2:] == [r"$\log_{10} k_p$", r"$\log_{10} \gamma_p$"]
        assert len(m.bio_bounds["phys_lb"]) == 9 and m.bio_bounds["phys_ub"][-2:] == [3.5, 3.5]
        assert m.filter_bounds["min_means"] == [0.01, 0.01, 1, 0.0]
        assert len(m.seq_bounds["samp_lb"]) == 3                        # two capture rates and the dropout
        d = m.to_dict()
        assert d["protein"] is True and d["protein_limit"] == 20
        m2 = CMEModel.from_dict(d)
        assert m2.protein and m2.protein_limit == 20 and m2.model_modalities == m.model_modalities
        assert not CMEModel("Chromatin", "None").protein
        with pytest.raises(ValueError, match="fit_unspliced"):
            CMEModel("Chromatin", "None", protein=True, fit_unspliced=False)

    def test_compose(self):
        c = CMEModel.compose("bursty", locus=dict(n_sites=2, rule="switch"), protein=dict(protein_limit=20), seq_model="Poisson")
        assert c.bio_model == "Chromatin" and c.protein and c.bursty and c.protein_limit == 20
        assert c.composition["protein"] == {"protein_limit": 20} and c.composition["locus"] == dict(n_sites=2, rule="switch")
        with pytest.raises(ValueError, match="fit_unspliced"):
            CMEModel.compose("bursty", locus=dict(n_sites=1), protein=dict(fit_unspliced=False))

    def test_simulator_says_not_yet(self):
        from monod.simulate import simulate_counts
        with pytest.raises(NotImplementedError, match="stage 2"):
            simulate_counts(_model(), THETA, 10)


class TestKernel:
    def test_rna_and_configuration_marginal_is_the_chromatin_model(self):
        pss = _model().eval_model_pss(THETA, L)
        assert pss.shape == tuple(L) and pss.sum() == pytest.approx(1.0, abs=1e-6)
        assert pss.sum(axis=(0, 1, 3))[-1] < 1e-4                        # the protein grid holds the mass
        ref = _model(protein=False).eval_model_pss(_rna(THETA), RNA_L)
        np.testing.assert_allclose(pss.sum(axis=2), ref, atol=2e-6)

    def test_constitutive_emission_too(self):
        theta = cf.join_theta([0.5, 0.8], 0.6, [2.0], 0.8, 0.3, None, (1.0, 0.5))
        pss = _model(bursty=False).eval_model_pss(theta, L)
        ref = _model(bursty=False, protein=False).eval_model_pss(theta[:-2], RNA_L)
        np.testing.assert_allclose(pss.sum(axis=2), ref, atol=2e-6)

    def test_table_with_equal_rates_is_protein_bursty(self):
        # one site transcribing at the same mean rate b whether open or closed: the locus is
        # invisible and the RNA + protein law is ProteinBursty's, burst frequency 1
        b, beta, gamma, k_p, gamma_p = 2.0, 0.8, 0.3, 1.0, 0.5
        m = CMEModel("Chromatin", "None", n_sites=1, rule="table", bursty=True, protein=True)
        theta = cf.join_theta([0.7], 1.0, [b, b], beta, gamma, b, (k_p, gamma_p))
        ref = CMEModel("ProteinBursty", "None").eval_model_pss(np.log10([b, beta, gamma, k_p, gamma_p]), [24, 28, 20])
        pss = m.eval_model_pss(theta, [24, 28, 20, 2]).sum(axis=-1)
        np.testing.assert_allclose(pss, ref, atol=2e-5)

    def test_constitutive_table_is_the_network_form(self):
        beta, gamma, k_p, gamma_p = 0.5, 0.3, 1.0, 0.5
        m = CMEModel("Chromatin", "None", n_sites=1, rule="table", protein=True)
        theta = cf.join_theta([0.7], 1.0, [1.0, 1.0], beta, gamma, None, (k_p, gamma_p))
        ref = CMEModel.compose("constitutive", protein=True).eval_model_pss(np.log10([beta, gamma, k_p, gamma_p]), [18, 20, 24])
        pss = m.eval_model_pss(theta, [18, 20, 24, 2]).sum(axis=-1)
        np.testing.assert_allclose(pss, ref, atol=2e-5)

    def test_no_translation_puts_the_protein_at_zero(self):
        theta = THETA.copy()
        theta[-2] = -8.0                                                  # k_p = 1e-8
        pss = _model().eval_model_pss(theta, L)
        assert pss[:, :, 1:, :].sum() < 1e-5
        ref = _model(protein=False).eval_model_pss(_rna(THETA), RNA_L)
        np.testing.assert_allclose(pss[:, :, 0, :], ref, atol=2e-6)

    def test_fast_protein_tracks_the_mature_count(self):
        # gamma_p far above every other rate with k_p / gamma_p = r: P | S is Poisson(r S)
        r, gamma_p = 2.0, 300.0
        theta = cf.join_theta([0.5, 0.8], 0.6, [3.0], 0.8, 0.3, 2.0, (r * gamma_p, gamma_p))
        pss = _model().eval_model_pss(theta, [24, 28, 60, 4])
        joint = pss.sum(axis=(0, 3))                                      # (S, P)
        pS = joint.sum(axis=1)
        cond_mean = (joint @ np.arange(60)) / np.maximum(pS, 1e-300)
        s = np.arange(28)
        ok = (pS > 1e-3) & (s > 0)
        np.testing.assert_allclose(cond_mean[ok], r * s[ok], rtol=0.05)
        assert cond_mean[0] < 0.1

    def test_sampling_and_dropout_act_on_their_axes(self):
        samp = np.log10([0.6, 0.8, 0.1])
        pss = _model(seq="Poisson").eval_model_pss(THETA, L, samp)
        ref = _model(seq="Poisson", protein=False).eval_model_pss(_rna(THETA), RNA_L, samp)
        np.testing.assert_allclose(pss.sum(axis=2), ref, atol=2e-6)
        plain = _model().eval_model_pss(THETA, L)                         # the protein is not captured
        np.testing.assert_allclose(pss.sum(axis=(0, 1, 3)), plain.sum(axis=(0, 1, 3)), atol=2e-6)

    def test_coarse_protein_grid_is_protein_burstys(self):
        # protein_limit keeps the low protein frequencies and zero-pads the inverse transform, the
        # approximation ProteinBursty makes for wide protein distributions; under the same setting
        # the two kernels agree, and against the full grid it is only an approximation
        b, beta, gamma, k_p, gamma_p = 2.0, 0.8, 0.3, 1.0, 0.5
        theta = cf.join_theta([0.7], 1.0, [b, b], beta, gamma, b, (k_p, gamma_p))
        p5 = np.log10([b, beta, gamma, k_p, gamma_p])
        lim = [24, 28, 60]
        coarse = CMEModel("Chromatin", "None", n_sites=1, rule="table", bursty=True, protein=True, protein_limit=20)
        pss = coarse.eval_model_pss(theta, lim + [2]).sum(axis=-1)
        ref = CMEModel("ProteinBursty", "None", protein_limit=20).eval_model_pss(p5, lim)
        np.testing.assert_allclose(pss, ref, atol=2e-5)
        fine = CMEModel("ProteinBursty", "None").eval_model_pss(p5, lim)
        assert 1e-4 < np.abs(pss - fine).max() < 1e-2

    def test_pinned_schedule_replays_and_the_gradient_is_clean(self):
        m = _model()
        pss = m.eval_model_pss(SMALL, L_SMALL)
        assert pss.sum(axis=(0, 1, 3))[-1] < 1e-4
        with m.pinned_schedule(SMALL, L_SMALL):
            np.testing.assert_allclose(m.eval_model_pss(SMALL, L_SMALL), pss, atol=1e-13)   # the contract suite's replay tolerance
        P2 = m.eval_model_pss(SMALL + 0.1, L_SMALL)                       # data from the kernel at a displaced point
        keep = P2 > 1e-6
        data = (np.argwhere(keep), P2[keep] / P2[keep].sum())
        _, grad = m.eval_model_kld_and_grad(SMALL, L_SMALL, None, data)
        h = 1e-3
        fd = np.array([(m.eval_model_kld(SMALL + h * e, L_SMALL, None, data) - m.eval_model_kld(SMALL - h * e, L_SMALL, None, data)) / (2 * h)
                       for e in np.eye(SMALL.size)])
        np.testing.assert_allclose(grad, fd, rtol=0.02, atol=1e-4)

    def test_ambiguous_channel_goes_through_the_generic_route(self):
        m = _model(amb_model="Equal")
        assert m.model_modalities == ["unspliced", "spliced", "protein", "configuration", "ambiguous"]
        theta = np.concatenate([SMALL, [np.log10(0.05)]])                 # a small share, so an ambiguous axis of 6 loses nothing
        lim = [22, 24, L_SMALL[2], 4, 6]
        pss = m.eval_model_pss(theta, lim)
        assert pss.shape == tuple(lim) and pss.sum() == pytest.approx(1.0, abs=1e-6)
        ref = _model().eval_model_pss(SMALL, [lim[0] + 5, lim[1] + 5, lim[2], 4])
        np.testing.assert_allclose(pss.sum(axis=(0, 1, 4)), ref.sum(axis=(0, 1)), atol=2e-5)    # the split leaves protein and locus alone


class TestStart:
    def test_moments_start_from_the_kernel_histogram(self):
        m = _model(dropout_bounds=(0.0, 0.0, 1))
        P = m.eval_model_pss(THETA, L)
        hist = cf.sparse_histogram(P)
        lb, ub = np.array(m.bio_bounds["phys_lb"]), np.array(m.bio_bounds["phys_ub"])
        x0 = m.get_MoM(None, lb, ub, None, hist=hist, limits=L)
        assert x0.size == 9
        np.testing.assert_allclose(x0[:3], THETA[:3], atol=0.3)          # the chain from the configuration marginal
        assert abs((x0[-2] - x0[-1]) - (THETA[-2] - THETA[-1])) < 0.2    # k_p / gamma_p from the means
        assert np.all(x0 >= lb) and np.all(x0 <= ub)
