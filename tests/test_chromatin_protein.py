"""The Chromatin model's protein layer (docs/chromatin.md, stage 1,
2026-11-02): the chromatin stepper fed by the protein characteristics, checked
at the limits where another kernel gives the answer."""
import warnings

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
        # the box is in the locus clock (k_off = 1), not Bursty's burst clock: beta, gamma and gamma_p reach lower
        assert m.bio_bounds["phys_lb"][4:6] == [-2.5, -2.5] and m.bio_bounds["phys_lb"][-2:] == [-1.8, -3.0]
        assert m.filter_bounds["min_means"] == [0.01, 0.01, 1, 0.0]
        assert len(m.seq_bounds["samp_lb"]) == 3                        # two capture rates and the dropout
        d = m.to_dict()
        assert d["protein"] is True and d["protein_limit"] == 20
        m2 = CMEModel.from_dict(d)
        assert m2.protein and m2.protein_limit == 20 and m2.model_modalities == m.model_modalities
        assert not CMEModel("Chromatin", "None").protein
        with pytest.raises(ValueError, match="fit_unspliced"):
            CMEModel("Chromatin", "None", protein=True, fit_unspliced=False)

    def test_protein_background_is_a_fourth_sampling_parameter(self):
        m = _model(protein_background=(-1.0, 0.5, 4))
        assert m.protein_background == (-1.0, 0.5, 4)
        assert m.seq_bounds["samp_lb"] == [0, 0, -2.0, -1.0] and m.seq_bounds["gridsize"] == [1, 1, 3, 4]
        assert m.chromatin_background([0, 0, 0, np.log10(1.5)]) == pytest.approx(1.5)
        assert m.chromatin_background([0, 0, 0]) is None and m.chromatin_background([0, 0, 0, -np.inf]) is None
        assert _model().chromatin_background([0, 0, 0, 0.0]) is None            # no grid, no background
        m2 = CMEModel.from_dict(m.to_dict())
        assert m2.protein_background == (-1.0, 0.5, 4) and len(m2.seq_bounds["samp_lb"]) == 4
        assert "protein_background" not in _model().to_dict()
        c = CMEModel.compose("bursty", locus=dict(n_sites=2, rule="switch"), protein=dict(protein_limit=20, protein_background=(-1.0, 0.5, 2)))
        assert c.protein_background == (-1.0, 0.5, 2)
        with pytest.raises(ValueError, match="protein=True"):
            CMEModel("Chromatin", "None", protein_background=(-1.0, 0.5, 2))
        with pytest.raises(ValueError, match="grid points"):
            _model(protein_background=(-1.0, 0.5, 0))

    def test_compose(self):
        c = CMEModel.compose("bursty", locus=dict(n_sites=2, rule="switch"), protein=dict(protein_limit=20), seq_model="Poisson")
        assert c.bio_model == "Chromatin" and c.protein and c.bursty and c.protein_limit == 20
        assert c.composition["protein"] == {"protein_limit": 20} and c.composition["locus"] == dict(n_sites=2, rule="switch")
        with pytest.raises(ValueError, match="fit_unspliced"):
            CMEModel.compose("bursty", locus=dict(n_sites=1), protein=dict(fit_unspliced=False))

    def test_simulator_draws_the_four_modalities(self):
        # the agreement with the kernel is test_generative_checks' business; here the layout
        from monod.simulate import simulate_counts
        c = simulate_counts(_model(seq="Poisson"), THETA, 2000, np.log10([0.7, 0.9, 0.1]), seed=0)
        assert list(c) == ["unspliced", "spliced", "protein", "configuration"]
        assert all(v.dtype == np.int64 and v.size == 2000 for v in c.values())
        assert c["protein"].mean() > 0.5 * c["spliced"].mean() / 0.9          # k_p / gamma_p = 1, mature capture 0.9
        assert c["configuration"].max() == 3

    def test_simulator_adds_the_protein_background(self):
        from monod.simulate import simulate_counts
        lam = 2.5
        plain = simulate_counts(_model(), THETA, 4000, np.array([0.0, 0.0, 0.0]), seed=3)
        m = _model(protein_background=(-1.0, 1.0, 3))
        bg = simulate_counts(m, THETA, 4000, np.array([0.0, 0.0, 0.0, np.log10(lam)]), seed=3)
        none = simulate_counts(m, THETA, 4000, np.array([0.0, 0.0, 0.0, -np.inf]), seed=3)
        assert np.array_equal(none["protein"], plain["protein"])                  # -inf: no background
        for k in ("unspliced", "spliced", "configuration"):
            assert np.array_equal(bg[k], plain[k])                                  # the same latent cells
        extra = bg["protein"] - plain["protein"]
        assert extra.min() >= 0 and abs(extra.mean() - lam) < 0.1 and abs(extra.var() - lam) < 0.3

    def test_cell_factor_is_a_model_option(self):
        m = _model(cell_factor=(0.3, 5))
        assert m.cell_factor == (0.3, 5)
        f, w = m.chromatin_factor_nodes()
        assert f.shape == (5,) and abs(w.sum() - 1) < 1e-12 and abs(w @ f - 1) < 1e-12      # unit mean: the rates keep their meaning
        assert abs(w @ np.log(f) ** 2 - (w @ np.log(f)) ** 2 - 0.3 ** 2) < 1e-3               # the spread is sigma
        assert _model().chromatin_factor_nodes() == (None, None) and _model(cell_factor=(0.0, 5)).chromatin_factor_nodes() == (None, None)
        m2 = CMEModel.from_dict(m.to_dict())
        assert m2.cell_factor == (0.3, 5) and m2.model_modalities == m.model_modalities
        assert "cell_factor" not in _model().to_dict()
        assert m._chromatin_rna_twin().cell_factor == (0.3, 5)                               # the RNA scales with the cell too
        with pytest.raises(ValueError, match="Chromatin"):
            CMEModel("Bursty", "None", cell_factor=(0.3, 5))
        with pytest.raises(ValueError, match="cell_factor"):
            _model(cell_factor=(-0.1, 5))

    def test_simulator_draws_the_cell_factor(self):
        from monod.simulate import simulate_counts
        sigma = 0.25
        plain = simulate_counts(_model(), THETA, 6000, np.array([0.0, 0.0, 0.0]), seed=5)
        fac = simulate_counts(_model(cell_factor=(sigma, 5)), THETA, 6000, np.array([0.0, 0.0, 0.0]), seed=5)
        # the top quadrature node doubles the burst size, so the kernel needs a wider RNA grid than L:
        # on L the mixture loses 1% of the mature mean and a tenth of its variance to the grid edge
        pss = _model(cell_factor=(sigma, 5)).eval_model_pss(THETA, [24, 36, 60, 4])
        for k, ax in (("spliced", 1), ("protein", 2)):
            mp, mf = plain[k].mean(), fac[k].mean()
            assert abs(mf / mp - 1) < 0.08                                                   # unit-mean factor: the means stay
            # the factor adds at least (e^sigma^2 - 1) mean^2 to the variance (more under bursty
            # emission, whose intrinsic variance grows with the burst size); the kernel's own
            # mixture says how much
            marg = pss.sum(axis=tuple(j for j in range(4) if j != ax)); x = np.arange(marg.size)
            v_kernel = marg @ x ** 2 - (marg @ x) ** 2
            assert fac[k].var() - plain[k].var() > 0.5 * (np.exp(sigma ** 2) - 1) * mp ** 2
            assert abs(fac[k].var() / v_kernel - 1) < 0.15
        m = _model(protein_background=(-1.0, 1.0, 3), cell_factor=(sigma, 5))
        bg = simulate_counts(m, THETA, 6000, np.array([0.0, 0.0, 0.0, np.log10(3.0)]), seed=5)
        assert abs(bg["protein"].mean() - fac["protein"].mean() - 3.0) < 0.2                # the background scales with the cell too


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

    def test_background_is_a_poisson_convolution_on_the_protein_axis(self):
        from scipy.stats import poisson
        lam = 1.5
        m = _model(protein_background=(-1.0, 1.0, 3))
        pss = m.eval_model_pss(THETA, L, np.array([0.0, 0.0, 0.0, np.log10(lam)]))
        plain = _model().eval_model_pss(THETA, L)
        np.testing.assert_allclose(pss.sum(axis=2), plain.sum(axis=2), atol=2e-6)     # RNA and locus untouched
        lp = L[2]
        pmf = poisson.pmf(np.arange(lp), lam)
        conv = np.zeros_like(plain)
        for k in range(lp):                                                       # P_obs = P + background, truncated
            conv[:, :, k:, :] += pmf[k] * plain[:, :, :lp - k, :]
        np.testing.assert_allclose(pss, conv, atol=2e-6)
        assert m.eval_model_pss(THETA, L, np.array([0.0, 0.0, 0.0, -np.inf])) == pytest.approx(plain, abs=1e-12)

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

    def test_step_cap_is_reported(self, monkeypatch):
        # a truncated integration must not pass silently (before the Magnus step the Mimitou fits found a
        # divergence of 3.8 at gamma_p = -4 where -3.5 gave 0.5, from the stepper's cap)
        m = _model()                                                                        # join_theta takes linear rates
        theta = cf.join_theta([0.22, 0.05], 0.63, [0.02], 0.32, 0.003, 0.25, (0.08, 1e-2))
        monkeypatch.setattr(ch, "_MAX_STEPS", 12)
        with pytest.warns(RuntimeWarning, match="cap of"):
            pss = m.eval_model_pss(theta, [4, 4, 6, 4])
        assert pss.shape == (4, 4, 6, 4)

    @pytest.mark.parametrize("gamma_p", [1e-3, 10 ** -3.5])
    def test_slow_protein_costs_a_few_hundred_steps(self, gamma_p):
        # a protein living thousands of locus clocks: the Magnus steps are exact for a constant forcing, so
        # the tail goes in steps of hundreds of clocks and the whole run stays near 500 steps, Strang finishing
        # the fast transient (Strang alone took about one clock per step, 12,000 steps at gamma_p = 1e-3 and
        # its cap of 20,000 at 10^-3.5); the RNA and configuration law does not depend on the protein, so the
        # marginal must still be the chromatin model's
        beta, gamma, k_p, b = 1.0, 10 ** -2.47, 10 ** -0.66, 10 ** -0.5                  # CD27 at capture 0.1
        H, A, C = ch.nascent_mature_system([0.1, 0.1], 1.0, 1.0, "switch", beta, gamma, rule_params=[10 ** -1.6])
        with warnings.catch_warnings():
            warnings.simplefilter("error")                                                 # no cap, stepper or table
            pss, sched = ch.eval_joint_pss(H, A, C, [6, 8, 16], burst=b, protein=(k_p, gamma_p), return_schedule=True)
        assert sched.size < 1000 and sched.magnus >= 50
        assert float(np.asarray(sched).max()) > 100.0
        ref = ch.eval_joint_pss(H, A, C, [6, 8], burst=b)
        np.testing.assert_allclose(pss.sum(axis=3), ref, atol=2e-6)

    def test_ambiguous_channel_goes_through_the_generic_route(self):
        m = _model(amb_model="Equal")
        assert m.model_modalities == ["unspliced", "spliced", "protein", "configuration", "ambiguous"]
        theta = np.concatenate([SMALL, [np.log10(0.05)]])                 # a small share, so an ambiguous axis of 6 loses nothing
        lim = [22, 24, L_SMALL[2], 4, 6]
        pss = m.eval_model_pss(theta, lim)
        assert pss.shape == tuple(lim) and pss.sum() == pytest.approx(1.0, abs=1e-6)
        ref = _model().eval_model_pss(SMALL, [lim[0] + 5, lim[1] + 5, lim[2], 4])
        np.testing.assert_allclose(pss.sum(axis=(0, 1, 4)), ref.sum(axis=(0, 1)), atol=2e-5)    # the split leaves protein and locus alone

    def test_cell_factor_is_a_mixture_over_scaled_production(self):
        sigma = 0.3
        m = _model(protein_background=(-1.0, 1.0, 3), cell_factor=(sigma, 5))
        samp = np.array([0.0, 0.0, 0.0, np.log10(1.5)])
        pss = m.eval_model_pss(THETA, L, samp)
        f, w = m.chromatin_factor_nodes()
        plain = _model(protein_background=(-1.0, 1.0, 3))
        mix = 0.0
        for fi, wi in zip(f, w):
            theta = THETA.copy()
            theta[3] += np.log10(fi)                                       # the switch rule's production rate
            theta[6] += np.log10(fi)                                       # the burst size: frequency unchanged, mean x f
            samp_f = samp.copy(); samp_f[3] += np.log10(fi)                # the background scales with the cell
            mix = mix + wi * plain.eval_model_pss(theta, L, samp_f)
        np.testing.assert_allclose(pss, mix, atol=1e-9)
        assert isinstance(m._last_schedule, list) and len(m._last_schedule) == 5
        np.testing.assert_allclose(_model(cell_factor=(0.0, 5)).eval_model_pss(THETA, L), _model().eval_model_pss(THETA, L), atol=1e-13)
        # the RNA mean is kept (unit-mean factor), the protein variance grows by about (e^sigma^2 - 1) mean^2
        base = _model().eval_model_pss(THETA, L)
        x = np.arange(L[1]); y = np.arange(L[2])
        ps, pf = base.sum(axis=(0, 2, 3)), _model(cell_factor=(sigma, 5)).eval_model_pss(THETA, L).sum(axis=(0, 2, 3))
        assert abs((pf @ x) / (ps @ x) - 1) < 0.04                          # unit mean up to what L's edge takes from the top node
        qs, qf = base.sum(axis=(0, 1, 3)), _model(cell_factor=(sigma, 5)).eval_model_pss(THETA, L).sum(axis=(0, 1, 3))
        mean = qs @ y
        extra = (qf @ y ** 2 - (qf @ y) ** 2) - (qs @ y ** 2 - mean ** 2)
        assert extra > 0.6 * (np.exp(sigma ** 2) - 1) * mean ** 2          # more than the mean's spread alone: the bursts grow too

    def test_cell_factor_pins_every_node_schedule(self):
        m = _model(cell_factor=(0.25, 3))
        pss = m.eval_model_pss(SMALL, L_SMALL)
        with m.pinned_schedule(SMALL, L_SMALL):
            assert len(m._fixed_schedule) == 3
            np.testing.assert_allclose(m.eval_model_pss(SMALL, L_SMALL), pss, atol=1e-13)
        P2 = m.eval_model_pss(SMALL + 0.1, L_SMALL)
        keep = P2 > 1e-6
        data = (np.argwhere(keep), P2[keep] / P2[keep].sum())
        _, grad = m.eval_model_kld_and_grad(SMALL, L_SMALL, None, data)
        h = 1e-3
        fd = np.array([(m.eval_model_kld(SMALL + h * e, L_SMALL, None, data) - m.eval_model_kld(SMALL - h * e, L_SMALL, None, data)) / (2 * h)
                       for e in np.eye(SMALL.size)])
        np.testing.assert_allclose(grad, fd, rtol=0.02, atol=1e-4)


class TestPipeline:
    """perform_inference on simulated four-layer data (design doc, stage 4): a three-point
    dropout grid under Poisson capture, the ambiguous channel through the generic route,
    the model definition through h5ad."""
    TRUTH = cf.join_theta([1.0, 1.5], 0.6, [1.0], 1.0, 0.8, 1.2, (0.6, 1.0))    # the contract suite's point: 0.3 s per evaluation
    CAPTURE = np.log10([0.7, 0.9])
    P_DROP = 0.1

    def _adata(self, model, theta, n_cells, seed):
        from monod.simulate import simulate_adata
        return simulate_adata(model, [theta], n_cells, np.append(self.CAPTURE, np.log10(self.P_DROP)), seed=seed, gene_names=["geneA"])

    def test_dropout_grid_poisson_capture_and_h5ad_round_trip(self, tmp_path, monkeypatch):
        import anndata as ad
        from monod.inference import perform_inference
        monkeypatch.chdir(tmp_path)
        grid = np.log10([0.05, 0.1, 0.2])
        m = _model(seq="Poisson", dropout_bounds=(grid[0], grid[-1], 3))
        path = tmp_path / "four_layers.h5ad"
        self._adata(m, self.TRUTH, 2000, 7).write_h5ad(path)
        gp = {"max_iterations": 3, "init_pattern": "moments", "num_restarts": 1, "num_gene_cores": 1, "check_identifiability": False}
        out = perform_inference(str(path), m, n_genes=1, viz=False, gradient_params=gp, dataset_string="chrom_prot", exclude_sigma=True,
                                filt_param={"min_means": [0] * 4, "max_maxes": [10_000] * 4, "min_maxes": [0] * 4},
                                samp_lb=list(self.CAPTURE) + [grid[0]], samp_ub=list(self.CAPTURE) + [grid[-1]], gridsize=[1, 1, 3])
        assert list(out.layers) == ["unspliced", "spliced", "protein", "configuration"]
        names = [p.replace(r"\log_{10} ", "") for p in m.param_str]
        est = np.array([float(out.var[n].iloc[0]) for n in names])
        assert est.size == 9 and np.all(np.isfinite(est))
        np.testing.assert_allclose(est[:3], self.TRUTH[:3], atol=0.3)                       # the chain
        assert abs((est[-2] - est[-1]) - (self.TRUTH[-2] - self.TRUTH[-1])) < 0.3          # k_p / gamma_p
        # one gene cannot tell dropout from its opening rates (the two-site marginal is saturated by the chain),
        # so the grid is checked as plumbing: three dropout points scanned, capture held, a grid value reported
        sr = out.uns["search_result"]
        assert list(np.asarray(sr["sp"]["gridsize"])) == [1, 1, 3] and 0 <= int(sr["samp_optimum_ind"]) < 3
        opt = np.asarray(sr["samp_optimum"], dtype=float)
        np.testing.assert_allclose(opt[:2], self.CAPTURE)
        assert np.any(np.isclose(opt[2], grid))
        out.write_h5ad(tmp_path / "fitted.h5ad")                                            # the model definition survives the disk
        m2 = CMEModel.from_dict(dict(ad.read_h5ad(tmp_path / "fitted.h5ad").uns["model"]))
        assert m2.protein and m2.bursty and m2.model_modalities == m.model_modalities and m2.dropout_bounds[2] == 3
        # the fitted PMF, its marginals and the rejection test on four modalities (the Mimitou pass-2
        # fit crashed here: the test indexed two or three modalities, and the protein marginal
        # collapsed the configuration axis, which the kernel refuses)
        from monod.inference import search_result_from_uns, searchdata_from_adata
        sr_ = search_result_from_uns(out.uns["search_result"]); sd = searchdata_from_adata(out)
        limits = np.asarray(sd.M[:, 0])
        joint = sr_.gene_pss(sd, 0)
        assert joint.shape == tuple(limits) and abs(joint.sum() - 1) < 1e-6
        np.testing.assert_allclose(joint, m2.eval_model_pss(sr_.phys_optimum[0], limits, sr_.regressor_optimum[0]), rtol=0, atol=1e-12)
        other = np.array(sr_.regressor_optimum[0], dtype=float); other[2] = -4.0             # the dropout optimum is part of the PMF
        assert np.abs(joint - m2.eval_model_pss(sr_.phys_optimum[0], limits, other)).max() > 1e-6
        for k, mod in enumerate(m2.model_modalities):
            marg = sr_.gene_pss(sd, 0, marg=mod)
            assert marg.shape == (limits[k],)
            np.testing.assert_allclose(marg, joint.sum(axis=tuple(j for j in range(4) if j != k)), atol=1e-12)
        csq, pval, hel = sr_.chisquare_testing(sd, reject_at_bounds=True)
        assert np.all(np.isfinite(csq)) and 0 <= hel[0] < 1.5 and sr_.rejected_genes.shape == (1,)
        # the statistic's sampling floor at this cell count, and the excess the threshold is applied to
        assert 0 < sr_.hellinger_floor[0] < hel[0] + 1e-9 and sr_.hellinger_excess[0] == pytest.approx(hel[0] - sr_.hellinger_floor[0])
        back = search_result_from_uns(sr_.to_dict())
        np.testing.assert_allclose(back.hellinger_floor, sr_.hellinger_floor)

    def test_per_gene_protein_background_rides_on_the_sampling_vector(self, tmp_path, monkeypatch):
        import anndata as ad
        from monod.inference import perform_inference, search_result_from_uns, searchdata_from_adata, InferenceParameters
        monkeypatch.chdir(tmp_path)
        lam = np.log10(2.0)
        m = _model(seq="Poisson", dropout_bounds=(np.log10(0.1), np.log10(0.1), 1), protein_background=(0.0, 0.0, 1))
        a = self._adata(m, self.TRUTH, 1500, 11)
        a.var["log10_protein_background"] = [lam]                                           # the antibody's measured level
        path = tmp_path / "bg.h5ad"; a.write_h5ad(path)
        gp = {"max_iterations": 2, "init_pattern": "moments", "num_restarts": 1, "num_gene_cores": 1, "check_identifiability": False}
        kw = dict(viz=False, gradient_params=gp, exclude_sigma=True, n_genes=1,
                  filt_param={"min_means": [0] * 4, "max_maxes": [10_000] * 4, "min_maxes": [0] * 4})
        out = perform_inference(str(path), m, dataset_string="bg_fixed", samp_lb=list(self.CAPTURE) + [np.log10(0.1), 0.0],
                                samp_ub=list(self.CAPTURE) + [np.log10(0.1), 0.0], gridsize=[1, 1, 1, 1], **kw)
        sr = search_result_from_uns(out.uns["search_result"])
        np.testing.assert_allclose(sr.regressor_optimum[0], list(self.CAPTURE) + [np.log10(0.1), lam])      # grid value + per-gene offset
        assert np.asarray(sr.sp.gene_samp_offsets).shape == (1, 4)                           # survives the h5ad round trip
        sd = searchdata_from_adata(out)
        np.testing.assert_allclose(sr.gene_pss(sd, 0), m.eval_model_pss(sr.phys_optimum[0], np.asarray(sd.M[:, 0]), sr.regressor_optimum[0]), atol=1e-12)
        # a shared factor scanned on top of the per-gene level
        m2 = _model(seq="Poisson", dropout_bounds=(np.log10(0.1), np.log10(0.1), 1), protein_background=(-0.3, 0.3, 3))
        out2 = perform_inference(str(path), m2, dataset_string="bg_scan", samp_lb=list(self.CAPTURE) + [np.log10(0.1), -0.3],
                                 samp_ub=list(self.CAPTURE) + [np.log10(0.1), 0.3], gridsize=[1, 1, 1, 3], **kw)
        sr2 = search_result_from_uns(out2.uns["search_result"])
        assert np.any(np.isclose(sr2.regressor_optimum[0][3] - lam, [-0.3, 0.0, 0.3]))
        # the column without a background model is an error; a wrong shape is an error
        with pytest.raises(ValueError):
            perform_inference(str(path), _model(seq="Poisson"), dataset_string="bg_wrong", **kw)
        with pytest.raises(ValueError):
            InferenceParameters("x", m, gene_samp_offsets=np.zeros((2, 3)), samp_lb=[0, 0, -1, 0], samp_ub=[0, 0, -1, 0], gridsize=[1, 1, 1, 1]) \
                .fit_all_grid_points(searchdata_from_adata(out), num_cores=1)

    def test_extract_warns_when_an_aggregate_sets_the_protein_limit(self, tmp_path, monkeypatch, caplog):
        import logging
        from monod.extract_data import extract_data, grid_limits
        monkeypatch.chdir(tmp_path)
        m = _model(seq="Poisson")
        a = self._adata(m, self.TRUTH, 500, 3)
        prot = np.asarray(a.layers["protein"]).copy()
        prot[0, 0] = 2000                                                                    # one antibody aggregate
        a.layers["protein"] = prot
        path = tmp_path / "aggregate.h5ad"; a.write_h5ad(path)
        filt = {"min_means": [0] * 4, "max_maxes": [10_000] * 4, "min_maxes": [0] * 4}
        with caplog.at_level(logging.WARNING, logger="monod"):
            out = extract_data(str(path), m, n_genes=1, viz=False, dataset_name="agg", filt_param=filt)
        assert grid_limits(out)[2, 0] == 2010                                               # the aggregate sets the limit
        assert any("aggregate" in r.getMessage() and "geneA" in r.getMessage() for r in caplog.records)
        caplog.clear()
        prot[0, 0] = 0; a.layers["protein"] = prot; a.write_h5ad(path)
        with caplog.at_level(logging.WARNING, logger="monod"):
            extract_data(str(path), m, n_genes=1, viz=False, dataset_name="agg", filt_param=filt)
        assert not any("aggregate" in r.getMessage() for r in caplog.records)

    def test_ambiguous_channel_through_the_pipeline(self, tmp_path, monkeypatch):
        from monod.inference import perform_inference
        monkeypatch.chdir(tmp_path)
        m = _model(seq="Poisson", amb_model="Equal", dropout_bounds=(np.log10(self.P_DROP), np.log10(self.P_DROP), 1))
        theta = np.append(self.TRUTH, np.log10(0.15))
        a = self._adata(m, theta, 2000, 8)
        assert list(a.layers) == ["unspliced", "spliced", "protein", "configuration", "ambiguous"]
        gp = {"max_iterations": 1, "init_pattern": "moments", "num_restarts": 1, "num_gene_cores": 1, "check_identifiability": False}
        out = perform_inference(a, m, n_genes=1, viz=False, gradient_params=gp, dataset_string="chrom_prot_amb", exclude_sigma=True,
                                filt_param={"min_means": [0] * 5, "max_maxes": [10_000] * 5, "min_maxes": [0] * 5},
                                samp_lb=list(self.CAPTURE) + [np.log10(self.P_DROP)], samp_ub=list(self.CAPTURE) + [np.log10(self.P_DROP)],
                                gridsize=[1, 1, 1])
        assert "$p$" in out.var.columns and abs(float(out.var["$p$"].iloc[0]) - np.log10(0.15)) < 0.3
        assert CMEModel.from_dict(dict(out.uns["model"])).amb_model == "Equal"


needs_rust_protein = pytest.mark.skipif(not ch._rust_available(True), reason="monod_core without the protein layer of eval_chromatin_pgf")


@needs_rust_protein
class TestRustBackend:
    """The Rust kernel (design doc, stage 5) follows the NumPy one step for step:
    the same tabulation schedule for the characteristics and, on a replayed
    schedule, the same steps to rounding, through the affine-map table instead
    of a table over the mesh.  The adaptive runs may part: the Magnus step's
    error estimate is a small difference of large exponentials, so the last
    bit of rounding (the two tables are different arithmetic) can move a step
    by 1e-9 and the controllers then drift apart, both within tolerance."""

    def _system(self, rule, rates, bursty, k_p=0.5, gamma_p=0.5, beta=0.8, gamma=0.3, k_on=(0.5, 0.8), eps=0.6):
        H, A, C = ch.nascent_mature_system(list(k_on), 1.0, eps, rule, beta, gamma, rule_params=rates)
        return H, A, C, (2.0 if bursty else None), (k_p, gamma_p)

    @pytest.mark.parametrize("case", ["switch-bursty", "table", "additive-bursty", "independent"])
    def test_matches_numpy_in_every_mode(self, case):
        rule, bursty = {"switch-bursty": ("switch", True), "table": ("table", False),
                        "additive-bursty": ("additive", True), "independent": ("additive", False)}[case]
        rates = {"switch": [1.5], "table": [0.3, 2.0, 1.0, 1.5], "additive": [1.5, 0.8]}[rule]
        H, A, C, burst, protein = self._system(rule, rates, bursty, eps=1.0 if case == "independent" else 0.6)
        lim = [14, 16, 14, 4][:3]
        p_py, s_py = ch.eval_joint_pss(H, A, C, lim, burst=burst, protein=protein, backend="python", return_schedule=True)
        p_rs, s_rs = ch.eval_joint_pss(H, A, C, lim, burst=burst, protein=protein, backend="rust", return_schedule=True)
        assert abs(s_rs.size - s_py.size) <= 0.03 * s_py.size + 2 and s_rs.magnus >= 1
        np.testing.assert_allclose(p_rs, p_py, atol=1e-8)                 # two adaptive runs within tolerance
        replay = ch.eval_joint_pss(H, A, C, lim, burst=burst, protein=protein, fixed_schedule=s_py, backend="rust")
        np.testing.assert_allclose(replay, p_py, atol=1e-11)              # the same steps give the same numbers
        back = ch.eval_joint_pss(H, A, C, lim, burst=burst, protein=protein, fixed_schedule=s_rs, backend="python")
        np.testing.assert_allclose(back, p_rs, atol=1e-11)

    def test_small_grid_runs_the_sequential_path(self):
        H, A, C, burst, protein = self._system("switch", [1.5], True, k_on=(0.7,), eps=1.0)
        lim = [8, 8, 6]                                                   # 2 states x 8 x 8 x 4 points < 4096
        p_py = ch.eval_joint_pss(H, A, C, lim, burst=burst, protein=protein, backend="python")
        p_rs = ch.eval_joint_pss(H, A, C, lim, burst=burst, protein=protein, backend="rust")
        np.testing.assert_allclose(p_rs, p_py, atol=1e-8)

    def test_model_takes_the_rust_path_with_capture_and_a_coarse_protein_axis(self):
        m = _model(seq="Poisson", protein_limit=10)
        samp = np.log10([0.6, 0.8, 0.1])
        lim = [16, 18, 30, 4]
        fast = m.eval_model_pss(THETA, lim, samp)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(ch, "_rust_available", lambda protein=False: False)
            slow = m.eval_model_pss(THETA, lim, samp)
        np.testing.assert_allclose(fast, slow, atol=1e-8)
        with pytest.raises(ValueError):
            ch.eval_joint_pss(*self._system("switch", [1.5], False)[:3], [8, 8, 6], protein=(-1.0, 0.5), backend="rust")


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

    def test_moments_start_with_a_protein_background(self):
        # the RNA twin used for the chain and RNA rates must drop the background with the
        # protein layer; it used to refuse it, and every gene then started from the middle
        # of the box without any sign of it (pass 4 on the DOGMA-seq cells)
        m = _model(dropout_bounds=(0.0, 0.0, 1), protein_background=(0.0, 0.0, 1))
        lam = 3.0
        samp = np.array([0.0, 0.0, 0.0, np.log10(lam)])
        P = m.eval_model_pss(THETA, L, samp)
        hist = cf.sparse_histogram(P)
        lb, ub = np.array(m.bio_bounds["phys_lb"]), np.array(m.bio_bounds["phys_ub"])
        x0 = m.get_MoM(None, lb, ub, samp, hist=hist, limits=L)
        mid = 0.5 * (lb + ub)
        assert not np.allclose(x0, mid), "the start fell back to the middle of the box"
        np.testing.assert_allclose(x0[:3], THETA[:3], atol=0.3)
        assert abs((x0[-2] - x0[-1]) - (THETA[-2] - THETA[-1])) < 0.2    # background taken off the protein mean
        no_bg = _model(dropout_bounds=(0.0, 0.0, 1)).get_MoM(None, lb, ub, samp[:3], hist=hist, limits=L)
        assert x0[-2] < no_bg[-2]                                        # ignoring the background would overstate k_p

    def test_moments_start_without_nascent_counts_in_open_cells(self):
        # CD27 in the DOGMA-seq cells: three nascent counts in 6,508 cells, all in closed cells,
        # so the per-configuration nascent means put the transcription rate at its lower bound
        # and gamma followed it into the corner of the box; the optimizer never left it
        m = _model(dropout_bounds=(0.0, 0.0, 1))
        P = np.asarray(m.eval_model_pss(THETA, L))                       # (U, S, P, configuration)
        P[1:, :, :, 1:] = 0.0                                            # no nascent counts in open configurations
        P /= P.sum()
        hist = cf.sparse_histogram(P)
        lb, ub = np.array(m.bio_bounds["phys_lb"]), np.array(m.bio_bounds["phys_ub"])
        x0 = m.get_MoM(None, lb, ub, None, hist=hist, limits=L)
        assert x0[3] > lb[3] + 0.5, "the transcription rate start sits at its lower bound"
        assert np.isfinite(m.eval_model_kld(x0, L, None, hist))
        P[1:] = 0.0                                                      # no nascent counts at all
        P /= P.sum()
        x0 = m.get_MoM(None, lb, ub, None, hist=cf.sparse_histogram(P), limits=L)
        assert x0[3] > lb[3] + 0.5
        assert abs(x0[5] - 0.5 * (lb[5] + ub[5])) < 0.01                # gamma from the middle of its box


class TestRnaOnlyCheck:
    """``inference.rna_only_check``: the joint fit's shared parameters against a
    chromatin-plus-RNA refit of the same cells."""
    TRUTH = TestPipeline.TRUTH
    CAPTURE = TestPipeline.CAPTURE
    GP = {"max_iterations": 3, "init_pattern": "moments", "num_restarts": 1, "num_gene_cores": 1, "check_identifiability": False}

    def _joint(self, tmp_path, monkeypatch, **model_kw):
        from monod.inference import perform_inference
        from monod.simulate import simulate_adata
        monkeypatch.chdir(tmp_path)
        d = np.log10(0.1)
        m = _model(seq="Poisson", dropout_bounds=(d, d, 1), **model_kw)
        n_samp = 4 if model_kw.get("protein_background") is not None else 3
        samp = list(self.CAPTURE) + [d] + ([0.0] if n_samp == 4 else [])
        a = simulate_adata(m, [self.TRUTH], 1500, np.append(self.CAPTURE, d), seed=5, gene_names=["geneA"])
        if n_samp == 4:
            a.var["log10_protein_background"] = [np.log10(0.5)]
        path = tmp_path / "joint.h5ad"; a.write_h5ad(path)
        return perform_inference(str(path), m, n_genes=1, viz=False, gradient_params=dict(self.GP), dataset_string="joint",
                                 exclude_sigma=True, filt_param={"min_means": [0] * 4, "max_maxes": [10_000] * 4, "min_maxes": [0] * 4},
                                 samp_lb=samp, samp_ub=samp, gridsize=[1] * n_samp)

    def test_columns_shared_parameters_and_result(self, tmp_path, monkeypatch):
        from monod.inference import rna_only_check, search_result_from_uns
        out = self._joint(tmp_path, monkeypatch)
        table = rna_only_check(out, gradient_params=dict(self.GP), num_cores=1)
        names = [p.replace(r"\log_{10} ", "") for p in _model(seq="Poisson").param_str]
        shared = names[:-2]                                                 # everything but k_p and gamma_p
        for n in shared:
            for c in ("rna_only_" + n, "rna_only_error_" + n, "shift_" + n):
                assert c in table.columns and c in out.var.columns and np.isfinite(table[c].iloc[0])
            assert table["shift_" + n].iloc[0] == pytest.approx(float(out.var[n].iloc[0]) - table["rna_only_" + n].iloc[0])
        for n in names[-2:]:                                                 # k_p and gamma_p
            assert "rna_only_" + n not in table.columns
        for c in ("rna_only_error_unreliable", "rna_only_hellinger_excess", "rna_only_rejected", "b_shift_sigma", "protein_biases_b"):
            assert c in out.var.columns
        assert table["b_shift_sigma"].iloc[0] == pytest.approx(table["shift_$b$"].iloc[0] / table["rna_only_error_$b$"].iloc[0])
        # the RNA-only result is a Chromatin fit without a protein layer, at the joint's capture and dropout
        sr_rna = search_result_from_uns(out.uns["rna_only_search_result"])
        assert not sr_rna.model.protein and list(sr_rna.model.model_modalities) == ["unspliced", "spliced", "configuration"]
        np.testing.assert_allclose(np.asarray(sr_rna.samp_optimum, float), list(self.CAPTURE) + [np.log10(0.1)])
        assert search_result_from_uns(out.uns["search_result"]).model.protein and out.uns["model"]["protein"]   # the joint result is untouched

    def test_joint_start_and_background_entry(self, tmp_path, monkeypatch):
        """A joint fit with a per-gene protein background has a fourth sampling entry;
        the RNA-only fit drops it. Started at the joint optimum the shared entries begin
        there, so one optimizer step leaves them within a few hundredths (a moments start
        moves them by tenths)."""
        from monod.inference import rna_only_check, search_result_from_uns
        out = self._joint(tmp_path, monkeypatch, protein_background=(0.0, 0.0, 1))
        gp = dict(self.GP, max_iterations=1)
        table = rna_only_check(out, gradient_params=gp, start="joint")
        sr_rna = search_result_from_uns(out.uns["rna_only_search_result"])
        assert len(np.asarray(sr_rna.samp_optimum)) == 3
        names = [p.replace(r"\log_{10} ", "") for p in sr_rna.model.param_str]
        for n in names:
            assert abs(table["shift_" + n].iloc[0]) < 0.05

    def test_needs_a_protein_fit(self, tmp_path, monkeypatch):
        from monod.inference import perform_inference, rna_only_check
        from monod.simulate import simulate_adata
        monkeypatch.chdir(tmp_path)
        d = np.log10(0.1)
        m = CMEModel("Chromatin", "Poisson", n_sites=2, rule="switch", bursty=True, dropout_bounds=(d, d, 1))
        theta = self.TRUTH[:-2]
        a = simulate_adata(m, [theta], 500, np.append(self.CAPTURE, d), seed=3, gene_names=["geneA"])
        path = tmp_path / "rna.h5ad"; a.write_h5ad(path)
        out = perform_inference(str(path), m, n_genes=1, viz=False, gradient_params=dict(self.GP), dataset_string="rna", exclude_sigma=True,
                                filt_param={"min_means": [0] * 3, "max_maxes": [10_000] * 3, "min_maxes": [0] * 3},
                                samp_lb=list(self.CAPTURE) + [d], samp_ub=list(self.CAPTURE) + [d], gridsize=[1, 1, 1])
        with pytest.raises(ValueError, match="protein layer"):
            rna_only_check(out, gradient_params=dict(self.GP))


class TestBatchedStencil:
    """The finite-difference stencil batched into one kernel call
    (``monod_core.eval_chromatin_pgf_batch``) and the forward stencil."""
    SAMP = [-0.3, -0.3, 0.0]

    @staticmethod
    def _data(m, theta, lim, samp, n_cells=600, seed=3):
        pss = m.eval_model_pss(theta, lim, samp)
        flat = pss.ravel() / pss.sum()
        rng = np.random.default_rng(seed)
        idx = rng.choice(flat.size, size=n_cells, p=flat)
        cells = np.column_stack(np.unravel_index(idx, pss.shape))
        coords, counts = np.unique(cells, axis=0, return_counts=True)
        return (coords.astype(np.int64), counts / counts.sum())

    @staticmethod
    def _needs_batch():
        if not ch._rust_batch_available(True):
            pytest.skip("monod_core without eval_chromatin_pgf_batch")

    def test_batch_matches_single_points(self):
        self._needs_batch()
        m = _model(seq="Poisson", protein_limit=10, dropout_bounds=(0, 0, 1))
        th2 = SMALL.copy(); th2[3] += 0.1
        points, g, _ = m._chromatin_points(SMALL, L_SMALL, self.SAMP)
        points2, _, _ = m._chromatin_points(th2, L_SMALL, self.SAMP)
        batch = ch.eval_joint_pgf_batch(points + points2, g)
        assert len(batch) == 2
        for pt, (G, sched) in zip(points + points2, batch):
            G1, sched1 = ch.eval_joint_pgf(pt["H"], pt["A"], pt["C"], g, tol=pt["tol"], burst=pt["burst"],
                                           structure=pt["structure"], protein=pt["protein"])
            np.testing.assert_allclose(G, G1, rtol=1e-12, atol=1e-14)
            assert list(sched.steps) == list(sched1.steps) and sched.magnus == sched1.magnus
        for th, pss in zip([SMALL, th2], m._eval_model_pss_chromatin_batch([SMALL, th2], L_SMALL, self.SAMP)):
            np.testing.assert_allclose(pss, m.eval_model_pss(th, L_SMALL, self.SAMP), rtol=1e-12, atol=1e-15)
        assert ch.eval_joint_pgf_batch([], g) == []
        with pytest.raises(ValueError, match="one row per RNA species|protein layer"):   # a point without the layer on a protein mesh
            ch.eval_joint_pgf_batch([points[0], dict(points[0], protein=None)], g)

    def test_batched_gradient_matches_serial(self, monkeypatch):
        self._needs_batch()
        m = _model(seq="Poisson", protein_limit=10, dropout_bounds=(0, 0, 1))
        data = self._data(m, SMALL, L_SMALL, self.SAMP)
        k_b, g_b = m.eval_model_kld_and_grad(SMALL, L_SMALL, self.SAMP, data)
        assert not m._schedule_is_pinned()
        monkeypatch.setattr(ch, "_rust_batch_available", lambda *a, **k: False)
        k_s, g_s = m.eval_model_kld_and_grad(SMALL, L_SMALL, self.SAMP, data)
        assert k_b == pytest.approx(k_s, rel=1e-12)
        np.testing.assert_allclose(g_b, g_s, rtol=1e-8, atol=1e-10)
        # the same under an enclosing pin (the Hessian's stencil)
        monkeypatch.undo()
        with m.pinned_schedule(SMALL, L_SMALL, self.SAMP):
            _, g_p = m.eval_model_kld_and_grad(SMALL, L_SMALL, self.SAMP, data)
        np.testing.assert_allclose(g_p, g_b, rtol=1e-8, atol=1e-10)

    def test_cell_factor_batches_every_node(self, monkeypatch):
        self._needs_batch()
        m = _model(seq="Poisson", protein_limit=10, dropout_bounds=(0, 0, 1), cell_factor=(0.3, 3))
        th2 = SMALL.copy(); th2[6] -= 0.1
        for th, pss in zip([SMALL, th2], m._eval_model_pss_chromatin_batch([SMALL, th2], L_SMALL, self.SAMP)):
            np.testing.assert_allclose(pss, m.eval_model_pss(th, L_SMALL, self.SAMP), rtol=1e-12, atol=1e-15)
        data = self._data(m, SMALL, L_SMALL, self.SAMP)
        _, g_b = m.eval_model_kld_and_grad(SMALL, L_SMALL, self.SAMP, data)
        monkeypatch.setattr(ch, "_rust_batch_available", lambda *a, **k: False)
        _, g_s = m.eval_model_kld_and_grad(SMALL, L_SMALL, self.SAMP, data)
        np.testing.assert_allclose(g_b, g_s, rtol=1e-8, atol=1e-10)

    def test_forward_differences(self):
        m_c = _model(seq="Poisson", protein_limit=10, dropout_bounds=(0, 0, 1))
        m_f = _model(seq="Poisson", protein_limit=10, dropout_bounds=(0, 0, 1), chromatin_fd="forward")
        data = self._data(m_c, SMALL, L_SMALL, self.SAMP)
        k_c, g_c = m_c.eval_model_kld_and_grad(SMALL, L_SMALL, self.SAMP, data)
        k_f, g_f = m_f.eval_model_kld_and_grad(SMALL, L_SMALL, self.SAMP, data)
        assert k_f == pytest.approx(k_c, rel=1e-12)
        np.testing.assert_allclose(g_f, g_c, rtol=2e-2, atol=2e-2 * np.abs(g_c).max())
        d = m_f.to_dict()
        assert d["chromatin_fd"] == "forward" and CMEModel.from_dict(d).chromatin_fd == "forward"
        assert m_c.chromatin_fd == "central"
        with pytest.raises(ValueError, match="chromatin_fd"):
            _model(chromatin_fd="backward")


class TestAdjoint:
    """The adjoint gradient (``monod_core.eval_chromatin_pgf_adjoint``) against
    central differences of the divergence under the same pinned schedule."""
    SAMP = [-0.3, -0.3, -0.6, 0.0]

    @staticmethod
    def _needs_adjoint():
        if not ch._rust_adjoint_available(True):
            pytest.skip("monod_core without eval_chromatin_pgf_adjoint")

    @staticmethod
    def _fd(m, theta, lim, samp, data, eps=1e-5):
        theta = np.asarray(theta, dtype=float)
        g = np.zeros(theta.size)
        with m.pinned_schedule(theta, lim, samp):
            for j in range(theta.size):
                e = np.zeros(theta.size); e[j] = eps
                g[j] = (m.eval_model_kld(theta + e, lim, samp, data) - m.eval_model_kld(theta - e, lim, samp, data)) / (2 * eps)
        return g

    def _check(self, m, theta, lim, samp, rtol=3e-4, atol=1e-8):
        """The adjoint against the pinned stencil.  The adjoint differentiates
        the protein table at its pinned nodes while the stencil re-tabulates at
        every point, so with the table at its working tolerance the two differ
        by the table's error (up to 1e-4 relative on the protein rates); with a fine
        table (``test_exact_with_a_fine_table``) they agree to 1e-6."""
        data = TestBatchedStencil._data(m, theta, lim, samp)
        th = np.asarray(theta, dtype=float) + 0.05                      # off the generating point: a gradient to speak of
        k_a, g_a = m.eval_model_kld_and_grad(th, lim, samp, data)
        assert not m._schedule_is_pinned()
        assert k_a == pytest.approx(m.eval_model_kld(th, lim, samp, data), rel=1e-12)
        np.testing.assert_allclose(g_a, self._fd(m, th, lim, samp, data), rtol=rtol, atol=atol)
        return th, data, g_a

    def test_exact_with_a_fine_table(self, monkeypatch):
        self._needs_adjoint()
        monkeypatch.setattr(ch, "_PROTEIN_TABLE_TOL", 1e-4)
        m = _model(seq="Poisson", protein_limit=10, dropout_bounds=(-1.5, -0.3, 3), cell_factor=(0.3, 3), protein_background=(-1.0, 1.0, 3))
        self._check(m, SMALL, L_SMALL, self.SAMP, rtol=1e-6, atol=1e-9)

    def test_switch_cell_factor_background_dropout(self):
        self._needs_adjoint()
        kw = dict(seq="Poisson", protein_limit=10, dropout_bounds=(-1.5, -0.3, 3), cell_factor=(0.3, 3), protein_background=(-1.0, 1.0, 3))
        m = _model(**kw)
        assert m.chromatin_grad == "auto"
        th, data, g_a = self._check(m, SMALL, L_SMALL, self.SAMP)
        # the stencil at its 1e-4 step carries its truncation error; the adjoint agrees to that
        _, g_s = _model(chromatin_grad="stencil", **kw).eval_model_kld_and_grad(th, L_SMALL, self.SAMP, data)
        np.testing.assert_allclose(g_a, g_s, rtol=2e-3, atol=1e-6)

    def test_additive_and_table_rules(self):
        """The factored and the general production structure."""
        self._needs_adjoint()
        for rule, rates in (("additive", [1.0, 2.0]), ("table", [0.2, 1.0, 1.5, 3.0])):
            m = _model(rule=rule, seq="Poisson", protein_limit=10)
            theta = cf.join_theta([0.5, 0.8], 0.6, rates, 0.8, 0.4, 1.5, (0.4, 0.8))
            self._check(m, theta, L_SMALL, [-0.3, -0.3, 0.0, 0.0])

    def test_single_molecules_and_independent_sites(self):
        """Constitutive emission under a protein layer; a Kronecker chain (eps = 1)."""
        self._needs_adjoint()
        m = _model(seq="Poisson", protein_limit=10, bursty=False)
        theta = cf.join_theta([0.5, 0.8], 0.6, [1.5], 0.8, 0.4, None, (0.4, 0.8))
        self._check(m, theta, L_SMALL, [-0.3, -0.3, 0.0, 0.0])
        m = _model(seq="Poisson", protein_limit=10)
        theta = cf.join_theta([0.5, 0.8], 1.0, [1.5], 0.8, 0.4, 1.5, (0.4, 0.8))
        self._check(m, theta, L_SMALL, [-0.3, -0.3, 0.0, 0.0])

    def test_enclosing_pin_and_magnus_tail(self):
        self._needs_adjoint()
        m = _model(seq="Poisson", protein_limit=10, dropout_bounds=(-1.5, -0.3, 3))
        samp = self.SAMP[:3]
        th = SMALL.copy(); th[5] = -1.0; th[-1] = -1.3                  # slow degradation: Magnus steps in the tail
        data = TestBatchedStencil._data(m, th, L_SMALL, samp)
        k_a, g_a = m.eval_model_kld_and_grad(th, L_SMALL, samp, data)
        with m.pinned_schedule(th, L_SMALL, samp):
            assert getattr(m._fixed_schedule, "magnus", 0) >= 1
            k_p, g_p = m.eval_model_kld_and_grad(th, L_SMALL, samp, data)
            assert m._schedule_is_pinned()
        assert k_p == pytest.approx(k_a, rel=1e-12)
        np.testing.assert_allclose(g_p, g_a, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(g_a, self._fd(m, th, L_SMALL, samp, data), rtol=1e-4, atol=1e-8)

    def test_options(self, monkeypatch):
        with pytest.raises(ValueError, match="chromatin_grad"):
            _model(chromatin_grad="exact")
        m = _model(chromatin_grad="stencil")
        d = m.to_dict()
        assert d["chromatin_grad"] == "stencil" and CMEModel.from_dict(d).chromatin_grad == "stencil"
        m = _model(seq="Poisson", protein_limit=10, chromatin_grad="adjoint")
        data = TestBatchedStencil._data(m, SMALL, L_SMALL, [-0.3, -0.3, 0.0, 0.0])
        monkeypatch.setattr(ch, "_rust_adjoint_available", lambda *a, **k: False)
        with pytest.raises(RuntimeError, match="adjoint"):
            m.eval_model_kld_and_grad(SMALL, L_SMALL, [-0.3, -0.3, 0.0, 0.0], data)
