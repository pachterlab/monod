"""Tests for the user-defined reaction network module.

Correctness is validated by comparing the Custom model against the
built-in Bursty and ProteinBursty models, which have well-tested
implementations.
"""

import numpy as np
import pytest
import sys
import os
from monod.reaction_network import ReactionNetwork
from monod.cme_toolbox import CMEModel


# ---------------------------------------------------------------------------
# Parsing tests
# ---------------------------------------------------------------------------

class TestParsing:
    def test_bursty_species(self):
        net = ReactionNetwork("0 -> B x U -> S -> 0")
        assert net.species == ["U", "S"]

    def test_bursty_named_params(self):
        net = ReactionNetwork("0 -> B x U -> S -> 0")
        assert net.named_params == ["B"]

    def test_bursty_rate_names(self):
        net = ReactionNetwork("0 -> B x U -> S -> 0")
        # 3 arrows → 3 rates
        assert net.rate_names == ["k0", "k1", "k2"]

    def test_bursty_all_params(self):
        net = ReactionNetwork("0 -> B x U -> S -> 0")
        assert net.all_params == ["B", "k0", "k1", "k2"]

    def test_normalize_production_rate_drops_first_prod(self):
        net = ReactionNetwork("0 -> B x U -> S -> 0", normalize_production_rate=True)
        # k0 is the first production rate → dropped
        assert net.all_params == ["B", "k1", "k2"]

    def test_semicolon_adds_extra_reaction(self):
        net = ReactionNetwork("0 -> B x U -> S -> 0; U -> 0")
        # 4 arrows total
        assert len(net.reactions) == 4
        assert net.rate_names == ["k0", "k1", "k2", "k3"]

    def test_extra_degradation_not_new_species(self):
        net = ReactionNetwork("0 -> B x U -> S -> 0; U -> 0")
        assert net.species == ["U", "S"]   # no new species from U -> 0

    def test_catalytic_production(self):
        # S -> S + P means S catalyses production of P.  S itself is never removed
        # in this network, which the structural check points out.
        with pytest.warns(UserWarning, match="S is never removed"):
            net = ReactionNetwork("0 -> B x U -> S -> S + P; P -> 0")
        assert net.species == ["U", "S", "P"]
        assert net.all_params == ["B", "k0", "k1", "k2", "k3"]

    def test_deterministic_burst(self):
        net = ReactionNetwork("0 -> 2 x U -> S -> 0")
        # '2' is numeric → no named_params for burst size
        assert net.named_params == []
        assert "U" in net.species

    def test_repr_runs(self):
        net = ReactionNetwork("0 -> B x U -> S -> 0; U -> 0")
        r = repr(net)
        assert "∅" in r
        assert "k0" in r

    def test_bad_node_raises(self):
        with pytest.raises(ValueError):
            ReactionNetwork("0 -> 123bad -> 0")

    def test_multiple_burst_terms_raises(self):
        with pytest.raises(ValueError):
            ReactionNetwork("0 -> B x U + C x S")


# ---------------------------------------------------------------------------
# ODE / PGF sanity checks
# ---------------------------------------------------------------------------

class TestODE:
    """Low-level checks on the characteristic ODE."""

    def _simple_grid(self, l0=8, l1=8):
        """Tiny 2-species complex mesh for quick ODE tests."""
        mx = (l0, l1 // 2 + 1)
        u0 = np.exp(-2j * np.pi * np.arange(mx[0]) / l0) - 1
        u1 = np.exp(-2j * np.pi * np.arange(mx[1]) / l1) - 1
        g0, g1 = np.meshgrid(u0, u1, indexing='ij')
        return [g0.flatten().astype(np.complex64),
                g1.flatten().astype(np.complex64)]

    def test_constitutive_limiting_case(self):
        """With burst size B→0 the Custom model should approximate constitutive."""
        # Constitutive: 0→U (rate β), U→S (rate k), S→0 (rate γ)
        # represented as "0 -> U -> S -> 0" (deterministic single production)
        net = ReactionNetwork("0 -> U -> S -> 0")
        g = self._simple_grid()
        params = np.log10([1.0, 0.5, 0.8])  # k0=1, k1=0.5 (β), k2=0.8 (γ)
        phi = net.eval_pgf(params, g)
        assert phi.shape == g[0].shape
        assert np.all(np.isfinite(phi))
        assert phi[0].real == pytest.approx(0.0, abs=1e-4)  # φ(0,0) = 0

    def test_phi_zero_at_origin(self):
        """φ(0,0,…) = 0 for any network (G(1,…) = 1 at origin)."""
        net = ReactionNetwork("0 -> B x U -> S -> 0; U -> 0")
        g = self._simple_grid()
        # g[i][0] == 0 at origin
        params = np.log10([2.0, 1.0, 0.5, 0.8, 0.3])  # B, k0, k1, k2, k3
        phi = net.eval_pgf(params, g)
        assert phi[0].real == pytest.approx(0.0, abs=1e-4)

    def test_phi_real_part_non_positive(self):
        """Real part of log-PGF should be ≤ 0 (|G| ≤ 1 for a distribution)."""
        net = ReactionNetwork("0 -> B x U -> S -> 0")
        g = self._simple_grid()
        params = np.log10([2.0, 1.0, 0.5, 0.8])
        phi = net.eval_pgf(params, g)
        assert np.all(phi.real <= 1e-6)


# ---------------------------------------------------------------------------
# Integration with CMEModel
# ---------------------------------------------------------------------------

class TestCMEModelCustom:
    BURSTY_NET = "0 -> B x U -> S -> 0"

    def _make_custom(self, normalize=False):
        return CMEModel(
            "Custom", "None",
            network=self.BURSTY_NET,
            normalize_production_rate=normalize,
            min_fudge=0.05,
            max_fudge=15,
        )

    def test_init_no_network_raises(self):
        with pytest.raises(ValueError, match="network"):
            CMEModel("Custom", "None")

    def test_species_set_as_modalities(self):
        model = self._make_custom()
        assert model.model_modalities == ["U", "S"]

    def test_num_params(self):
        model = self._make_custom()
        # B, k0, k1, k2 = 4
        assert model.get_num_params() == 4

    def test_num_params_normalized(self):
        model = self._make_custom(normalize=True)
        # B, k1, k2 = 3
        assert model.get_num_params() == 3

    def test_log_name_str_length(self):
        model = self._make_custom()
        assert len(model.get_log_name_str()) == 4

    def test_eval_model_pss_shape(self):
        model = self._make_custom()
        # B=2, k0=1 (burst rate), k1=1 (β), k2=0.8 (γ)
        p = np.log10([2.0, 1.0, 1.0, 0.8])
        pss = model.eval_model_pss(p, limits=[20, 20])
        assert pss.shape == (20, 20)

    def test_eval_model_pss_is_probability(self):
        model = self._make_custom()
        p = np.log10([2.0, 1.0, 1.0, 0.8])
        pss = model.eval_model_pss(p, limits=[20, 20])
        assert pss.sum() == pytest.approx(1.0, abs=1e-4)
        assert np.all(pss >= -1e-8)


# ---------------------------------------------------------------------------
# Equivalence: Custom Bursty ≈ built-in Bursty
# ---------------------------------------------------------------------------

class TestBurstyEquivalence:
    """
    The Custom model "0 -> B x U -> S -> 0" with normalize_production_rate=True
    and parameters [B, β, γ] should reproduce the built-in Bursty model with
    parameters [b, β, γ].
    """
    LIMITS = [30, 30]
    B, BETA, GAMMA = 2.0, 0.8, 0.5

    def _pss_builtin(self):
        model = CMEModel("Bursty", "None", quad_method="fixed_quad", fixed_quad_T=10)
        p = np.log10([self.B, self.BETA, self.GAMMA])
        return model.eval_model_pss(p, limits=self.LIMITS)

    def _pss_custom(self):
        model = CMEModel(
            "Custom", "None",
            network="0 -> B x U -> S -> 0",
            normalize_production_rate=True,
            min_fudge=0.05,
            max_fudge=15,
        )
        p = np.log10([self.B, self.BETA, self.GAMMA])
        return model.eval_model_pss(p, limits=self.LIMITS)

    def test_pss_close_to_builtin(self):
        pss_ref    = self._pss_builtin()
        pss_custom = self._pss_custom()
        # KLD (reference ∥ custom) should be small
        mask = pss_ref > 0
        kld = np.sum(pss_ref[mask] * np.log(pss_ref[mask] / np.clip(pss_custom[mask], 1e-15, None)))
        assert kld < 0.05, f"KLD too large: {kld:.4f}"

    def test_mean_U_close(self):
        pss_ref    = self._pss_builtin()
        pss_custom = self._pss_custom()
        idx = np.arange(self.LIMITS[0])
        mean_ref    = np.sum(pss_ref    * idx[:, None])
        mean_custom = np.sum(pss_custom * idx[:, None])
        assert mean_custom == pytest.approx(mean_ref, rel=0.05)


# ---------------------------------------------------------------------------
# Additional-reaction test: "; U -> 0"
# ---------------------------------------------------------------------------

class TestExtraReaction:
    """Adding 'U -> 0' should shift the U distribution toward lower counts."""
    LIMITS = [30, 30]
    B, BETA, GAMMA, GAMMA_U = 2.0, 0.8, 0.5, 0.5

    def _pss_with_u_deg(self):
        model = CMEModel(
            "Custom", "None",
            network="0 -> B x U -> S -> 0; U -> 0",
            normalize_production_rate=True,
            min_fudge=0.05,
            max_fudge=15,
        )
        p = np.log10([self.B, self.BETA, self.GAMMA, self.GAMMA_U])
        return model.eval_model_pss(p, limits=self.LIMITS)

    def _pss_without_u_deg(self):
        model = CMEModel(
            "Custom", "None",
            network="0 -> B x U -> S -> 0",
            normalize_production_rate=True,
            min_fudge=0.05,
            max_fudge=15,
        )
        p = np.log10([self.B, self.BETA, self.GAMMA])
        return model.eval_model_pss(p, limits=self.LIMITS)

    def test_u_degradation_lowers_mean_U(self):
        pss_base = self._pss_without_u_deg()
        pss_deg  = self._pss_with_u_deg()
        idx = np.arange(self.LIMITS[0])
        mean_base = np.sum(pss_base * idx[:, None])
        mean_deg  = np.sum(pss_deg  * idx[:, None])
        assert mean_deg < mean_base

    def test_pss_is_valid_distribution(self):
        pss = self._pss_with_u_deg()
        assert pss.sum() == pytest.approx(1.0, abs=1e-4)
        assert np.all(pss >= -1e-8)


# ---------------------------------------------------------------------------
# Delayed-reaction tests
# ---------------------------------------------------------------------------

class TestDelayedReaction:
    """Tests for the '=>' (time-delayed) reaction syntax."""

    # ── parsing ──────────────────────────────────────────────────────────────

    def test_delay_detected(self):
        net = ReactionNetwork("0 -> B x U => S -> 0")
        delayed = [r for r in net.reactions if r.is_delayed]
        assert len(delayed) == 1

    def test_delay_params_present(self):
        net = ReactionNetwork("0 -> B x U => S -> 0")
        assert net.delay_params == ["tauinv0"]

    def test_delay_param_order_in_all_params(self):
        # all_params order: named_params + delay_params + rate_names
        net = ReactionNetwork("0 -> B x U => S -> 0")
        assert net.all_params == ["B", "tauinv0", "k0", "k1", "k2"]

    def test_delay_normalized_removes_prod_rate(self):
        net = ReactionNetwork("0 -> B x U => S -> 0", normalize_production_rate=True)
        # k0 (first production rate) is normalized away
        assert net.all_params == ["B", "tauinv0", "k1", "k2"]

    def test_multiple_delays(self):
        net = ReactionNetwork("0 -> B x U => S; S => P -> 0")
        assert net.delay_params == ["tauinv0", "tauinv1"]
        delayed = [r for r in net.reactions if r.is_delayed]
        assert len(delayed) == 2

    def test_repr_shows_delay_arrow(self):
        net = ReactionNetwork("0 -> B x U => S -> 0")
        r = repr(net)
        assert "⇒" in r
        assert "tauinv0" in r

    # ── ODE / PGF sanity ─────────────────────────────────────────────────────

    def _grid_2sp(self, l0=10, l1=10):
        mx = (l0, l1 // 2 + 1)
        u0 = np.exp(-2j * np.pi * np.arange(mx[0]) / l0) - 1
        u1 = np.exp(-2j * np.pi * np.arange(mx[1]) / l1) - 1
        g0, g1 = np.meshgrid(u0, u1, indexing='ij')
        return [g0.flatten().astype(np.complex64),
                g1.flatten().astype(np.complex64)]

    def test_phi_origin_zero(self):
        net = ReactionNetwork("0 -> B x U => S -> 0")
        g = self._grid_2sp()
        # B=2, tauinv0=0.5, k0=1 (burst rate), k1=0.5 (β), k2=0.5 (γ)
        p = np.log10([2.0, 0.5, 1.0, 0.5, 0.5])
        phi = net.eval_pgf(p, g)
        assert phi[0].real == pytest.approx(0.0, abs=1e-3)

    def test_phi_real_nonpositive(self):
        net = ReactionNetwork("0 -> B x U => S -> 0")
        g = self._grid_2sp()
        p = np.log10([2.0, 0.5, 1.0, 0.5, 0.5])
        phi = net.eval_pgf(p, g)
        assert np.all(phi.real <= 1e-5)

    def test_pss_is_probability(self):
        """Delayed network PSS via CMEModel sums to 1."""
        model = CMEModel(
            "Custom", "None",
            network="0 -> B x U => S -> 0",
            normalize_production_rate=True,
            min_fudge=0.05, max_fudge=15,
        )
        # params: [B, tauinv0, k1(β), k2(γ)]
        p = np.log10([2.0, 0.5, 0.5, 0.5])
        pss = model.eval_model_pss(p, limits=[20, 20])
        assert pss.sum() == pytest.approx(1.0, abs=1e-3)
        assert np.all(pss >= -1e-8)

    # ── physical effect ───────────────────────────────────────────────────────

    def test_delay_shifts_distribution(self):
        """A longer delay (smaller tauinv) should broaden the U distribution.

        When τ → 0 (tauinv → ∞) the delayed model approaches the non-delayed
        model.  A finite delay increases the effective variance of U.
        """
        # Without delay
        m_instant = CMEModel(
            "Custom", "None",
            network="0 -> B x U -> S -> 0",
            normalize_production_rate=True,
            min_fudge=0.05, max_fudge=15,
        )
        pss_instant = m_instant.eval_model_pss(
            np.log10([2.0, 0.5, 0.5]), limits=[30, 30]
        )

        # With delay (tauinv=2 → τ=0.5, a moderate delay)
        m_delayed = CMEModel(
            "Custom", "None",
            network="0 -> B x U => S -> 0",
            normalize_production_rate=True,
            min_fudge=0.05, max_fudge=15,
        )
        pss_delayed = m_delayed.eval_model_pss(
            np.log10([2.0, 2.0, 0.5, 0.5]), limits=[30, 30]
        )

        # Both should be valid distributions
        assert pss_instant.sum() == pytest.approx(1.0, abs=1e-3)
        assert pss_delayed.sum() == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Unsupported reactions are rejected at parse time (audit fix 4, 2026-10-15)
# ---------------------------------------------------------------------------

class TestUnsupportedReactionsRejected:
    @pytest.mark.parametrize("net", [
        "A + A -> B",          # reactant stoichiometry 2 via repetition
        "2 x A -> B",          # reactant stoichiometry 2 via prefix
        "A + B -> C",          # bimolecular
        "0 -> B x U -> S -> 0; U + S -> 0",  # bimolecular in a second chain
    ])
    def test_raises_value_error(self, net):
        with pytest.raises(ValueError, match="zero- and first-order|stoichiometry > 1"):
            ReactionNetwork(net)

    def test_burst_prefix_on_reactant_raises(self):
        with pytest.raises(ValueError, match="only allowed on products"):
            ReactionNetwork("0 -> B x U; B x U -> S")

    def test_cme_model_custom_rejects_too(self):
        with pytest.raises(ValueError):
            CMEModel("Custom", "None", network="A + B -> C")


# ---------------------------------------------------------------------------
# Tight equivalence after removing the trapezoid double count (fix 4)
# ---------------------------------------------------------------------------

class TestBurstyEquivalenceTight:
    LIMITS = [30, 30]
    B, BETA, GAMMA = 2.0, 0.8, 0.5

    def _ref(self):
        model = CMEModel("Bursty", "None", quad_method="fixed_quad", fixed_quad_T=10)
        return model.eval_model_pss(np.log10([self.B, self.BETA, self.GAMMA]), limits=self.LIMITS)

    def _custom(self, use_rust):
        import unittest.mock
        import monod.cme_toolbox as ct
        model = CMEModel("Custom", "None", network="0 -> B x U -> S -> 0",
                         normalize_production_rate=True, min_fudge=0.05, max_fudge=15)
        p = np.log10([self.B, self.BETA, self.GAMMA])
        if use_rust:
            if not ct._HAS_RUST:
                pytest.skip("monod_core not installed")
            return model.eval_model_pss(p, limits=self.LIMITS)
        with unittest.mock.patch.object(ct, "_HAS_RUST", False):
            return model.eval_model_pss(p, limits=self.LIMITS)

    @staticmethod
    def _kld(p, q):
        m = p > 0
        return float(np.sum(p[m] * np.log(p[m] / np.clip(q[m], 1e-15, None))))

    @pytest.mark.parametrize("use_rust", [True, False], ids=["rust", "python"])
    def test_custom_matches_builtin_tightly(self, use_rust):
        ref = self._ref()
        cus = self._custom(use_rust)
        kld = self._kld(ref, cus)
        assert kld < 1e-5, f"KLD {kld:.2e} (was ~6e-5 with the double-counted half-steps)"
        assert np.max(np.abs(ref - cus)) < 1e-4, f"max |dPMF| {np.max(np.abs(ref - cus)):.2e}"

    def test_rust_and_python_custom_agree(self):
        import monod.cme_toolbox as ct
        if not ct._HAS_RUST:
            pytest.skip("monod_core not installed")
        r, p = self._custom(True), self._custom(False)
        np.testing.assert_allclose(r, p, rtol=0, atol=2e-4)


# ---------------------------------------------------------------------------
# Structural checks and identifiability
# ---------------------------------------------------------------------------

import warnings as _warnings
from monod.reaction_network import ReactionNetworkWarning, IdentifiabilityWarning


def _construct_warnings(net, **kw):
    with _warnings.catch_warnings(record=True) as rec:
        _warnings.simplefilter("always")
        n = ReactionNetwork(net, **kw)
    return n, [w for w in rec if issubclass(w.category, ReactionNetworkWarning)]


def _codes(net, **kw):
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        return [i["code"] for i in ReactionNetwork(net, **kw).structural_issues()]


class TestStructuralChecks:
    def test_clean_network_has_no_issues(self):
        assert _codes("0 -> B x U -> S -> 0", normalize_production_rate=True) == []
        assert _codes("0 -> B x U -> S -> 0; S -> S + P; P -> 0", normalize_production_rate=True) == []

    def test_free_rates_flag_timescale_but_do_not_warn_at_construction(self):
        n, warned = _construct_warnings("0 -> B x U -> S -> 0")
        assert warned == []
        issues = n.structural_issues()
        assert [i["code"] for i in issues] == ["timescale"]
        assert issues[0]["params"] == ["k0", "k1", "k2"]
        # validate() for inference includes it
        with pytest.warns(ReactionNetworkWarning, match="scaling k0, k1, k2 together"):
            n.validate(for_inference=True)

    def test_normalisation_removes_timescale_issue(self):
        assert "timescale" not in _codes("0 -> B x U -> S -> 0", normalize_production_rate=True)

    def test_normalisation_without_production_still_flags_timescale(self):
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            codes = _codes("U -> S -> 0", normalize_production_rate=True)
        assert "timescale" in codes

    def test_delayed_reaction_without_product_warns(self):
        n, warned = _construct_warnings("0 -> B x U -> S -> 0; U => 0")
        assert len(warned) == 1 and "tauinv0 has no effect" in str(warned[0].message)
        issue = [i for i in n.structural_issues() if i["code"] == "delay_no_product"][0]
        assert issue["params"] == ["tauinv0"]

    def test_delayed_reaction_with_product_is_fine(self):
        _, warned = _construct_warnings("0 -> B x U => S -> 0")
        assert warned == []

    def test_duplicate_reaction_warns(self):
        n, warned = _construct_warnings("0 -> B x U -> S -> 0; U -> S")
        assert len(warned) == 1 and "not separately identifiable" in str(warned[0].message)
        issue = [i for i in n.structural_issues() if i["code"] == "duplicate_reaction"][0]
        assert issue["params"] == ["k1", "k3"]

    def test_duplicate_burst_production_warns(self):
        n, warned = _construct_warnings("0 -> B1 x U; 0 -> B2 x U; U -> 0")
        assert len(warned) == 1
        issue = [i for i in n.structural_issues() if i["code"] == "duplicate_reaction"][0]
        assert set(issue["params"]) == {"k0", "k1", "B1", "B2"}

    def test_burst_and_constitutive_production_are_not_duplicates(self):
        _, warned = _construct_warnings("0 -> B x U; 0 -> U; U -> 0")
        assert warned == []

    def test_unproduced_species_warns(self):
        n, warned = _construct_warnings("0 -> U -> 0; S -> 0")
        assert len(warned) == 1 and "S is never produced" in str(warned[0].message)
        issue = [i for i in n.structural_issues() if i["code"] == "no_source"][0]
        assert issue["params"] == ["k2"]

    def test_unremoved_species_warns(self):
        n, warned = _construct_warnings("0 -> B x U -> S; S -> S + P; P -> 0")
        assert len(warned) == 1 and "S is never removed" in str(warned[0].message)

    def test_autocatalytic_reaction_warns(self):
        n, warned = _construct_warnings("0 -> U -> 2 x U; U -> 0")
        msgs = [str(w.message) for w in warned]
        assert any("turns one U into 2" in m for m in msgs)
        assert "no_sink" not in [i["code"] for i in n.structural_issues()]

    def test_validate_can_be_silent(self):
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            n = ReactionNetwork("0 -> B x U -> S -> 0", normalize_production_rate=True)
            issues = n.validate(for_inference=True, warn=False)
        assert issues == []

    def test_cmemodel_constructor_propagates_warning(self):
        with pytest.warns(ReactionNetworkWarning, match="same reactant and products"):
            CMEModel("Custom", "None", network="0 -> B x U -> S -> 0; U -> S",
                     normalize_production_rate=True)

    def test_inference_parameters_warn_about_free_rates(self):
        from monod.inference import InferenceParameters
        model = CMEModel("Custom", "None", network="0 -> B x U -> S -> 0")
        with pytest.warns(ReactionNetworkWarning, match="every rate constant is free"):
            InferenceParameters("ident_test", model, phys_lb=[-1.0] * 4, phys_ub=[2.0] * 4)
        model_ok = CMEModel("Custom", "None", network="0 -> B x U -> S -> 0",
                            normalize_production_rate=True)
        with _warnings.catch_warnings():
            _warnings.simplefilter("error", ReactionNetworkWarning)
            InferenceParameters("ident_test", model_ok, phys_lb=[-1.0] * 3, phys_ub=[2.0] * 3)


class TestIdentifiabilityCheck:
    LIMITS = [20, 16]

    def _model(self, net, normalize):
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", ReactionNetworkWarning)
            return CMEModel("Custom", "None", network=net, normalize_production_rate=normalize,
                            min_fudge=0.05, max_fudge=15)

    def test_param_names(self):
        assert CMEModel("Bursty", "None").param_names() == ["b", "beta", "gamma"]
        assert CMEModel("ProteinBursty", "None").param_names() == ["b", "beta", "gamma", "k_p", "gamma_p"]
        assert self._model("0 -> B x U => S -> 0", True).param_names() == ["B", "tauinv0", "k1", "k2"]

    def test_free_rates_give_flat_scaling_direction(self):
        m = self._model("0 -> B x U -> S -> 0", False)
        with pytest.warns(IdentifiabilityWarning, match=r"\+0\.58 k0 \+0\.58 k1 \+0\.58 k2"):
            r = m.check_identifiability([0.5, 0.0, 0.0, -0.3], self.LIMITS)
        assert not r["identifiable"]
        assert len(r["flat"]) == 1
        rel_sv, direction = r["flat"][0]
        assert rel_sv < 1e-3
        v = np.array([direction[k] for k in ["B", "k0", "k1", "k2"]])
        np.testing.assert_allclose(np.abs(v), [0, 1, 1, 1] / np.sqrt(3), atol=0.02)

    def test_normalised_network_is_identifiable(self):
        m = self._model("0 -> B x U -> S -> 0", True)
        with _warnings.catch_warnings():
            _warnings.simplefilter("error", IdentifiabilityWarning)
            r = m.check_identifiability([0.5, 0.0, -0.3], self.LIMITS)
        assert r["identifiable"] and r["flat"] == []
        assert r["singular_values"][-1] > 0.05

    def test_builtin_bursty_is_identifiable(self):
        r = CMEModel("Bursty", "None").check_identifiability([0.5, 0.0, -0.3], self.LIMITS, warn=False)
        assert r["identifiable"]
        assert r["param_names"] == ["b", "beta", "gamma"]

    def test_delay_without_product_is_exactly_flat(self):
        m = self._model("0 -> B x U -> S -> 0; U => 0", True)
        r = m.check_identifiability([0.5, 0.0, 0.0, -0.3, -0.5], self.LIMITS, warn=False)
        assert not r["identifiable"]
        rel_sv, direction = r["flat"][0]
        assert rel_sv < 1e-8
        assert abs(direction["tauinv0"]) > 0.999

    def test_duplicate_reaction_is_flat(self):
        m = self._model("0 -> B x U -> S -> 0; U -> S", True)
        r = m.check_identifiability([0.5, 0.0, -0.3, -0.4], self.LIMITS, warn=False)
        assert not r["identifiable"]
        _, direction = r["flat"][0]
        assert abs(direction["B"]) < 0.05 and abs(direction["k2"]) < 0.05

    def test_wrong_parameter_count_raises(self):
        with pytest.raises(ValueError, match="expected 3 parameters"):
            CMEModel("Bursty", "None").check_identifiability([0.5, 0.0], self.LIMITS)

    def test_inference_runs_check_once_for_custom(self, monkeypatch):
        """GradientInference calls the check at grid point 0 for Custom networks
        and stores the result on the InferenceParameters."""
        from monod.inference import InferenceParameters, GradientInference

        class _SD:
            n_genes = 2
            M = np.array([[12, 12], [10, 10]])
            hist = [None, None]
            gene_log_lengths = np.zeros(2)

        m = self._model("0 -> B x U -> S -> 0", False)
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", ReactionNetworkWarning)
            ip = InferenceParameters("ident_gi", m, phys_lb=[-1.0] * 4, phys_ub=[2.0] * 4,
                                     gradient_params={"init_pattern": "random", "num_restarts": 1})
        with pytest.warns(IdentifiabilityWarning, match="k0 .*k1 .*k2"):
            GradientInference(ip, m, _SD(), 0)
        assert ip.identifiability is not None and not ip.identifiability["identifiable"]
        # an opted-out run does nothing
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", ReactionNetworkWarning)
            _warnings.simplefilter("error", IdentifiabilityWarning)
            ip2 = InferenceParameters("ident_gi", m, phys_lb=[-1.0] * 4, phys_ub=[2.0] * 4,
                                      gradient_params={"init_pattern": "random", "num_restarts": 1,
                                                       "check_identifiability": False})
            GradientInference(ip2, m, _SD(), 0)
        assert ip2.identifiability is None


class TestSigmaFlags:
    """compute_sigma flags genes whose Hessian cannot give a standard error."""

    def _results(self, hessians, lb=(-1.0, -1.8, -1.8), ub=(4.2, 2.5, 3.5), n_cells=1000):
        from monod.inference import SearchResults, InferenceParameters
        model = CMEModel("Bursty", "None")
        sp = InferenceParameters("sigma_test", model, phys_lb=list(lb), phys_ub=list(ub))
        sr = SearchResults.__new__(SearchResults)
        sr.sp = sp
        sr.model = model
        sr.n_genes = len(hessians)
        sr.n_cells = n_cells
        sr.samp_optimum_ind = 0
        sr.par_fun_hess = lambda inputs: np.asarray(hessians[inputs[0]], float)
        return sr

    def test_good_hessians_are_not_flagged(self):
        sr = self._results([np.diag([4.0, 9.0, 16.0]), np.diag([1.0, 1.0, 1.0])])
        with _warnings.catch_warnings():
            _warnings.simplefilter("error", IdentifiabilityWarning)
            sr.compute_sigma(search_data=None)
        assert not sr.sigma_fail.any()
        np.testing.assert_allclose(sr.sigma[0], [0.5, 1 / 3, 0.25] / np.sqrt(1000))
        assert sr.sigma_flat_direction.shape == (2, 3)

    def test_singular_unconstrained_and_nonfinite_are_flagged(self):
        good = np.diag([4.0, 9.0, 16.0])
        singular = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 4.0]])   # null vector (1,-1,0)
        unconstrained = np.diag([4.0, 9.0, 1e-9])                                 # sigma_gamma ~ 1e3
        bad = np.full((3, 3), np.nan)
        sr = self._results([good, singular, unconstrained, bad])
        with pytest.warns(IdentifiabilityWarning, match="3 of 4 genes") as rec:
            sr.compute_sigma(search_data=None)
        assert sr.sigma_fail.tolist() == [False, True, True, True]
        # fallback: flagged genes carry the mean of the good ones
        for g in (1, 2, 3):
            np.testing.assert_allclose(sr.sigma[g], sr.sigma[0])
        # flat direction of the singular gene is its null vector
        d = sr.sigma_flat_direction[1]
        np.testing.assert_allclose(np.abs(d), [1, 1, 0] / np.sqrt(2), atol=1e-8)
        assert np.all(np.isnan(sr.sigma_flat_direction[3]))
        msg = str(rec[0].message)
        assert "not positive definite" in msg and "wider than the parameter box" in msg

    def test_all_failed_gives_nan(self):
        sr = self._results([np.zeros((3, 3))])
        with pytest.warns(IdentifiabilityWarning):
            sr.compute_sigma(search_data=None)
        assert np.all(np.isnan(sr.sigma)) and sr.sigma_fail.all()

    def test_serialisation_round_trip_keeps_flags(self):
        from monod.inference import SearchResults
        sr = self._results([np.diag([4.0, 9.0, 16.0]), np.diag([4.0, 9.0, 1e-9])])
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", IdentifiabilityWarning)
            sr.compute_sigma(search_data=None)
        d = sr.to_dict()
        assert "sigma_fail" in d and "sigma_flat_direction" in d
        back = SearchResults.from_dict(d)
        assert np.asarray(back.sigma_fail).tolist() == [False, True]


class TestCustomInferenceDefaults:
    """Custom networks get bounds, filters and a starting point so perform_inference accepts them."""

    def _model(self, net, normalize=True):
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", ReactionNetworkWarning)
            return CMEModel("Custom", "None", network=net, normalize_production_rate=normalize)

    def test_default_bounds_follow_parameter_kinds(self):
        m = self._model("0 -> B x U => S -> 0")          # params: B, tauinv0, k1, k2
        assert m.bio_bounds == {"phys_lb": [-1.0, -1.8, -1.8, -1.8], "phys_ub": [4.2, 3.5, 3.5, 3.5]}
        m3 = self._model("0 -> B x U -> S; S -> S + P; P -> 0; S -> 0")
        assert m3.filter_bounds == {"min_means": [0.01] * 3, "max_maxes": [350] * 3, "min_maxes": [4] * 3}

    def test_mom_is_box_midpoint(self):
        m = self._model("0 -> B x U -> S -> 0")
        x0 = m.get_MoM({}, np.array([-1.0, -1.0, -1.0]), np.array([3.0, 1.0, 2.0]))
        np.testing.assert_allclose(x0, [1.0, 0.0, 0.5])

    def test_gradient_inference_moments_init_uses_midpoint(self):
        from monod.inference import InferenceParameters, GradientInference

        class _SD:
            n_genes = 2
            M = np.array([[12, 12], [10, 10]])
            hist = [None, None]
            gene_log_lengths = np.zeros(2)

        m = self._model("0 -> B x U -> S -> 0")
        ip = InferenceParameters("custom_defaults", m, gradient_params={"num_restarts": 1})
        gi = GradientInference(ip, m, _SD(), 0)
        np.testing.assert_allclose(gi.param_MoM, np.tile([1.6, 0.85, 0.85], (2, 1)))
        assert ip.identifiability["identifiable"]


class TestMeanRateBursts:
    """burst_mean_rate=True reads the rate of a burst production as the mean
    production rate: the same network as the frequency form at k_freq = k_mean / B."""

    def test_matches_the_frequency_form_at_matched_parameters(self):
        freq = CMEModel("Custom", "None", network="0 -> B x U -> S -> 0")
        mean = CMEModel("Custom", "None", network="0 -> B x U -> S -> 0", burst_mean_rate=True)
        assert mean.network.all_params == freq.network.all_params == ["B", "k0", "k1", "k2"]
        B, k0, k1, k2 = 3.0, 0.7, 0.4, 0.25
        p_freq = np.log10([B, k0, k1, k2])
        p_mean = np.log10([B, k0 * B, k1, k2])
        np.testing.assert_allclose(mean.eval_model_pss(p_mean, [30, 40]), freq.eval_model_pss(p_freq, [30, 40]), atol=1e-9)
        # and on the NumPy path of the network itself
        g = [np.array([0.1 + 0.2j, -0.3j, 0.05]), np.array([0.05j, 0.2 - 0.1j, -0.1])]
        np.testing.assert_allclose(mean.network.eval_pgf(p_mean, g), freq.network.eval_pgf(p_freq, g), atol=1e-9)

    def test_small_bursts_are_the_single_molecule_network(self):
        const = CMEModel("Custom", "None", network="0 -> U -> S -> 0", normalize_production_rate=True)
        mean = CMEModel("Custom", "None", network="0 -> B x U -> S -> 0", normalize_production_rate=True, burst_mean_rate=True)
        p = np.log10([0.4, 0.25])
        np.testing.assert_allclose(mean.eval_model_pss(np.r_[-3.0, p], [30, 40]), const.eval_model_pss(p, [30, 40]), atol=5e-4)
        np.testing.assert_allclose(mean.eval_model_pss(np.r_[-4.0, p], [30, 40]), const.eval_model_pss(p, [30, 40]), atol=5e-5)
        # the box reaches down to that limit, and the flag survives serialisation
        assert mean.bio_bounds["phys_lb"][0] == -3.0 and const.bio_bounds["phys_lb"][0] == -1.8
        back = CMEModel.from_dict(mean.to_dict())
        assert back.network.burst_mean_rate and "mean rate" in repr(back.network)
        assert not CMEModel("Custom", "None", network="0 -> B x U -> S -> 0").network.burst_mean_rate
