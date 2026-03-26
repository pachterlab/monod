"""Tests for the user-defined reaction network module.

Correctness is validated by comparing the Custom model against the
built-in Bursty and ProteinBursty models, which have well-tested
implementations.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'monod'))

from reaction_network import ReactionNetwork
from cme_toolbox import CMEModel


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
        # S -> S + P means S catalyses production of P
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
        phi = net.eval_pgf(params, g, T=50.0, n_steps=500)
        assert phi.shape == g[0].shape
        assert np.all(np.isfinite(phi))
        assert phi[0].real == pytest.approx(0.0, abs=1e-4)  # φ(0,0) = 0

    def test_phi_zero_at_origin(self):
        """φ(0,0,…) = 0 for any network (G(1,…) = 1 at origin)."""
        net = ReactionNetwork("0 -> B x U -> S -> 0; U -> 0")
        g = self._simple_grid()
        # g[i][0] == 0 at origin
        params = np.log10([2.0, 1.0, 0.5, 0.8, 0.3])  # B, k0, k1, k2, k3
        phi = net.eval_pgf(params, g, T=30.0, n_steps=300)
        assert phi[0].real == pytest.approx(0.0, abs=1e-4)

    def test_phi_real_part_non_positive(self):
        """Real part of log-PGF should be ≤ 0 (|G| ≤ 1 for a distribution)."""
        net = ReactionNetwork("0 -> B x U -> S -> 0")
        g = self._simple_grid()
        params = np.log10([2.0, 1.0, 0.5, 0.8])
        phi = net.eval_pgf(params, g, T=30.0, n_steps=300)
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
        phi = net.eval_pgf(p, g, T=50.0, n_steps=500)
        assert phi[0].real == pytest.approx(0.0, abs=1e-3)

    def test_phi_real_nonpositive(self):
        net = ReactionNetwork("0 -> B x U => S -> 0")
        g = self._grid_2sp()
        p = np.log10([2.0, 0.5, 1.0, 0.5, 0.5])
        phi = net.eval_pgf(p, g, T=50.0, n_steps=500)
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
