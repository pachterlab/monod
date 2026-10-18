"""Delayed reactions (``=>``) in custom networks on the Rust accelerator.

The delayed state is read from the history by linear interpolation between
stored steps in both implementations, so the result depends smoothly on the
delay and the two implementations agree to rounding.
"""
import numpy as np
import pytest

import monod.cme_toolbox as ct
from monod._core import _mc
from monod.cme_toolbox import CMEModel
from monod.reaction_network import ReactionNetwork

pytestmark = pytest.mark.skipif(_mc is None, reason="monod_core not available")

NET2 = "0 -> B x U => S -> 0"            # params: B, tauinv0, k1, k2 (k0 normalised)
NET3 = "0 -> B x U => S => P -> 0"       # params: B, tauinv0, tauinv1, k1, k2, k3


def _model(net):
    return CMEModel("Custom", "None", network=net, normalize_production_rate=True,
                    min_fudge=0.05, max_fudge=15)


def _pss_both(model, p, limits, monkeypatch):
    rust = model.eval_model_pss(p, limits)
    monkeypatch.setattr(ct, "_HAS_RUST", False)
    py = model.eval_model_pss(p, limits)
    monkeypatch.setattr(ct, "_HAS_RUST", True)
    return rust, py


class TestTopology:
    def test_delayed_reaction_is_kind_3_with_delay_index(self):
        net = ReactionNetwork(NET2, normalize_production_rate=True)
        topo = net.rust_topology()
        assert topo["rxn_kinds"] == [0, 3, 2]
        assert topo["param_names"] == ["B", "tauinv0", "k1", "k2", "k0"]
        assert topo["rxn_extra2"][1] == topo["param_names"].index("tauinv0")

    def test_two_delays_get_distinct_indices(self):
        topo = ReactionNetwork(NET3, normalize_production_rate=True).rust_topology()
        delayed = [e for k, e in zip(topo["rxn_kinds"], topo["rxn_extra2"]) if k == 3]
        assert len(delayed) == 2 and delayed[0] != delayed[1]


class TestParity:
    @pytest.mark.parametrize("net,p,limits", [
        (NET2, np.log10([2.0, 2.0, 0.5, 0.5]), [16, 14]),
        (NET2, np.log10([3.0, 0.7, 1.5, 0.4]), [16, 14]),
        (NET3, np.log10([2.0, 2.0, 1.0, 0.6, 0.5, 0.8]), [10, 8, 8]),
    ])
    def test_rust_matches_python(self, net, p, limits, monkeypatch):
        rust, py = _pss_both(_model(net), p, limits, monkeypatch)
        assert rust.shape == py.shape
        np.testing.assert_allclose(rust, py, rtol=1e-9, atol=1e-12)
        assert rust.sum() == pytest.approx(1.0, abs=1e-6)


class TestPhysics:
    def test_pss_depends_smoothly_on_the_delay(self):
        """Interpolated history: no staircase in the inverse-delay parameter."""
        model = _model(NET2)
        taus = np.linspace(np.log10(2.0), np.log10(2.0) + 0.01, 6)
        psss = [model.eval_model_pss(np.log10([2.0, 10 ** t, 0.5, 0.5]), [20, 20]) for t in taus]
        diffs = [np.max(np.abs(a - b)) for a, b in zip(psss[:-1], psss[1:])]
        assert max(diffs) < 1e-3
        assert max(diffs) / max(min(diffs), 1e-12) < 20, diffs

    def test_delay_changes_the_joint_but_not_the_marginals(self):
        """Shifting the arrival times of S leaves both stationary marginals
        unchanged and only reshapes the joint distribution."""
        delayed = _model(NET2).eval_model_pss(np.log10([2.0, 1.0, 0.5, 0.5]), [30, 30])
        instant = CMEModel("Custom", "None", network="0 -> B x U -> S -> 0",
                           normalize_production_rate=True, min_fudge=0.05, max_fudge=15
                           ).eval_model_pss(np.log10([2.0, 0.5, 0.5]), [30, 30])
        np.testing.assert_allclose(delayed.sum(1), instant.sum(1), atol=2e-3)
        np.testing.assert_allclose(delayed.sum(0), instant.sum(0), atol=2e-3)
        u = np.arange(30)[:, None]; s = np.arange(30)[None, :]
        def cov(p):
            return (p * u * s).sum() - (p * u).sum() * (p * s).sum()
        assert abs(cov(delayed) - cov(instant)) > 1e-2


@pytest.fixture(scope="module")
def sd():
    rng = np.random.default_rng(5)
    layers = [np.ascontiguousarray(rng.integers(0, 5, size=(150, 1)), dtype=np.int64) for _ in range(2)]
    coords, freqs = _mc.make_state_dist(layers)
    limits = np.array([[8], [8]], dtype=np.int64)
    return _mc.searchdata_from_arrays(layers, ["unspliced", "spliced"], limits, coords, freqs, ["G0"], 150, "unique")


class TestOptimizer:
    def test_rust_optimizer_improves_kld(self, sd):
        model = _model(NET2)
        net = model.network
        topo = net.rust_topology()
        x0 = np.log10([2.0, 2.0, 0.5, 0.5])
        kld0 = model.eval_model_kld(x0, [8, 8], None, sd.hist[0])
        params, klds, status = _mc.optimize_genes_custom_sd(
            sd, x0_list=[[x0.tolist()]], lb=[-1.0] * 4, ub=[2.0] * 4, n_species=2,
            rxn_kinds=topo["rxn_kinds"], rxn_rate_idxs=topo["rxn_rate_idxs"],
            rxn_extra1=topo["rxn_extra1"], rxn_extra2=topo["rxn_extra2"],
            prod_sp=topo["prod_sp"], prod_st=topo["prod_st"], prod_off=topo["prod_off"],
            has_norm_rate=True, min_fudge=0.05, max_fudge=15.0, fd_eps=1e-6, maxiter=15,
            ftol=1e-9, gtol=1e-6, eps=1e-15, m_lbfgs=10, num_threads=None, return_status=True,
        )
        assert status[0] in (0, 1, 2)
        assert klds[0] < kld0
        # the reported KLD is the KLD at the reported parameters
        assert klds[0] == pytest.approx(model.eval_model_kld(np.asarray(params[0]), [8, 8], None, sd.hist[0]), rel=1e-6)

    def test_inference_routes_delayed_network_to_rust(self, sd, monkeypatch):
        from monod.inference import InferenceParameters, GradientInference
        model = _model(NET2)
        # custom networks have no moments estimator: start from the random box
        ip = InferenceParameters("delayed_rust_test", model, phys_lb=[-1.0] * 4, phys_ub=[2.0] * 4,
                                 gradient_params={"max_iterations": 3, "init_pattern": "random",
                                                  "num_restarts": 1, "num_gene_cores": 1, "use_rust_lbfgsb": True})
        gi = GradientInference(ip, model, sd, 0)
        np.random.seed(0)
        calls = []
        real = _mc.optimize_genes_custom_sd

        def spy(*a, **k):
            calls.append(k.get("rxn_kinds"))
            return real(*a, **k)

        monkeypatch.setattr(_mc, "optimize_genes_custom_sd", spy)
        params, klds, obj, _ = gi.iterate_over_genes(model, sd)
        assert calls and 3 in calls[0]
        assert np.isfinite(klds).all()
