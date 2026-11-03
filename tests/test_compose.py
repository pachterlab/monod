"""CMEModel.compose: mechanism pieces map onto the kernel families, or onto a
reaction network where no family exists (2026-11-02)."""
import numpy as np
import pytest

from monod.cme_toolbox import CMEModel

L2 = [20, 24]


class TestFamilies:
    @pytest.mark.parametrize("kwargs,family", [
        (dict(transcription="constitutive"), "Constitutive"),
        (dict(transcription="bursty"), "Bursty"),
        (dict(transcription="bursty", regulation="cir"), "CIR"),
        (dict(transcription="bursty", regulation="extrinsic"), "Extrinsic"),
        (dict(transcription="bursty", lifetime="delayed_splicing"), "DelayedSplicing"),
        (dict(transcription="bursty", lifetime="delayed_degradation"), "Delay"),
        (dict(transcription="bursty", protein=True), "ProteinBursty"),
    ])
    def test_maps_onto_the_family(self, kwargs, family):
        m = CMEModel.compose(seq_model="Poisson", **kwargs)
        direct = CMEModel(family, "Poisson")
        assert m.bio_model == family and m.to_dict() == direct.to_dict()
        assert m.composition["transcription"] == kwargs["transcription"]

    def test_protein_options_and_ambiguous(self):
        m = CMEModel.compose("bursty", protein=dict(protein_limit=10, fit_unspliced=False), amb_model="Unequal")
        assert m.bio_model == "ProteinBursty" and m.protein_limit == 10 and m.fit_unspliced is False
        assert m.model_modalities == ["unspliced", "spliced", "protein", "ambiguous"] and m.amb_species == ("unspliced", "spliced")

    def test_locus(self):
        m = CMEModel.compose("bursty", locus=dict(n_sites=2, rule="switch"), seq_model="Poisson")
        assert m.bio_model == "Chromatin" and m.bursty and m.n_sites == 2 and m.rule == "switch"
        c = CMEModel.compose("constitutive", locus=dict(n_sites=1, rule="additive", periodic=True))
        assert c.bio_model == "Chromatin" and not c.bursty and c.periodic
        with pytest.raises(NotImplementedError, match="designed"):
            CMEModel.compose("bursty", locus=dict(n_sites=2), protein=True)
        with pytest.raises(ValueError, match="cannot be combined"):
            CMEModel.compose("bursty", locus=dict(n_sites=2), regulation="cir")


class TestNetworkForm:
    def test_constitutive_protein_is_a_network(self):
        m = CMEModel.compose("constitutive", protein=True)
        assert m.bio_model == "Custom" and m.model_modalities == ["unspliced", "spliced", "protein"]
        assert m.network.network_str == "0 -> unspliced -> spliced -> 0; spliced -> spliced + protein; protein -> 0"
        assert m.get_num_params() == 4                                 # beta, gamma, k_p, gamma_p; production normalised
        pss = m.eval_model_pss(np.log10([0.5, 0.3, 1.0, 0.5]), [12, 14, 10])
        assert pss.shape == (12, 14, 10) and pss.sum() == pytest.approx(1.0)

    def test_network_form_reproduces_the_bursty_kernel(self):
        theta = np.log10([3.0, 0.4, 0.25])
        ref = CMEModel("Bursty", "None").eval_model_pss(theta, [40, 50])
        m = CMEModel.compose("bursty", prefer_network=True)
        assert m.bio_model == "Custom" and m.model_modalities == ["unspliced", "spliced"]
        np.testing.assert_allclose(m.eval_model_pss(theta, [40, 50]), ref, atol=5e-6)

    def test_network_form_takes_the_ambiguous_channel(self):
        m = CMEModel.compose("constitutive", protein=True, amb_model="Equal")
        assert m.model_modalities == ["unspliced", "spliced", "protein", "ambiguous"]
        assert m.amb_species == ("unspliced", "spliced") and m.get_num_params() == 5

    def test_moments_start_is_the_familys(self):
        """The network form starts where its family would; a plain network starts mid-box."""
        from monod.simulate import simulate_counts
        mom = {"MOM_unspliced_mean": 2.0, "MOM_unspliced_var": 8.0, "MOM_spliced_mean": 6.0, "MOM_spliced_var": 30.0,
               "MOM_protein_mean": 12.0, "MOM_protein_var": 90.0, "MOM_cov_unspliced_protein": 1.5,
               "MOM_cov_unspliced_spliced": 2.0, "MOM_cov_spliced_protein": 10.0}
        for seq, samp in [("None", None), ("Poisson", np.log10([0.6, 0.8, 0.5]))]:
            fam = CMEModel("ProteinBursty", seq)
            net = CMEModel.compose("bursty", protein=True, seq_model=seq, prefer_network=True)
            lb, ub = np.array(net.bio_bounds["phys_lb"]), np.array(net.bio_bounds["phys_ub"])
            np.testing.assert_allclose(net.get_MoM(mom, lb, ub, samp), fam.get_MoM(mom, lb, ub, samp))
        plain = CMEModel("Custom", "None", network="0 -> B x unspliced -> spliced -> 0; spliced -> spliced + protein; protein -> 0",
                         normalize_production_rate=True)
        lb, ub = np.array(plain.bio_bounds["phys_lb"]), np.array(plain.bio_bounds["phys_ub"])
        np.testing.assert_allclose(plain.get_MoM(mom, lb, ub), 0.5 * (lb + ub))
        # the constitutive form from simulated cells: within a third of a decade of the truth
        m = CMEModel.compose("constitutive", protein=True)
        theta = np.log10([0.5, 0.3, 1.0, 0.5])
        c = simulate_counts(m, theta, 4000, None, seed=1)
        X = np.column_stack([c[k] for k in m.model_modalities]).astype(float)
        mom = {f"MOM_{k}_mean": X[:, i].mean() for i, k in enumerate(m.model_modalities)}
        mom.update({f"MOM_{k}_var": X[:, i].var() for i, k in enumerate(m.model_modalities)})
        mom["MOM_cov_unspliced_protein"] = np.cov(X[:, 0], X[:, 2])[0, 1]
        lb, ub = np.array(m.bio_bounds["phys_lb"]), np.array(m.bio_bounds["phys_ub"])
        x0 = m.get_MoM(mom, lb, ub)
        np.testing.assert_allclose(x0, theta, atol=0.3)                  # gamma_p from the protein Fano factor
        np.testing.assert_allclose(m.get_MoM(mom, lb, ub, np.zeros(2)), x0)   # the pipeline's placeholder samp for seq None

    def test_fixed_lifetimes_need_the_bursty_kernels(self):
        with pytest.raises(ValueError, match="fixed lifetime"):
            CMEModel.compose("constitutive", lifetime="delayed_degradation")
        with pytest.raises(ValueError, match="fixed lifetime"):
            CMEModel.compose("bursty", lifetime="delayed_splicing", protein=True)


class TestArguments:
    def test_bad_values(self):
        with pytest.raises(ValueError, match="transcription"):
            CMEModel.compose("telegraph")
        with pytest.raises(ValueError, match="regulation"):
            CMEModel.compose("bursty", regulation="locus")
        with pytest.raises(ValueError, match="lifetime"):
            CMEModel.compose("bursty", lifetime="gamma")
        with pytest.raises(ValueError, match="regulation='cir'"):
            CMEModel.compose("constitutive", regulation="cir")

    def test_round_trip_and_record(self):
        m = CMEModel.compose("bursty", protein=True, seq_model="Poisson", amb_model="Equal")
        back = CMEModel.from_dict(m.to_dict())
        assert back.to_dict() == m.to_dict()
        assert m.composition == dict(transcription="bursty", regulation=None, lifetime="exponential", protein={},
                                     locus=None, seq_model="Poisson", amb_model="Equal")
