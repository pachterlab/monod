"""
Parity tests: new Rust optimizer functions must agree with their Python
reference paths.

Strategy
--------
Each test:
1. Builds a small synthetic SearchData (using monod_core helpers so it
   is exactly the same object the optimizer will read from).
2. Runs the new Rust _sd optimizer.
3. Runs a reference that independently computes KLD at the found params
   using the corresponding PSS function that *already* has its own parity
   tests (eval_model_pss_protein_bursty / eval_model_pss_2d / eval_custom_network_pgf).
4. Asserts that the KLD returned by the optimizer matches the independently
   computed value (self-consistency), and that optimization actually improved
   over the starting point.

Additionally, for protein-bursty and amb-model, a scipy L-BFGS-B reference
with the same PSS function is run from the same x0 to verify that both
optimizers converge to the same (or better) minimum.

Models covered
--------------
* optimize_genes_protein_bursty_sd  — ProteinBursty (3-D PSS)
* optimize_genes_2d_amb_sd          — 2-D models with amb_model=Equal/Unequal
* optimize_genes_custom_sd          — Custom reaction network (2- and 3-species)
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "monod"))

try:
    import monod_core as _mc
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

pytestmark = pytest.mark.skipif(not _HAS_RUST, reason="monod_core not available")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(123)


def _make_layers(n_cells, n_genes, n_layers, high=25):
    return [
        np.ascontiguousarray(
            RNG.integers(0, high, size=(n_cells, n_genes)), dtype=np.int64
        )
        for _ in range(n_layers)
    ]


def _build_sd(layers, layer_names, limits_arr):
    """Build SearchData from pre-computed layers + limits (n_layers × n_genes)."""
    coords_list, freqs_list = _mc.make_state_dist(layers)
    n_cells = layers[0].shape[0]
    n_genes = layers[0].shape[1]
    gene_names = [f"G{i}" for i in range(n_genes)]
    return _mc.SearchData(
        layers, layer_names,
        np.ascontiguousarray(limits_arr, dtype=np.int64),
        coords_list, freqs_list,
        gene_names, n_cells, "unique",
    ), coords_list, freqs_list


def _compute_kld_from_pss(pss_flat, limits, coords, freqs, eps=1e-15):
    """Sparse KLD from a flat PSS array and coord/freq lists."""
    strides = []
    s = 1
    for l in reversed(limits):
        strides.insert(0, s)
        s *= l
    kld = 0.0
    for ms, f in zip(coords, freqs):
        idx = sum(c * st for c, st in zip(ms, strides))
        p = max(pss_flat[idx], eps)
        kld += f * np.log(f / p)
    return kld


# ---------------------------------------------------------------------------
# ProteinBursty parity
# ---------------------------------------------------------------------------

class TestOptimizeGenesProteinBurstySd:
    """optimize_genes_protein_bursty_sd must:
    - Return a KLD that matches independently computed KLD at found params.
    - Reduce KLD below the starting-point value.
    - Agree with a scipy L-BFGS-B reference from the same x0 (loose tol).
    """

    # Fixed model config used in all tests.
    FIT_UNSPLICED  = True
    PROTEIN_LIMIT  = np.inf
    MIN_FUDGE      = 0.1
    MAX_FUDGE      = 10.0
    FIXED_QUAD_T   = 20.0
    QUAD_ORDER     = 10

    # ProteinBursty has 5 params: [log10_b, log10_s_u, log10_s_s, log10_k_on, log10_k_prot]
    # Bounds matching inference defaults.
    LB = np.array([-2.0, -2.0, -2.0, -2.0, -2.0])
    UB = np.array([ 2.0,  2.0,  2.0,  2.0,  2.0])

    @pytest.fixture(scope="class")
    def synthetic_sd(self):
        # Use a tiny grid (8×6×5 = 240 points) to keep each PSS call fast
        # (~240 pts × ~1000 tsteps at x0=[0,0,0,0,0]).  Parity is tested via
        # self-consistency, not numerical accuracy, so small grids are fine.
        n_cells, n_genes, n_layers = 100, 1, 3
        layers = _make_layers(n_cells, n_genes, n_layers, high=4)
        layer_names = ["unspliced", "spliced", "protein"]
        # limits > high so all values are in-bounds
        limits_arr = np.array(
            [[8]],  # unspliced
            dtype=np.int64
        )
        # Build a proper 3-row limits array: [unspliced, spliced, protein]
        limits_arr = np.array([[8], [6], [5]], dtype=np.int64)
        sd, coords_list, freqs_list = _build_sd(layers, layer_names, limits_arr)
        return sd, coords_list, freqs_list, limits_arr

    def _x0_all(self, n_genes, n_restarts=1):
        # Use a fixed mild x0 (log10=0 → all rates=1) so num_tsteps stays
        # small (~100) during FD gradient evaluation.
        x0 = [[0.0] * len(self.LB)] * n_restarts
        return [x0 for _ in range(n_genes)]

    def _kld_at_params(self, x, limits, coords, freqs):
        """Compute KLD at x using eval_model_pss_protein_bursty."""
        pss_flat = _mc.eval_model_pss_protein_bursty(
            x,
            [int(l) for l in limits],
            self.FIT_UNSPLICED,
            float(self.PROTEIN_LIMIT),
            self.MIN_FUDGE,
            self.MAX_FUDGE,
        )
        return _compute_kld_from_pss(pss_flat, [int(l) for l in limits], coords, freqs)

    def test_kld_self_consistent(self, synthetic_sd):
        """KLD returned by optimizer matches independently computed KLD at found params."""
        sd, coords_list, freqs_list, limits_arr = synthetic_sd
        n_genes = sd.n_genes
        x0_all = self._x0_all(n_genes)

        params_out, klds_out = _mc.optimize_genes_protein_bursty_sd(
            sd, x0_list=x0_all,
            lb=self.LB.tolist(), ub=self.UB.tolist(),
            fit_unspliced=self.FIT_UNSPLICED,
            protein_limit=float(self.PROTEIN_LIMIT),
            min_fudge=self.MIN_FUDGE, max_fudge=self.MAX_FUDGE,
            fd_eps=1e-6, maxiter=5, ftol=1e-10, gtol=1e-6, eps=1e-15,
        )

        for gi in range(n_genes):
            limits = limits_arr[:, gi]
            kld_check = self._kld_at_params(
                params_out[gi], limits, coords_list[gi], freqs_list[gi]
            )
            assert abs(klds_out[gi] - kld_check) < 1e-10, (
                f"gene {gi}: returned KLD {klds_out[gi]:.6g} != "
                f"independently computed KLD {kld_check:.6g}"
            )

    def test_kld_improves_over_x0(self, synthetic_sd):
        """KLD at found params must be ≤ KLD at starting point."""
        sd, coords_list, freqs_list, limits_arr = synthetic_sd
        n_genes = sd.n_genes
        x0_all = self._x0_all(n_genes)

        params_out, klds_out = _mc.optimize_genes_protein_bursty_sd(
            sd, x0_list=x0_all,
            lb=self.LB.tolist(), ub=self.UB.tolist(),
            fit_unspliced=self.FIT_UNSPLICED,
            protein_limit=float(self.PROTEIN_LIMIT),
            min_fudge=self.MIN_FUDGE, max_fudge=self.MAX_FUDGE,
            fd_eps=1e-6, maxiter=20, ftol=1e-10, gtol=1e-6, eps=1e-15,
        )

        for gi in range(n_genes):
            limits = limits_arr[:, gi]
            kld_x0 = self._kld_at_params(
                x0_all[gi][0], limits, coords_list[gi], freqs_list[gi]
            )
            assert klds_out[gi] <= kld_x0 + 1e-10, (
                f"gene {gi}: KLD did not improve: {klds_out[gi]:.6g} > {kld_x0:.6g}"
            )

    def test_scipy_parity(self, synthetic_sd):
        """Rust optimizer KLD ≤ scipy L-BFGS-B from the same x0 (within 1e-3 rel)."""
        from scipy.optimize import minimize as scipy_minimize

        sd, coords_list, freqs_list, limits_arr = synthetic_sd
        n_genes = sd.n_genes
        x0_all = self._x0_all(n_genes)

        params_out, klds_out = _mc.optimize_genes_protein_bursty_sd(
            sd, x0_list=x0_all,
            lb=self.LB.tolist(), ub=self.UB.tolist(),
            fit_unspliced=self.FIT_UNSPLICED,
            protein_limit=float(self.PROTEIN_LIMIT),
            min_fudge=self.MIN_FUDGE, max_fudge=self.MAX_FUDGE,
            fd_eps=1e-6, maxiter=200, ftol=1e-10, gtol=1e-6, eps=1e-15,
        )

        for gi in range(n_genes):
            limits = [int(l) for l in limits_arr[:, gi]]
            x0 = np.array(x0_all[gi][0])

            def obj(x):
                return self._kld_at_params(x, limits, coords_list[gi], freqs_list[gi])

            res = scipy_minimize(
                obj, x0,
                method="L-BFGS-B",
                bounds=list(zip(self.LB, self.UB)),
                options={"maxiter": 200, "ftol": 1e-15, "gtol": 1e-9},
            )
            # Rust should reach at most the same KLD as scipy (within rounding).
            assert klds_out[gi] <= res.fun + 1e-3 * abs(res.fun) + 1e-8, (
                f"gene {gi}: Rust KLD {klds_out[gi]:.6g} > scipy KLD {res.fun:.6g}"
            )

    def test_three_layer_required(self):
        """Must raise if SearchData has fewer than 3 layers."""
        n_cells, n_genes = 50, 2
        layers = [np.ascontiguousarray(RNG.integers(0, 10, (n_cells, n_genes)), dtype=np.int64),
                  np.ascontiguousarray(RNG.integers(0, 10, (n_cells, n_genes)), dtype=np.int64)]
        limits = np.ascontiguousarray(np.array([[20, 20], [20, 20]], dtype=np.int64))
        sd, _, _ = _build_sd(layers, ["unspliced", "spliced"], limits)
        with pytest.raises(Exception, match="3 layers"):
            _mc.optimize_genes_protein_bursty_sd(
                sd, x0_list=[[[-1.0]*5]]*n_genes,
                lb=self.LB.tolist(), ub=self.UB.tolist(),
                fit_unspliced=True, protein_limit=np.inf,
                min_fudge=0.1, max_fudge=10.0,
            )


# ---------------------------------------------------------------------------
# amb_model parity
# ---------------------------------------------------------------------------

class TestOptimizeGenes2dAmbSd:
    """optimize_genes_2d_amb_sd must agree with eval_model_pss_2d(amb_model=…)
    and scipy L-BFGS-B from the same starting point."""

    FIXED_QUAD_T = 20.0
    QUAD_ORDER   = 10

    # Constitutive has 2 bio params + 1 (Equal) or 2 (Unequal) amb params.
    # Bounds: bio params [-2,2], amb params [-2,0].
    LB_EQUAL   = np.array([-2.0, -2.0, -2.0])       # [b, s_del, p_amb]
    UB_EQUAL   = np.array([ 2.0,  2.0,  0.0])
    LB_UNEQUAL = np.array([-2.0, -2.0, -2.0, -2.0])  # [b, s_del, p0, p1]
    UB_UNEQUAL = np.array([ 2.0,  2.0,  0.0,  0.0])

    @pytest.fixture(scope="class")
    def synthetic_sd_equal(self):
        """1-gene, tiny-grid (8×6×5=240 pts) SearchData for amb_model=Equal."""
        n_cells, n_genes = 100, 1
        layers = _make_layers(n_cells, n_genes, 3, high=4)
        layer_names = ["unspliced", "spliced", "ambient"]
        limits_arr = np.array([[8], [6], [5]], dtype=np.int64)
        sd, coords_list, freqs_list = _build_sd(layers, layer_names, limits_arr)
        return sd, coords_list, freqs_list, limits_arr

    @pytest.fixture(scope="class")
    def synthetic_sd_unequal(self):
        """Same as equal but used for Unequal tests."""
        n_cells, n_genes = 100, 1
        layers = _make_layers(n_cells, n_genes, 3, high=4)
        layer_names = ["unspliced", "spliced", "ambient"]
        limits_arr = np.array([[8], [6], [5]], dtype=np.int64)
        sd, coords_list, freqs_list = _build_sd(layers, layer_names, limits_arr)
        return sd, coords_list, freqs_list, limits_arr

    def _kld_at_params(self, x, limits, coords, freqs, bio_model, amb_model):
        """KLD using eval_model_pss_2d with amb support."""
        import cme_toolbox as ct
        n_amb = 1 if amb_model == "Equal" else 2
        n_bio = len(x) - n_amb
        pss_flat = _mc.eval_model_pss_2d(
            bio_model, x[:n_bio].tolist(), [int(l) for l in limits],
            self.FIXED_QUAD_T, self.QUAD_ORDER,
            None, amb_model, x[n_bio:].tolist(),
        )
        return _compute_kld_from_pss(pss_flat, [int(l) for l in limits], coords, freqs)

    @pytest.mark.parametrize("bio_model", ["Constitutive", "Bursty", "Extrinsic"])
    def test_equal_kld_self_consistent(self, bio_model, synthetic_sd_equal):
        sd, coords_list, freqs_list, limits_arr = synthetic_sd_equal
        n_genes = sd.n_genes
        lb, ub = self.LB_EQUAL, self.UB_EQUAL
        # Adjust param length for bio_model (Constitutive=2, others=3 bio params)
        n_bio = 2 if bio_model == "Constitutive" else 3
        lb_use = np.concatenate([lb[:n_bio], lb[-1:]])
        ub_use = np.concatenate([ub[:n_bio], ub[-1:]])
        x0_mid = ((lb_use + ub_use) / 2).tolist()
        x0_all = [[x0_mid] for _ in range(n_genes)]

        params_out, klds_out = _mc.optimize_genes_2d_amb_sd(
            sd, bio_model=bio_model, amb_model="Equal",
            x0_list=x0_all, lb=lb_use.tolist(), ub=ub_use.tolist(),
            fixed_quad_t=self.FIXED_QUAD_T, quad_order=self.QUAD_ORDER,
            fd_eps=1e-6, maxiter=5, ftol=1e-10, gtol=1e-6, eps=1e-15,
        )

        for gi in range(n_genes):
            limits = limits_arr[:, gi]
            kld_check = self._kld_at_params(
                np.array(params_out[gi]), limits,
                coords_list[gi], freqs_list[gi], bio_model, "Equal",
            )
            assert abs(klds_out[gi] - kld_check) < 1e-10, (
                f"{bio_model}/Equal gene {gi}: returned KLD {klds_out[gi]:.6g} "
                f"!= check {kld_check:.6g}"
            )

    @pytest.mark.parametrize("bio_model", ["Constitutive", "Bursty"])
    def test_unequal_kld_self_consistent(self, bio_model, synthetic_sd_unequal):
        sd, coords_list, freqs_list, limits_arr = synthetic_sd_unequal
        n_genes = sd.n_genes
        lb, ub = self.LB_UNEQUAL, self.UB_UNEQUAL
        n_bio = 2 if bio_model == "Constitutive" else 3
        lb_use = np.concatenate([lb[:n_bio], lb[-2:]])
        ub_use = np.concatenate([ub[:n_bio], ub[-2:]])
        x0_mid = ((lb_use + ub_use) / 2).tolist()
        x0_all = [[x0_mid] for _ in range(n_genes)]

        params_out, klds_out = _mc.optimize_genes_2d_amb_sd(
            sd, bio_model=bio_model, amb_model="Unequal",
            x0_list=x0_all, lb=lb_use.tolist(), ub=ub_use.tolist(),
            fixed_quad_t=self.FIXED_QUAD_T, quad_order=self.QUAD_ORDER,
            fd_eps=1e-6, maxiter=5, ftol=1e-10, gtol=1e-6, eps=1e-15,
        )

        for gi in range(n_genes):
            limits = limits_arr[:, gi]
            kld_check = self._kld_at_params(
                np.array(params_out[gi]), limits,
                coords_list[gi], freqs_list[gi], bio_model, "Unequal",
            )
            assert abs(klds_out[gi] - kld_check) < 1e-10, (
                f"{bio_model}/Unequal gene {gi}: returned KLD {klds_out[gi]:.6g} "
                f"!= check {kld_check:.6g}"
            )

    def test_equal_kld_improves(self, synthetic_sd_equal):
        sd, coords_list, freqs_list, limits_arr = synthetic_sd_equal
        n_genes = sd.n_genes
        lb, ub = self.LB_EQUAL, self.UB_EQUAL
        x0_mid = ((lb + ub) / 2).tolist()
        x0_all = [[x0_mid] for _ in range(n_genes)]

        params_out, klds_out = _mc.optimize_genes_2d_amb_sd(
            sd, bio_model="Constitutive", amb_model="Equal",
            x0_list=x0_all, lb=lb.tolist(), ub=ub.tolist(),
            fixed_quad_t=self.FIXED_QUAD_T, quad_order=self.QUAD_ORDER,
            fd_eps=1e-6, maxiter=50, ftol=1e-10, gtol=1e-6, eps=1e-15,
        )

        for gi in range(n_genes):
            limits = limits_arr[:, gi]
            kld_x0 = self._kld_at_params(
                np.array(x0_all[gi][0]), limits,
                coords_list[gi], freqs_list[gi], "Constitutive", "Equal",
            )
            assert klds_out[gi] <= kld_x0 + 1e-10, (
                f"Equal gene {gi}: KLD did not improve: {klds_out[gi]:.6g} > {kld_x0:.6g}"
            )

    def test_scipy_parity_equal(self, synthetic_sd_equal):
        """Rust Equal optimizer KLD ≤ scipy L-BFGS-B from the same x0."""
        from scipy.optimize import minimize as scipy_minimize

        sd, coords_list, freqs_list, limits_arr = synthetic_sd_equal
        n_genes = sd.n_genes
        lb, ub = self.LB_EQUAL, self.UB_EQUAL
        x0_mid = ((lb + ub) / 2).tolist()
        x0_all = [[x0_mid] for _ in range(n_genes)]

        params_out, klds_out = _mc.optimize_genes_2d_amb_sd(
            sd, bio_model="Constitutive", amb_model="Equal",
            x0_list=x0_all, lb=lb.tolist(), ub=ub.tolist(),
            fixed_quad_t=self.FIXED_QUAD_T, quad_order=self.QUAD_ORDER,
            fd_eps=1e-6, maxiter=200, ftol=1e-10, gtol=1e-6, eps=1e-15,
        )

        for gi in range(n_genes):
            limits = limits_arr[:, gi]
            x0 = np.array(x0_all[gi][0])

            def obj(x, gi=gi):
                return self._kld_at_params(
                    x, limits, coords_list[gi], freqs_list[gi], "Constitutive", "Equal"
                )

            res = scipy_minimize(
                obj, x0,
                method="L-BFGS-B",
                bounds=list(zip(lb, ub)),
                options={"maxiter": 200, "ftol": 1e-15, "gtol": 1e-9},
            )
            assert klds_out[gi] <= res.fun + 1e-3 * abs(res.fun) + 1e-8, (
                f"gene {gi}: Rust KLD {klds_out[gi]:.6g} > scipy KLD {res.fun:.6g}"
            )

    def test_wrong_amb_model_raises(self, synthetic_sd_equal):
        sd = synthetic_sd_equal[0]
        with pytest.raises(Exception):
            _mc.optimize_genes_2d_amb_sd(
                sd, bio_model="Constitutive", amb_model="None",
                x0_list=[[[-1.0, -1.0, -1.0]]] * sd.n_genes,
                lb=[-2.0]*3, ub=[2.0]*3,
                fixed_quad_t=20.0, quad_order=10,
            )

    def test_two_layer_raises(self):
        n_cells, n_genes = 50, 2
        layers = [np.ascontiguousarray(RNG.integers(0, 10, (n_cells, n_genes)), dtype=np.int64),
                  np.ascontiguousarray(RNG.integers(0, 10, (n_cells, n_genes)), dtype=np.int64)]
        limits = np.ascontiguousarray(np.array([[20, 20], [20, 20]], dtype=np.int64))
        sd, _, _ = _build_sd(layers, ["unspliced", "spliced"], limits)
        with pytest.raises(Exception, match="3 layers"):
            _mc.optimize_genes_2d_amb_sd(
                sd, bio_model="Constitutive", amb_model="Equal",
                x0_list=[[[-1.0, -1.0, -1.0]]] * n_genes,
                lb=[-2.0]*3, ub=[2.0]*3,
                fixed_quad_t=20.0, quad_order=10,
            )


# ---------------------------------------------------------------------------
# Custom network parity
# ---------------------------------------------------------------------------

class TestOptimizeGenesCustomSd:
    """optimize_genes_custom_sd must agree with eval_custom_network_pgf
    (same ODE, same params → same PSS → same KLD)."""

    MIN_FUDGE = 0.1
    MAX_FUDGE = 10.0

    # Bursty network topology: 0 -> B x U -> S -> 0
    # params: [b, s_u, s_s] (all log10)
    LB = np.array([-2.0, -2.0, -2.0])
    UB = np.array([ 1.0,  1.0,  1.0])

    # Flat topology arrays for the bursty-equivalent custom network.
    # Reactions:
    #   0: burst prod U (kind=0, rate=s_u idx=1, burst_param=b idx=0, burst_sp=U idx=0)
    #   1: first-order U->S (kind=2, rate=s_u idx=1, reactant=U idx=0, product S idx=1)
    #   2: first-order S->0 (kind=2, rate=s_s idx=2, reactant=S idx=1, no products)
    # prod_sp/prod_st encode products; prod_off[i..i+1] is the range.
    RXN_KINDS      = [0, 2, 2]
    RXN_RATE_IDXS  = [1, 1, 2]   # s_u for bursting + splicing, s_s for degradation
    RXN_EXTRA1     = [0, 0, 1]   # burst_param=b, reactant U, reactant S
    RXN_EXTRA2     = [0, -1, -1] # burst_sp=U for kind 0, else -1
    PROD_SP        = [1, 1]      # U->S produces S (sp 1); S->0 has no products
    PROD_ST        = [1, 1]
    PROD_OFF       = [0, 0, 1, 1]   # rxn0: no products [0,0); rxn1: S [0,1); rxn2: empty [1,1)
    HAS_NORM_RATE  = False
    N_SPECIES      = 2

    @pytest.fixture(scope="class")
    def synthetic_sd(self):
        # Tiny 2D grid (10×8=80 pts) keeps each PSS call fast.
        n_cells, n_genes = 100, 1
        layers = _make_layers(n_cells, n_genes, 2, high=4)
        layer_names = ["U", "S"]
        limits_arr = np.array([[10], [8]], dtype=np.int64)
        sd, coords_list, freqs_list = _build_sd(layers, layer_names, limits_arr)
        return sd, coords_list, freqs_list, limits_arr

    def _kld_at_params(self, x, limits, coords, freqs):
        """KLD using eval_custom_network_pgf + manual irfftn via Python scipy."""
        from scipy.fft import irfftn
        p_lin = 10.0 ** np.asarray(x)
        dt    = float(np.min(1.0 / p_lin) * self.MIN_FUDGE)
        t_max = float(np.max(1.0 / p_lin) * self.MAX_FUDGE)
        n_steps = int(np.ceil(t_max / dt))
        max_while = 10 * n_steps + 10_000

        re_e, im_e, mx_shape = _mc.eval_custom_network_pgf(
            self.N_SPECIES,
            [int(l) for l in limits],
            self.RXN_KINDS, self.RXN_RATE_IDXS,
            self.RXN_EXTRA1, self.RXN_EXTRA2,
            self.PROD_SP, self.PROD_ST, self.PROD_OFF,
            p_lin.tolist(), dt, n_steps, max_while,
        )
        exp_phi = (np.array(re_e) + 1j * np.array(im_e)).reshape(mx_shape)
        pss_raw = np.abs(irfftn(exp_phi, s=[int(l) for l in limits]))
        pss = (pss_raw / pss_raw.sum()).ravel()
        return _compute_kld_from_pss(pss, [int(l) for l in limits], coords, freqs)

    def test_kld_self_consistent(self, synthetic_sd):
        """KLD returned by optimizer matches independently computed KLD at found params."""
        sd, coords_list, freqs_list, limits_arr = synthetic_sd
        n_genes = sd.n_genes
        x0_all = [[[0.0, 0.0, 0.0]] for _ in range(n_genes)]

        params_out, klds_out = _mc.optimize_genes_custom_sd(
            sd, x0_list=x0_all,
            lb=self.LB.tolist(), ub=self.UB.tolist(),
            n_species=self.N_SPECIES,
            rxn_kinds=self.RXN_KINDS, rxn_rate_idxs=self.RXN_RATE_IDXS,
            rxn_extra1=self.RXN_EXTRA1, rxn_extra2=self.RXN_EXTRA2,
            prod_sp=self.PROD_SP, prod_st=self.PROD_ST, prod_off=self.PROD_OFF,
            has_norm_rate=self.HAS_NORM_RATE,
            min_fudge=self.MIN_FUDGE, max_fudge=self.MAX_FUDGE,
            fd_eps=1e-6, maxiter=5, ftol=1e-10, gtol=1e-6, eps=1e-15,
        )

        for gi in range(n_genes):
            limits = limits_arr[:, gi]
            kld_check = self._kld_at_params(
                np.array(params_out[gi]), limits, coords_list[gi], freqs_list[gi]
            )
            assert abs(klds_out[gi] - kld_check) < 1e-8, (
                f"gene {gi}: returned KLD {klds_out[gi]:.6g} != check {kld_check:.6g}"
            )

    def test_kld_improves_over_x0(self, synthetic_sd):
        sd, coords_list, freqs_list, limits_arr = synthetic_sd
        n_genes = sd.n_genes
        x0_all = [[[0.0, 0.0, 0.0]] for _ in range(n_genes)]

        params_out, klds_out = _mc.optimize_genes_custom_sd(
            sd, x0_list=x0_all,
            lb=self.LB.tolist(), ub=self.UB.tolist(),
            n_species=self.N_SPECIES,
            rxn_kinds=self.RXN_KINDS, rxn_rate_idxs=self.RXN_RATE_IDXS,
            rxn_extra1=self.RXN_EXTRA1, rxn_extra2=self.RXN_EXTRA2,
            prod_sp=self.PROD_SP, prod_st=self.PROD_ST, prod_off=self.PROD_OFF,
            has_norm_rate=self.HAS_NORM_RATE,
            min_fudge=self.MIN_FUDGE, max_fudge=self.MAX_FUDGE,
            fd_eps=1e-6, maxiter=20, ftol=1e-10, gtol=1e-6, eps=1e-15,
        )

        for gi in range(n_genes):
            limits = limits_arr[:, gi]
            kld_x0 = self._kld_at_params(
                np.array(x0_all[gi][0]), limits, coords_list[gi], freqs_list[gi]
            )
            assert klds_out[gi] <= kld_x0 + 1e-10, (
                f"gene {gi}: KLD did not improve: {klds_out[gi]:.6g} > {kld_x0:.6g}"
            )

    def test_scipy_parity(self, synthetic_sd):
        """Rust optimizer KLD ≤ scipy L-BFGS-B from the same x0 (loose tol)."""
        from scipy.optimize import minimize as scipy_minimize

        sd, coords_list, freqs_list, limits_arr = synthetic_sd
        n_genes = sd.n_genes
        x0_all = [[[0.0, 0.0, 0.0]] for _ in range(n_genes)]

        params_out, klds_out = _mc.optimize_genes_custom_sd(
            sd, x0_list=x0_all,
            lb=self.LB.tolist(), ub=self.UB.tolist(),
            n_species=self.N_SPECIES,
            rxn_kinds=self.RXN_KINDS, rxn_rate_idxs=self.RXN_RATE_IDXS,
            rxn_extra1=self.RXN_EXTRA1, rxn_extra2=self.RXN_EXTRA2,
            prod_sp=self.PROD_SP, prod_st=self.PROD_ST, prod_off=self.PROD_OFF,
            has_norm_rate=self.HAS_NORM_RATE,
            min_fudge=self.MIN_FUDGE, max_fudge=self.MAX_FUDGE,
            fd_eps=1e-6, maxiter=200, ftol=1e-10, gtol=1e-6, eps=1e-15,
        )

        for gi in range(n_genes):
            limits = limits_arr[:, gi]
            x0 = np.array(x0_all[gi][0])

            def obj(x, gi=gi):
                return self._kld_at_params(x, limits, coords_list[gi], freqs_list[gi])

            res = scipy_minimize(
                obj, x0,
                method="L-BFGS-B",
                bounds=list(zip(self.LB, self.UB)),
                options={"maxiter": 200, "ftol": 1e-15, "gtol": 1e-9},
            )
            assert klds_out[gi] <= res.fun + 1e-3 * abs(res.fun) + 1e-8, (
                f"gene {gi}: Rust KLD {klds_out[gi]:.6g} > scipy KLD {res.fun:.6g}"
            )

    def test_unsupported_species_raises(self, synthetic_sd):
        sd = synthetic_sd[0]
        with pytest.raises(Exception, match="2- and 3-species"):
            _mc.optimize_genes_custom_sd(
                sd, x0_list=[[[-1.0, -1.0, -1.0]]] * sd.n_genes,
                lb=self.LB.tolist(), ub=self.UB.tolist(),
                n_species=4,
                rxn_kinds=self.RXN_KINDS, rxn_rate_idxs=self.RXN_RATE_IDXS,
                rxn_extra1=self.RXN_EXTRA1, rxn_extra2=self.RXN_EXTRA2,
                prod_sp=self.PROD_SP, prod_st=self.PROD_ST, prod_off=self.PROD_OFF,
                has_norm_rate=self.HAS_NORM_RATE,
                min_fudge=self.MIN_FUDGE, max_fudge=self.MAX_FUDGE,
            )
