"""
This script provides convenience functions for evaluating RNA distributions.
"""

import logging
import numpy as np
from numba import jit
import scipy
from scipy import integrate

from scipy.fft import irfftn

# from .nn_toolbox import basic_ml_bivariate, ml_microstate_logP
from extract_data import log
from reaction_network import ReactionNetwork

_MESH_CACHE: dict = {}


def _build_mesh(limits_tuple, mx_tuple):
    """Build and cache the complex evaluation mesh for eval_model_pss."""
    key = (limits_tuple, mx_tuple)
    if key not in _MESH_CACHE:
        u = []
        for i, m in enumerate(mx_tuple):
            l = np.arange(m)
            u_ = np.exp(-2j * np.pi * l / limits_tuple[i]) - 1
            u.append(u_)
        g = np.meshgrid(*u, indexing="ij")
        _MESH_CACHE[key] = np.array([arr.flatten() for arr in g])
    return _MESH_CACHE[key]

# ---------------------------------------------------------------------------
# Optional Rust backend (monod_core).  When available, eval_model_pss for
# 2-modality models with seq_model="None" and amb_model="None" is delegated
# to the Rust implementation (same results, faster computation).
# ---------------------------------------------------------------------------
try:
    import monod_core as _mc
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

# ---------------------------------------------------------------------------
# Optional GPU backend (PyTorch).  When a CUDA or MPS GPU is available,
# eval_model_pss for 2-modality models with seq_model="None" is delegated
# to a fully-vectorised torch implementation that broadcasts the grid ×
# quadrature computation onto the GPU.  Results match the Python/Rust
# baseline within rtol=1e-5.
# ---------------------------------------------------------------------------
try:
    import torch as _torch

    def _probe_float64(device_str):
        """Return True if the device supports float64 tensors."""
        try:
            t = _torch.ones(1, dtype=_torch.float64, device=device_str)
            _ = t + t   # force a computation
            return True
        except Exception:
            return False

    if _torch.cuda.is_available() and _probe_float64("cuda"):
        _GPU_DEVICE = _torch.device("cuda")
        _HAS_GPU = True
    elif hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available() and _probe_float64("mps"):
        _GPU_DEVICE = _torch.device("mps")
        _HAS_GPU = True
    else:
        _HAS_GPU = False
        _GPU_DEVICE = None
except ImportError:
    _HAS_GPU = False
    _GPU_DEVICE = None


def _gpu_build_mesh_2d(l0, l1, device):
    """Return (g0, g1) flat complex128 tensors of length l0*(l1//2+1)."""
    mx1 = l1 // 2 + 1
    pi = np.pi
    k0 = _torch.arange(l0, dtype=_torch.float64, device=device)
    k1 = _torch.arange(mx1, dtype=_torch.float64, device=device)
    a0 = k0 * (-2.0 * pi / l0)
    a1 = k1 * (-2.0 * pi / l1)
    u0 = _torch.complex(_torch.cos(a0) - 1.0, _torch.sin(a0))   # [l0]
    u1 = _torch.complex(_torch.cos(a1) - 1.0, _torch.sin(a1))   # [mx1]
    g0 = u0.unsqueeze(1).expand(l0, mx1).reshape(-1)             # [n_grid]
    g1 = u1.unsqueeze(0).expand(l0, mx1).reshape(-1)             # [n_grid]
    return g0, g1


def _gpu_pgf_bursty(g0, g1, b, beta, gamma, T, quad_order):
    """Vectorised Bursty log-PGF on GPU (broadcast grid × quadrature)."""
    from numpy.polynomial.legendre import leggauss
    device = g0.device
    xi_np, wi_np = leggauss(quad_order)
    xi = _torch.tensor(xi_np, dtype=_torch.float64, device=device)
    wi = _torch.tensor(wi_np, dtype=_torch.float64, device=device)
    t_half = T / 2.0
    x = t_half + t_half * xi                             # [Q]
    close = bool(np.isclose(beta, gamma))
    eb = _torch.exp(-beta * x).unsqueeze(0)              # [1, Q]
    eg = _torch.exp(-gamma * x).unsqueeze(0)             # [1, Q]
    g0_ = g0.unsqueeze(1)                                # [N, 1]
    g1_ = g1.unsqueeze(1)                                # [N, 1]
    if close:
        x_ = x.unsqueeze(0)
        u = b * (g0_ * eb + g1_ * (beta * x_ * eg))
    else:
        f = beta / (beta - gamma)
        c2 = g1_ * f
        c1 = g0_ - c2
        u = b * (c1 * eb + c2 * eg)
    integrand = u / (1.0 - u)                            # [N, Q]
    return (integrand * (wi * t_half)).sum(dim=1)         # [N]


def _gpu_pgf_cir(g0, g1, b, beta, gamma, T, quad_order):
    """Vectorised CIR log-PGF on GPU (broadcast grid × quadrature)."""
    from numpy.polynomial.legendre import leggauss
    device = g0.device
    xi_np, wi_np = leggauss(quad_order)
    xi = _torch.tensor(xi_np, dtype=_torch.float64, device=device)
    wi = _torch.tensor(wi_np, dtype=_torch.float64, device=device)
    t_half = T / 2.0
    x = t_half + t_half * xi                             # [Q]
    close = bool(np.isclose(beta, gamma))
    eb = _torch.exp(-beta * x).unsqueeze(0)              # [1, Q]
    eg = _torch.exp(-gamma * x).unsqueeze(0)             # [1, Q]
    g0_ = g0.unsqueeze(1)                                # [N, 1]
    g1_ = g1.unsqueeze(1)                                # [N, 1]
    if close:
        x_ = x.unsqueeze(0)
        u = b * (g0_ * eb + g1_ * (beta * x_ * eg))
    else:
        f = beta / (beta - gamma)
        c2 = g1_ * f
        c1 = g0_ - c2
        u = b * (c1 * eb + c2 * eg)
    integrand = 1.0 - _torch.sqrt(1.0 - 4.0 * u)        # [N, Q]
    return (integrand * (wi * t_half)).sum(dim=1) / 2.0  # [N]


def _eval_model_pss_gpu(bio_model, p_log, limits, fixed_quad_T, quad_order, device):
    """Evaluate 2D PSS on GPU for the six supported bio_models.

    p_log is an array of log10 parameters (same convention as Python/Rust).
    Returns a numpy array of shape (l0, l1).
    """
    p = 10.0 ** np.asarray(p_log)
    l0, l1 = int(limits[0]), int(limits[1])

    g0, g1 = _gpu_build_mesh_2d(l0, l1, device)

    if bio_model == "Constitutive":
        beta, gamma = p[0], p[1]
        gf = g0 / beta + g1 / gamma

    elif bio_model == "Extrinsic":
        alpha, beta, gamma = p[0], p[1], p[2]
        one = _torch.ones(1, dtype=_torch.complex128, device=device)
        gf = -alpha * _torch.log(one - g0 / beta - g1 / gamma)

    elif bio_model == "Delay":
        b, beta, tauinv = p[0], p[1], p[2]
        tau = 1.0 / tauinv
        one = _torch.ones(1, dtype=_torch.complex128, device=device)
        exp_bt = float(np.exp(-beta * tau))
        u = g1 + (g0 - g1) * exp_bt
        term1 = -_torch.log(one - u * b) / beta
        ratio = (u * b - one) / (g0 * b - one)
        term2 = _torch.log(ratio) / (beta * (one - g1 * b))
        term3 = g1 * b * tau / (one - g1 * b)
        gf = term1 + term2 + term3

    elif bio_model == "DelayedSplicing":
        b, tauinv, gamma = p[0], p[1], p[2]
        tau = 1.0 / tauinv
        one = _torch.ones(1, dtype=_torch.complex128, device=device)
        gf = g0 * b * tau / (one - g0 * b) - _torch.log(one - g1 * b) / gamma

    elif bio_model == "Bursty":
        b, beta, gamma = p[0], p[1], p[2]
        T = fixed_quad_T * (1.0 / beta + 1.0 / gamma + 1.0)
        gf = _gpu_pgf_bursty(g0, g1, b, beta, gamma, T, quad_order)

    elif bio_model == "CIR":
        b, beta, gamma = p[0], p[1], p[2]
        T = fixed_quad_T * (1.0 / beta + 1.0 / gamma + 1.0)
        gf = _gpu_pgf_cir(g0, g1, b, beta, gamma, T, quad_order)

    else:
        raise ValueError(f"Unknown bio_model for GPU path: {bio_model}")

    gf = _torch.exp(gf).reshape(l0, l1 // 2 + 1)
    pss = _torch.fft.irfftn(gf, s=(l0, l1))
    pss = pss.abs()
    pss = pss / pss.sum()
    # Use tolist() then np.array to avoid ABI issues with some torch+numpy combos.
    return np.array(pss.cpu().tolist(), dtype=np.float64)

class CMEModel:
    """Stores and evaluates biological and technical variation models.

    This class is used as a general wrapper around modular, generating function-based
    solutions for chemical master equation models.

    We use "sampling," "sequencing," and "technical" parameters interchangeably.
    We use "biological," "biophysical," and "transcriptional" parameters interchangeably.
    When used in the context of CMEModel, a sampling parameter "samp" is gene-specific,
    and typically corresponds to "regressor" in inference.py.

    The sampling parameters use the following conventions:
        Null model uses samp=None.
        Bernoulli model uses log-sampling rates.
        Poisson model uses log-sampling rates.

    Attributes
    ----------
    bio_model: str
        type of transcriptional model describing biological variation.
    seq_model: str
        type of sequencing model describing technical variation.
    amb_model: str
        type of model describing ambiguity in RNA assignements.
    available_biomodels: tuple of str
        implemented transcriptional model types.
    available_seqmodels: tuple of str
        implemented technical model types.
    fixed_quad_T: float
        time horizon used for integration with Gaussian quadature.
    quad_order: int
        Gaussian quadrature order.
    quad_vec_T: float or np.inf
        time horizon used for adaptive integration.
    quad_method: str
        quadrature method to use.
        if 'fixed_quad', use Gaussian quadrature via scipy.integrate.fixed_quad.
        if 'quad_vec', use adaptive integration via scipy.integrate.quad_vec.
    prot
    """
    def __init__(
        self,
        bio_model,
        seq_model,
        amb_model="None",
        quad_method="fixed_quad",
        fixed_quad_T=10,
        quad_order=60,
        quad_vec_T=np.inf,
        protein_limit = np.inf,
        fit_unspliced = True,
        min_fudge = 0.1,
        max_fudge = 10,
        network = None,
        normalize_production_rate = False,
    ):
        """Initialize the CMEModel instance.

        A typical mammalian or bacterial cell will be effectively described by bio_model = "Bursty".
        A typical sequencing workflow will be effectively described by seq_model = "Poisson";
            this model is mandatory for incorporating a length bias.
        A high-fidelity fit uses quad_method = "quad_vec". A low-fidelity fit uses "fixed_quad".
            High-fidelity fits are several orders of magnitude slower.
        amb_model is not used in a typical workflow.

        Parameters
        ----------
        bio_model: str
            type of transcriptional model describing biological variation.
        seq_model: str
            type of sequencing model describing technical variation.
        amb_model: str, optional
            type of model describing ambiguity in RNA assignments.
        quad_method: str, optional
            quadrature method to use.
            if 'fixed_quad', use Gaussian quadrature via scipy.integrate.fixed_quad.
            if 'quad_vec', use adaptive integration via scipy.integrate.quad_vec.
        fixed_quad_T: float, optional
            time horizon used for integration with Gaussian quadature.
        quad_order: int, optional
            Gaussian quadrature order.
        quad_vec_T: float or np.inf, optional
            time horizon used for adaptive integration.
        protein_limit: int
            coarse grid for pgf only relevant for protein model.
        """
        self.bio_model = bio_model
        self.available_biomodels = (
            "Delay",
            "Bursty",
            "Extrinsic",
            "Constitutive",
            "CIR",
            "DelayedSplicing",
            "ProteinBursty",
            "Custom",
        )
        self.available_seqmodels = ("None", "Bernoulli", "Poisson")
        self.available_ambmodels = ("None", "Equal", "Unequal")

        # Store the custom reaction network (required when bio_model == "Custom").
        if bio_model == "Custom":
            if network is None:
                raise ValueError(
                    "bio_model='Custom' requires a network string passed as network=..."
                )
            self.network = ReactionNetwork(
                network, normalize_production_rate=normalize_production_rate
            )
        else:
            self.network = None

        # Define the modalities used for each model, and their order.
        CMEModel.available_model_modalities = {"Delay":['unspliced', 'spliced'],
            "Bursty":['unspliced', 'spliced'],
            "Extrinsic":['unspliced', 'spliced'],
            "Constitutive":['unspliced', 'spliced'],
            "CIR":['unspliced', 'spliced'],
            "DelayedSplicing":['unspliced', 'spliced'],
            "ProteinBursty":['unspliced', 'spliced', 'protein']}

        try:
            self.model_modalities = CMEModel.available_model_modalities[self.bio_model]

        except KeyError:
            if self.bio_model == "Custom":
                self.model_modalities = self.network.species
            else:
                log.error("Modalities unknown for model: {}".format(self.bio_model))

        print('The expected modalities for this model are:', self.model_modalities)
        print('If your anndata layers have different names, please give a modality dictionary of the form: modality_name_dict  = {\'spliced\':your_spliced_layer_name, \'unspliced\':your_unspliced_layer_name} ')

        # TODO: make nascent and mature also detected automatically


        # Define the expression filter each biophysical model.
        # TODO: check reasonableness of these.
        CMEModel.available_filter_bounds = {"Delay":{'min_means':[0.01, 0.01], 'max_maxes':[350, 350], 'min_maxes':[4,4]},
            "Bursty":{'min_means':[0.01, 0.01], 'max_maxes':[350, 350], 'min_maxes':[4,4]},
            "Extrinsic":{'min_means':[0.01, 0.01], 'max_maxes':[350, 350], 'min_maxes':[4,4]},
            "Constitutive":{'min_means':[0.01, 0.01], 'max_maxes':[350, 350], 'min_maxes':[4,4]},
            "CIR":{'min_means':[0.01, 0.01], 'max_maxes':[350, 350], 'min_maxes':[4,4]},
            "DelayedSplicing":{'min_means':[0.01, 0.01], 'max_maxes':[350, 350], 'min_maxes':[4,4]},
            "ProteinBursty":{'min_means':[0.01, 0.01, 1], 'max_maxes':[350, 350, 1000], 'min_maxes':[4,4,10]}}
        
        try:
            self.filter_bounds = CMEModel.available_filter_bounds[self.bio_model]
            
        except KeyError:
            log.info("Biophysical bounds unknown for model: {}".format(self.bio_model))
            
            
        # Define the parameter bounds used for each biophysical model.
        # TODO: check reasonableness of these bounds.
        CMEModel.available_bio_bounds = {"Delay":{'phys_lb':[-1.0, -1.8, -1.8 ], 'phys_ub':[4.2, 2.5, 3.5]},
            "Bursty":{'phys_lb':[-1.0, -1.8, -1.8 ], 'phys_ub':[4.2, 2.5, 3.5]},
            "Extrinsic":{'phys_lb':[-1.0, -1.8, -1.8 ], 'phys_ub':[4.2, 2.5, 3.5]},
            "Constitutive":{'phys_lb':[-1.8, -1.8 ], 'phys_ub':[2.5, 3.5]},
            "CIR":{'phys_lb':[-1.0, -1.8, -1.8 ], 'phys_ub':[4.2, 2.5, 3.5]},
            "DelayedSplicing":{'phys_lb':[-1.0, -1.8, -1.8 ], 'phys_ub':[4.2, 2.5, 3.5]},
            "ProteinBursty":{'phys_lb':[-1.0, -1.8, -1.8, -1.8, -1.8], 'phys_ub':[4.2, 2.5, 3.5, 3.5, 3.5]}}
        
        try:
            self.bio_bounds = CMEModel.available_bio_bounds[self.bio_model]
            
        except KeyError:
            log.info("Biophysical bounds unknown for model: {}".format(self.bio_model))

        self.amb_model = amb_model
        if seq_model in ["None", None, "Null"]:
            self.seq_model = "None"
        else:
            self.seq_model = seq_model
        self.set_integration_parameters(
            fixed_quad_T, quad_order, quad_vec_T, quad_method
        )
        
        if self.bio_model in ("ProteinBursty", "Custom"):
            self.protein_limit = protein_limit
            self.fit_unspliced = fit_unspliced
            self.min_fudge = min_fudge
            self.max_fudge = max_fudge
            if self.bio_model == "ProteinBursty":
                log.info("Protein grid limit: {}".format(self.protein_limit))
            
        # Define the parameter bounds used for each technical noise model.
        # TODO: check reasonableness of these bounds.
        CMEModel.available_seq_bounds = {"None":{'samp_lb':[1, 1],'samp_ub':[1, 1],'gridsize':[1, 1]}, 
                                         "Bernoulli":{'samp_lb':[-8, -3], 'samp_ub':[-5, 0],'gridsize':[6, 7]}, 
                                         "Poisson":{'samp_lb':[-8, -3], 'samp_ub':[-5, 0],'gridsize':[6, 7]}}
        
        try:
            self.seq_bounds = CMEModel.available_seq_bounds[self.seq_model]
        except KeyError:
            log.info("Technical sequencing bounds unknown for model: {}".format(self.seq_model))
        
        # Get all biological and technical parameters.
        self.param_str = self.get_log_name_str()

    def set_integration_parameters(
        self, fixed_quad_T, quad_order, quad_vec_T, quad_method
    ):
        """Set quadrature parameters.

        Parameters
        ----------
        fixed_quad_T: float
            time horizon used for integration with Gaussian quadature.
        quad_order: int
            Gaussian quadrature order.
        quad_vec_T: float or np.inf
            time horizon used for adaptive integration.
        quad_method: str
            quadrature method to use.
            if 'fixed_quad', use Gaussian quadrature via scipy.integrate.fixed_quad.
            if 'quad_vec', use adaptive integration via scipy.integrate.quad_vec.
        """
        self.fixed_quad_T = fixed_quad_T
        self.quad_order = quad_order
        self.quad_vec_T = quad_vec_T
        self.quad_method = quad_method

    def get_log_name_str(self):
        """Return the names of log-parameters for the model instance.

        Returns
        -------
        name: tuple of str
            TeX-formatted log-parameter names for the current transcriptional model.
            Useful for plot labels.
        """
        param_str = []
        if self.bio_model == "Constitutive":
            param_str += [r"$\log_{10} \beta$", r"$\log_{10} \gamma$"]
        elif self.bio_model == "Delay":
            param_str += [
                r"$\log_{10} b$",
                r"$\log_{10} \beta$",
                r"$\log_{10} \tau^{-1}$",
            ]
        elif self.bio_model == "DelayedSplicing":
            param_str += [
                r"$\log_{10} b$",
                r"$\log_{10} \tau^{-1}$",
                r"$\log_{10} \gamma$",
            ]
        elif self.bio_model == "Bursty":
            param_str += [r"$\log_{10} b$", r"$\log_{10} \beta$", r"$\log_{10} \gamma$"]
        elif self.bio_model == "Extrinsic":
            param_str += [
                r"$\log_{10} \alpha$",
                r"$\log_{10} \beta$",
                r"$\log_{10} \gamma$",
            ]
        elif self.bio_model == "CIR":
            param_str += [r"$\log_{10} b$", r"$\log_{10} \beta$", r"$\log_{10} \gamma$"]

        elif self.bio_model == "ProteinBursty":
            param_str += [r"$\log_{10} b$", r"$\log_{10} \beta$", r"$\log_{10} \gamma$", r"$\log_{10} k_p$", r"$\log_{10} \gamma_p$"]

        elif self.bio_model == "Custom":
            param_str += self.network.get_log_name_str()

        else:
            raise ValueError(
                "Please select a biological noise model from {}.".format(
                    self.available_biomodels
                )
            )
        
        if self.amb_model == "Equal":
            param_str += [r"$\log_{10} p$"]
        elif self.amb_model == "Unequal":
            param_str += [r"$\log_{10} p_N$", r"$\log_{10} p_M$"]
        return param_str

    def get_num_params(self):
        """Return the number of parameters for the model instance.
        Note that these are gene-specific parameters, i.e., global technical noise
        parameters are not used here.

        Returns
        -------
        numpars: int
            The number of gene-specific parameters.
        """
        numpars = 0
        if self.bio_model == "Constitutive":
            numpars += 2
        elif self.bio_model == "ProteinBursty":
            numpars += 5
        elif self.bio_model == "Custom":
            numpars += len(self.network.all_params)
        else:
            numpars += 3
            
        if self.amb_model == "Equal":
            numpars += 1
        elif self.amb_model == "Unequal":
            numpars += 2
        return numpars

    def eval_model_logL(self, p, limits, samp, data, n_cells, EPS=1e-15):
        """Compute the log-likelihood of data under a set of parameters.

        Parameters
        ----------
        p: np.ndarray
            log10 biological parameters.
        limits: list of int
            grid size for PMF evaluation, size n_species.
        samp: None or np.ndarray
            sampling parameters, if applicable.
        data: tuple or np.ndarray
            experimental data histogram.
        EPS: float, optional
            minimum allowed proposal probability mass. anything below this value is rounded up to EPS.

        Returns
        -------
        logL: float
            log-likelihood.
        """
        _LOGL_RUST_MODELS = {
            "Constitutive", "Bursty", "CIR",
            "Extrinsic", "Delay", "DelayedSplicing",
        }
        if (
            _HAS_RUST
            and self.bio_model in _LOGL_RUST_MODELS
            and self.amb_model == "None"
            and self.quad_method == "fixed_quad"
            and (samp is None or self.seq_model in ("Poisson", "Bernoulli"))
        ):
            x, f = data
            x_np = np.asarray(x, dtype=np.int64)
            return _mc.eval_logl_2d(
                self.bio_model,
                np.asarray(p, dtype=float).tolist(),
                [int(v) for v in limits],
                x_np[:, 0].tolist(),
                x_np[:, 1].tolist(),
                np.asarray(f, dtype=float).tolist(),
                float(n_cells),
                float(self.fixed_quad_T),
                int(self.quad_order),
                samp.tolist() if samp is not None else None,
                float(EPS),
                self.seq_model,
            )

        x, f = data
        proposal = self.eval_model_pss(p, limits, samp)
        proposal[proposal < EPS] = EPS
        proposal = proposal[tuple(x.T)]
        return np.sum(np.log(proposal) * f * n_cells)

    def eval_model_kld(self, p, limits, samp, data, EPS=1e-15):
        """Compute the Kullback-Leibler divergence between data and a fit.

        Parameters
        ----------
        p: np.ndarray
            log10 biological parameters.
        limits: list of int
            grid size for PMF evaluation, size n_species.
        samp: None or np.ndarray
            sampling parameters, if applicable.
        data: tuple or np.ndarray
            experimental data histogram.
        EPS: float, optional
            minimum allowed proposal probability mass. anything below this value is rounded up to EPS.

        Returns
        -------
        kld: float
            Kullback-Leibler divergence.
        """
        _KLD_RUST_MODELS = {
            "Constitutive", "Bursty", "CIR",
            "Extrinsic", "Delay", "DelayedSplicing",
        }
        # Rust fast-path: compute PSS and KLD in one call, avoiding the
        # Python round-trip for histogram indexing and log/sum operations.
        if (
            _HAS_RUST
            and self.bio_model in _KLD_RUST_MODELS
            and self.amb_model == "None"
            and self.quad_method == "fixed_quad"
            and (samp is None or self.seq_model in ("Poisson", "Bernoulli"))
        ):
            x, f = data
            x_np = np.asarray(x, dtype=np.int64)
            return _mc.eval_kld_2d(
                self.bio_model,
                np.asarray(p, dtype=float).tolist(),
                [int(v) for v in limits],
                x_np[:, 0].tolist(),
                x_np[:, 1].tolist(),
                np.asarray(f, dtype=float).tolist(),
                float(self.fixed_quad_T),
                int(self.quad_order),
                samp.tolist() if samp is not None else None,
                float(EPS),
                self.seq_model,
            )

        x, f = data
        proposal = self.eval_model_pss(p, limits, samp)
        proposal[proposal < EPS] = EPS
        proposal = proposal[tuple(x.T)]
        d = f * np.log(f / proposal)

        if log.isEnabledFor(logging.DEBUG):
            log.debug('The KL divergence with parameter %s is %.10f', np.array2string(10**p), np.sum(d))

        return np.sum(d)

    def eval_model_kld_and_grad(self, p, limits, samp, data,
                                EPS=1e-15, eps=1e-6, n_jobs=1):
        """Compute KLD and its gradient w.r.t. log10 parameters.

        For Constitutive, Extrinsic, Delay, and DelayedSplicing (Python path only),
        uses an analytical gradient derived through the IFFT via the chain rule —
        one shared PGF evaluation plus n_params IFFTs, no extra quadrature.

        For Bursty, CIR, and ProteinBursty, uses forward finite differences,
        optionally parallelised with joblib.

        Parameters
        ----------
        p: np.ndarray
            log10 biological parameters.
        limits: list of int
            grid size for PMF evaluation.
        samp: None or np.ndarray
            sampling parameters.
        data: tuple or np.ndarray
            experimental histogram.
        EPS: float
            minimum allowed probability mass.
        eps: float
            finite-difference step size (used only for FD fallback).
        n_jobs: int
            number of parallel joblib workers for FD gradient evaluation.
            1 = serial (default); -1 = all available cores.

        Returns
        -------
        kld: float
        grad: np.ndarray, shape (n_params,)
        """
        from joblib import Parallel, delayed

        p = np.asarray(p, dtype=float)

        # Analytical gradient path.
        # Closed-form models (Constitutive, Extrinsic, Delay, DelayedSplicing) use
        # pure algebra — fast and exact, always preferred over FD.
        # Quadrature-based models (Bursty, CIR) require a quadrature pass in Python
        # which is slower than the Rust FD path; use them only when Rust is unavailable.
        _FAST_ANALYTIC_MODELS = {"Constitutive", "Extrinsic",
                                  "Delay", "DelayedSplicing"}
        _QUAD_ANALYTIC_MODELS = {"Bursty", "CIR"}
        _use_analytic = (
            self.bio_model in _FAST_ANALYTIC_MODELS
            or (self.bio_model in _QUAD_ANALYTIC_MODELS and not _HAS_RUST)
        )
        if (_use_analytic
                and self.amb_model == "None"):
            result = self._eval_kld_analytic_grad(p, limits, samp, data, EPS)
            if result is not None:
                return result

        # Rust FD gradient path: evaluates base + all perturbed PSS in parallel
        # (rayon, GIL released), eliminating the Python round-trip per perturbation.
        _RUST_FD_MODELS = {
            "Constitutive", "Bursty", "CIR",
            "Extrinsic", "Delay", "DelayedSplicing",
        }
        if (
            _HAS_RUST
            and self.bio_model in _RUST_FD_MODELS
            and self.amb_model == "None"
            and self.quad_method == "fixed_quad"
            and (samp is None or self.seq_model == "Poisson")
        ):
            x, f_data = data
            x_np = np.asarray(x, dtype=np.int64)
            kld0, grad_list = _mc.eval_kld_grad_2d(
                self.bio_model,
                p.tolist(),
                [int(v) for v in limits],
                x_np[:, 0].tolist(),
                x_np[:, 1].tolist(),
                np.asarray(f_data, dtype=float).tolist(),
                float(self.fixed_quad_T),
                int(self.quad_order),
                eps,
                samp.tolist() if samp is not None else None,
                float(EPS),
            )
            return kld0, np.array(grad_list)

        # Python fallback: forward finite differences.
        kld0 = self.eval_model_kld(p, limits, samp, data, EPS)

        def _perturbed(i):
            p_eps = p.copy()
            p_eps[i] += eps
            return self.eval_model_kld(p_eps, limits, samp, data, EPS)

        n_params = len(p)
        if n_jobs == 1:
            klds_eps = [_perturbed(i) for i in range(n_params)]
        else:
            klds_eps = Parallel(n_jobs=min(n_params, n_jobs) if n_jobs > 0 else n_jobs)(
                delayed(_perturbed)(i) for i in range(n_params)
            )
        grad = (np.array(klds_eps) - kld0) / eps
        return kld0, grad

    def _eval_kld_analytic_grad(self, p, limits, samp, data, EPS):
        """Analytical KLD gradient via chain rule through log-PGF → IFFT.

        Computes d KLD / d log10(θ_i) = -(1/N) * dot(f/pss, dR_i) + dN_i/N
        where dR_i = irfftn(exp(phi) * d phi / d log10(θ_i)).

        Returns (kld, grad) or None to signal fallback to finite differences.
        """
        LN10 = np.log(10.0)
        p_lin = 10.0 ** p

        # Build mesh (reads from cache; no copy needed since g is read-only here).
        mx = np.array(limits, dtype=int)
        mx[-1] = mx[-1] // 2 + 1
        g = _build_mesh(tuple(limits), tuple(mx))

        # Apply technical noise transform to the mesh.
        if samp is not None and self.seq_model == "Poisson":
            g = np.exp(np.power(10.0, samp)[:, None] * g) - 1
        elif samp is not None and self.seq_model == "Bernoulli":
            g = g * np.asarray(samp)[:, None]
        elif self.seq_model not in ("None", "Poisson", "Bernoulli"):
            return None  # unsupported seq_model

        g0, g1 = g[0], g[1]

        # Compute log-PGF (phi) and its gradient w.r.t. each log10 parameter.
        if self.bio_model == "Constitutive":
            beta, gamma = p_lin
            phi = g0 / beta + g1 / gamma
            dphi = [
                -g0 / beta * LN10,   # d phi / d log10(beta)
                -g1 / gamma * LN10,  # d phi / d log10(gamma)
            ]
        elif self.bio_model == "Extrinsic":
            alpha, beta, gamma = p_lin
            D = 1.0 - g0 / beta - g1 / gamma
            phi = -alpha * np.log(D)
            dphi = [
                phi * LN10,                          # d phi / d log10(alpha)
                -alpha * g0 / (beta * D) * LN10,    # d phi / d log10(beta)
                -alpha * g1 / (gamma * D) * LN10,   # d phi / d log10(gamma)
            ]
        elif self.bio_model == "DelayedSplicing":
            b, tauinv, gamma = p_lin
            tau = 1.0 / tauinv
            D0 = 1.0 - b * g0
            D1 = 1.0 - b * g1
            phi = b * g0 * tau / D0 - np.log(D1) / gamma
            dphi = [
                (g0 * tau / D0 ** 2 + g1 / (gamma * D1)) * b * LN10,
                -b * g0 * tau / D0 * LN10,
                np.log(D1) / gamma * LN10,
            ]
        elif self.bio_model == "Delay":
            b, beta, tauinv = p_lin
            tau = 1.0 / tauinv
            E = np.exp(-beta * tau)
            U = g1 + (g0 - g1) * E
            A = 1.0 - b * U
            B = 1.0 - b * g0
            C = 1.0 - b * g1
            phi = -np.log(A) / beta + np.log(A / B) / (beta * C) + tau * b * g1 / C
            logAB = np.log(A / B)
            # dA/dbeta via dU/dbeta = (g0-g1)*(-tau)*E
            dA_dbeta = b * tau * (g0 - g1) * E
            # dA/dtau  via dU/dtau  = (g0-g1)*(-beta)*E
            dA_dtau = b * beta * (g0 - g1) * E
            # d phi / d b  (A, B, C all depend on b; U does not)
            dphi_db = (
                U / (A * beta)
                - U / (A * beta * C)
                + g0 / (B * beta * C)
                + logAB * g1 / (beta * C ** 2)
                + tau * g1 / C
                + tau * b * g1 ** 2 / C ** 2
            ) * b * LN10
            # d phi / d beta  (B, C do not depend on beta)
            dphi_dbeta = (
                np.log(A) / beta ** 2
                - dA_dbeta / (A * beta)
                + dA_dbeta / (A * beta * C)
                - logAB / (beta ** 2 * C)
            ) * beta * LN10
            # d phi / d tauinv:
            #   d phi / d tau  = b*g1/C * (dA_dtau/(A*beta) + 1)
            #   d phi / d tauinv = d phi / d tau * (-tau^2)
            #   dphi_dtauinv   = d phi / d tauinv * tauinv * LN10
            #                  = -b*g1*tau/C * (dA_dtau/(A*beta) + 1) * LN10
            dphi_dtauinv = (
                -b * g1 * tau / C * (dA_dtau / (A * beta) + 1.0)
            ) * LN10
            dphi = [dphi_db, dphi_dbeta, dphi_dtauinv]
        elif self.bio_model in ("Bursty", "CIR"):
            # Semi-analytical gradient via the same Gauss-Legendre quadrature used
            # for the PSS.  phi = integral of integrand(U) dt over [0,T] where
            #   Bursty:  integrand = U / (1 - U)
            #   CIR:     integrand = (1/2) * (1 - sqrt(1 - 4*U))
            # and U(t) = b * (c1*exp(-beta*t) + c2*exp(-gamma*t)).
            # All three d phi / d log10(param) integrals are accumulated in the
            # same quadrature loop — no extra function evaluations.
            if self.quad_method != "fixed_quad":
                return None  # quad_vec integration: fall back to FD
            from numpy.polynomial.legendre import leggauss
            b, beta, gamma = p_lin
            T = self.fixed_quad_T * (1.0 / beta + 1.0 / gamma + 1.0)
            t_half = T / 2.0
            xi, wi = leggauss(self.quad_order)
            x = t_half + t_half * xi   # (Q,) quadrature points in [0, T]
            w_T = t_half * wi           # (Q,) weights (Jacobian of [0,T] transform)

            # Frequency-mesh arrays g0, g1 have shape (N_freq,); broadcast to
            # (N_freq, 1) so they multiply cleanly against (Q,) arrays.
            g0e = g0[:, None]          # (N_freq, 1)
            g1e = g1[:, None]          # (N_freq, 1)
            eb = np.exp(-beta  * x)    # (Q,)
            eg = np.exp(-gamma * x)    # (Q,)

            if np.isclose(beta, gamma):
                # Limiting form as gamma -> beta to avoid cancellation.
                # U = b * (g0 * exp(-beta*t) + g1 * beta * t * exp(-gamma*t))
                U = b * (g0e * eb + g1e * (beta * x * eg))  # (N_freq, Q)
                # dU/dbeta: differentiate b*(g0*exp(-beta*t) + g1*beta*t*exp(-gamma*t)) w.r.t. beta
                #   d/dbeta[g0*exp(-beta*t)] = -g0*t*exp(-beta*t)
                #   d/dbeta[g1*beta*t*exp(-gamma*t)] = g1*t*exp(-gamma*t)  (gamma fixed)
                dU_dbeta  = b * (-g0e * x * eb + g1e * x * eg)
                # dU/dgamma = b * g1 * beta * t * (-t) * eg = -b*g1*beta*t^2*eg
                dU_dgamma = -b * g1e * beta * x**2 * eg
            else:
                f  = beta / (beta - gamma)
                c2 = g1e * f            # (N_freq, 1)
                c1 = g0e - c2           # (N_freq, 1)
                U  = b * (c1 * eb + c2 * eg)  # (N_freq, Q)
                # ∂c2/∂beta = g1 * (-gamma/(beta-gamma)^2)
                # ∂c1/∂beta = -∂c2/∂beta = g1 * gamma/(beta-gamma)^2
                bg2 = b * g1e * gamma / (beta - gamma)**2
                dU_dbeta  = bg2 * (eb - eg) - b * c1 * x * eb
                # ∂c2/∂gamma = g1 * beta/(beta-gamma)^2
                # ∂c1/∂gamma = -∂c2/∂gamma
                bb2 = b * g1e * beta / (beta - gamma)**2
                dU_dgamma = bb2 * (eg - eb) - b * c2 * x * eg

            if self.bio_model == "Bursty":
                # phi = integral U/(1-U) dt;  d(integrand)/dU = 1/(1-U)^2
                inv1mU    = 1.0 / (1.0 - U)            # (N_freq, Q)
                phi       = (U * inv1mU     * w_T).sum(axis=1)  # (N_freq,)
                d_intg_dU = inv1mU**2                   # (N_freq, Q)
            else:  # CIR
                # phi = (1/2) * integral (1 - sqrt(1-4U)) dt;
                # d(integrand)/dU = 2/sqrt(1-4U)
                sqrt_term = np.sqrt(1.0 - 4.0 * U)     # (N_freq, Q)
                phi       = 0.5 * ((1.0 - sqrt_term) * w_T).sum(axis=1)
                d_intg_dU = 2.0 / sqrt_term             # (N_freq, Q)  [× 1/2 from phi]
                d_intg_dU = d_intg_dU * 0.5             # absorb the 1/2 prefactor

            # d phi / d log10(param) = param_lin * LN10 * integral d(integrand)/dU * dU/d(param_lin) dt
            # For b: dU/db = U/b, so param_lin * dU/d(param_lin) = b * (U/b) = U.
            dphi_db     = LN10         * (d_intg_dU * U         * w_T).sum(axis=1)
            dphi_dbeta  = beta  * LN10 * (d_intg_dU * dU_dbeta  * w_T).sum(axis=1)
            dphi_dgamma = gamma * LN10 * (d_intg_dU * dU_dgamma * w_T).sum(axis=1)
            dphi = [dphi_db, dphi_dbeta, dphi_dgamma]
        else:
            return None  # unsupported model

        # Shared forward pass: G = exp(phi), PSS via irfftn.
        G = np.exp(phi)
        shape_mx = tuple(mx)
        pss_unnorm = irfftn(G.reshape(shape_mx), s=tuple(limits))
        pss_unnorm_flat = pss_unnorm.flatten()
        norm = float(np.sum(np.abs(pss_unnorm_flat)))
        pss_flat = np.abs(pss_unnorm_flat) / norm

        # KLD value.
        coords, freqs = data
        proposal = pss_unnorm.reshape(limits)[tuple(coords.T)] / norm
        proposal_clipped = np.clip(proposal.real, EPS, None)
        kld = float(np.sum(freqs * np.log(freqs / proposal_clipped)))

        # Analytical gradient: d KLD / d θ_i = -(1/N)*dot(f/pss, dR_i) + dN_i/N
        grad = np.empty(len(dphi))
        for i, dphi_i in enumerate(dphi):
            dG_i = G * dphi_i
            dR_i = irfftn(dG_i.reshape(shape_mx), s=tuple(limits)).flatten().real
            dN_i = float(np.sum(dR_i))
            dR_at_data = dR_i.reshape(limits)[tuple(coords.T)]
            grad[i] = float(
                -(1.0 / norm) * np.sum(freqs / proposal_clipped * dR_at_data)
                + dN_i / norm
            )
        return kld, grad

    def eval_model_pss(self, p, limits, samp=None):
        """Evaluate the PMF of the model over a grid at a set of parameters.

        Parameters
        ----------
        p: np.ndarray
            log10 biological parameters.
        limits: list of int
            grid size for PMF evaluation, size n_species.
        samp: None or np.ndarray, optional
            sampling parameters, if applicable.

        Returns
        -------
        Pss: np.ndarray
            the steady-state model PMF over a grid.
        """

        # This was formerly the interface with nn_toolbox neural likelihood approximation methods.
        # It will be implemented in a future version.

        # if (
        #     (self.quad_method == "nn")
        #     and (self.bio_model == "Bursty")
        #     and (self.seq_model == "None")
        # ):
        #     return basic_ml_bivariate(p, limits)
        # else:

        # Fast path: GPU (PyTorch) if available, then Rust (quadrature models only), then Python.
        # Analytical models (Constitutive, Extrinsic, Delay, DelayedSplicing) are faster in
        # Python because their PGF is a single numpy broadcast and scipy/FFTW beats RustFFT.
        _FAST_MODELS_2D = {
            "Constitutive", "Bursty", "CIR",
            "Extrinsic", "Delay", "DelayedSplicing",
        }
        _RUST_MODELS_2D = {"Bursty", "CIR"}
        _fast_cond = (
            self.bio_model in _FAST_MODELS_2D
            and self.seq_model == "None"
            and self.amb_model == "None"
            and self.quad_method == "fixed_quad"
            and samp is None
        )
        if _HAS_GPU and _fast_cond:
            pss = _eval_model_pss_gpu(
                self.bio_model,
                p.tolist(),
                limits,
                float(self.fixed_quad_T),
                int(self.quad_order),
                _GPU_DEVICE,
            )
            return pss.reshape(int(limits[0]), int(limits[1])).squeeze()

        if (
            _HAS_RUST
            and _fast_cond
            and self.bio_model in _RUST_MODELS_2D
        ):
            pss_flat = _mc.eval_model_pss_2d(
                self.bio_model,
                p.tolist(),
                [int(x) for x in limits],
                float(self.fixed_quad_T),
                int(self.quad_order),
            )
            pss = np.array(pss_flat).reshape(int(limits[0]), int(limits[1]))
            return pss.squeeze()

        # Rust fast-path for Poisson or Bernoulli technical noise (all six 2-D models).
        if (
            _HAS_RUST
            and self.bio_model in _FAST_MODELS_2D
            and self.seq_model in ("Poisson", "Bernoulli")
            and self.amb_model == "None"
            and self.quad_method == "fixed_quad"
            and samp is not None
        ):
            pss_flat = _mc.eval_model_pss_2d(
                self.bio_model,
                p.tolist(),
                [int(x) for x in limits],
                float(self.fixed_quad_T),
                int(self.quad_order),
                samp.tolist(),
                seq_model=self.seq_model,
            )
            pss = np.array(pss_flat).reshape(int(limits[0]), int(limits[1]))
            return pss.squeeze()

        # Rust fast-path for ambient models (Equal/Unequal) + seq_model="None".
        # All 6 2-D bio_models are supported; p_log must have the amb params stripped.
        _RUST_AMB_MODELS = {
            "Constitutive", "Bursty", "CIR",
            "Extrinsic", "Delay", "DelayedSplicing",
        }
        if (
            _HAS_RUST
            and self.bio_model in _RUST_AMB_MODELS
            and self.seq_model == "None"
            and self.amb_model in ("Equal", "Unequal")
            and self.quad_method == "fixed_quad"
            and samp is None
        ):
            n_amb = 1 if self.amb_model == "Equal" else 2
            p_bio = p[:-n_amb]
            amb_params = p[-n_amb:].tolist()
            pss_flat = _mc.eval_model_pss_2d(
                self.bio_model,
                p_bio.tolist(),
                [int(x) for x in limits],
                float(self.fixed_quad_T),
                int(self.quad_order),
                None,
                self.amb_model,
                amb_params,
            )
            pss = np.array(pss_flat).reshape([int(l) for l in limits])
            return pss.squeeze()

        # Rust fast-path for ProteinBursty (seq_model="None", amb_model="None").
        if (
            _HAS_RUST
            and self.bio_model == "ProteinBursty"
            and self.seq_model == "None"
            and self.amb_model == "None"
            and samp is None
        ):
            pss_flat = _mc.eval_model_pss_protein_bursty(
                p.tolist(),
                [int(x) for x in limits],
                bool(self.fit_unspliced),
                float(self.protein_limit),
                float(self.min_fudge),
                float(self.max_fudge),
            )
            pss = np.array(pss_flat).reshape(
                int(limits[0]), int(limits[1]), int(limits[2])
            )
            return pss.squeeze()

        # Rust fast-path for Custom networks (non-delayed, seq_model="None",
        # amb_model="None").  Parallelises RK4 over grid points via rayon.
        if (
            _HAS_RUST
            and self.bio_model == "Custom"
            and not self.network._has_delays
            and self.seq_model == "None"
            and self.amb_model == "None"
            and samp is None
        ):
            p_lin = np.power(10.0, p)
            dt      = float(np.min(1.0 / p_lin) * self.min_fudge)
            t_max   = float(np.max(1.0 / p_lin) * self.max_fudge)
            n_steps = int(np.ceil(t_max / dt))
            max_while = 10 * n_steps + 10_000

            # Serialise network topology to flat arrays for Rust.
            net       = self.network
            params_d  = net._param_map(p)          # name → linear float
            if net._norm_rate is not None:
                params_d[net._norm_rate] = 1.0
            all_names = list(net.all_params) + ([net._norm_rate] if net._norm_rate else [])
            params_linear = [float(params_d[name]) for name in all_names]
            name_to_idx   = {name: i for i, name in enumerate(all_names)}
            sp_idx_map    = {s: i for i, s in enumerate(net.species)}

            rxn_kinds, rxn_rate_idxs = [], []
            rxn_extra1, rxn_extra2   = [], []
            prod_sp_flat, prod_st_flat, prod_off = [], [], [0]

            for rxn in net.reactions:
                rate_idx = name_to_idx[rxn.rate_name]
                if not rxn.reactants and rxn.burst_param is not None:
                    rxn_kinds.append(0)
                    rxn_rate_idxs.append(rate_idx)
                    rxn_extra1.append(name_to_idx[rxn.burst_param])
                    rxn_extra2.append(sp_idx_map[rxn.burst_species])
                elif not rxn.reactants:
                    rxn_kinds.append(1)
                    rxn_rate_idxs.append(rate_idx)
                    rxn_extra1.append(-1)
                    rxn_extra2.append(-1)
                else:
                    src = next(iter(rxn.reactants))
                    rxn_kinds.append(2)
                    rxn_rate_idxs.append(rate_idx)
                    rxn_extra1.append(sp_idx_map[src])
                    rxn_extra2.append(-1)
                for sp, st in rxn.products.items():
                    prod_sp_flat.append(sp_idx_map[sp])
                    prod_st_flat.append(st)
                prod_off.append(len(prod_sp_flat))

            re_e, im_e, mx_shape = _mc.eval_custom_network_pgf(
                len(net.species),
                [int(x) for x in limits],
                rxn_kinds,
                rxn_rate_idxs,
                rxn_extra1,
                rxn_extra2,
                prod_sp_flat,
                prod_st_flat,
                prod_off,
                params_linear,
                dt,
                n_steps,
                max_while,
            )
            exp_phi = (np.array(re_e) + 1j * np.array(im_e)).reshape(mx_shape)
            pss = np.abs(irfftn(exp_phi, s=[int(x) for x in limits]))
            pss = pss / pss.sum()
            return pss.squeeze()

        if (self.amb_model != "None") and (len(limits) == 2):
            raise ValueError("Please specify a limit for the ambiguous species.")

        mx = np.copy(limits)

        ### if protein model, then decrease the grids of pgf
        if self.bio_model == "ProteinBursty":
            scale = mx[-1]//self.protein_limit + 1
            mx[-1] = (mx[-1]+scale-1)//scale

            if not self.fit_unspliced:
                mx[0]=1


        mx[-1] = mx[-1] // 2 + 1
        g = _build_mesh(tuple(limits), tuple(mx))

        
        if self.amb_model == "Unequal":
            g_ = np.zeros((2, g.shape[1]), dtype=np.complex128)
            p_amb = np.power(10, p[-2:])
            g_[0] = p_amb[0] * g[2] + (1 - p_amb[0]) * g[0]
            g_[1] = p_amb[1] * g[2] + (1 - p_amb[1]) * g[1]
            g = g_
            p = np.copy(p[:-2])  # better safe
        elif self.amb_model == "Equal":
            g_ = np.zeros((2, g.shape[1]), dtype=np.complex128)
            p_amb = np.power(10, p[-1])
            g_[0] = p_amb * g[2] + (1 - p_amb) * g[0]
            g_[1] = p_amb * g[2] + (1 - p_amb) * g[1]
            g = g_
            p = np.copy(p[:-1])

        # For now add zero for protein sampling parameter.
        if samp is not None:
            num_excess = np.shape(g)[0] - len(samp)
            samp_use = np.pad(samp, (0, num_excess))
        
        if self.seq_model == "Poisson":
            g = np.exp((np.power(10, samp))[:, None] * g) - 1
        elif self.seq_model == "Bernoulli":
            g = g * np.asarray(samp)[:, None]
        elif self.seq_model == "None":
            pass
        else:
            raise ValueError(
                "Please select a technical noise model from {}.".format(
                    self.available_seqmodels
                )
            )
            
        gf = self.eval_model_pgf(p, g) # this gf is actually phi
        gf = np.exp(gf)
        gf = gf.reshape(tuple(mx))
        Pss = irfftn(gf, s=tuple(limits))
        Pss = np.abs(Pss) / np.sum(np.abs(Pss))
        Pss = Pss.squeeze()
        return Pss

    def eval_model_pss_batch(self, params_list, limits_list, samp_list=None, num_threads=None):
        """Evaluate PSS for N genes in a single batched Rust call (rayon-parallel).

        All N evaluations run in parallel on the Rust thread pool (rayon), with
        the GIL released for the entire computation.  Falls back to sequential
        `eval_model_pss` calls when the Rust batch path is not available.

        Supported for all six 2-D bio_models with seq_model in {"None","Poisson"}
        and amb_model="None", quad_method="fixed_quad".

        Parameters
        ----------
        params_list : list of np.ndarray, length N
            log10 biological parameters per gene.
        limits_list : list of array-like, length N
            grid dimensions per gene.
        samp_list : list of (np.ndarray or None), length N, optional
            Poisson sampling parameters per gene; pass None per entry (or
            omit the argument) when seq_model=="None".
        num_threads : int or None, optional
            Number of rayon threads to use. None (default) uses all available
            cores. Ignored when falling back to the sequential Python path.

        Returns
        -------
        list of np.ndarray, length N
            PSS reshaped to limits_list[i] for each gene.
        """
        _RUST_BATCH_MODELS = {
            "Constitutive", "Bursty", "CIR",
            "Extrinsic", "Delay", "DelayedSplicing",
        }
        n = len(params_list)
        if samp_list is None:
            samp_list = [None] * n

        _batch_ok = (
            _HAS_RUST
            and self.bio_model in _RUST_BATCH_MODELS
            and self.amb_model == "None"
            and self.quad_method == "fixed_quad"
            and self.seq_model in ("None", "Poisson", "Bernoulli")
        )
        if not _batch_ok:
            return [
                self.eval_model_pss(p, lim, s)
                for p, lim, s in zip(params_list, limits_list, samp_list)
            ]

        p_py   = [p.tolist() for p in params_list]
        lim_py = [[int(x) for x in lim] for lim in limits_list]

        if self.seq_model == "None":
            samp_py = None
        else:
            samp_py = [
                (s.tolist() if s is not None else None) for s in samp_list
            ]

        results_flat = _mc.eval_model_pss_2d_batch(
            self.bio_model,
            p_py,
            lim_py,
            float(self.fixed_quad_T),
            int(self.quad_order),
            samp_py,
            num_threads,
            self.seq_model,
        )
        return [
            np.array(flat).reshape(int(lim[0]), int(lim[1])).squeeze()
            for flat, lim in zip(results_flat, lim_py)
        ]

    def eval_model_pgf(self, p_, g):
        """Evaluate the log-PGF of the model over the complex unit sphere at a set of parameters.

        Parameters
        ----------
        p_: np.ndarray
            log10 biological parameters.
        g: np.ndarray
            complex PGF arguments, adjusted for sampling.

        Returns
        -------
        gf: np.ndarray
            generating function values at each argument.
        """
        p = 10**p_
        if self.bio_model == "Constitutive":  # constitutive production
            beta, gamma = p
            gf = g[0] / beta + g[1] / gamma
        elif self.bio_model == "Bursty":  # bursty production
            b, beta, gamma = p
            fun = lambda x: self.burst_intfun(x, g, b, beta, gamma)
            if self.quad_method == "quad_vec":
                T = self.quad_vec_T * (1 / beta + 1 / gamma + 1)
                gf = scipy.integrate.quad_vec(fun, 0, T)[0]
            elif self.quad_method == "fixed_quad":
                T = self.fixed_quad_T * (1 / beta + 1 / gamma + 1)
                gf = scipy.integrate.fixed_quad(fun, 0, T, n=self.quad_order)[0]
            else:
                raise ValueError("Please use one of the specified quadrature methods.")
        elif (
            self.bio_model == "Extrinsic"
        ):  # constitutive production with extrinsic noise
            alpha, beta, gamma = p
            gf = -alpha * np.log(1 - g[0] / beta - g[1] / gamma)
        elif self.bio_model == "Delay":  # bursty production with delayed degradation
            b, beta, tauinv = p
            tau = 1 / tauinv
            U = g[1] + (g[0] - g[1]) * np.exp(-beta * tau)
            gf = (
                -1 / beta * np.log(1 - b * U)
                + 1 / beta / (1 - b * g[1]) * np.log((b * U - 1) / (b * g[0] - 1))
                + tau * b * g[1] / (1 - b * g[1])
            )
        elif self.bio_model == "DelayedSplicing":
            b, tauinv, gamma = p
            tau = 1 / tauinv
            gf = tau * b * g[0] / (1 - b * g[0]) - 1 / gamma * np.log(1 - b * g[1])
            
        elif self.bio_model == "CIR":  # CIR-like:
            b, beta, gamma = p
            fun = lambda x: self.cir_intfun(x, g, b, beta, gamma)
            if self.quad_method == "quad_vec":
                T = self.quad_vec_T * (1 / beta + 1 / gamma + 1)
                gf = scipy.integrate.quad_vec(fun, 0, T)[0]
            elif self.quad_method == "fixed_quad":
                T = self.fixed_quad_T * (1 / beta + 1 / gamma + 1)
                gf = scipy.integrate.fixed_quad(fun, 0, T, n=self.quad_order)[0]
            else:
                raise ValueError("Please use one of the specified quadrature methods.")
            gf /= 2

        elif self.bio_model == "ProteinBursty":  # bursty production
            gf = self.protein_pgf(g, p)

        elif self.bio_model == "Custom":
            dt    = np.min(1.0 / p) * self.min_fudge
            t_max = np.max(1.0 / p) * self.max_fudge
            n_steps = int(np.ceil(t_max / dt))
            gf = self.network.eval_pgf(
                p_,
                [g[i] for i in range(len(self.network.species))],
                t_max,
                n_steps,
            )

        else:
            raise ValueError(
                "Please select a biological noise model from {}.".format(
                    self.available_biomodels
                )
            )
        return gf  # this is the log-generating function
    
    def protein_pgf(self, g, p):
        """Evaluates the log generating function.

        It is a helper function for the protein biological model.

        Parameters
        ----------
        g: np.ndarray
            complex PGF arguments, adjusted for sampling.
        p: float array of length 5
            (b, beta, gamma, k_p, gamma_p)

        Returns
        -------
        phi: np.ndarray
            log gf.
        """
        epsilon = 1e-10 
        
        def u_tilda_ode(u, du, beta, gamma, k_p, gamma_p):
            """
            Solve the characteristics ODE
            """
            # u (n_species, n_grids)
            du = np.zeros_like(u)
            du[0] = beta * (u[1]-u[0]) # Unspliced
            du[1] = - gamma * u[1] + k_p * u[2] * (u[1]+1) # Spliced
            du[2] = - gamma_p * u[2] # Proteins
            return du
        
        def RK2(x, dx, f, dt, beta, gamma, k_p, gamma_p):
            """
            2nd Order Runge-Kutta integration for updating x.
        
            Parameters:
            - x: Current state of the system
            - dx: Current derivative
            - f: Function defining the differential equation
            - dt: Time step
            - beta, gamma, k_p, gamma_p: Parameters for the ODE
        
            Returns:
            - x_new: Updated state after one step
            """
            # Calculate k1: the slope at the current position
            k1 = f(x, dx, beta, gamma, k_p, gamma_p)
        
            # Calculate k2: the slope at the midpoint
            k2 = f(x + (dt / 2) * k1, dx, beta, gamma, k_p, gamma_p)
        
            # Update x using the second-order approximation
            x_new = x + dt * k2
        
            return x_new
    
        def RK4(x, dx, f, dt, beta, gamma, k_p, gamma_p):
            
            j1 = f(x, dx, beta, gamma, k_p, gamma_p)
            j2 = f(x + (dt / 2) * j1, dx, beta, gamma, k_p, gamma_p)
            j3 = f(x + (dt / 2) * j2, dx, beta, gamma, k_p, gamma_p)
            j4 = f(x + dt * j3, dx, beta, gamma, k_p, gamma_p)
            
            x_new = x + (dt / 6) * (j1 + 2 * j2 + 2 * j3 + j4)
            return x_new
            
        b, beta, gamma, k_p, gamma_p = p
        
        dt = np.min(1 / np.array(p)) * self.min_fudge
        t_max = np.max(1 / np.array(p)) * self.max_fudge
        num_tsteps = int(np.ceil(t_max / dt))
        #log.debug('dt: %s, t_max: %s', np.array2string(dt), np.array2string(t_max))
        
        t = 0
        u_tilde = np.array(g, dtype=np.complex64)
        du_tilde = np.array(g, dtype=np.complex64)
        
        # Use numexpr for fast computation
        phi = b * u_tilde[0] / (1 - b * u_tilde[0]) * dt / 2
    
        # Solve ODE using RK4 method 
        for step in range(num_tsteps):
            t += dt
            u_tilde = RK4(u_tilde, du_tilde, u_tilda_ode, dt, beta, gamma, k_p, gamma_p)
            phi += b * u_tilde[0] / (1 - b * u_tilde[0]) * dt
        
        while np.max(np.abs(u_tilde[0]))>1e-3:
            t += dt
            u_tilde = RK4(u_tilde, du_tilde, u_tilda_ode, dt, beta, gamma, k_p, gamma_p)
            phi += b * u_tilde[0] / (1 - b * u_tilde[0]) * dt
            
        #log.debug('t: %s', np.array2string(t))
        u_tilde = RK4(u_tilde, du_tilde, u_tilda_ode,dt, beta, gamma, k_p, gamma_p)
        phi += b * u_tilde[0] / (1 - b * u_tilde[0]) * dt / 2
        
        return phi

    def cir_intfun(self, x, g, b, beta, gamma):
        """Evaluates the inverse Gaussian-driven CME process integrand at time x.

        This solution was reported by Gorin*, Vastola*, Fang, and Pachter (2021).
        It is a helper function for the CIR-like biological model.

        Parameters
        ----------
        x: float np.ndarray or float
            time or array of times to evaluate the integrand at.
        g: np.ndarray
            complex PGF arguments, adjusted for sampling.
        b: float
            burst size-like parameter.
        beta: float
            splicing rate.
        gamma: float
            degradation rate.

        Returns
        -------
        _: np.ndarray
            integrand value.
        """
        g = np.asarray(g)[:, :, None]
        if np.isclose(beta, gamma):  # compute prefactors for the ODE characteristics.
            c_1 = g[0]  # nascent
            c_2 = x * beta * g[1]
        else:
            f = beta / (beta - gamma)
            c_2 = g[1] * f
            c_1 = g[0] - c_2

        U = b * (np.exp(-beta * x) * c_1 + np.exp(-gamma * x) * c_2)
        return 1 - np.sqrt(1 - 4 * U)

    def burst_intfun(self, x, g, b, beta, gamma):
        """Evaluates the bursty CME process integrand at time x.

        This solution was reported by Singh and Bokes (2012).
        It is a helper function for the bursty biological model.

        Parameters
        ----------
        x: float np.ndarray or float
            time or array of times to evaluate the integrand at.
        g: np.ndarray
            complex PGF arguments, adjusted for sampling.
        b: float
            burst size.
        beta: float
            splicing rate.
        gamma: float
            degradation rate.

        Returns
        -------
        _: np.ndarray
            integrand value.
        """
        g = np.asarray(g)[:,:,None]
        if np.isclose(beta, gamma):  # compute prefactors for the ODE characteristics.
            c_1 = g[0]  # nascent
            c_2 = x * beta * g[1]
        else:
            f = beta / (beta - gamma)
            c_2 = g[1] * f
            c_1 = g[0] - c_2

        U = b * (np.exp(-beta * x) * c_1 + np.exp(-gamma * x) * c_2)
        return U / (1 - U)

    def get_MoM(self, moments, lb_log, ub_log, samp=None):
        """Compute method of moments parameter estimates.

        This method evaluates the method of moments biological parameter estimates
        at a particular set of sampling parameters and under the instantiated model.


        Parameters
        ----------
        moments: dict
            moments for the gene, including 'S_mean', 'U_mean', 'S_var', 'U_var'.
        lb_log: float np.ndarray
            log10 lower bounds on biological parameters.
        ub_log: float np.ndarray
            log10 upper bounds on biological parameters.
        samp: None or np.ndarray, optional
            sampling parameters, if applicable.

        Returns
        -------
        x0: np.ndarray
            log10 biological parameter estimates.
        """
        lb = 10**lb_log
        ub = 10**ub_log
        if self.seq_model == "Poisson" or "Bernoulli":
            samp = 10**samp

        # These can be defined per model (just happen to be shared by multiple models here).
        U_var, U_mean = moments['MOM_unspliced_var'], moments['MOM_unspliced_mean']
        S_var, S_mean = moments['MOM_spliced_var'], moments['MOM_spliced_mean']

        if self.bio_model == "Bursty" or self.bio_model == "CIR":
            b = (U_var / U_mean - 1) if U_mean > 0 else 1.0
            if not np.isfinite(b):
                b = 1.0
            
            if self.seq_model == "Bernoulli":
                b /= samp[0]
            elif self.seq_model == "Poisson":
                b = b / samp[0] - 1

            b = np.clip(b, lb[0], ub[0])
            beta = b / U_mean
            gamma = b / S_mean
            x0 = np.asarray([b, beta, gamma])
            
        elif self.bio_model == "ProteinBursty":
            U_var, U_mean, S_mean, P_mean, UP_covar = moments["MOM_unspliced_var"], moments["MOM_unspliced_mean"], moments["MOM_spliced_mean"], moments["MOM_protein_mean"], moments["MOM_cov_unspliced_protein"]
            b = (U_var / U_mean - 1) if U_mean > 0 else 1.0
            if not np.isfinite(b):
                b = 1.0
            if self.seq_model == "Bernoulli":
                b /= samp[0]
            elif self.seq_model == "Poisson":
                b = b / samp[0] - 1

            b = np.clip(b, lb[0], ub[0])
            beta = b / U_mean
            gamma = b / S_mean

            # TODO: add protein moments to the list of moments.
            # Define r = k_p/gamma_p
            r = P_mean*gamma/b
            C = UP_covar
            gamma_p = C*(beta + gamma)*beta/(b**2*r - C*(beta + gamma))
            gamma_p = np.clip(gamma_p, lb[4], ub[4])
            k_p = np.clip(r*gamma_p, lb[3], ub[3])
            gammap = k_p/r
            x0 = np.asarray([b, beta, gamma, k_p, gamma_p])

        elif self.bio_model == "Delay":
            b = U_var / U_mean - 1

            if self.seq_model == "Bernoulli":
                b /= samp[0]
            elif self.seq_model == "Poisson":
                b = b / samp[0] - 1

            b = np.clip(b, lb[0], ub[0])
            beta = b / U_mean
            tauinv = b / S_mean
            x0 = np.asarray([b, beta, tauinv])

        elif self.bio_model == "DelayedSplicing":
            # b = moments["S_var"] / moments["S_mean"] - 1
            b = (U_var / U_mean - 1) / 2
            b = np.clip(b, lb[0], ub[0])
            tauinv = b / U_mean
            gamma = b / S_mean
            x0 = np.asarray([b, tauinv, gamma])

            if self.seq_model == "Bernoulli":
                b /= 2*samp[0]
            elif self.seq_model == "Poisson":
                b = b / samp[0] - (1/2)

        elif self.bio_model == "Constitutive":
            beta = 1 / U_mean
            gamma = 1 / S_mean
            x0 = np.asarray([beta, gamma])

        elif self.bio_model == "Extrinsic":
            if self.seq_model == "Poisson":
                alpha = U_mean ** 2 / (
                    U_var - U_mean * (1 + samp[0])
                )
            else:
                alpha = U_mean ** 2 / (U_var - U_mean)

            beta = alpha / U_mean
            gamma = alpha / S_mean
            x0 = np.asarray([alpha, beta, gamma])
        else:
            raise ValueError("Please select from implemented models.")


        if self.seq_model in ("Bernoulli", "Poisson"):
            if self.bio_model == "Constitutive":
                x0 *= samp
            elif self.bio_model == "ProteinBursty":
                x0[[1,2]] *= samp[:2]
                x0[-1] *= samp[2]/samp[1]
            else:
                x0[1:] = x0[1:] * samp

        if self.amb_model == "Equal":  # basic
            x0 = np.concatenate((x0, [0.1]))  # just make a guess
        elif self.amb_model == "Unequal":
            x0 = np.concatenate((x0, [0.1, 0.1]))  # just make a guess

        for j in range(self.get_num_params()):
            x0[j] = np.clip(x0[j], lb[j], ub[j])

        x0 = np.log10(x0)
        if (~np.isfinite(x0)).any():
            x0 = (
                np.random.rand(self.get_num_params()) * (ub_log - lb_log) + lb_log
            )  # last resort -- makes it nondeterministic though
        return x0

    # TODO: Add ProteinBursty
    def eval_model_noise(self, p_, samp=None):
        """Compute CV2 fractions due to intrinsic, extrinsic, and technical noise.

        This method reports the fractions of normalized variance attributable to intrinsic (single-molecule),
        extrinsic (e.g., burst), and technical noise under the instantiated model.

        Parameters
        ----------
        p_: np.ndarray
            log10 biological parameters.
        samp: None or np.ndarray, optional
            sampling parameters, if applicable.

        Returns
        -------
        f: tuple of np.ndarrays
            array of size 3x2 or 2x2 reporting noise fractions for each species attributable to each source.
            dimension 0: variance fraction (intrinsic, extrinsic, technical)
            dimension 1: species (unspliced, spliced)
            The null technical noise model has dim 0 of size 2, as it has no technical noise component.
            dimension 0 sums to 1.

        """
        p = 10**p_
        if self.bio_model == "Constitutive":  # constitutive production
            beta, gamma = p
            mu = [1 / beta, 1 / gamma]
        elif self.bio_model == "Bursty":  # bursty production
            b, beta, gamma = p
            mu = [b / beta, b / gamma]
        elif (
            self.bio_model == "Extrinsic"
        ):  # constitutive production with extrinsic noise
            alpha, beta, gamma = p
            mu = [alpha / beta, alpha / gamma]
        elif self.bio_model == "Delay":  # bursty production with delayed degradation
            raise ValueError("Not yet implemented!")
        elif self.bio_model == "CIR":  # CIR-like:
            b, beta, gamma = p
            mu = [b / beta, b / gamma]

        mu = np.asarray(mu)
        noise_int = 1 / mu

        if self.bio_model == "Constitutive":  # constitutive production
            noise_ext = [0, 0]
        elif self.bio_model == "Bursty":  # bursty production
            noise_ext = [beta, beta * gamma / (beta + gamma)]
        elif (
            self.bio_model == "Extrinsic"
        ):  # constitutive production with extrinsic noise
            noise_ext = [1 / alpha, 1 / alpha]
        elif self.bio_model == "Delay":  # bursty production with delayed degradation
            raise ValueError("Not yet implemented!")
        elif self.bio_model == "CIR":  # CIR-like:
            noise_ext = [beta, beta * gamma / (beta + gamma)]
        noise_ext = np.asarray(noise_ext)

        if self.seq_model == "None":
            noise = noise_int + noise_ext
            return (noise_int / noise, noise_ext / noise)
        elif self.seq_model == "Bernoulli":
            noise = noise_int / samp + noise_ext
            return (
                noise_int / noise,
                noise_ext / noise,
                1 - noise_int / noise - noise_ext / noise,
            )
        elif self.seq_model == "Poisson":
            samp_ = 10**samp
            noise_tech = 1 / (mu * samp_)
            noise = noise_int + noise_ext + noise_tech
            return (noise_int / noise, noise_ext / noise, noise_tech / noise)
