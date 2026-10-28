"""
This script provides convenience functions for evaluating RNA distributions.
"""

import logging
import warnings
import collections as _collections
import contextlib
import threading
import numpy as np
from numba import jit
import scipy
from scipy import integrate

from scipy.fft import irfftn

# from .nn_toolbox import basic_ml_bivariate, ml_microstate_logP
from .extract_data import log
from .reaction_network import ReactionNetwork, IdentifiabilityWarning, StepSchedule

_MESH_CACHE: dict = {}
_MESH_CACHE_ENTRIES = [0]          # total complex128 entries currently cached
_MESH_CACHE_CAP_ENTRIES = 8 << 20  # 8M entries = 128 MB; cleared wholesale when exceeded


def clear_mesh_cache():
    """Drop every cached evaluation mesh (Python path)."""
    _MESH_CACHE.clear()
    _MESH_CACHE_ENTRIES[0] = 0


def _build_mesh(limits_tuple, mx_tuple):
    """Build and cache the complex evaluation mesh for eval_model_pss.

    Keyed by (limits, reduced shape).  With per-gene limits each distinct grid
    adds an entry, so the cache is bounded: when adding a mesh would push the
    total above ``_MESH_CACHE_CAP_ENTRIES`` the cache is cleared first.
    """
    key = (limits_tuple, mx_tuple)
    mesh = _MESH_CACHE.get(key)
    if mesh is None:
        u = []
        for i, m in enumerate(mx_tuple):
            l = np.arange(m)
            u_ = np.exp(-2j * np.pi * l / limits_tuple[i]) - 1
            u.append(u_)
        g = np.meshgrid(*u, indexing="ij")
        mesh = np.array([arr.flatten() for arr in g])
        if _MESH_CACHE and _MESH_CACHE_ENTRIES[0] + mesh.size > _MESH_CACHE_CAP_ENTRIES:
            clear_mesh_cache()
        _MESH_CACHE[key] = mesh
        _MESH_CACHE_ENTRIES[0] += mesh.size
    return mesh

# ---------------------------------------------------------------------------
# Optional Rust backend (monod_core).  When available, eval_model_pss for
# 2-modality models with seq_model="None" and amb_model="None" is delegated
# to the Rust implementation (same results, faster computation).
# ---------------------------------------------------------------------------
from ._core import _mc          # None when the compiled extension is not installed
_HAS_RUST = _mc is not None

# ---------------------------------------------------------------------------
# Optional GPU backend (PyTorch).  When a CUDA GPU with float64 support is
# available, eval_model_pss for 2-modality models (seq_model "None", "Poisson"
# or "Bernoulli"; no ambient model) is delegated to a fully-vectorised torch
# implementation that broadcasts the grid × quadrature computation onto the
# GPU.  Results match the Python/Rust baseline within rtol=1e-5.
# ---------------------------------------------------------------------------
# ProteinBursty integrator settings; keep in sync with monod_core (PROTEIN_ODE_*).
_PROTEIN_ODE_TOL = 1e-6        # per-step error target on the generating-function scale
_PROTEIN_ODE_C0 = 0.05         # first step as a fraction of the fastest time scale
_PROTEIN_ODE_MAX_STEPS = 20_000
_CUSTOM_FD_EPS = 1e-4          # central finite-difference step for custom-network gradients (log10 units)


def _samp_is_null(samp):
    """True when no sampling transform applies: samp is None or all zeros.

    Inference hands every model a sampling vector for the current grid point;
    for seq_model="None" that vector is zeros rather than None, and the
    noise-free fast paths must accept both spellings.
    """
    return samp is None or bool(np.all(np.asarray(samp, dtype=float) == 0))


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
    else:
        _HAS_GPU = False
        _GPU_DEVICE = None
except ImportError:
    _HAS_GPU = False
    _GPU_DEVICE = None


_GPU_NUMPY_OK = None


def _gpu_to_numpy(t):
    """Move a float64 tensor to host memory as a numpy array.

    ``Tensor.numpy()`` is a zero-copy view and is what we want; torch builds
    compiled against an older numpy raise ``RuntimeError`` from it, in which
    case the (much slower) ``tolist`` round trip is used instead.  The choice
    is probed once.
    """
    global _GPU_NUMPY_OK
    cpu = t.detach().cpu()
    if _GPU_NUMPY_OK is None:
        try:
            _GPU_NUMPY_OK = isinstance(cpu[:1].numpy(), np.ndarray)
        except Exception:  # noqa: BLE001 - any failure means "do not use numpy()"
            _GPU_NUMPY_OK = False
    if _GPU_NUMPY_OK:
        return np.array(cpu.numpy(), dtype=np.float64, copy=True)
    return np.array(cpu.tolist(), dtype=np.float64)


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


def _eval_model_pss_gpu(bio_model, p_log, limits, fixed_quad_T, quad_order, device,
                        samp=None, seq_model="None"):
    """Single-gene GPU evaluation: the batched evaluator with one gene.

    Returns a numpy array of shape (l0, l1)."""
    return _eval_model_pss_gpu_batch(
        bio_model, [p_log], [limits], fixed_quad_T, quad_order, device,
        samp_list=None if samp is None else [samp], seq_model=seq_model,
    )[0]


# Memory budget for the batched quadrature tensor [genes, grid, quad] on the GPU.
_GPU_BATCH_BYTES = 512 * 1024 ** 2


def _gpu_apply_noise(g0, g1, lam, seq_model):
    """Per-gene sequencing transform of the mesh.  g0/g1: [N]; lam: [G, 2] linear
    sampling rates.  Returns [G, N] tensors."""
    if seq_model == "None":
        return g0.unsqueeze(0), g1.unsqueeze(0)
    l0 = lam[:, 0:1]
    l1 = lam[:, 1:2]
    if seq_model == "Poisson":
        return _torch.exp(l0 * g0.unsqueeze(0)) - 1.0, _torch.exp(l1 * g1.unsqueeze(0)) - 1.0
    if seq_model == "Bernoulli":
        return l0 * g0.unsqueeze(0), l1 * g1.unsqueeze(0)
    raise ValueError(f"Unknown seq_model for GPU path: {seq_model}")


def _gpu_pgf_quadrature_batch(bio_model, g0, g1, b, beta, gamma, T, quad_order):
    """Batched Bursty/CIR log-PGF.  g0/g1: [G, N]; b/beta/gamma/T: [G]."""
    from numpy.polynomial.legendre import leggauss
    device = g0.device
    G, N = g0.shape
    xi_np, wi_np = leggauss(quad_order)
    xi = _torch.tensor(xi_np, dtype=_torch.float64, device=device)
    wi = _torch.tensor(wi_np, dtype=_torch.float64, device=device)
    t_half = (T / 2.0).reshape(G, 1)                          # [G, 1]
    x = t_half + t_half * xi.unsqueeze(0)                     # [G, Q]
    b_ = b.reshape(G, 1, 1)
    beta_ = beta.reshape(G, 1, 1)
    gamma_ = gamma.reshape(G, 1, 1)
    eb = _torch.exp(-beta_ * x.unsqueeze(1))                  # [G, 1, Q]
    eg = _torch.exp(-gamma_ * x.unsqueeze(1))                 # [G, 1, Q]
    close = _torch.isclose(beta, gamma).reshape(G, 1, 1)      # per-gene branch, as in the CPU paths
    denom = _torch.where(close, _torch.ones_like(beta_), beta_ - gamma_)
    f = beta_ / denom
    w = (wi.unsqueeze(0) * t_half).unsqueeze(1)               # [G, 1, Q]
    out = _torch.empty((G, N), dtype=_torch.complex128, device=device)
    # chunk the grid axis so [G, n, Q] complex128 stays within the memory budget
    per_point = G * quad_order * 16 * 3
    n_chunk = max(1, int(_GPU_BATCH_BYTES // per_point))
    for s in range(0, N, n_chunk):
        g0c = g0[:, s:s + n_chunk].unsqueeze(2)               # [G, n, 1]
        g1c = g1[:, s:s + n_chunk].unsqueeze(2)
        u_general = b_ * ((g0c - g1c * f) * eb + g1c * f * eg)
        u_close = b_ * (g0c * eb + g1c * (beta_ * x.unsqueeze(1) * eg))
        u = _torch.where(close, u_close, u_general)           # [G, n, Q]
        out[:, s:s + n_chunk] = (_burst_integrand(bio_model, u, xp=_torch) * w).sum(dim=2)
    return out


def _eval_model_pss_gpu_batch(bio_model, params_list, limits_list, fixed_quad_T, quad_order,
                              device, samp_list=None, seq_model="None"):
    """Evaluate the 2-D PSS of many genes on the GPU.

    Genes are grouped by grid size; within a group the log-PGF of all genes is
    computed with one broadcast expression and inverted with one batched
    irfftn.  Per gene this matches ``_eval_model_pss_gpu`` to rounding.

    Returns a list of numpy arrays shaped like each gene's limits.
    """
    n = len(params_list)
    if samp_list is None:
        samp_list = [None] * n
    out = [None] * n
    groups = {}
    for i, lim in enumerate(limits_list):
        groups.setdefault((int(lim[0]), int(lim[1])), []).append(i)
    one = _torch.ones(1, dtype=_torch.complex128, device=device)
    for (l0, l1), idx in groups.items():
        G = len(idx)
        p = _torch.tensor(np.power(10.0, np.asarray([np.asarray(params_list[i], dtype=float) for i in idx])),
                          dtype=_torch.float64, device=device)                 # [G, n_params]
        g0, g1 = _gpu_build_mesh_2d(l0, l1, device)
        if seq_model in ("Poisson", "Bernoulli"):
            samp = []
            for i in idx:
                if samp_list[i] is None:
                    raise ValueError(f"seq_model={seq_model!r} requires samp (log10 sampling parameters).")
                s_ = np.asarray(samp_list[i], dtype=float).ravel()
                if s_.size < 2:
                    raise ValueError("samp must hold one sampling parameter per modality (2 for 2-D models).")
                samp.append(s_[:2])
            lam = _torch.tensor(np.power(10.0, np.asarray(samp)), dtype=_torch.float64, device=device)
        else:
            lam = None
        g0b, g1b = _gpu_apply_noise(g0, g1, lam, seq_model)                    # [G or 1, N]
        g0b = g0b.expand(G, -1)
        g1b = g1b.expand(G, -1)

        if bio_model in _CLOSED_FORM_MODELS:
            gf = _closed_form_pgf(bio_model, g0b, g1b, [p[:, j:j + 1] for j in range(p.shape[1])], xp=_torch)
        elif bio_model in _QUADRATURE_MODELS:
            b, beta, gamma = p[:, 0], p[:, 1], p[:, 2]
            T = _quadrature_horizon(fixed_quad_T, beta, gamma)
            gf = _gpu_pgf_quadrature_batch(bio_model, g0b, g1b, b, beta, gamma, T, quad_order)
        else:
            raise ValueError(f"Unknown bio_model for GPU path: {bio_model}")

        gf = _torch.exp(gf).reshape(G, l0, l1 // 2 + 1)
        pss = _torch.fft.irfftn(gf, s=(l0, l1), dim=(1, 2)).abs()
        pss = pss / pss.sum(dim=(1, 2), keepdim=True)
        arr = _gpu_to_numpy(pss)
        for k, i in enumerate(idx):
            out[i] = arr[k]
    return out


_MODELS_2D = frozenset({"Constitutive", "Bursty", "CIR", "Extrinsic", "Delay", "DelayedSplicing"})
_CLOSED_FORM_MODELS = frozenset({"Constitutive", "Extrinsic", "Delay", "DelayedSplicing"})
_QUADRATURE_MODELS = frozenset({"Bursty", "CIR"})
_Backend = _collections.namedtuple("_Backend", "kind family samp")
_LN10 = np.log(10.0)

_PROTEIN_FUDGE = (0.1, 10.0)        # legacy step-size bounds the Rust protein kernels still take
_BIO_MODELS = ("Delay", "Bursty", "Extrinsic", "Constitutive", "CIR", "DelayedSplicing", "ProteinBursty", "Custom", "Chromatin")
_SEQ_MODELS = ("None", "Bernoulli", "Poisson")
_AMB_MODELS = ("None", "Equal", "Unequal")

# Per-family defaults for the built-in models: modalities, expression filter
# and log10 parameter box (burst size [-1, 4.2], rates [-1.8, 2.5 or 3.5]).
# Custom networks and Chromatin derive theirs from their structure in __init__.
_TWO_SPECIES = dict(modalities=['unspliced', 'spliced'],
                    filt={'min_means': [0.01, 0.01], 'max_maxes': [350, 350], 'min_maxes': [4, 4]},
                    lb=[-1.0, -1.8, -1.8], ub=[4.2, 2.5, 3.5])
_FAMILIES = {
    "Constitutive": {**_TWO_SPECIES, "lb": [-1.8, -1.8], "ub": [2.5, 3.5]},
    "Bursty": _TWO_SPECIES, "CIR": _TWO_SPECIES, "Extrinsic": _TWO_SPECIES,
    "Delay": _TWO_SPECIES, "DelayedSplicing": _TWO_SPECIES,
    "ProteinBursty": dict(modalities=['unspliced', 'spliced', 'protein'],
                          filt={'min_means': [0.01, 0.01, 1], 'max_maxes': [350, 350, 1000], 'min_maxes': [4, 4, 10]},
                          lb=[-1.0, -1.8, -1.8, -1.8, -1.8], ub=[4.2, 2.5, 3.5, 3.5, 3.5]),
}
# Sampling-parameter box per sequencing model (log10); "None" is the null grid.
_SEQ_BOUNDS = {"None": {'samp_lb': [0, 0], 'samp_ub': [0, 0], 'gridsize': [1, 1]},
               "Bernoulli": {'samp_lb': [-8, -3], 'samp_ub': [-5, 0], 'gridsize': [6, 7]},
               "Poisson": {'samp_lb': [-8, -3], 'samp_ub': [-5, 0], 'gridsize': [6, 7]}}


def _closed_form_pgf(bio_model, g0, g1, params, xp=np, grad=False):
    """log-PGF ``phi`` of the four closed-form two-species models at the mesh
    points ``(g0, g1)``, written once against an array namespace ``xp`` so the
    NumPy and torch paths evaluate the same expressions.  ``params`` are the
    linear parameters, each broadcastable against ``g0`` (scalars on the CPU,
    ``[G, 1]`` columns on the GPU).  With ``grad`` (NumPy only) also returns
    the list of ``d phi / d log10(param)``."""
    if bio_model == "Constitutive":
        beta, gamma = params
        phi = g0 / beta + g1 / gamma
        if not grad:
            return phi
        return phi, [-g0 / beta * _LN10, -g1 / gamma * _LN10]
    if bio_model == "Extrinsic":
        alpha, beta, gamma = params
        D = 1.0 - g0 / beta - g1 / gamma
        phi = -alpha * xp.log(D)
        if not grad:
            return phi
        return phi, [phi * _LN10, -alpha * g0 / (beta * D) * _LN10, -alpha * g1 / (gamma * D) * _LN10]
    if bio_model == "DelayedSplicing":
        b, tauinv, gamma = params
        tau = 1.0 / tauinv
        D0 = 1.0 - b * g0
        D1 = 1.0 - b * g1
        phi = b * g0 * tau / D0 - xp.log(D1) / gamma
        if not grad:
            return phi
        return phi, [(g0 * tau / D0 ** 2 + g1 / (gamma * D1)) * b * _LN10,
                     -b * g0 * tau / D0 * _LN10,
                     xp.log(D1) / gamma * _LN10]
    if bio_model == "Delay":
        b, beta, tauinv = params
        tau = 1.0 / tauinv
        E = xp.exp(-beta * tau)
        U = g1 + (g0 - g1) * E
        A = 1.0 - b * U
        B = 1.0 - b * g0
        C = 1.0 - b * g1
        logAB = xp.log(A / B)
        phi = -xp.log(A) / beta + logAB / (beta * C) + tau * b * g1 / C
        if not grad:
            return phi
        dA_dbeta = b * tau * (g0 - g1) * E
        dA_dtau = b * beta * (g0 - g1) * E
        dphi_db = (U / (A * beta) - U / (A * beta * C) + g0 / (B * beta * C)
                   + logAB * g1 / (beta * C ** 2) + tau * g1 / C + tau * b * g1 ** 2 / C ** 2) * b * _LN10
        dphi_dbeta = (xp.log(A) / beta ** 2 - dA_dbeta / (A * beta) + dA_dbeta / (A * beta * C)
                      - logAB / (beta ** 2 * C)) * beta * _LN10
        dphi_dtauinv = (-b * g1 * tau / C * (dA_dtau / (A * beta) + 1.0)) * _LN10
        return phi, [dphi_db, dphi_dbeta, dphi_dtauinv]
    raise ValueError(f"{bio_model!r} has no closed-form generating function")


def _burst_characteristic(x, g0, g1, b, beta, gamma, grad=False):
    """``U(t)`` of the Bursty and CIR characteristics at times ``x`` (Singh and
    Bokes 2012; Gorin, Vastola, Fang and Pachter 2021), with the limiting form
    at ``beta == gamma``; with ``grad`` also ``dU/dbeta`` and ``dU/dgamma``.
    ``g0``, ``g1`` broadcast against ``x`` (a trailing time axis)."""
    eb = np.exp(-beta * x)
    eg = np.exp(-gamma * x)
    if np.isclose(beta, gamma):
        U = b * (g0 * eb + g1 * (beta * x * eg))
        if not grad:
            return U
        return U, b * (-g0 * x * eb + g1 * x * eg), -b * g1 * beta * x ** 2 * eg
    f = beta / (beta - gamma)
    c2 = g1 * f
    c1 = g0 - c2
    U = b * (c1 * eb + c2 * eg)
    if not grad:
        return U
    dU_dbeta = b * g1 * gamma / (beta - gamma) ** 2 * (eb - eg) - b * c1 * x * eb
    dU_dgamma = b * g1 * beta / (beta - gamma) ** 2 * (eg - eb) - b * c2 * x * eg
    return U, dU_dbeta, dU_dgamma


def _burst_integrand(bio_model, U, xp=np):
    """The integrand of the log-PGF time integral: ``U/(1-U)`` for Bursty,
    ``(1 - sqrt(1-4U))/2`` for CIR."""
    if bio_model == "Bursty":
        return U / (1.0 - U)
    return 0.5 * (1.0 - xp.sqrt(1.0 - 4.0 * U))


def _quadrature_horizon(fixed_quad_T, beta, gamma):
    """Upper limit of the Bursty/CIR time integral."""
    return fixed_quad_T * (1.0 / beta + 1.0 / gamma + 1.0)


def _format_direction(direction, min_weight=0.15):
    """'+0.58 k0 +0.58 k1 +0.58 k2' for a {name: component} unit vector."""
    items = [(n, c) for n, c in direction.items() if abs(c) >= min_weight]
    if not items:
        items = [max(direction.items(), key=lambda kv: abs(kv[1]))]
    sign = 1.0 if items[0][1] >= 0 else -1.0     # canonical sign: first named entry positive
    return " ".join(f"{sign * c:+.2f} {n}" for n, c in items)


# Step schedules of the adaptive integrators are per-thread state: the
# Hessian and gradient stencils pin a schedule on the model while other threads
# may be evaluating other genes on the same model object.  Properties backed by
# thread-local storage keep the plain attribute syntax; the storage is not
# pickled, so worker processes start unpinned.
_SCHEDULE_TLS = threading.local()


def _tls_get(obj, key):
    d = getattr(_SCHEDULE_TLS, "d", None)
    return None if d is None else d.get((id(obj), key))


def _tls_set(obj, key, value):
    d = getattr(_SCHEDULE_TLS, "d", None)
    if d is None:
        d = _SCHEDULE_TLS.d = {}
    if value is None:
        d.pop((id(obj), key), None)
    else:
        d[(id(obj), key)] = value


def _tls_snapshot(obj):
    """This thread's thread-local entries for ``obj`` (pins, tolerance override)."""
    d = getattr(_SCHEDULE_TLS, "d", None)
    return {} if d is None else {k: v for k, v in d.items() if k[0] == id(obj)}


@contextlib.contextmanager
def _tls_carry(obj, snap):
    """Install another thread's snapshot of ``obj``'s thread-local state for
    the block and restore this thread's own afterwards.  Pool workers use it
    to evaluate on the calling thread's pinned schedule and tolerance."""
    d = getattr(_SCHEDULE_TLS, "d", None)
    if d is None:
        d = _SCHEDULE_TLS.d = {}
    keys = set(snap) | {k for k in d if k[0] == id(obj)}
    saved = {k: d.get(k) for k in keys}
    for k in keys:
        d.pop(k, None)
    d.update(snap)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                d.pop(k, None)
            else:
                d[k] = v


def _tls_property(key):
    return property(lambda self: _tls_get(self, key),
                    lambda self, value: _tls_set(self, key, value))


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
    protein_limit: float
        ProteinBursty only: protein counts above this are coarse-grained.
    fit_unspliced: bool
        ProteinBursty only: whether the unspliced layer is part of the fit.
    min_fudge, max_fudge: float
        Accepted for backward compatibility and ignored: since 0.4.0 the
        custom-network and protein integrators choose their steps by error
        control (see ``ReactionNetwork.eval_pgf`` and ``protein_pgf``).
        (The protein model chooses its steps by error control and ignores them.)
    network: ReactionNetwork or None
        the user-defined network for bio_model="Custom".
    normalize_production_rate, burst_mean_rate: bool
        ``bio_model="Custom"`` only: fix the first production rate to 1, and
        read the rate of a geometric-burst production as the mean production
        rate instead of the burst frequency (see :mod:`monod.reaction_network`).
        With ``burst_mean_rate=True`` the burst-size box reaches down to 1e-3,
        where the network is its single-molecule counterpart, so a constitutive
        network is nested in its bursty version and
        :func:`monod.analysis.compare_nested_models` can test the two.
    """

    # per-thread schedule state of the adaptive integrators (see _SCHEDULE_TLS)
    _fixed_protein_schedule = _tls_property("fixed_protein")
    _last_protein_schedule = _tls_property("last_protein")
    _fixed_custom_schedule = _tls_property("fixed_custom")
    _last_custom_schedule = _tls_property("last_custom")
    _fixed_chromatin_schedule = _tls_property("fixed_chromatin")
    _last_chromatin_schedule = _tls_property("last_chromatin")

    # The chromatin integrator's tolerance: the model's value, or the thread's
    # override while inside search_tolerance(), so that one model instance can
    # serve a pool of per-gene threads that enter and leave the search at
    # different times.
    @property
    def chromatin_tol(self):
        override = _tls_get(self, "chromatin_tol")
        return self._chromatin_tol if override is None else override

    @chromatin_tol.setter
    def chromatin_tol(self, value):
        self._chromatin_tol = float(value)

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
        burst_mean_rate = False,
        n_sites = 2,
        rule = "additive",
        periodic = False,
        chromatin_tol = 1e-5,
        chromatin_search_tol = 1e-4,
        dropout_bounds = (-2.0, -0.05, 3),
        bursty = False,
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
        n_sites, rule, periodic: int, str, bool
            ``bio_model="Chromatin"`` only: the number of binary chromatin sites
            of a gene's locus, the transcription rule (``"additive"``,
            ``"switch"`` or ``"table"``, see :mod:`monod.chromatin`) and whether
            the sites form a ring.  Parameters are ``k_on`` per site, ``epsilon``,
            the rule's rates, ``beta`` and ``gamma``, in units of the shared
            closing rate ``k_off = 1``.  Modalities are ``unspliced``, ``spliced``
            and ``configuration`` (the integer 0..2**n_sites-1 encoding which
            sites are open, site 0 the least significant bit).
        bursty: bool
            ``bio_model="Chromatin"`` only: transcribe in geometric bursts of a
            shared mean size ``b`` (one more parameter, last) instead of single
            molecules.  The rule's rates stay mean production rates, so the
            constitutive model is the small-``b`` limit.
        chromatin_tol, chromatin_search_tol: float
            step tolerance of the chromatin integrator for reported values and
            during the search (see :meth:`search_tolerance`).
        dropout_bounds: (float, float, int)
            ATAC dropout is a sampling parameter on the configuration axis,
            scanned on the grid like the sequencing parameters: log10 bounds and
            grid size of the per-open-site dropout probability.  A value of 0
            (the null sampling vector) means no dropout.
        """
        self.bio_model = bio_model
        self.available_biomodels = _BIO_MODELS
        self.available_seqmodels = _SEQ_MODELS
        self.available_ambmodels = _AMB_MODELS

        # Store the custom reaction network (required when bio_model == "Custom").
        if bio_model == "Custom":
            if network is None:
                raise ValueError(
                    "bio_model='Custom' requires a network string passed as network=..."
                )
            self.network = ReactionNetwork(
                network, normalize_production_rate=normalize_production_rate,
                burst_mean_rate=burst_mean_rate,
            )
        else:
            self.network = None

        if bio_model == "Chromatin":
            from . import chromatin as _chromatin
            if rule not in _chromatin.RULES:
                raise ValueError(f"rule must be one of {_chromatin.RULES}, got {rule!r}")
            if int(n_sites) < 1:
                raise ValueError("n_sites must be at least 1")
            self.n_sites = int(n_sites)
            self.rule = rule
            self.periodic = bool(periodic)
            self.chromatin_tol = float(chromatin_tol)
            self.chromatin_search_tol = float(chromatin_search_tol)
            self.dropout_bounds = tuple(dropout_bounds)
            self.bursty = bool(bursty)

        # Modalities (in order), expression filter and parameter box: from the
        # family table for the built-in models, from the structure otherwise.
        fam = _FAMILIES.get(self.bio_model)
        if fam is not None:
            self.model_modalities = list(fam["modalities"])
            self.filter_bounds = {k: list(v) for k, v in fam["filt"].items()}
            self.bio_bounds = {'phys_lb': list(fam["lb"]), 'phys_ub': list(fam["ub"])}
        elif self.bio_model == "Custom":
            # One filter entry per species with the mRNA defaults (override with
            # filt_param= when a species lives on a different scale); burst sizes
            # get the Bursty b range (down to the single-molecule limit for
            # mean-rate bursts), rates and inverse delays the union of the beta
            # and gamma ranges: the exponential integrator's cost does not
            # depend on the spread.
            net = self.network
            n_sp = len(net.species)
            self.model_modalities = net.species
            self.filter_bounds = {'min_means': [0.01] * n_sp, 'max_maxes': [350] * n_sp, 'min_maxes': [4] * n_sp}
            n_rates = len(net.delay_params) + len(net.rate_names)
            self.bio_bounds = {'phys_lb': [-3.0 if net.burst_mean_rate else -1.0] * len(net.named_params) + [-1.8] * n_rates,
                               'phys_ub': [4.2] * len(net.named_params) + [3.5] * n_rates}
        elif self.bio_model == "Chromatin":
            # k_on per site and the rule's mean production rates in units of
            # k_off (the floor low enough for "no transcription in this
            # configuration"); beta and gamma as Bursty; the burst size last
            from . import chromatin as _chromatin
            k = _chromatin.n_rule_params(self.rule, self.n_sites)
            self.model_modalities = ['unspliced', 'spliced', 'configuration']
            self.filter_bounds = {'min_means': [0.01, 0.01, 0.0], 'max_maxes': [350, 350, 2 ** self.n_sites],
                                  'min_maxes': [4, 4, 0]}
            self.bio_bounds = {'phys_lb': [-2.0] * self.n_sites + [-1.5] + [-4.0] * k + [-1.8, -1.8],
                               'phys_ub': [2.0] * self.n_sites + [0.5] + [2.0] * k + [2.5, 3.5]}
            if self.bursty:
                self.bio_bounds['phys_lb'].append(-3.0)
                self.bio_bounds['phys_ub'].append(2.0)
        else:
            raise ValueError(f"Please select a biological noise model from {_BIO_MODELS}; got {bio_model!r}")
        log.debug("Modalities for %s: %s", self.bio_model, self.model_modalities)

        if amb_model in ["None", None, "Null"]:
            amb_model = "None"
        if amb_model not in _AMB_MODELS:
            raise ValueError(f"amb_model must be one of {_AMB_MODELS}; got {amb_model!r}")
        self.amb_model = amb_model
        if seq_model in ["None", None, "Null"]:
            self.seq_model = "None"
        else:
            self.seq_model = seq_model
        self.set_integration_parameters(
            fixed_quad_T, quad_order, quad_vec_T, quad_method
        )
        
        if self.bio_model == "ProteinBursty":
            # min_fudge / max_fudge are accepted and ignored (step control is by
            # error since 0.4.0); the Rust kernels still take them as constants.
            self.protein_limit = protein_limit
            self.fit_unspliced = fit_unspliced
            log.info("Protein grid limit: {}".format(self.protein_limit))

        if self.seq_model not in _SEQ_BOUNDS:
            raise ValueError(f"seq_model must be one of {_SEQ_MODELS}; got {self.seq_model!r}")
        self.seq_bounds = {k: list(v) for k, v in _SEQ_BOUNDS[self.seq_model].items()}
        if self.bio_model == "Chromatin":
            # a third sampling parameter: log10 ATAC dropout per open site on the configuration axis
            lo, hi, ng = self.dropout_bounds
            self.seq_bounds['samp_lb'].append(float(lo))
            self.seq_bounds['samp_ub'].append(float(hi))
            self.seq_bounds['gridsize'].append(int(ng))
        
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

        elif self.bio_model == "Chromatin":
            param_str += [rf"$\log_{{10}} k_{{on,{j}}}$" for j in range(self.n_sites)]
            param_str += [r"$\log_{10} \epsilon$"]
            if self.rule == "switch":
                param_str += [r"$\log_{10} a$"]
            elif self.rule == "additive":
                param_str += [rf"$\log_{{10}} a_{{{j}}}$" for j in range(self.n_sites)]
            else:
                param_str += [rf"$\log_{{10}} a_{{{c}}}$" for c in range(2 ** self.n_sites)]
            param_str += [r"$\log_{10} \beta$", r"$\log_{10} \gamma$"]
            if self.bursty:
                param_str += [r"$\log_{10} b$"]

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

    # ------------------------------------------------------------------
    # Structured (pickle-free) serialisation
    # ------------------------------------------------------------------
    _CTOR_KEYS = (
        "bio_model", "seq_model", "amb_model", "quad_method", "fixed_quad_T", "quad_order",
        "quad_vec_T", "protein_limit", "fit_unspliced", "min_fudge", "max_fudge",
        "network", "normalize_production_rate", "burst_mean_rate",
        "n_sites", "rule", "periodic", "chromatin_tol", "chromatin_search_tol", "dropout_bounds", "bursty",
    )

    def to_dict(self):
        """Constructor arguments as a flat dict of scalars and strings.

        Round-trips through ``adata.uns`` and h5ad; see :meth:`from_dict`.
        """
        d = {
            "format_version": 1,
            "kind": "CMEModel",
            "bio_model": self.bio_model,
            "seq_model": self.seq_model,
            "amb_model": self.amb_model,
            "quad_method": self.quad_method,
            "fixed_quad_T": float(self.fixed_quad_T),
            "quad_order": int(self.quad_order),
            "quad_vec_T": float(self.quad_vec_T),
        }
        if self.bio_model == "ProteinBursty":
            d["protein_limit"] = float(self.protein_limit)
            d["fit_unspliced"] = bool(self.fit_unspliced)
        if getattr(self, "network", None) is not None:
            d["network"] = self.network.network_str
            d["normalize_production_rate"] = bool(self.network.normalize_production_rate)
            d["burst_mean_rate"] = bool(self.network.burst_mean_rate)
        if self.bio_model == "Chromatin":
            d.update(n_sites=int(self.n_sites), rule=self.rule, periodic=bool(self.periodic),
                     chromatin_tol=float(self._chromatin_tol), chromatin_search_tol=float(self.chromatin_search_tol),
                     dropout_bounds=[float(self.dropout_bounds[0]), float(self.dropout_bounds[1]), int(self.dropout_bounds[2])],
                     bursty=bool(self.bursty))
        return d

    @classmethod
    def from_dict(cls, d):
        """Rebuild a CMEModel from :meth:`to_dict` output (or an h5ad-loaded copy of it)."""
        kwargs = {}
        for k in cls._CTOR_KEYS:
            if k in d:
                v = d[k]
                kwargs[k] = v.item() if isinstance(v, np.generic) else v
        if kwargs.get("network", None) == "":
            kwargs["network"] = None
        if "dropout_bounds" in kwargs:
            kwargs["dropout_bounds"] = tuple(np.asarray(kwargs["dropout_bounds"]).tolist())
        return cls(**kwargs)

    def get_num_params(self):
        """Return the number of parameters for the model instance.
        Note that these are gene-specific parameters, i.e., global technical noise
        parameters are not used here.

        Returns
        -------
        numpars: int
            The number of gene-specific parameters.
        """
        # the parameter list is the single source of the count
        return len(getattr(self, "param_str", None) or self.get_log_name_str())

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
        # logL = n_cells * (sum f log f - KLD): one kernel for both quantities.
        f = np.asarray(data[1], dtype=float)
        entropy = float(np.sum(f * np.log(f)))
        return n_cells * (entropy - self.eval_model_kld(p, limits, samp, data, EPS))

    def _rust_2d_args(self, p, limits, data):
        """``(p, limits, x0, x1, f)`` as the 2-D Rust kernels take them."""
        x, f = data
        x_np = np.asarray(x, dtype=np.int64)
        return (np.asarray(p, dtype=float).tolist(), [int(v) for v in limits],
                x_np[:, 0].tolist(), x_np[:, 1].tolist(), np.asarray(f, dtype=float).tolist())

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
        # Rust fast-path: compute PSS and KLD in one call, avoiding the
        # Python round-trip for histogram indexing and log/sum operations.
        be = self._backend(samp)
        if _HAS_RUST and be.family == "2d":
            return _mc.eval_kld_2d(
                self.bio_model, *self._rust_2d_args(p, limits, data),
                float(self.fixed_quad_T), int(self.quad_order), be.samp, float(EPS), self.seq_model,
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
            finite-difference step size (used only for FD fallback); custom
            networks use central differences with at least 1e-4.
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
        _use_analytic = (
            self.bio_model in _CLOSED_FORM_MODELS
            or (self.bio_model in _QUADRATURE_MODELS and not _HAS_RUST)
        )
        if (_use_analytic
                and self.amb_model == "None"):
            result = self._eval_kld_analytic_grad(p, limits, samp, data, EPS)
            if result is not None:
                return result

        # Rust FD gradient path: evaluates base + all perturbed PSS in parallel
        # (rayon, GIL released), eliminating the Python round-trip per perturbation.
        be = self._backend(samp)
        if _HAS_RUST and be.family == "2d" and self.seq_model in ("None", "Poisson"):
            kld0, grad_list = _mc.eval_kld_grad_2d(
                self.bio_model, *self._rust_2d_args(p, limits, data),
                float(self.fixed_quad_T), int(self.quad_order), eps, be.samp, float(EPS),
            )
            return kld0, np.array(grad_list)

        # ProteinBursty: Rust forward FD with the integration schedule frozen at
        # the base point (see monod_core::eval_kld_grad_protein_bursty).
        if _HAS_RUST and be.family == "protein":
            x, f_data = data
            x_np = np.asarray(x, dtype=np.int64)
            if x_np.ndim == 2 and x_np.shape[1] == 3:
                outer = getattr(self, "_fixed_protein_schedule", None)   # an enclosing pinned_schedule block
                kld0, grad_list = _mc.eval_kld_grad_protein_bursty(
                    p.tolist(),
                    [int(v) for v in limits],
                    x_np.tolist(),
                    np.asarray(f_data, dtype=float).tolist(),
                    bool(self.fit_unspliced),
                    float(self.protein_limit),
                    _PROTEIN_FUDGE[0], _PROTEIN_FUDGE[1],
                    float(eps),
                    float(EPS),
                    None if outer is None else [float(x) for x in np.asarray(outer, dtype=float)],
                )
                return kld0, np.array(grad_list)

        # Python fallback: forward finite differences.
        outer_pin = self._schedule_is_pinned()
        if not outer_pin:
            self._fixed_protein_schedule = None
            self._fixed_custom_schedule = None
            self._fixed_chromatin_schedule = None
        kld0 = self.eval_model_kld(p, limits, samp, data, EPS)

        # Custom networks: central differences with a wide step.  Histogram
        # bins deep in the tail sit near the FFT's absolute precision and put a
        # noise floor of about 1e-9 on the KLD, which a 1e-6 forward difference
        # would turn into gradient errors of order 1e-2.
        central = self.bio_model in ("Custom", "Chromatin")
        if central:
            eps = max(float(eps), _CUSTOM_FD_EPS)

        n_params = len(p)
        # ProteinBursty, Custom and Chromatin: integrate every stencil point with
        # the base point's schedule so the quotient is not polluted by step-size
        # jumps (an enclosing ``pinned_schedule`` block already provides one).
        # The pin lives in thread-local state, so a stencil spread over a thread
        # pool carries it into each worker explicitly.
        pin_attr = {"ProteinBursty": "_fixed_protein_schedule", "Custom": "_fixed_custom_schedule",
                    "Chromatin": "_fixed_chromatin_schedule"}.get(self.bio_model)
        if pin_attr is not None and not outer_pin:
            setattr(self, pin_attr, getattr(self, pin_attr.replace("_fixed_", "_last_"), None))
        state = self.thread_state()

        def _perturbed(i):
            with self.carry_thread_state(state):                             # no-op on the calling thread
                return _perturbed_here(i)

        def _perturbed_here(i):
            p_eps = p.copy()
            p_eps[i] += eps
            k_plus = self.eval_model_kld(p_eps, limits, samp, data, EPS)
            if not central:
                return k_plus
            p_eps[i] -= 2 * eps
            return k_plus, self.eval_model_kld(p_eps, limits, samp, data, EPS)

        try:
            if n_jobs == 1:
                klds_eps = [_perturbed(i) for i in range(n_params)]
            else:
                # threads: the kernels release the GIL, and a process pool would
                # re-pickle the model and the histogram at every gradient
                klds_eps = Parallel(n_jobs=min(n_params, n_jobs) if n_jobs > 0 else n_jobs, prefer="threads")(
                    delayed(_perturbed)(i) for i in range(n_params)
                )
        finally:
            if not outer_pin and pin_attr is not None:
                setattr(self, pin_attr, None)
        if central:
            grad = np.array([(kp - km) / (2 * eps) for kp, km in klds_eps])
        else:
            grad = (np.array(klds_eps) - kld0) / eps
        return kld0, grad

    def _eval_kld_analytic_grad(self, p, limits, samp, data, EPS):
        """Analytical KLD gradient via chain rule through log-PGF → IFFT.

        Computes d KLD / d log10(θ_i) = -(1/N) * dot(f/pss, dR_i) + dN_i/N
        where dR_i = irfftn(exp(phi) * d phi / d log10(θ_i)).

        Returns (kld, grad) or None to signal fallback to finite differences.
        """
        p_lin = 10.0 ** p

        # Build mesh (reads from cache; no copy needed since g is read-only here).
        mx = np.array(limits, dtype=int)
        mx[-1] = mx[-1] // 2 + 1
        g = _build_mesh(tuple(limits), tuple(mx))

        # Apply technical noise transform to the mesh.
        if samp is not None and self.seq_model == "Poisson":
            g = np.exp(np.power(10.0, samp)[:, None] * g) - 1
        elif samp is not None and self.seq_model == "Bernoulli":
            # samp holds log10 capture probabilities (same convention as Poisson).
            g = g * np.power(10.0, samp)[:, None]
        elif self.seq_model not in ("None", "Poisson", "Bernoulli"):
            return None  # unsupported seq_model

        g0, g1 = g[0], g[1]

        # log-PGF phi and d phi / d log10(param) for every parameter.
        if self.bio_model in _CLOSED_FORM_MODELS:
            phi, dphi = _closed_form_pgf(self.bio_model, g0, g1, list(p_lin), grad=True)
        elif self.bio_model in _QUADRATURE_MODELS:
            # Semi-analytical gradient on the same Gauss-Legendre nodes as the
            # PSS: phi = integral of the integrand of U(t) over [0, T], and the
            # three d phi / d log10(param) integrals share the nodes.
            if self.quad_method != "fixed_quad":
                return None  # quad_vec integration: fall back to FD
            from numpy.polynomial.legendre import leggauss
            b, beta, gamma = p_lin
            t_half = _quadrature_horizon(self.fixed_quad_T, beta, gamma) / 2.0
            xi, wi = leggauss(self.quad_order)
            x = t_half + t_half * xi                                        # (Q,) nodes in [0, T]
            w_T = t_half * wi                                               # (Q,) weights
            U, dU_dbeta, dU_dgamma = _burst_characteristic(x, g0[:, None], g1[:, None], b, beta, gamma, grad=True)
            if self.bio_model == "Bursty":
                inv1mU = 1.0 / (1.0 - U)
                phi = (U * inv1mU * w_T).sum(axis=1)
                d_intg_dU = inv1mU ** 2
            else:
                sqrt_term = np.sqrt(1.0 - 4.0 * U)
                phi = 0.5 * ((1.0 - sqrt_term) * w_T).sum(axis=1)
                d_intg_dU = 1.0 / sqrt_term                                 # (1/2) * 2 / sqrt
            # d phi / d log10(param) = param * LN10 * integral d(integrand)/dU * dU/dparam; dU/db = U/b
            dphi = [_LN10 * (d_intg_dU * U * w_T).sum(axis=1),
                    beta * _LN10 * (d_intg_dU * dU_dbeta * w_T).sum(axis=1),
                    gamma * _LN10 * (d_intg_dU * dU_dgamma * w_T).sum(axis=1)]
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

    def _kernel_family(self):
        """Which accelerated kernel family this model configuration maps to.

        Returns ``"2d"`` (the six two-species models without ambient noise),
        ``"amb"`` (two-species with Equal/Unequal ambient noise and no or
        Poisson sequencing noise), ``"protein"``, ``"custom"`` (non-delayed or delayed network
        without sequencing or ambient noise), or ``None`` when only the
        Python implementation applies.
        """
        if self.bio_model in _MODELS_2D and self.quad_method == "fixed_quad":
            if self.amb_model == "None":
                return "2d"
            if self.amb_model in ("Equal", "Unequal") and self.seq_model in ("None", "Poisson"):
                return "amb"
            return None
        if self.seq_model == "None" and self.amb_model == "None":
            if self.bio_model == "ProteinBursty":
                return "protein"
            if self.bio_model == "Custom":
                return "custom"
        return None

    def _backend(self, samp):
        """Decide where an evaluation runs: ``(kind, family, samp)``.

        ``kind`` is ``"gpu"``, ``"rust"`` or ``"python"``; ``family`` is the
        kernel family from ``_kernel_family`` (``None`` when the Python path
        must be used); ``samp`` is the sampling vector to hand to a kernel: a
        list of log10 sampling parameters for the Poisson and Bernoulli models,
        ``None`` when no sequencing transform applies.  Inference passes a zero
        vector rather than ``None`` for noise-free models; both are accepted.
        Only the Python path handles a missing ``samp`` for a noise model (it
        raises the informative error) or a non-zero ``samp`` for a noise-free one.
        """
        samp_list = None
        if self.seq_model in ("Poisson", "Bernoulli"):
            if samp is None:
                return _Backend("python", None, None)
            samp_list = np.asarray(samp, dtype=float).ravel().tolist()
        elif not _samp_is_null(samp):
            return _Backend("python", None, None)
        family = self._kernel_family()
        if family is None:
            kind = "python"
        elif family == "2d" and _HAS_GPU:
            kind = "gpu"
        elif _HAS_RUST:
            kind = "rust"
        else:
            kind = "python"
        return _Backend(kind, family, samp_list)

    def _check_pss_args(self, p, limits):
        """Validate and normalise the (p, limits) arguments of eval_model_pss.

        Returns ``(p, limits)`` with ``p`` a 1-D float ndarray and ``limits``
        a list of Python ints, raising ``ValueError`` with a clear message
        instead of letting a shape mismatch surface as an opaque numpy or
        Rust error.
        """
        p = np.asarray(p, dtype=float)
        n_total = self.get_num_params()          # biological + ambient parameters
        n_amb = {"None": 0, "Equal": 1, "Unequal": 2}.get(self.amb_model, 0)
        if p.ndim != 1 or p.shape[0] != n_total:
            raise ValueError(
                f"{self.bio_model}/{self.seq_model}/{self.amb_model}: expected {n_total} log10 "
                f"parameters ({n_total - n_amb} biological"
                f"{' + ' + str(n_amb) + ' ambient' if n_amb else ''}), got shape {p.shape}"
            )
        if not np.all(np.isfinite(p)):
            raise ValueError("parameters must be finite log10 values; got {}".format(p))
        limits_arr = np.atleast_1d(np.asarray(limits))
        n_species = len(self.model_modalities) + (1 if self.amb_model != "None" else 0)
        if limits_arr.ndim != 1 or limits_arr.shape[0] != n_species:
            raise ValueError(
                f"{self.bio_model}: limits must have one grid size per species "
                f"({n_species} for modalities {self.model_modalities}"
                f"{' + ambient' if self.amb_model != 'None' else ''}), got {list(limits_arr)}"
            )
        if np.any(limits_arr < 1) or np.any(limits_arr != np.round(limits_arr)):
            raise ValueError(f"limits must be positive integers, got {list(limits_arr)}")
        return p, [int(x) for x in limits_arr]

    def param_names(self):
        """Plain-text parameter names, one per entry of the parameter vector.

        ``param_str`` holds TeX labels for plotting; this strips the log prefix
        and the math delimiters (``b``, ``beta``, ``gamma``, or the network's
        own names for ``Custom``).
        """
        out = []
        for s in self.param_str:
            s = s.replace(r"\log_{10}", "").replace("$", "").replace("\\,", "").replace("\\", "")
            out.append(s.strip())
        return out

    def thread_state(self):
        """The calling thread's pinned schedule and tolerance override for
        this model, to be installed in worker threads with
        :meth:`carry_thread_state`."""
        return _tls_snapshot(self)

    def carry_thread_state(self, state):
        """Context manager: evaluate on another thread's :meth:`thread_state`
        (pool workers of a stencil or Hessian replaying the caller's pinned
        schedule at the caller's tolerance)."""
        return _tls_carry(self, state)

    def _schedule_is_pinned(self):
        return (getattr(self, "_fixed_protein_schedule", None) is not None
                or getattr(self, "_fixed_custom_schedule", None) is not None
                or getattr(self, "_fixed_chromatin_schedule", None) is not None)

    @contextlib.contextmanager
    def search_tolerance(self):
        """Run the block at the integrator's search tolerance (Chromatin:
        ``chromatin_search_tol`` instead of ``chromatin_tol``).  The optimiser
        searches here and reports its optimum at the fine tolerance; with the
        Richardson extrapolation the coarse setting is still accurate to about
        1e-6 on the distribution at half the steps.  No-op for other models."""
        if self.bio_model != "Chromatin":
            yield
            return
        previous = _tls_get(self, "chromatin_tol")                      # this thread's enclosing block, if any
        _tls_set(self, "chromatin_tol", float(self.chromatin_search_tol))
        try:
            yield
        finally:
            _tls_set(self, "chromatin_tol", previous)

    def has_search_tolerance(self):
        return self.bio_model == "Chromatin" and self.chromatin_search_tol != self._chromatin_tol

    def null_sampling_grid(self):
        """``(samp_lb, samp_ub, gridsize)`` for ``seq_model="None"``: zeros, plus
        the dropout grid for the Chromatin model's configuration axis."""
        if self.bio_model == "Chromatin":
            return list(self.seq_bounds['samp_lb']), list(self.seq_bounds['samp_ub']), list(self.seq_bounds['gridsize'])
        return [0, 0], [0, 0], [1, 1]

    @contextlib.contextmanager
    def pinned_schedule(self, p, limits, samp=None):
        """Replay the step schedule adapted at ``p`` for every evaluation in the block.

        The protein and custom-network integrators choose their steps by error
        control, so two evaluations at nearby parameters use different
        discretisations and differ by up to the integrator's tolerance.  A
        finite difference across such evaluations is polluted by that; this
        evaluates the base point once and replays its schedule at every stencil
        point, which makes the differences clean to rounding.  Nested blocks
        and the finite-difference gradient keep an outer pin.  No-op for the
        other models.
        """
        if self.bio_model not in ("ProteinBursty", "Custom", "Chromatin") or self._schedule_is_pinned():
            yield
            return
        try:
            self.eval_model_pss(np.asarray(p, dtype=float), limits, samp)
            if self.bio_model == "ProteinBursty":
                self._fixed_protein_schedule = getattr(self, "_last_protein_schedule", None)
            elif self.bio_model == "Chromatin":
                self._fixed_chromatin_schedule = getattr(self, "_last_chromatin_schedule", None)
            else:
                self._fixed_custom_schedule = getattr(self, "_last_custom_schedule", None)
            yield
        finally:
            self._fixed_protein_schedule = None
            self._fixed_custom_schedule = None
            self._fixed_chromatin_schedule = None

    def check_identifiability(self, p, limits, samp=None, h=1e-4, rtol=1e-3, warn=True):
        """Test whether every parameter direction changes the steady state.

        Builds the Jacobian of the distribution over the grid with respect to
        the log10 parameters by central finite differences, takes its singular
        value decomposition and reports the directions whose singular value is
        below ``rtol`` times the largest. A direction with singular value zero
        is structurally non-identifiable: moving along it leaves the
        distribution, and therefore the likelihood of any data, unchanged. A
        small but nonzero value marks a direction the data will constrain
        poorly at this point in parameter space.

        This is a local test at one parameter vector. It catches exact
        degeneracies (all rates scaled together, duplicate reactions, a delay
        with nothing to delay) wherever it is evaluated, and value-dependent
        ones (the protein model's ``k_p``/``gamma_p`` valley) only when
        evaluated in the regime where they occur.

        Parameters
        ----------
        p : array-like
            log10 parameters at which to test.
        limits : array-like of int
            grid size per species, as for :meth:`eval_model_pss`.
        samp : array-like or None, optional
            log10 sampling parameters when the sequencing model needs them.
        h : float, optional
            finite-difference step in log10 units.
        rtol : float, optional
            relative singular-value threshold below which a direction is
            reported as flat.
        warn : bool, optional
            emit an :class:`monod.reaction_network.IdentifiabilityWarning`
            naming each flat direction.

        Returns
        -------
        dict
            ``identifiable`` (bool), ``param_names``, ``singular_values``
            (relative to the largest, descending), ``directions`` (rows are the
            right singular vectors in the same order) and ``flat`` (a list of
            ``(relative singular value, {name: component})`` for each direction
            below the threshold).
        """
        p = np.asarray(p, dtype=float)
        names = self.param_names()
        if p.shape != (len(names),):
            raise ValueError(f"expected {len(names)} parameters {names}, got shape {p.shape}")

        def f(x):
            return np.asarray(self.eval_model_pss(x, limits, samp), dtype=float).ravel()

        # Replay the base point's integration schedule at every stencil point:
        # otherwise step-size changes swamp the small differences being measured.
        cols = []
        with self.pinned_schedule(p, limits, samp):
            for i in range(p.size):
                e = np.zeros_like(p)
                e[i] = h
                cols.append((f(p + e) - f(p - e)) / (2 * h))
        J = np.column_stack(cols)
        if not np.all(np.isfinite(J)):
            raise ValueError("the steady state is not finite at this point; cannot assess identifiability")
        _, S, Vt = np.linalg.svd(J, full_matrices=False)
        rel = S / S[0] if S[0] > 0 else np.zeros_like(S)
        flat = []
        for k in range(len(rel) - 1, -1, -1):
            if rel[k] < rtol:
                flat.append((float(rel[k]), dict(zip(names, Vt[k].tolist()))))
            else:
                break
        result = dict(identifiable=not flat, param_names=names,
                      singular_values=rel, directions=Vt, flat=flat)
        if warn and flat:
            for rel_sv, direction in flat:
                warnings.warn(
                    f"{self.bio_model} model: the steady state is insensitive to the parameter direction "
                    f"{_format_direction(direction)} (relative singular value {rel_sv:.1e}, threshold {rtol:.0e}); "
                    "this combination of parameters cannot be inferred from the data.",
                    IdentifiabilityWarning, stacklevel=2,
                )
        return result

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

        p, limits = self._check_pss_args(p, limits)

        if self.bio_model == "Chromatin":
            return self._eval_model_pss_chromatin(p, limits, samp)

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
        be = self._backend(samp)
        if be.kind == "gpu":
            pss = _eval_model_pss_gpu(
                self.bio_model, p.tolist(), limits, float(self.fixed_quad_T),
                int(self.quad_order), _GPU_DEVICE, samp=be.samp, seq_model=self.seq_model,
            )
            return pss.reshape(int(limits[0]), int(limits[1])).squeeze()
        if be.kind == "rust" and be.family == "2d":
            pss_flat = _mc.eval_model_pss_2d(
                self.bio_model, p.tolist(), [int(x) for x in limits],
                float(self.fixed_quad_T), int(self.quad_order), be.samp, seq_model=self.seq_model,
            )
            return np.array(pss_flat).reshape(int(limits[0]), int(limits[1])).squeeze()
        if be.kind == "rust" and be.family == "amb":
            n_amb = 1 if self.amb_model == "Equal" else 2
            pss_flat = _mc.eval_model_pss_2d(
                self.bio_model, p[:-n_amb].tolist(), [int(x) for x in limits],
                float(self.fixed_quad_T), int(self.quad_order), be.samp, self.amb_model,
                p[-n_amb:].tolist(), seq_model=self.seq_model,
            )
            return np.array(pss_flat).reshape([int(l) for l in limits]).squeeze()
        if be.kind == "rust" and be.family == "protein":
            fixed = getattr(self, "_fixed_protein_schedule", None)
            pss_flat, steps = _mc.eval_model_pss_protein_bursty_sched(
                p.tolist(), [int(x) for x in limits], bool(self.fit_unspliced),
                float(self.protein_limit), _PROTEIN_FUDGE[0], _PROTEIN_FUDGE[1],
                None if fixed is None else [float(x) for x in np.asarray(fixed, dtype=float)],
            )
            self._last_protein_schedule = np.asarray(steps, dtype=float)
            return np.array(pss_flat).reshape([int(l) for l in limits]).squeeze()
        if be.kind == "rust" and be.family == "custom":
            net  = self.network
            topo = net.rust_topology()
            params_d = net._param_map(p)          # name → linear float (frequency form, normalised rate = 1)
            params_linear = [float(params_d[name]) for name in topo["param_names"]]
            fixed = getattr(self, "_fixed_custom_schedule", None)
            taus = [1.0 / float(params_d[name]) for name in net.delay_params]
            re_e, im_e, mx_shape, steps = _mc.eval_custom_network_pgf(
                len(net.species), [int(x) for x in limits],
                topo["rxn_kinds"], topo["rxn_rate_idxs"], topo["rxn_extra1"], topo["rxn_extra2"],
                topo["prod_sp"], topo["prod_st"], topo["prod_off"],
                params_linear,
                None if fixed is None else [float(x) for x in np.asarray(fixed, dtype=float)],
                None if fixed is None else list(getattr(fixed, "breakpoints", ())),
            )
            self._last_custom_schedule = StepSchedule(steps, taus)
            exp_phi = (np.array(re_e) + 1j * np.array(im_e)).reshape(mx_shape)
            pss = np.abs(irfftn(exp_phi, s=[int(x) for x in limits]))
            return (pss / pss.sum()).squeeze()

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

        g = self._apply_seq_transform(g, samp)
            
        gf = self.eval_model_pgf(p, g) # this gf is actually phi
        gf = np.exp(gf)
        gf = gf.reshape(tuple(mx))
        Pss = irfftn(gf, s=tuple(limits))
        Pss = np.abs(Pss) / np.sum(np.abs(Pss))
        Pss = Pss.squeeze()
        return Pss

    def _apply_seq_transform(self, g, samp):
        """Technical-noise transform of the PGF arguments.  ``samp`` holds one
        log10 sampling parameter per sequenced modality; modalities beyond
        ``len(samp)`` (protein in ProteinBursty) are left untouched, as is a
        modality whose entry is 0 (the null vector inference passes for
        ``seq_model="None"``).  The mesh is cached, so a copy is transformed."""
        if self.seq_model in ("Poisson", "Bernoulli"):
            if samp is None:
                raise ValueError(
                    f"seq_model={self.seq_model!r} requires samp (log10 sampling parameters, "
                    "one per sequenced modality); got None."
                )
            samp = np.asarray(samp, dtype=float).ravel()
            n_s = min(len(samp), g.shape[0])
            lam = np.power(10.0, samp[:n_s])[:, None]
            g = np.array(g, copy=True)
            if self.seq_model == "Poisson":
                g[:n_s] = np.exp(lam * g[:n_s]) - 1
            else:
                g[:n_s] = g[:n_s] * lam
        elif self.seq_model == "None":
            pass
        else:
            raise ValueError(
                "Please select a technical noise model from {}.".format(
                    self.available_seqmodels
                )
            )
        return g

    def chromatin_system(self, p):
        """``(H, A, C)`` of the Chromatin model at log10 parameters ``p``
        (``A`` holds mean production rates; see :meth:`chromatin_burst`)."""
        from . import chromatin as _chromatin
        n = self.n_sites
        k = _chromatin.n_rule_params(self.rule, n)
        lin = np.power(10.0, np.asarray(p, dtype=float))
        k_on, eps, rates, beta, gamma = lin[:n], float(lin[n]), lin[n + 1:n + 1 + k], float(lin[n + 1 + k]), float(lin[n + 2 + k])
        return _chromatin.nascent_mature_system(k_on, 1.0, eps, self.rule, beta, gamma, periodic=self.periodic, rule_params=rates)

    def chromatin_burst(self, p):
        """Mean burst size at log10 parameters ``p`` (the last entry) for a
        bursty Chromatin model, else None."""
        if not getattr(self, "bursty", False):
            return None
        return float(10.0 ** float(np.asarray(p, dtype=float)[-1]))

    def chromatin_dropout(self, samp):
        """Per-open-site dropout probability from the sampling vector (its third
        entry, log10), or None when absent or null."""
        if samp is None:
            return None
        samp = np.asarray(samp, dtype=float).ravel()
        if samp.size < 3 or samp[2] == 0:
            return None
        return float(10.0 ** samp[2])

    def _eval_model_pss_chromatin(self, p, limits, samp):
        """Joint distribution over (unspliced, spliced, configuration): the
        chromatin kernel on the RNA mesh, transformed by the sequencing model,
        then dropout on the configuration axis."""
        from . import chromatin as _chromatin
        n = self.n_sites
        N = 2 ** n
        if int(limits[2]) != N:
            raise ValueError(f"Chromatin: the configuration axis must have {N} = 2**n_sites states, got limits {limits}")
        lu, ls = int(limits[0]), int(limits[1])
        mx = (lu, ls // 2 + 1)
        g = _build_mesh((lu, ls), mx)
        g = self._apply_seq_transform(g, samp)
        H, A, C = self.chromatin_system(p)
        G, sched = _chromatin.eval_joint_pgf(H, A, C, g, tol=self.chromatin_tol,
                                             fixed_schedule=getattr(self, "_fixed_chromatin_schedule", None),
                                             burst=self.chromatin_burst(p))
        self._last_chromatin_schedule = sched
        pss = _chromatin.invert_joint_pgf(G, mx, (lu, ls))              # (N, lu, ls)
        p_drop = self.chromatin_dropout(samp)
        if p_drop is not None:
            pss = _chromatin.apply_dropout(pss, n, p_drop)
        return np.moveaxis(pss, 0, -1)                                  # (lu, ls, N)

    def eval_model_pss_batch(self, params_list, limits_list, samp_list=None, num_threads=None):
        """Evaluate PSS for N genes in one batched call.

        On a CUDA device with float64 support the genes are evaluated on the
        GPU, grouped by grid size, with one batched inverse FFT per group.
        Otherwise all N evaluations run in parallel on the Rust thread pool
        (rayon), with the GIL released for the entire computation.  Falls back
        to sequential `eval_model_pss` calls when neither is available.

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
        n = len(params_list)
        if samp_list is None:
            samp_list = [None] * n

        _model_ok = self._kernel_family() == "2d"
        if _HAS_GPU and _model_ok and (self.seq_model == "None" or all(s is not None for s in samp_list)):
            return _eval_model_pss_gpu_batch(
                self.bio_model, params_list, limits_list, float(self.fixed_quad_T),
                int(self.quad_order), _GPU_DEVICE,
                samp_list=None if self.seq_model == "None" else samp_list,
                seq_model=self.seq_model,
            )
        _batch_ok = _HAS_RUST and _model_ok
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
        if self.bio_model in _CLOSED_FORM_MODELS:
            gf = _closed_form_pgf(self.bio_model, g[0], g[1], list(p))
        elif self.bio_model in _QUADRATURE_MODELS:
            b, beta, gamma = p
            g2 = np.asarray(g)[:, :, None]                                   # a trailing time axis
            fun = lambda x: _burst_integrand(self.bio_model, _burst_characteristic(x, g2[0], g2[1], b, beta, gamma))
            if self.quad_method == "quad_vec":
                gf = scipy.integrate.quad_vec(fun, 0, _quadrature_horizon(self.quad_vec_T, beta, gamma))[0]
            elif self.quad_method == "fixed_quad":
                gf = scipy.integrate.fixed_quad(fun, 0, _quadrature_horizon(self.fixed_quad_T, beta, gamma), n=self.quad_order)[0]
            else:
                raise ValueError("Please use one of the specified quadrature methods.")

        elif self.bio_model == "ProteinBursty":  # bursty production
            gf = self.protein_pgf(g, p)

        elif self.bio_model == "Custom":
            gf = self.network.eval_pgf(
                p_,
                [g[i] for i in range(len(self.network.species))],
                fixed_schedule=getattr(self, "_fixed_custom_schedule", None),
            )
            self._last_custom_schedule = self.network.last_schedule

        else:
            raise ValueError(
                "Please select a biological noise model from {}.".format(
                    self.available_biomodels
                )
            )
        return gf  # this is the log-generating function
    
    def protein_pgf(self, g, p):
        """Evaluate the log generating function of the protein model.

        Characteristics in the PGF variables v = u0 + 1 (unspliced) and
        w = u1 + 1 (spliced), with the protein characteristic u2 = g2 e^{-gamma_p t}
        solved analytically::

            v' = beta (w - v)
            w' = a(t) w + gamma,     a(t) = -gamma + k_p g2 e^{-gamma_p t}
            phi = int_0^inf b u0 / (1 - b u0) dt,   u0 = v - 1

        The system is linear and the integral of a(t) is known in closed form,
        so every step propagates the homogeneous part of w exactly and treats the
        forcing and the v-coupling with the step-average coefficient (exact for
        constant coefficients): the stiff rates never limit the step.  Simpson's
        rule on two half-steps accumulates phi; the step size is controlled by
        comparing one full step with the two half-steps, taking the maximum error
        over all grid points so the whole grid shares one schedule.  The accepted
        steps are recorded in ``_last_protein_schedule``; setting
        ``_fixed_protein_schedule`` replays a given sequence (finite-difference
        stencils use the base point's).  Mirrors ``monod_core::protein_pgf_sched``.

        Parameters
        ----------
        g: np.ndarray
            complex PGF arguments (3, n_points), adjusted for sampling.
        p: float array of length 5
            (b, beta, gamma, k_p, gamma_p)

        Returns
        -------
        phi: np.ndarray
            log gf.
        """
        b, beta, gamma, k_p, gamma_p = (float(x) for x in p)
        g = np.asarray(g, dtype=np.complex128)
        g0, g1, g2 = g[0], g[1], g[2]

        def phi1r(z):
            return np.expm1(z) / z if abs(z) > 1e-4 else 1.0 + z * 0.5 + z * z * (1.0 / 6.0)

        def f_int(v):
            u = v - 1.0
            return u * b / (1.0 - u * b)

        def consts(t, h):
            # per-step scalars shared by all grid points (mirrors ProteinStepConsts)
            return (h, -gamma * h,
                    (k_p / gamma_p) * np.exp(-gamma_p * t) * (-np.expm1(-gamma_p * h)),
                    np.exp(-beta * h), h * phi1r(-beta * h))

        def step(c, v, w):
            h, dar, dac, eb, i0 = c
            da = dar + g2 * dac          # exact integral of a(t) over the step
            abar = da / h
            e = np.exp(da)
            big = np.abs(da) > 1e-4
            p1 = np.empty_like(da)
            p1[big] = (e[big] - 1.0) / da[big]
            zs = da[~big]
            p1[~big] = 1.0 + zs * 0.5 + zs * zs * (1.0 / 6.0)
            w_new = e * w + p1 * (gamma * h)
            s_ = abar + beta
            z = s_ * h
            big = np.abs(z) > 1e-3
            i1 = np.empty_like(z)
            i1[big] = (e[big] - eb) / s_[big]
            zs = z[~big]
            i1[~big] = (1.0 + zs * 0.5 + zs * zs * (1.0 / 6.0)) * (h * eb)
            goa = gamma / abar
            v_new = v * eb + ((w + goa) * i1 - goa * i0) * beta
            return v_new, w_new

        lam_fast = max(beta, gamma + 2.0 * k_p, gamma_p)
        lam_slow = min(beta, gamma, gamma_p)
        t_end = np.log(1.0 / _PROTEIN_ODE_TOL) / lam_slow + max(0.0, np.log(2.0 * k_p / gamma)) / gamma_p

        fixed = getattr(self, "_fixed_protein_schedule", None)
        v = g0 + 1.0
        w = g1 + 1.0
        phi = np.zeros_like(v)
        t = 0.0
        h = _PROTEIN_ODE_C0 / lam_fast
        h_min = 1e-12 / lam_fast
        steps = []
        while True:
            if fixed is not None:
                if len(steps) >= len(fixed):
                    break
                h = float(fixed[len(steps)])
            else:
                if t >= t_end or len(steps) >= _PROTEIN_ODE_MAX_STEPS:
                    break
                h = min(h, t_end - t)
            f0 = f_int(v)
            c_half1 = consts(t, 0.5 * h)
            c_half2 = consts(t + 0.5 * h, 0.5 * h)
            vm, wm = step(c_half1, v, w)
            v2, w2 = step(c_half2, vm, wm)
            fm = f_int(vm)
            f2 = f_int(v2)
            simpson = (f0 + fm * 4.0 + f2) * (h / 6.0)
            fac = 1.0
            if fixed is None:
                trap = (f0 + fm * 2.0 + f2) * (h / 4.0)
                v1, _ = step(consts(t, h), v, w)
                err = float(np.max(np.exp(phi.real) * (np.abs(simpson - trap) + b * h * np.abs(v2 - v1) / 3.0)))
                if err > _PROTEIN_ODE_TOL and h > h_min:
                    h *= max(0.2, 0.9 * (_PROTEIN_ODE_TOL / err) ** (1.0 / 3.0))
                    continue
                fac = min(2.0, max(0.2, 0.9 * (_PROTEIN_ODE_TOL / max(err, 1e-300)) ** (1.0 / 3.0)))
            v, w = v2, w2
            phi = phi + simpson
            t += h
            steps.append(h)
            h *= fac

        self._last_protein_schedule = np.asarray(steps, dtype=float)
        return phi

    def cir_intfun(self, x, g, b, beta, gamma):
        """The CIR log-PGF integrand ``1 - sqrt(1 - 4U)`` at times ``x`` (the
        half in the log-PGF is applied by the caller); ``g`` the complex mesh
        arguments.  See :func:`_burst_characteristic`."""
        g = np.asarray(g)[:, :, None]
        return 1 - np.sqrt(1 - 4 * _burst_characteristic(x, g[0], g[1], b, beta, gamma))

    def burst_intfun(self, x, g, b, beta, gamma):
        """The bursty log-PGF integrand ``U / (1 - U)`` at times ``x``."""
        g = np.asarray(g)[:, :, None]
        return _burst_integrand("Bursty", _burst_characteristic(x, g[0], g[1], b, beta, gamma))

    def get_MoM(self, moments, lb_log, ub_log, samp=None, hist=None, limits=None):
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
        hist, limits: optional
            the gene's histogram ``(coords, freqs)`` and grid, which the
            Chromatin model needs: its chain parameters come from the
            configuration marginal and its rates from the conditional nascent
            means per configuration (see :mod:`monod.chromatin_fit`).

        Returns
        -------
        x0: np.ndarray
            log10 biological parameter estimates.
        """
        if self.bio_model == "Chromatin":
            mid = 0.5 * (np.asarray(lb_log, float) + np.asarray(ub_log, float))[: self.get_num_params()]
            if hist is None or limits is None:
                return mid
            from . import chromatin_fit as _cf
            try:
                x0 = _cf.chromatin_start(self, hist, limits, samp, lb_log, ub_log)
            except Exception as exc:                      # a start is a convenience, never a failure
                log.info("Chromatin moments start failed (%s); starting from the middle of the box", exc)
                return mid
            return np.clip(x0, np.asarray(lb_log, float), np.asarray(ub_log, float))
        if self.bio_model == "Custom":
            # No moment estimator for an arbitrary network: start every fit from
            # the middle of the parameter box (log scale).
            return 0.5 * (np.asarray(lb_log, float) + np.asarray(ub_log, float))[: self.get_num_params()]

        lb = 10**lb_log
        ub = 10**ub_log
        if self.seq_model in ("Poisson", "Bernoulli"):
            if samp is None:
                raise ValueError(
                    "get_MoM requires samp (log10 sampling parameters) for seq_model={!r}.".format(
                        self.seq_model
                    )
                )
            samp = 10 ** np.asarray(samp, dtype=float)

        # These can be defined per model (just happen to be shared by multiple models here).
        U_var, U_mean = moments['MOM_unspliced_var'], moments['MOM_unspliced_mean']
        S_var, S_mean = moments['MOM_spliced_var'], moments['MOM_spliced_mean']

        if self.bio_model == "Bursty" or self.bio_model == "CIR":
            b = np.clip(self._mom_burst_size(U_var, U_mean, samp), lb[0], ub[0])
            beta = b / U_mean
            gamma = b / S_mean
            x0 = np.asarray([b, beta, gamma])
            
        elif self.bio_model == "ProteinBursty":
            U_var, U_mean, S_mean, P_mean, UP_covar = moments["MOM_unspliced_var"], moments["MOM_unspliced_mean"], moments["MOM_spliced_mean"], moments["MOM_protein_mean"], moments["MOM_cov_unspliced_protein"]
            b = np.clip(self._mom_burst_size(U_var, U_mean, samp), lb[0], ub[0])
            beta = b / U_mean
            gamma = b / S_mean

            # Define r = k_p/gamma_p
            r = P_mean*gamma/b
            C = UP_covar
            gamma_p = C*(beta + gamma)*beta/(b**2*r - C*(beta + gamma))
            gamma_p = np.clip(gamma_p, lb[4], ub[4])
            k_p = np.clip(r*gamma_p, lb[3], ub[3])
            x0 = np.asarray([b, beta, gamma, k_p, gamma_p])

        elif self.bio_model == "Delay":
            b = np.clip(self._mom_burst_size(U_var, U_mean, samp), lb[0], ub[0])
            beta = b / U_mean
            tauinv = b / S_mean
            x0 = np.asarray([b, beta, tauinv])

        elif self.bio_model == "DelayedSplicing":
            # the nascent Fano excess splits between the burst and the delay
            b = np.clip(self._mom_burst_size(U_var, U_mean, samp, half=True), lb[0], ub[0])
            tauinv = b / U_mean
            gamma = b / S_mean
            x0 = np.asarray([b, tauinv, gamma])

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

    def _mom_burst_size(self, U_var, U_mean, samp, half=False):
        """Moments estimate of the burst size from the nascent Fano factor,
        corrected for the sequencing model (``samp`` linear); ``half`` for the
        DelayedSplicing model, where the delay takes half of the excess.  A
        degenerate gene (no nascent counts, or a non-finite ratio) gives 1."""
        b = (U_var / U_mean - 1) if U_mean > 0 else 1.0
        if not np.isfinite(b):
            b = 1.0
        if half:
            b = b / 2
        if self.seq_model == "Bernoulli":
            b = b / samp[0] / (2 if half else 1)
        elif self.seq_model == "Poisson":
            b = b / samp[0] - (0.5 if half else 1)
        return b

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
        # means and the extrinsic CV2 per species, per family
        if self.bio_model == "Constitutive":
            beta, gamma = p
            mu, noise_ext = [1 / beta, 1 / gamma], [0, 0]
        elif self.bio_model in ("Bursty", "CIR"):
            b, beta, gamma = p
            mu, noise_ext = [b / beta, b / gamma], [beta, beta * gamma / (beta + gamma)]
        elif self.bio_model == "Extrinsic":
            alpha, beta, gamma = p
            mu, noise_ext = [alpha / beta, alpha / gamma], [1 / alpha, 1 / alpha]
        else:
            raise NotImplementedError(f"eval_model_noise is defined for Constitutive, Bursty, CIR and Extrinsic, not {self.bio_model}")
        mu = np.asarray(mu, dtype=float)
        noise_int = 1 / mu
        noise_ext = np.asarray(noise_ext, dtype=float)

        if self.seq_model == "None":
            noise = noise_int + noise_ext
            return (noise_int / noise, noise_ext / noise)
        elif self.seq_model == "Bernoulli":
            samp_ = 10**samp
            noise = noise_int / samp_ + noise_ext
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
