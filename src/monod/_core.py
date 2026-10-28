"""Locate the compiled ``monod_core`` extension, ignoring look-alikes.

When Python runs from the repository root, the ``monod_core/`` *source*
directory is importable as an empty namespace package, so a bare
``import monod_core`` succeeds even though no extension is installed.  Every
module in monod therefore goes through :func:`load`, which only accepts a
module that actually exposes the Rust entry points.

The module also holds the few utilities every other module needs (the
library logger, directory creation, densifying, the sampling grid), so that
``monod.cme_toolbox`` does not import the AnnData stack to get at them.
"""
import logging
import os

import numpy as np
from scipy.sparse import issparse

# Library logger.  monod does not configure handlers or touch the root logger;
# messages propagate to whatever logging setup the application has.  To see
# progress messages: ``logging.getLogger("monod").setLevel(logging.INFO)`` after
# ``logging.basicConfig()``.
log = logging.getLogger("monod")


def make_dir(dir_string):
    """Create *dir_string* (and any missing parents); a no-op if it already exists."""
    try:
        os.makedirs(dir_string, exist_ok=True)
        log.debug("Directory %s ready.", dir_string)
    except OSError as e:
        log.error("Directory %s could not be created: %s", dir_string, e)
        raise


def to_dense(arr):
    """Return ``arr`` as a dense numpy array, whether it arrived sparse or dense."""
    return arr.toarray() if issparse(arr) else np.asarray(arr)


def _build_sampling_grid(samp_lb, samp_ub, gridsize):
    """Build a flat meshgrid over the sampling-parameter space.

    Returns (grid_values, sampl_vals, n_grid_points) where grid_values is a
    list of flattened per-axis arrays, sampl_vals is a list of (v0, v1, ...)
    tuples, and n_grid_points is the total number of points.
    """
    linspaces = [np.linspace(samp_lb[i], samp_ub[i], gridsize[i]) for i in range(len(gridsize))]
    grid_values = np.meshgrid(*linspaces, indexing="ij")
    grid_values = [g.flatten() for g in grid_values]
    sampl_vals = list(zip(*grid_values))
    return grid_values, sampl_vals, len(grid_values[0])


def step_control(err, h, h_min, tol):
    """Step-size control shared by the adaptive integrators (protein, custom
    network, chromatin), the mirror of the Rust ``control_step``.

    Returns ``(accept, value)``: ``(False, h_new)`` when the step must be
    retried with ``h_new``, ``(True, factor)`` when it is accepted and the next
    step is ``h * factor``.  The floating-point order is part of the contract
    with the Rust kernels: ``0.9 * (tol / err) ** (1/3)``, then max 0.2, then
    min 2.
    """
    if err > tol and h > h_min:
        return False, h * max(0.2, 0.9 * (tol / err) ** (1.0 / 3.0))
    return True, min(2.0, max(0.2, 0.9 * (tol / max(err, 1e-300)) ** (1.0 / 3.0)))


def load():
    try:
        import monod_core as mc
    except ImportError:
        return None
    if not all(hasattr(mc, name) for name in ("eval_model_pss_2d", "searchdata_from_arrays", "make_state_dist")):
        return None
    return mc


_mc = load()
