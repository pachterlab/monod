"""Locate the compiled ``monod_core`` extension, ignoring look-alikes.

When Python runs from the repository root, the ``monod_core/`` *source*
directory is importable as an empty namespace package, so a bare
``import monod_core`` succeeds even though no extension is installed.  Every
module in monod therefore goes through :func:`load`, which only accepts a
module that actually exposes the Rust entry points.
"""


def load():
    try:
        import monod_core as mc
    except ImportError:
        return None
    if not all(hasattr(mc, name) for name in ("eval_model_pss_2d", "searchdata_from_arrays", "make_state_dist")):
        return None
    return mc


_mc = load()
HAS_RUST = _mc is not None
