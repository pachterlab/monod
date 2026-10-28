# Installation

Python 3.9 or newer is required.

```bash
pip install monod monod-core
```

`monod` is the Python package. `monod-core` is an optional accelerator written in Rust: it replaces the expensive parts of a fit (evaluating the model's steady-state distribution, histogramming, the L-BFGS-B optimizer, and goodness-of-fit scoring) with multithreaded native code. Monod detects it at import time and uses it automatically; without it, everything runs in pure Python with identical results and longer run times.

Prebuilt `monod-core` wheels are published for Linux x86_64 and Apple-silicon macOS, for Python 3.9 to 3.12. On other platforms `pip install monod` alone is enough to get started; to build the accelerator yourself see [below](#building-the-rust-accelerator-from-source).

## GPU

If PyTorch is installed and a CUDA device with float64 support is available, the two-species steady-state evaluation runs on the GPU, for all six transcription models with no sequencing noise or with Poisson or Bernoulli noise (the ambiguous-assignment models, the protein model and custom networks stay on the CPU). Apple MPS does not support float64 and is not used. The order of preference is GPU, then the Rust accelerator, then pure Python; no configuration is needed. On a single evaluation the GPU is only modestly faster than the Rust path (about 1.6 ms against 3 ms on a 120 by 80 grid with an RTX 3070 and 24 CPU cores). It pays off when many genes are evaluated at once: `CMEModel.eval_model_pss_batch` groups genes by grid size and evaluates each group in one batched call, about 0.4 ms per gene for 1000 genes against 0.7 ms on the Rust thread pool. The `gpu` notebook in `demos/` shows the detection and the timings on a real dataset.

## Checking what is active

```python
from monod._core import _mc
print("Rust accelerator:", _mc is not None)
```

## Installing from a checkout

```bash
git clone https://github.com/pachterlab/monod.git
cd monod
pip install -e .[test]
python -m pytest tests/ -q      # optional: the test suite runs with or without the accelerator
```

## Building the Rust accelerator from source

Use the prebuilt wheel when one exists for your platform. Build it yourself on platforms without a wheel (Intel Macs, for example) or to work on it.

Requirements: a Rust toolchain (`rustup`), a C compiler, a system HDF5 library discoverable via `pkg-config` or `HDF5_DIR`, and [maturin](https://www.maturin.rs/).

```bash
pip install maturin
cd monod_core
maturin build --release --compatibility linux -i "$(which python)"   # drop --compatibility on macOS
pip install --force-reinstall --no-deps target/wheels/monod_core-*.whl
cd ..
python -c "import monod_core; print('Rust extension OK, meK-Means init:', hasattr(monod_core, 'initialize_q_kmeans'))"
```

Notes:

- The `kmeans` cargo feature (the Rust k-means initializer used by meK-Means) is on by default; building with `--no-default-features` drops it, in which case meK-Means initializes through scikit-learn instead.
- The `ruanndata` crate is fetched from GitHub at a pinned commit by cargo. If you are developing ruanndata alongside monod, see the commented `[patch]` block at the end of `monod_core/Cargo.toml`.
- On Apple Silicon with an x86_64 (Rosetta) Homebrew HDF5, set `HDF5_NO_RUNTIME_CHECK=1` for the build.
- The published wheels link HDF5 statically via the `static-hdf5` cargo feature; `.github/workflows/wheels.yml` has the exact invocation.
- `MONOD_LBFGSB_VERBOSE=1` makes every Rust optimizer print a per-iteration trace to stderr (iteration, function evaluations, skipped curvature pairs, memory size, projected-gradient norm, objective), useful when a fit misbehaves.
