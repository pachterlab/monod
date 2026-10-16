# monod-core

Rust extension for [monod](https://github.com/pachterlab/monod): multithreaded
steady-state PMF evaluation, histogramming, L-BFGS-B fitting, chi-square and
log-likelihood scoring, and the meK-Means initializer. `monod` detects it at
import time and falls back to pure Python when it is missing.

Published wheels are built with the `static-hdf5` feature, so they do not need a
system HDF5 library. To build from source you need a Rust toolchain, a C
compiler and either a system HDF5 (`pkg-config`/`HDF5_DIR`) or `cmake` for the
static build:

```bash
pip install maturin
maturin build --release                      # links the system HDF5
maturin build --release --features static-hdf5   # self-contained
```
