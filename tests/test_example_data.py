"""combine_chunks reassembles the chunked h5ad files of the monod_examples repository."""
import zlib
from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from monod.extract_data import combine_chunks


def test_combine_chunks_round_trip(tmp_path):
    a = ad.AnnData(X=np.arange(12.0).reshape(4, 3))
    a.var_names = ["g0", "g1", "g2"]
    src = tmp_path / "src.h5ad"
    a.write_h5ad(src)
    raw = src.read_bytes()
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    step = len(raw) // 3 + 1
    for i in range(3):
        (chunk_dir / f"chunk_{i:04d}.gz").write_bytes(zlib.compress(raw[i * step:(i + 1) * step]))
    out = combine_chunks(chunk_dir, tmp_path / "out.h5ad")
    assert Path(out).read_bytes() == raw
    np.testing.assert_array_equal(ad.read_h5ad(out).X, a.X)
    with pytest.raises(FileNotFoundError):
        combine_chunks(tmp_path, tmp_path / "none.h5ad")
