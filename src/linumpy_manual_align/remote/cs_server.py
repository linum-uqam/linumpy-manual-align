"""Remote cross-section server script.

This file is uploaded verbatim to the server via SSH and executed there by
``open_remote_slice_reader`` in ``remote_reader.py``.  It is a standalone
Python script — not imported by the package locally.

The script self-deletes immediately on startup so ``/tmp`` never accumulates
stale copies even if the widget crashes; the kernel keeps the process alive
from its in-memory file buffer after the unlink.
"""

# ruff: noqa
import os
import sys
import base64
import io
import numpy as np
from linumpy.io.zarr import read_omezarr

try:
    os.unlink(sys.argv[0])  # self-delete; kernel keeps process alive
except OSError:
    pass  # already deleted by a concurrent reader (harmless)

zarr_path = sys.argv[1]
level = int(sys.argv[2])
vol, sc = read_omezarr(zarr_path, level=level)
arr = np.asarray(vol)  # (Z, Y, X)
nz, ny, nx = arr.shape
scale_str = ",".join(str(s) for s in sc)
print(f"ready {nz} {ny} {nx} {scale_str}", flush=True)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    axis, pos = line.split(",")
    pos = int(pos)
    if axis == "xz":
        img = arr[:, pos, :][::-1]  # (Z, X), flip Z
    else:
        img = arr[:, :, pos][::-1]  # (Z, Y), flip Z
    buf = io.BytesIO()
    np.savez_compressed(buf, img=img.astype("float32"), scale=np.array(sc, dtype="float64"))
    print(base64.b64encode(buf.getvalue()).decode(), flush=True)
