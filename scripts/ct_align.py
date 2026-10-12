"""FL4-C C0: independent check of the teacher-canvas -> native-canvas offset with CT itself (no ink models).

    ct_align.py HI_ZARR_URL LEVEL NATIVE.zarr OUT.json [--max-shift 30]

HI_ZARR_URL: the hi-res surface volume the teacher map was predicted on (2.215 / 2.401 um canvas, s3:// path);
its pyramid level LEVEL, middle slice (the sheet), is resized (INTER_AREA) onto the native (H, W) grid. The native
image is the mean of layers 13-15 of the 28-layer native volume. Both are high-passed (sigma 4 px) inside their
footprints, and the NCC peak over |dy|, |dx| <= max-shift is reported with the convention of align116.py:
native(y, x) ~ hires(y + dy, x + dx).
"""
import argparse, json, sys, os
import numpy as np, cv2, zarr, s3fs
from scipy.ndimage import gaussian_filter

ap = argparse.ArgumentParser(); ap.add_argument('url'); ap.add_argument('level', type=int); ap.add_argument('native'); ap.add_argument('out')
ap.add_argument('--max-shift', type=int, default=30)
a = ap.parse_args()
fs = s3fs.S3FileSystem(anon=True)
hz = zarr.open_array(store=zarr.storage.FsspecStore(fs, path=a.url.replace('s3://', '') + f'/{a.level}', read_only=True), mode='r')
hi = np.asarray(hz[hz.shape[0] // 2]).astype(np.float32)
nz = zarr.open(a.native, mode='r'); nz = nz if hasattr(nz, 'shape') else nz['0']
nat = np.asarray(nz[13:16]).astype(np.float32).mean(0)
Hh, Ww = nat.shape
hi = cv2.resize(hi, (Ww, Hh), interpolation=cv2.INTER_AREA)


def hp(x, m, s=4.0):
    w = gaussian_filter(m.astype(np.float32), s)
    return np.where(m, x - gaussian_filter(x * m, s) / np.maximum(w, 1e-3), 0).astype(np.float32)


mh, mn = hi > 0, nat > 0
A, B = hp(hi, mh), hp(nat, mn)
m = a.max_shift
r = cv2.matchTemplate(A, B[m:-m, m:-m], cv2.TM_CCOEFF_NORMED)
iy, ix = np.unravel_index(np.argmax(r), r.shape)
rec = dict(url=a.url, level=a.level, native=a.native, shift=[int(iy - m), int(ix - m)], ncc=float(r[iy, ix]),
           ncc_at_zero=float(r[m, m]), second_best=float(np.sort(r.ravel())[-30]))
json.dump(rec, open(a.out, 'w'), indent=1)
print(os.path.basename(a.native), 'CT shift', rec['shift'], f"NCC {rec['ncc']:.3f} (at 0 {rec['ncc_at_zero']:.3f})")
