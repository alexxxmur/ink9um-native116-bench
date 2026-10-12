"""FL4-C C1: inputs of the two new 116 keV controls (run once, before the benchmark is frozen).

    bench_prep.py

0009B : native 28-layer volume cropped to the manual-label supervision bbox + 64 px (native grid, CT-registered);
        truth = manual inklabels / supervision on the crop (from labels116.py calib, >= 128 after INTER_AREA + shift).
h0343P: held-out 0343P 20250511003658-tifxyz, whole canvas; truth = teacher (2.215 um map, INTER_AREA + CT shift [7, 7])
        classes of native_bench (freeze_thresholds / classes) - the teacher-as-key upper bound (Nieuwlaar §8 caveat).
Writes ~/data/fl4c/bench/<ctrl>/{ct28.zarr, valid.npy, label.npy, support.npy, teacher.npy (h0343P), meta.json} and the
null volumes ct28_shuf.zarr (seed 0) / ct28_slr.zarr (layer 14) with scripts/fl4/nulls.py.
"""
import json, os, subprocess, sys
import numpy as np, tifffile, zarr

H = os.path.expanduser
R = H('~/data/fl4c'); B = f'{R}/bench'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labels116 as L  # noqa: E402
NULLS = '<repo>/scripts/nulls.py'
PY = H('~/envs/fl3/bin/python')


def nulls(d):
    for kind, extra in (('shuffle', ['--seed', '0']), ('slr', ['--layer', '14'])):
        out = f'{d}/ct28_{"shuf" if kind == "shuffle" else kind}.zarr'
        if not os.path.exists(out):
            subprocess.run([PY, NULLS, f'{d}/ct28.zarr', out, kind] + extra, check=True)


# 0009B
n = '0009B_20250919125754'; d = f'{B}/0009B'; os.makedirs(d, exist_ok=True)
ink, sup, val = np.load(f'{R}/labels/0009B_native_masks.npy')
valid = L.valid_of(n)
ys, xs = np.nonzero(sup)
y0, y1 = max(0, ys.min() - 64), min(sup.shape[0], ys.max() + 65)
x0, x1 = max(0, xs.min() - 64), min(sup.shape[1], xs.max() + 65)
if not os.path.exists(f'{d}/ct28.zarr'):
    subprocess.run([PY, NULLS, f'{R}/volumes/native116/{n}.zarr', f'{d}/ct28.zarr', 'crop', '--box', f'{y0},{y1},{x0},{x1}'], check=True)
c = (slice(y0, y1), slice(x0, x1))
v = valid[c]; s = sup[c] & v
np.save(f'{d}/valid.npy', v); np.save(f'{d}/support.npy', s); np.save(f'{d}/label.npy', ink[c] & s)
np.save(f'{d}/validation.npy', val[c] & s)
json.dump(dict(segment=n, box=[int(y0), int(y1), int(x0), int(x1)], shift=L.ct_shift(n), n_support=int(s.sum()),
               n_ink=int((ink[c] & s).sum()), truth='manual labels 20260918 (2.401 um) on the native grid'), open(f'{d}/meta.json', 'w'), indent=1)
nulls(d)
print('0009B crop', (y1 - y0, x1 - x0), 'support', int(s.sum()), 'ink', int((ink[c] & s).sum()))

# held-out 0343P
n = L.HELD_OUT_0343P; d = f'{B}/h0343P'; os.makedirs(d, exist_ok=True)
if not os.path.exists(f'{d}/ct28.zarr'):
    os.symlink(f'{R}/volumes/native116/{n}.zarr', f'{d}/ct28.zarr')
valid = L.valid_of(n)
t = L.on_grid(tifffile.imread(f'{R}/teacher/{n}.tif'), valid.shape, L.ct_shift(n))
np.save(f'{d}/valid.npy', valid); np.save(f'{d}/teacher.npy', t)
json.dump(dict(segment=n, shift=L.ct_shift(n), truth='2.215 um canonical teacher, native_bench classes'), open(f'{d}/meta.json', 'w'), indent=1)
nulls(d)
print('h0343P', valid.shape)
