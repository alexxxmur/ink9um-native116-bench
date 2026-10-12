"""FL4-C C0: native 8.64 um / 116 keV surface volumes (31 layers) -> the 0139 native113 layout (28 layers).

    prep116.py RAW.zarr OUT.zarr [--flip]

The 116 keV surface volumes on S3 have 31 layers with the sheet at layer 15 (= shape // 2, villa's convention);
0139 native113 volumes have 28 layers with the sheet at 14. Layers [1, 29) of the 31 keep the sheet at the new
layer 14 with the same 14 / 13 layers before / after as 0139, so Nieuwlaar's label format (28 x H x W, plane 14),
the villa flat loader (surface = shape // 2) and the published inference windows 0:20 3:23 5:25 8:27 apply unchanged.
--flip reverses the depth order first (index 30 - k), for scrolls whose reading side is villa 'reverse'.
Writes OUT.zarr (group, array '0', zarr v2, chunks (28, 128, 128)), OUT.zarr/prep.json and valid.npy next to it
(valid = max_z > 0).
"""
import argparse, json, os
import numpy as np, zarr

ap = argparse.ArgumentParser(); ap.add_argument('raw'); ap.add_argument('out'); ap.add_argument('--flip', action='store_true')
a = ap.parse_args()
z = zarr.open(a.raw, mode='r'); src = z if hasattr(z, 'shape') else z['0']
D, H, W = src.shape
assert D == 31, src.shape
g = zarr.open_group(a.out, mode='w', zarr_format=2)
dst = g.create_array('0', shape=(28, H, W), chunks=(28, 128, 128), dtype=np.uint8, fill_value=0)
valid = np.zeros((H, W), bool)
for y in range(0, H, 1024):
    v = np.asarray(src[:, y:y + 1024, :])
    if a.flip:
        v = v[::-1]
    v = v[1:29]
    dst[:, y:y + 1024, :] = v
    valid[y:y + 1024] = v.max(0) > 0
np.save(os.path.splitext(a.out)[0] + '_valid.npy', valid)
json.dump(dict(source=os.path.realpath(a.raw), layers='[1, 29) of 31', flip=a.flip, shape=[28, H, W],
               valid_frac=float(valid.mean())), open(os.path.join(a.out, 'prep.json'), 'w'))
print(a.out, (28, H, W), 'flip' if a.flip else 'noflip', f'valid {valid.mean():.3f}')
