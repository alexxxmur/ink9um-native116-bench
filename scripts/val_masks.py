"""FL4-C C2: a validation region shared by both arms (the native corpus ships no validation masks).

    val_masks.py [w046 w049] [--frac 0.15]

For each listed native-dense 0139 segment: validation = the bottom FRAC of the canvas rows inside the supervision
mask, written as <w>_validation_mask.zarr next to its labels (official native9 layout, plane 14). villa's loader
(segment.py suffix '_validation_mask') turns those patches into validation patches and removes the voxels from
training supervision, identically in both arms (0139nat is in both). Val labels = the 78 keV teacher pseudo-labels.
"""
import argparse, json, os
import numpy as np, zarr
from numcodecs import Blosc

H = os.path.expanduser
L = H('~/data/fl4c/labels/native-dense-0139-78keV')
ap = argparse.ArgumentParser(); ap.add_argument('segs', nargs='*', default=['w046', 'w049']); ap.add_argument('--frac', type=float, default=0.15)
a = ap.parse_args()
rep = {}
for w in a.segs:
    sup = np.asarray(zarr.open(f'{L}/{w}/{w}_supervision_mask.zarr', mode='r')['0'][14]) > 0
    h = sup.shape[0]; y0 = int(round(h * (1 - a.frac)))
    val = np.zeros_like(sup); val[y0:] = sup[y0:]
    g = zarr.open_group(f'{L}/{w}/{w}_validation_mask.zarr', mode='w', zarr_format=2)
    arr = g.create_array('0', shape=(28,) + sup.shape, chunks=(28, 128, 128), dtype=np.uint8, fill_value=0,
                         compressors=Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE))
    arr.attrs['_ARRAY_DIMENSIONS'] = ['z', 'y', 'x']
    arr[14] = (val * 255).astype(np.uint8)
    g.attrs.update(dict(source='FL4-C shared validation region', rows=[y0, h], frac=a.frac))
    rep[w] = dict(rows=[y0, h], val_px=int(val.sum()), sup_px=int(sup.sum()))
    print(w, rep[w])
json.dump(rep, open('<repo>/labels/val_masks.json', 'w'), indent=1)
