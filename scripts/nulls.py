"""Null input volumes for FL4-A (protocol §5.2), fixed seeds.

    nulls.py IN.zarr OUT.zarr shuffle [--seed 0]          layer order permuted (as scripts/fl3/shuffle_layers.py)
    nulls.py IN.zarr OUT.zarr slr [--layer K]             single_layer_repeat: layer K (default centre) in every layer
    nulls.py IN.zarr OUT.zarr crop --box y0,y1,x0,x1      plain crop (to keep big controls small)

IN.zarr: group with array '0' (pyramid levels ignored) or a bare array. OUT.zarr: group with array '0', zarr v2,
chunks (Z, 128, 128). Writes OUT.zarr/null.json with the recipe.
"""
import argparse, json, os
import numpy as np, zarr

ap = argparse.ArgumentParser(); ap.add_argument('inp'); ap.add_argument('out'); ap.add_argument('kind', choices=['shuffle', 'slr', 'crop'])
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--layer', type=int, default=None); ap.add_argument('--box', default=None)
a = ap.parse_args()
z = zarr.open(a.inp, mode='r'); src = z if hasattr(z, 'shape') else z['0']
D = src.shape[0]
box = [int(v) for v in a.box.split(',')] if a.box else [0, src.shape[1], 0, src.shape[2]]
vol = np.asarray(src[:, box[0]:box[1], box[2]:box[3]])
rec = dict(source=os.path.realpath(a.inp), kind=a.kind, box=box)
if a.kind == 'shuffle':
    perm = np.random.default_rng(a.seed).permutation(D); vol = vol[perm]; rec.update(seed=a.seed, perm=perm.tolist())
elif a.kind == 'slr':
    k = D // 2 if a.layer is None else a.layer
    vol = np.repeat(vol[k:k + 1], D, axis=0); rec.update(layer=int(k))
g = zarr.open_group(a.out, mode='w', zarr_format=2)
g.create_array('0', shape=vol.shape, chunks=(D, 128, 128), dtype=np.uint8, fill_value=0)[:] = vol
json.dump(rec, open(os.path.join(a.out, 'null.json'), 'w'))
print(a.out, a.kind, vol.shape)
