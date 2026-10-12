"""FL4-C C0.3: dense native 116 keV pseudo-labels (Nieuwlaar's build_native_dense_labels.py recipe, adapted).

    labels116.py calib  OUT.json                      t* on PHerc0009B manual labels (balanced accuracy)
    labels116.py build  CALIB.json NAME [NAME ...]    0343P label zarrs + selection stats + pictures

teacher(native grid) = canonical 2.2 / 2.4 um map, cv2.INTER_AREA onto the native (H, W) grid, then shifted by the
CT-to-CT registration of ct_align.py (artifacts/fl4c/align/ct_<name>.json): t(y, x) = T(y + dy, x + dx).
calib: manual inklabels / supervision / validation (2.401 um canvas, same canvas as the teacher) go through the
same resize (INTER_AREA, >= 128) and shift; t* = argmax BA of (teacher >= t) vs manual ink over
supervision minus validation (the validation region stays clean for the benchmark), valid & teacher > 0.
build: ink = teacher >= t*; supervision = valid & teacher > 0 (valid = max_z > 0 of the 28-layer volume);
format = official native9 (zarr v2 group, '0' (28, H, W) u1, chunks (28, 128, 128), blosc zstd-3 bitshuffle, 0/255,
plane z = 14 only). Selection (protocol C0.3, fixed before training): ink fraction of supervision in [2, 40] % and
at least MIN_LETTERS connected ink components of letter scale (max bbox side 0.3-1.5 mm = 35-174 px at 8.64 um).
"""
import json, os, sys
import numpy as np, cv2, tifffile, zarr
from numcodecs import Blosc

H = os.path.expanduser
R = H('~/data/fl4c'); A = '<repo>/labels'
sys.path.insert(0, H('~/data/fl4a/dense-native/training/scripts'))
from native_common import balanced_accuracy_curve  # noqa: E402
sys.path.insert(0, H('~/data/fl4a/dense-native/native-eval'))
import native_bench as NB  # noqa: E402

OUT_LABELS = f'{R}/labels/native-dense-0343P-116keV'
HELD_OUT_0343P = '0343P_20250511003658-tifxyz'
MIN_LETTERS = 10
LET_PX = (int(round(300 / 8.64)), int(round(1500 / 8.64)))
D, ZP = 28, 14


def shift_img(a, dy, dx):
    out = np.zeros_like(a); Hh, Ww = a.shape
    ys, yd = (slice(dy, Hh), slice(0, Hh - dy)) if dy >= 0 else (slice(0, Hh + dy), slice(-dy, Hh))
    xs, xd = (slice(dx, Ww), slice(0, Ww - dx)) if dx >= 0 else (slice(0, Ww + dx), slice(-dx, Ww))
    out[yd, xd] = a[ys, xs]
    return out


def ct_shift(name):
    return json.load(open(f'{A}/align/ct_{name}.json'))['shift']


def on_grid(img, hw, shift):
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    g = cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
    return shift_img(g, *shift)


def valid_of(name):
    return np.load(f'{R}/volumes/native116/{name}_valid.npy')


def write_label(path, plane, attrs):
    g = zarr.open_group(path, mode='w', zarr_format=2)
    arr = g.create_array('0', shape=(D,) + plane.shape, chunks=(D, 128, 128), dtype=np.uint8, fill_value=0,
                         compressors=Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE))
    arr.attrs['_ARRAY_DIMENSIONS'] = ['z', 'y', 'x']
    for y in range(0, plane.shape[0], 1024):
        arr[ZP, y:y + 1024, :] = plane[y:y + 1024]
    g.attrs.update(attrs)
    a = zarr.open_group(path, mode='r')['0']
    assert int((np.asarray(a[ZP]) > 0).sum()) == int((plane > 0).sum()) and int(np.asarray(a[0]).sum()) == 0


def letter_components(ink):
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    side = np.maximum(st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT])
    return int(((side >= LET_PX[0]) & (side <= LET_PX[1])).sum()), int(n - 1)


def calib(out):
    name = '0009B_20250919125754'
    valid = valid_of(name); hw = valid.shape; sh = ct_shift(name)
    t = on_grid(tifffile.imread(f'{R}/teacher/{name}.tif'), hw, sh)
    m = {}
    for k in ('inklabels', 'supervision', 'validation'):
        g = zarr.open(f'{R}/raw0009B/{k}.zarr', mode='r'); z = g['0'] if hasattr(g, 'keys') else g
        m[k] = on_grid(np.asarray(z[:]), hw, sh) >= 128
    dom = m['supervision'] & ~m['validation'] & valid & (t > 0)
    t_star, ba_star, ba, n_pos, n_neg = balanced_accuracy_curve(t, m['inklabels'], dom)
    auc = NB.roc_auc(m['inklabels'][dom], t[dom].astype(np.float64))
    rec = dict(segment=name, shift=sh, t_star=t_star, t_star_frac=round(t_star / 255, 4), ba_at_t_star=round(ba_star, 4),
               ba_at_47=round(float(ba[47]), 4), ba_at_128=round(float(ba[128]), 4), teacher_auc_vs_manual=round(float(auc), 4),
               n_pos=n_pos, n_neg=n_neg, manual_ink_frac=round(float(m['inklabels'][dom].mean()), 4),
               in_range_0p12_0p26=bool(0.12 <= t_star / 255 <= 0.26),
               domain='supervision minus validation, valid, teacher > 0 (native grid)')
    np.save(f'{R}/labels/0009B_native_masks.npy', np.stack([m['inklabels'], m['supervision'], m['validation']]))
    json.dump(rec, open(out, 'w'), indent=1)
    print(json.dumps(rec, indent=1))


def build(calib_json, names):
    c = json.load(open(calib_json)); th = int(c.get("t_used", c["t_star"]))
    rep_p = f'{A}/labels116_report.json'
    rep = json.load(open(rep_p)) if os.path.exists(rep_p) else {}
    os.makedirs(f'{A}/labels116', exist_ok=True)
    for name in names:
        valid = valid_of(name); hw = valid.shape; sh = ct_shift(name)
        t = on_grid(tifffile.imread(f'{R}/teacher/{name}.tif'), hw, sh)
        sup = valid & (t > 0); ink = (t >= th) & sup
        nlet, ncomp = letter_components(ink)
        frac = float(ink.sum() / max(1, sup.sum()))
        sel = (0.02 <= frac <= 0.40) and nlet >= MIN_LETTERS
        rec = dict(shape=[D] + list(hw), theta=th, shift=sh, valid_frac=round(float(valid.mean()), 4),
                   supervised_px=int(sup.sum()), ink_px=int(ink.sum()), ink_frac_of_sup=round(frac, 4),
                   letter_scale_components=nlet, components=ncomp, selected_by_rule=bool(sel),
                   held_out=name == HELD_OUT_0343P, train=bool(sel and name != HELD_OUT_0343P))
        if name != HELD_OUT_0343P:
            od = f'{OUT_LABELS}/{name}'; os.makedirs(od, exist_ok=True)
            attrs = dict(source='native-dense-0343P-116keV', teacher=f'{name}.tif', theta=th, shift=sh,
                         teacher_resample='cv2.INTER_AREA 2.215um->8.64um grid, CT-registered shift',
                         annotated_plane=ZP, layout='official native9 21slices (28-deep, z=14)')
            write_label(f'{od}/{name}_inklabels.zarr', (ink * 255).astype(np.uint8), attrs)
            write_label(f'{od}/{name}_supervision_mask.zarr', (sup * 255).astype(np.uint8), attrs)
        rep[name] = rec
        s = 3
        img = np.concatenate([np.pad(cv2.resize(x.astype(np.float32), (hw[1] // s, hw[0] // s), interpolation=cv2.INTER_AREA), 4, constant_values=1)
                              for x in (t.astype(np.float32) / 255 * valid, ink.astype(np.float32))], 1)
        cv2.imwrite(f'{A}/labels116/{name}.png', (img * 255).astype(np.uint8))
        print(f"{name}: ink {frac*100:.1f}% of sup, sup {sup.sum()/1e6:.2f} Mpx, letter-scale comps {nlet}/{ncomp} -> "
              f"{'SELECTED' if sel else 'EXCLUDED'}{' (HELD OUT)' if name == HELD_OUT_0343P else ''}")
    json.dump(rep, open(rep_p, 'w'), indent=1)
    tr = [k for k, v in rep.items() if v['train']]
    print('train segments:', len(tr), 'supervised Mpx:', round(sum(rep[k]['supervised_px'] for k in tr) / 1e6, 2),
          'canvas Mpx:', round(sum(rep[k]['shape'][1] * rep[k]['shape'][2] for k in tr) / 1e6, 2))


if __name__ == '__main__':
    if sys.argv[1] == 'calib':
        calib(sys.argv[2])
    else:
        build(sys.argv[2], sys.argv[3:])
