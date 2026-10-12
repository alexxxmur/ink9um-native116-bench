"""FL4-A measurements on positive controls (protocol §5.1): every component and stack combination, both directions.

    measure.py CTRL [--variant orig] [--out OUT.json]

Components (maps from stack_infer.py under ~/work/fl4a/maps/<ctrl>/<variant>/):
  R0      ink_9um s42-75k, published baseline window
  S42z    soup42, mean of the 4 window probabilities (zavg_infer.sh: p = clip((u8 - 64) / 128, 0, 1))
  S43z, DN, KL   same for soup43, dense_native, KLAVIS dense
  H       Hecate 9.6 2D map (png, 9.6 um grid) resized to the control grid (bilinear)
  V8      v8-in, mean of the available windows (float npy)
  F*      the F+ variants (depth-sharpened input), when present
Combinations: each map z-scored over the control's valid pixels, then averaged:
  STACK-A = mean(z S42z, z S43z, z DN, z KL, z H);  STACK-B = mean(z STACK-A, z V8)  (1:1)
  STACK-F = STACK-B with S42z/S43z/DN/KL replaced by their F+ versions
Metrics:
  AUC     tie-aware Mann-Whitney (Nieuwlaar native_bench.roc_auc; identical to AndreasHad's histogram AUC for uint8)
          title/w024: native_bench classes (78 keV teacher);  0841, w043: label vs support;
          c343: pixels inside letter boxes vs valid pixels outside all boxes (dilated by 20 px)
  hp_r    ShribyrLabs letter-scale score (hp_score.py: 48 um high-pass of map and key, Pearson r over the core of the
          valid area); key = 78 keV teacher (0139) or 255 * label (0841, w043); not defined for c343 (no key)
  sep     AndreasHad label-free separation on the mean component probability, for the direction rule
"""
import argparse, glob, json, os, sys
import numpy as np, tifffile, cv2
from scipy.ndimage import gaussian_filter, binary_dilation

H = os.path.expanduser
sys.path.insert(0, H('~/data/fl4a/dense-native/native-eval'))
import native_bench as NB  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan import CTRL, windows, r0_window  # noqa: E402

D = H('~/data/fl4a'); MAPS = H('~/work/fl4a/maps')
SIG = 20 / 3.9


def u8prob(a):
    return np.clip((a.astype(np.float32) - 64.0) / 128.0, 0.0, 1.0)


def read_map(path, shape):
    if path.endswith('.npy'):
        m = np.load(path).astype(np.float32)
    elif path.endswith('.png'):
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    else:
        m = u8prob(tifffile.imread(path))
    if m.shape != tuple(shape):
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return m


def control_truth(c, shape):
    """valid (bool), and for scoring: (pos, neg) masks; key for hp_r (or None)."""
    if c in ('title', 'w024'):
        sd = f'{D}/bench0139/{c}'
        valid = NB.load_valid(sd); t = NB.load_teacher(sd, valid.shape)
        lo, hi = NB.freeze_thresholds(t, valid); ink, noink = NB.classes(t, valid, lo, hi)
        return valid, ink, noink, t.astype(np.float32), valid & (t > 0)
    if c == 'c343':
        import zarr
        z = zarr.open(CTRL[c][0], mode='r')['0']
        valid = np.asarray(z[33]) > 0
        boxes = json.load(open(H('~/work/fl3/n343/inputs/concat_w047-w048_R5B2_z9500-11000_letter_boxes.json')))['letters']
        inb = np.zeros(shape, bool)
        for l in boxes:
            x0, y0, x1, y1 = (int(round(v)) for v in l['box']); inb[y0:y1, x0:x1] = True
        far = ~binary_dilation(inb, iterations=20)
        return valid, inb & valid, far & valid, None, None
    sd = f'{D}/w043' if c == 'w043' else f'{D}/p0841/{c}'
    lab = np.load(f'{sd}/label.npy'); sup = np.load(f'{sd}/support.npy')
    return sup, lab & sup, sup & ~lab, lab.astype(np.float32) * 255, sup


def hp(img, m):
    w = gaussian_filter(m.astype(np.float32), SIG)
    return img - gaussian_filter(img * m, SIG) / np.maximum(w, 1e-3)


def hp_r(p, key, kvalid):
    core = kvalid & (gaussian_filter(kvalid.astype(np.float32), 2 * SIG) > 0.999)
    k = np.where(kvalid, key, 0)
    a, b = hp(p, core), hp(k, kvalid)
    return float(np.corrcoef(a[core], b[core])[0, 1])


def zs(m, valid):
    v = m[valid]; return (m - v.mean()) / (v.std() + 1e-9)


def sep(p, valid):
    v = p[valid]; hi, lo = v[v > 0.5], v[v <= 0.5]
    return float(hi.mean() - lo.mean()) if hi.size and lo.size else float('nan')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('ctrl'); ap.add_argument('--variant', default='orig'); ap.add_argument('--out')
    a = ap.parse_args()
    c = a.ctrl; src, n, um, true_dir = CTRL[c]
    md = f'{MAPS}/{c}/{a.variant}'; pre = f'{md}/{c}'
    shape = tifffile.TiffFile(glob.glob(f'{pre}__r0__w*.tif')[0]).pages[0].shape
    valid, pos, neg, key, kvalid = control_truth(c, shape)
    suf = {'forward': '', 'reverse': '_reverse'}

    def comp(d):
        """probability maps of every component for villa-direction d (v8 and hecate use their own flag, same order)."""
        out = {}
        ls, le = r0_window(c, n)
        f = f'{pre}__r0__w{ls}-{le}{suf[d]}.tif'
        if os.path.exists(f): out['R0'] = read_map(f, shape)
        for key_, m in (('S42z', 'soup42'), ('S43z', 'soup43'), ('DN', 'dn'), ('KL', 'kl')):
            for F in ('', 'F'):
                fs = [f'{pre}__{F}{m}__{wn}{suf[d]}.tif' for wn in windows(n)]
                if all(os.path.exists(x) for x in fs):
                    out[F + key_] = np.mean([read_map(x, shape) for x in fs], 0)
                    if not F:
                        for wn, x in zip(windows(n), fs): out[f'{key_}@{wn}'] = read_map(x, shape)
        hs = glob.glob(f'{pre}__hecate__zc*{suf[d]}.png')
        hs = [h for h in hs if (d == 'reverse') == h.endswith('_reverse.png')]
        if hs: out['H'] = read_map(hs[0], shape)
        vs = sorted(x for x in glob.glob(f'{pre}__v8__l*.npy') if (d == 'reverse') == x.endswith('_reverse.npy'))
        if vs:
            out['V8'] = np.mean([read_map(x, shape) for x in vs], 0)
            for x in vs: out['V8@' + x.split('__v8__')[1].replace('_reverse', '').replace('.npy', '')] = read_map(x, shape)
        return out

    res = {'ctrl': c, 'variant': a.variant, 'true_dir': true_dir, 'n_pos': int(pos.sum()), 'n_neg': int(neg.sum()), 'rows': {}}
    maps = {d: comp(d) for d in ('forward', 'reverse')}
    for d in maps:
        M = maps[d]; Z = {k: zs(v, valid) for k, v in M.items() if '@' not in k}
        fam = [k for k in ('S42z', 'S43z', 'DN', 'KL', 'H') if k in Z]
        if fam:
            M['STACK-A'] = np.mean([Z[k] for k in fam], 0); Z['STACK-A'] = zs(M['STACK-A'], valid)
            ffa = [('F' + k if 'F' + k in Z else None) for k in ('S42z', 'S43z', 'DN', 'KL')]
            if all(ffa) and 'H' in Z:
                M['STACK-AF'] = np.mean([Z[k] for k in ffa] + [Z['H']], 0)      # STACK-A with the F+ family
            if 'V8' in Z:
                M['STACK-B'] = (Z['STACK-A'] + Z['V8']) / 2
                ff = [('F' + k if 'F' + k in Z else None) for k in ('S42z', 'S43z', 'DN', 'KL')]
                if all(ff):
                    fa = zs(np.mean([Z[k] for k in ff] + [Z['H']], 0), valid)
                    M['STACK-F'] = (fa + Z['V8']) / 2
            # probability for the label-free direction rule: mean of the component probabilities
            M['_prob'] = np.mean([maps[d][k] for k in ('S42z', 'S43z', 'DN', 'KL', 'H', 'V8') if k in maps[d]], 0)
    for k in sorted(set(maps['forward']) & set(maps['reverse'])):
        if k == '_prob': continue
        row = {}
        for d in ('forward', 'reverse'):
            m = maps[d][k]; sel = pos | neg
            r = {'auc': NB.roc_auc(pos[sel], m[sel])}
            if key is not None: r['hp_r'] = hp_r(m, key, kvalid)
            if k.startswith('STACK'):
                r['sep'] = sep(maps[d]['_prob'], valid)
            elif m.max() <= 1.0 and m.min() >= 0.0:
                r['sep'] = sep(m, valid)
            row[d] = r
        row['true'] = row[true_dir]; row['null_dir'] = row['reverse' if true_dir == 'forward' else 'forward']
        sf, sr = row['forward'].get('sep', np.nan), row['reverse'].get('sep', np.nan)
        if not (np.isnan(sf) and np.isnan(sr)):
            ch = 'forward' if (np.isnan(sr) or (not np.isnan(sf) and sf >= sr)) else 'reverse'
            row['rule_choice'] = ch; row['rule_correct'] = ch == true_dir
        res['rows'][k] = row
    for k, row in res['rows'].items():
        t = row['true']; nd = row['null_dir']
        print(f"{c:6s} {k:14s} AUC {t['auc']:.4f} (rev {nd['auc']:.4f})" + (f"  hp_r {t['hp_r']:+.4f} (rev {nd['hp_r']:+.4f})" if 'hp_r' in t else '')
              + (f"  rule {'ok' if row.get('rule_correct') else 'WRONG'}" if 'rule_correct' in row else ''), flush=True)
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
