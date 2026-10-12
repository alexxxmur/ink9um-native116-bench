"""FL4-C frozen benchmark (protocol C1/C3): job lists for stack_infer.py and the measurements.

    bench.py jobs base                         > jobs.tsv   baseline maps the benchmark still lacks
    bench.py jobs TAG=MODEL.pth [--ctrls ...] [--nulls] [--depth]  > jobs.tsv   maps of a new model
    bench.py measure OUT.json [--models TAG ...] [--nulls]      metrics (baseline always included)

Controls (all native 28-layer except c343):
  title          PHerc0139 native 113 keV, Nieuwlaar bench (78 keV teacher classes), maps of FL4-A
  w00 ag144 ag174  PHerc0841 native 9.366 um, labels 20260918, maps of FL4-A (regression controls, 113 keV)
  c343           PHerc0343 66-layer render (8.64 um / 116 keV), letter boxes (box-AUC), reading side villa 'reverse'
  0009B          PHerc0009B 20250919125754 native 8.64 um / 116 keV, manual labels 20260918 (bench_prep.py crop)
  h0343P         held-out PHerc0343P 20250511003658 native 116 keV, teacher classes (upper bound, Nieuwlaar §8)
Model keys: R0 (s42-75k, published window), S42z S43z DN KL (mean of the 4 depth windows, zavg_infer.sh probability),
H (Hecate 9.6), STACK-A (z-mean of S42z S43z DN KL H); a new model TAG gives X (4-window mean), STACK-A116 (DN -> X)
and STACK-A+ (X added as a sixth member).
Metrics, true direction (and the other direction as 'rev'): AUC (tie-aware Mann-Whitney), hp_r (ShribyrLabs 48 um
high-pass Pearson r; not for c343). """
import argparse, glob, json, os, sys, time
import numpy as np, tifffile

H = os.path.expanduser
sys.path.insert(0, '<repo>/scripts')
import plan as PL  # noqa: E402
import measure as MS  # noqa: E402
NB = MS.NB

A_MAPS = H('~/work/fl4a/maps'); C_MAPS = H('~/work/fl4c/maps'); BD = H('~/data/fl4c/bench')
FL4A = ('title', 'w00', 'ag144', 'ag174', 'c343')
NEW = ('0009B', 'h0343P')
CTRLS = FL4A + NEW
N_LAYERS = {c: (66 if c == 'c343' else 28) for c in CTRLS}
TRUE_DIR = {c: ('reverse' if c == 'c343' else 'forward') for c in CTRLS}
UM = {'title': 9.362, 'w00': 9.366, 'ag144': 9.366, 'ag174': 9.366, 'c343': 8.64, '0009B': 8.64, 'h0343P': 8.64}
BASE_TAGS = ('soup42', 'soup43', 'dn', 'kl')
FAMKEY = {'soup42': 'S42z', 'soup43': 'S43z', 'dn': 'DN', 'kl': 'KL'}
VARIANTS = ('orig', 'shuf', 'slr')
SUF = {'forward': '', 'reverse': '_reverse'}


def r0_window(c):
    return (25, 42) if c == 'c343' else ((0, 28) if c == 'title' else (4, 25))


def src(c, v):
    if c in NEW:
        return f'{BD}/{c}/ct28.zarr' if v == 'orig' else f'{BD}/{c}/ct28_{v}.zarr'
    return PL.vol(c, v)


def mapdir(c, v, tag):
    base = A_MAPS if (c in FL4A and (tag in BASE_TAGS + ('r0', 'hecate'))) else C_MAPS
    return f'{base}/{c}/{v}/{c}'


# --------------------------------------------------------------------------------------------------------- jobs
def jobs_base():
    out = []
    for c in NEW:
        for v in VARIANTS:
            s, o = src(c, v), mapdir(c, v, 'r0')
            ls, le = r0_window(c); out.append(('villa', s, 'r0', ls, le, f'{o}__r0__w{ls}-{le}'))
            for m in BASE_TAGS:
                for wn, (ls, le) in PL.windows(28).items():
                    out.append(('villa', s, m, ls, le, f'{o}__{m}__{wn}'))
            if v != 'slr':
                out.append(('hecate', s, '-', 13.5, UM[c], f'{o}__hecate__zc13.5'))
    # environment control (protocol C1): dense_native on the title, native_bench default window [0, 28)
    out.append(('villa', src('title', 'orig'), 'dn', 0, 28, f'{C_MAPS}/title/orig/title__dn__w0-28'))
    return out


def jobs_model(tag, model, ctrls, nulls, depth):
    out = []
    for c in ctrls:
        for v in (VARIANTS if nulls else ('orig',)):
            if v != 'orig' and c == 'title':
                continue
            s, o = src(c, v), mapdir(c, v, tag)
            for wn, (ls, le) in PL.windows(N_LAYERS[c]).items():
                out.append(('villa', s, model, ls, le, f'{o}__{tag}__{wn}'))
        if depth and c == 'c343':
            o = mapdir(c, 'orig', tag)
            for d in (-10, -8, -2, 2, 8, 10):
                out.append(('villa', src(c, 'orig'), model, 25 + d, 42 + d, f'{o}__{tag}__d{d:+d}'))
    return out


# ------------------------------------------------------------------------------------------------------ truth
def truth(c, shape):
    """valid, pos, neg, key (or None), kvalid (or None), pos_tile_mask (for tile recall)"""
    if c in FL4A:
        valid, pos, neg, key, kvalid = MS.control_truth(c, shape)
        return valid, pos, neg, key, kvalid, pos
    d = f'{BD}/{c}'
    valid = np.load(f'{d}/valid.npy')
    if c == '0009B':
        lab, sup = np.load(f'{d}/label.npy'), np.load(f'{d}/support.npy')
        return valid, lab & sup, sup & ~lab, lab.astype(np.float32) * 255, sup, lab & sup
    t = np.load(f'{d}/teacher.npy')
    lo, hi = NB.freeze_thresholds(t, valid)
    ink, noink = NB.classes(t, valid, lo, hi)
    return valid, ink, noink, t.astype(np.float32), valid & (t > 0), ink


def stats_valid(c, shape):
    import zarr
    z = zarr.open(src(c, 'orig'), mode='r'); z = z if hasattr(z, 'shape') else z['0']
    return np.asarray(z[z.shape[0] // 2]) > 0


# ------------------------------------------------------------------------------------------------- components
def comps(c, v, d, shape, tags=()):
    out = {}
    s = SUF[d]
    for m in BASE_TAGS:
        fs = [f'{mapdir(c, v, m)}__{m}__{wn}{s}.tif' for wn in PL.windows(N_LAYERS[c])]
        if all(os.path.exists(x) for x in fs):
            out[FAMKEY[m]] = np.mean([MS.read_map(x, shape) for x in fs], 0)
    hs = [h for h in glob.glob(f'{mapdir(c, v, "hecate")}__hecate__zc*.png') if (d == 'reverse') == h.endswith('_reverse.png')]
    if hs:
        out['H'] = MS.read_map(hs[0], shape)
    ls, le = r0_window(c)
    f = f'{mapdir(c, v, "r0")}__r0__w{ls}-{le}{s}.tif'
    if os.path.exists(f):
        out['R0'] = MS.read_map(f, shape)
    for t in tags:
        fs = [f'{mapdir(c, v, t)}__{t}__{wn}{s}.tif' for wn in PL.windows(N_LAYERS[c])]
        if all(os.path.exists(x) for x in fs):
            out['X:' + t] = np.mean([MS.read_map(x, shape) for x in fs], 0)
    return out



FAM_A = ('S42z', 'S43z', 'DN', 'KL', 'H')


def stack_members(M, kind, tag=None):
    """component dict for a stack variant, returns (dict, fam, agree_set)."""
    base = {k: M[k] for k in FAM_A if k in M}
    if kind == 'A':
        return base, FAM_A, ('S42z', 'S43z', 'DN', 'H')
    x = M['X:' + tag]
    if kind == 'A116':
        base['DN'] = x
        return base, FAM_A, ('S42z', 'S43z', 'DN', 'H')
    base['X'] = x                                          # 'A+'
    return base, FAM_A + ('X',), ('S42z', 'S43z', 'DN', 'H', 'X')



def stack_map(M_read, M_other, valid, fam):
    """stack map for AUC / hp_r: members z-scored over the valid pixels of their own side (as measure.py, FL4-A)."""
    return np.mean([MS.zs(M_read[k], valid) for k in fam if k in M_read], 0)



# ---------------------------------------------------------------------------------------------------- measure
def measure(models, nulls):
    res = {'computed_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'models': list(models), 'controls': {}}
    for c in CTRLS:
        t0 = time.time()
        r0f = glob.glob(f'{mapdir(c, "orig", "r0")}__r0__w*.tif')
        shape = tifffile.TiffFile(r0f[0]).pages[0].shape
        valid, pos, neg, key, kvalid, postile = truth(c, shape)
        sval = stats_valid(c, shape) if c not in NEW else valid
        td = TRUE_DIR[c]; od = 'reverse' if td == 'forward' else 'forward'
        M = {d: comps(c, 'orig', d, shape, models) for d in ('forward', 'reverse')}
        out = {'n_pos': int(pos.sum()), 'n_neg': int(neg.sum()), 'true_dir': td, 'maps': {}}
        sel = pos | neg

        def score(mf, mo):
            r = {'auc': float(NB.roc_auc(pos[sel], mf[sel])), 'auc_rev': float(NB.roc_auc(pos[sel], mo[sel]))}
            if key is not None:
                r['hp_r'] = MS.hp_r(mf, key, kvalid)
            return r
        for k in ('R0', 'S42z', 'S43z', 'DN', 'KL', 'H'):
            if k in M[td] and k in M[od]:
                out['maps'][k] = score(M[td][k], M[od][k])
        kinds = [('STACK-A', 'A', None)] + [(f'{n}:{t}', n2, t) for t in models for n, n2 in (('X', None), ('STACK-A116', 'A116'), ('STACK-A+', 'A+'))]
        for name, kind, tag in kinds:
            if kind is None:
                if 'X:' + tag in M[td]:
                    out['maps'][name] = score(M[td]['X:' + tag], M[od]['X:' + tag])
                continue
            if tag and 'X:' + tag not in M[td]:
                continue
            R_, fam, agree = stack_members(M[td], kind, tag)
            B_, _, _ = stack_members(M[od], kind, tag)
            sr, sb = stack_map(R_, B_, valid, fam), stack_map(B_, R_, valid, fam)
            out['maps'][name] = score(sr, sb)
        res['controls'][c] = out
        line = '  '.join(f"{k} {v['auc']:.4f}" for k, v in out['maps'].items())
        print(f'{c:7s} {line}  ({time.time() - t0:.0f}s)', flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('arg')
    ap.add_argument('--models', nargs='*', default=[]); ap.add_argument('--ctrls', nargs='*', default=list(CTRLS))
    ap.add_argument('--nulls', action='store_true'); ap.add_argument('--depth', action='store_true')  # kept for job lists
    a = ap.parse_args()
    if a.cmd == 'jobs':
        J = jobs_base() if a.arg == 'base' else jobs_model(*a.arg.split('=', 1), a.ctrls, a.nulls, a.depth)
        for j in J:
            print('\t'.join(str(x) for x in j))
    elif a.cmd == 'measure':
        json.dump(measure(a.models, a.nulls), open(a.arg, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
