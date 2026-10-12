"""FL4-A stack inference: many ink maps (villa ink_9um family, Hecate 9.6, v8-in) in one process.

    stack_infer.py JOBS.tsv [--batch-size 32] [--workers 4]

JOBS.tsv (tab separated, '#' comments):
    villa   INPUT.zarr  MODEL  LS  LE  OUT_PREFIX          -> OUT_PREFIX.tif, OUT_PREFIX_reverse.tif (uint8, villa encoding)
    villaF  INPUT.zarr  MODEL  LS:LE  UM  OUT_PREFIX   -> same as villa, input depth-sharpened first (run_villa_f)
    hecate  INPUT.zarr  -      ZC  UM  OUT_PREFIX         -> OUT_PREFIX.png, _reverse.png (9.6 um grid, Nieuwlaar run_ink.sh)
    v8      INPUT.zarr  -      LS  STRIDE|-  OUT_PREFIX     -> OUT_PREFIX.npy/.png, _reverse.npy/.png (24 layers LS..LS+23;
                                                          tile stride, '-' = his 21; batch 8: 16 overflows 16 GB)

INPUT.zarr: a zarr group with array '0' (Z,Y,X uint8; pyramid levels ignored) or a bare array.
villa: the window [LS, LE) is centre-cropped by villa to the model's 17 layers (select_layer_indices); forward and
reverse are both written. Same code path as scripts/fl3/fast_ink.py (RAM volume, GPU robust normalisation, fork
workers, RAM accumulators); fast_ink.py is byte-checked against run_ink.sh in FL3.
MODEL is a key of MODELS below or a path to a villa .pth.
Existing outputs are skipped.
"""
import argparse, os, subprocess, sys, tempfile, time
import numpy as np, zarr

H = os.path.expanduser
MODELS = {
    'r0': H('~/data/fl3/ckpt/hybrid_3d2d-seed42_step-075000.pth'),
    's43': H('~/data/fl3/ckpt/hybrid_3d2d-seed43_step-075000.pth'),
    'soup42': H('~/data/fl4a/ckpt/soup42_last4.pth'),
    'soup43': H('~/data/fl4a/ckpt/soup43_last4.pth'),
    'dn': H('~/data/fl3/ckpt/dense_native-016000.rebuilt.pth'),
    'kl': H('~/data/fl4a/ckpt/klavis_dense9um-w016excluded-step075000.pth'),
}
HECATE_CK = H('~/data/fl3/ckpt/hecate_9.6um.pth')
HECATE_PY = H('~/data/fl3/ckpt/hecate.py')
HECATE_PREP = H('~/work/fl3/n343/ink/scripts/hecate_prep.py')
V8_DIR = H('~/data/fl4a/v8in')


def open_l0(path):
    z = zarr.open(path, mode='r')
    return z if hasattr(z, 'shape') else z['0']


# ----------------------------------------------------------------------------------------------- villa (fast_ink)
_villa = None

def villa():
    """Install the fast_ink.py patches into villa's infer module once; return the module."""
    global _villa
    if _villa is not None:
        return _villa
    import torch
    from torch import nn
    import vesuvius.ink_detection.inference.infer as infer
    CACHE = {}
    orig_open = infer.open_volume

    def ram_open(path, resolution, *args, **kw):
        key = (os.path.realpath(str(path)), str(resolution))
        if key not in CACHE:
            CACHE.clear()
            CACHE[key] = np.asarray(orig_open(path, resolution, *args, **kw)[:])
        return CACHE[key]
    infer.open_volume = ram_open
    infer._ram_open = ram_open
    orig_norm = infer.normalize_flat_patch

    def defer_norm(patch, preprocessing):
        if preprocessing == 'tifxyz_robust':
            return np.ascontiguousarray(patch, dtype=np.float32)
        return orig_norm(patch, preprocessing)
    infer.normalize_flat_patch = defer_norm

    def _q(sv, q):
        n = sv.shape[1]; pos = q * (n - 1); lo = int(np.floor(pos)); hi = min(lo + 1, n - 1); f = pos - lo
        return sv[:, lo] * (1 - f) + sv[:, hi] * f

    def gpu_robust(x):
        B = x.shape[0]; v = x.reshape(B, -1).float()
        sv = torch.sort(v, dim=1).values
        std_pre = v.std(dim=1, unbiased=False)
        lo, hi = _q(sv, 0.01), _q(sv, 0.99)
        v = torch.maximum(torch.minimum(v, hi[:, None]), lo[:, None])
        sv = torch.sort(v, dim=1).values
        med = _q(sv, 0.5)
        mad = _q(torch.sort((v - med[:, None]).abs(), dim=1).values, 0.5)
        smad = 1.4826 * mad
        span = (hi - lo).abs() / 2
        fb = torch.where(std_pre.abs() >= 1e-6, std_pre.abs(), torch.where(span >= 1e-6, span, torch.ones_like(span)))
        smad = torch.where(torch.isfinite(smad) & (smad >= 1e-6), smad, fb)
        out = (v - med[:, None]) / smad[:, None]
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).reshape(x.shape)

    class NormThenModel(nn.Module):
        def __init__(self, inner): super().__init__(); self.inner = inner
        def forward(self, x):
            with torch.autocast('cuda', enabled=False):
                x = gpu_robust(x)
            return self.inner(x)

    orig_conf = infer.configure_model
    MC = {}

    def conf_cached(args):
        k = os.path.realpath(str(args.checkpoint))
        if k not in MC:
            MC.clear(); torch.cuda.empty_cache()           # one model on the GPU at a time
            cm = orig_conf(args)
            if cm.preprocessing == 'tifxyz_robust':
                cm = infer.replace(cm, model=NormThenModel(cm.model).eval())
            MC[k] = cm
        return MC[k]
    infer.configure_model = conf_cached
    infer._mc = MC
    from torch.utils.data import DataLoader as _DL

    def dl_fork(*args, **kw):
        if kw.get('num_workers', 0) > 0:
            kw['multiprocessing_context'] = 'fork'
        return _DL(*args, **kw)
    infer.DataLoader = dl_fork
    infer.open_temp_zarr_array = lambda path, *, shape, chunks: np.zeros(shape, dtype=np.float32)
    _villa = infer
    return infer


def level0_view(render, tmp, i):
    """A render with pyramid levels is presented as level 0 only (same patch selection as run_ink.sh)."""
    render = os.path.realpath(render)
    if os.path.isdir(os.path.join(render, '1')) and os.path.exists(os.path.join(render, '0', '.zarray')):
        zin = os.path.join(tmp, f'l0_{i}.zarr'); os.makedirs(zin, exist_ok=True)
        open(os.path.join(zin, '.zgroup'), 'w').write('{"zarr_format": 2}')
        if not os.path.exists(os.path.join(zin, '0')):
            os.symlink(os.path.join(render, '0'), os.path.join(zin, '0'))
        return zin
    return render


def run_villa(render, model, ls, le, prefix, tmp, i, a):
    infer = villa()
    ck = MODELS.get(model, model)
    zin = level0_view(render, tmp, i)
    infer._ram_open(zin, '0')
    out = os.path.join(tmp, 'out.tif')
    rc = infer.main([zin, ck, out, '--overlap', '0.5', '--blend-mode', 'hann', '--batch-size', str(a.batch_size),
                     '--direction', 'both', '--layer-start', str(ls), '--layer-end', str(le), '--num-workers', str(a.workers)])
    os.replace(out, prefix + '.tif'); os.replace(os.path.join(tmp, 'out_reverse.tif'), prefix + '_reverse.tif')
    return rc


DEPTH_SHARPEN = H('~/data/fl4a/repos/villa-apple-silicon/bench')

def run_villa_f(render, model, ls, le, um, prefix, tmp, i, a):
    """F+: AndreasHad04 depth_sharpen_9um.sharpen(--no-map, layer_um=um) on the 21 layers around the villa window,
    then villa on exactly the same 17 effective layers as the unfiltered job [ls, le)."""
    infer = villa()
    sys.path.insert(0, DEPTH_SHARPEN)
    import depth_sharpen_9um as DS
    src = open_l0(render); D = src.shape[0]
    eff = infer.select_layer_indices(D, layer_start=ls, layer_end=le, output_depth=17, direction='forward')
    s0 = min(max(int(eff[8]) - 10, 0), D - 21)
    assert s0 <= eff[0] and eff[-1] < s0 + 21, (eff, s0)
    filt = DS.sharpen(np.asarray(src[s0:s0 + 21]), um, use_map=False)
    zf = os.path.join(tmp, f'f_{i}.zarr')
    g = zarr.open_group(zf, mode='w', zarr_format=2)
    g.create_array('0', shape=filt.shape, chunks=(21, 128, 128), dtype=np.uint8, fill_value=0)[:] = filt
    rc = run_villa(zf, model, int(eff[0]) - s0, int(eff[0]) - s0 + 17, prefix, tmp, i, a)
    subprocess.run(['rm', '-rf', zf])
    return rc


# ----------------------------------------------------------------------------------------------- hecate (run_ink.sh)
def run_hecate(render, zc, um, prefix, tmp, a):
    py = sys.executable
    prep = os.path.join(tmp, 'prep.zarr')
    subprocess.run([py, HECATE_PREP, '--src', render, '--out', prep, '--native-um', str(um), '--native-um-z', str(um),
                    '--target-um', '9.6', '--z-center', f'{zc:.2f}', '--nz', '16'], check=True, cwd=tmp)
    for suf, extra in (('', []), ('_reverse', ['--reverse'])):
        subprocess.run([py, HECATE_PY, '--checkpoint', HECATE_CK, '--input', prep, '--spacing-um', '9.6',
                        '--output', os.path.join(tmp, f'out{suf}.png'), '--device', 'cuda', '--batch-size', '8'] + extra,
                       check=True, cwd=tmp, env=dict(os.environ, TMPDIR=tmp))
        os.replace(os.path.join(tmp, f'out{suf}.png'), prefix + f'{suf}.png')
        if os.path.exists(os.path.join(tmp, f'out{suf}.json')):
            os.replace(os.path.join(tmp, f'out{suf}.json'), prefix + f'{suf}.json')
    subprocess.run(['rm', '-rf', prep])
    return 0


# ----------------------------------------------------------------------------------------------- v8-in (ink8um)
_v8 = None

def run_v8(render, ls, stride, prefix, a):
    global _v8
    import cv2, torch
    sys.path.insert(0, V8_DIR)
    from ink8um import InkDetector
    from ink8um.inference import predict_stack
    if _v8 is None:
        if _villa is not None:                                         # free villa models before the 83 M v8-in
            _villa._mc.clear()
        import gc; gc.collect(); torch.cuda.empty_cache()
        _v8 = InkDetector.from_pretrained(V8_DIR).cuda().eval()
    D = int(_v8.in_depth)
    vol = open_l0(render)
    assert 0 <= ls and ls + D <= vol.shape[0], (ls, vol.shape)
    st = np.asarray(vol[ls:ls + D])                                   # Z,Y,X
    st = np.clip(np.ascontiguousarray(st.transpose(1, 2, 0)), 0, int(_v8.clip_max)).astype(np.uint8)   # H,W,D
    for suf, rev in (('', False), ('_reverse', True)):
        t0 = time.time(); last = [0.0]

        def prog(done, total):
            if time.time() - last[0] > 60 or done == total:
                last[0] = time.time(); print(f'  v8{suf} {done}/{total} tiles {time.time() - t0:.0f}s', flush=True)
        p = predict_stack(_v8, st, reverse=rev, stride=stride, batch_size=a.v8_batch, num_workers=a.workers, device='cuda', progress=prog)
        np.save(prefix + f'{suf}.npy', p.astype(np.float32))
        cv2.imwrite(prefix + f'{suf}.png', (p * 255).astype(np.uint8))
    return 0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('jobs'); ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--workers', type=int, default=4); ap.add_argument('--v8-batch', type=int, default=8)
    a = ap.parse_args()
    jobs = [l.rstrip('\n').split('\t') for l in open(a.jobs) if l.strip() and not l.startswith('#')]
    tmp = tempfile.mkdtemp(prefix='stack_infer_', dir=os.environ.get('TMPDIR'))
    for i, (kind, render, model, p1, p2, prefix) in enumerate(jobs):
        ext = {'villa': '.tif', 'villaF': '.tif', 'hecate': '.png', 'v8': '.npy'}[kind]
        if os.path.exists(prefix + ext) and os.path.exists(prefix + '_reverse' + ext):
            continue
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        t0 = time.time()
        if kind == 'villa':
            rc = run_villa(render, model, int(p1), int(p2), prefix, tmp, i, a)
        elif kind == 'villaF':
            ls, le, um = p1.split(':')[0], p1.split(':')[1], p2
            rc = run_villa_f(render, model, int(ls), int(le), float(um), prefix, tmp, i, a)
        elif kind == 'hecate':
            rc = run_hecate(render, float(p1), float(p2), prefix, tmp, a)
        else:
            rc = run_v8(render, int(p1), None if p2 == '-' else int(p2), prefix, a)
        print(f'{i + 1}/{len(jobs)}\t{kind}\t{model}\t{p1}\t{p2}\t{os.path.basename(prefix)}\trc={rc}\t{time.time() - t0:.1f}s',
              flush=True)


if __name__ == '__main__':
    main()
