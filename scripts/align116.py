"""FL4-C C0: teacher-to-native registration and reading direction for one native 116 keV segment.

    align116.py NAME VOL.zarr TEACHER.tif PRED_PREFIX OUT.json OUT.png [--max-shift 30]

teacher  = canonical 2.2/2.4 um ink map, cv2.INTER_AREA onto the native (H, W) grid (Nieuwlaar native_common)
pred     = dense_native_016000 on the native volume, window [0, 28), villa forward (PRED_PREFIX.tif) and reverse
           (PRED_PREFIX_reverse.tif), probability = clip((u8 - 64) / 128, 0, 1) (zavg_infer.sh)
direction: the villa direction whose prediction has the higher AUC against the teacher classes (native_bench
           freeze_thresholds / classes on the teacher, the title-benchmark rule); also the label-free separation.
shift    = peak of the normalised cross-correlation (cv2 TM_CCOEFF_NORMED) of the prediction (true direction, inner
           part, max-shift border removed) inside the teacher map, over |dy|, |dx| <= max-shift; convention:
           pred(y, x) ~ teacher(y + dy, x + dx). Protocol C0.3: segment excluded if |shift| > 20 px or NCC < 0.2.
           The prediction-based peak is flat on 116 keV (weak predictions; log 2026-10-12 11:10), so two more
           estimates are reported: 'mask' = NCC peak of the render footprints (teacher > 0 vs native valid; both are
           renders of the same mesh, edges are sharp) and 'hp' = NCC peak of the 48 um high-passed maps. The applied
           shift is --shift when given (the CT-to-CT registration of ct_align.py, log 11:25), else the mask shift;
           the exclusion rule uses the applied shift and the prediction NCC at that shift.
"""
import argparse, json, os, sys
import numpy as np, cv2, tifffile

H = os.path.expanduser
sys.path.insert(0, H('~/data/fl4a/dense-native/native-eval'))
import native_bench as NB  # noqa: E402


def u8prob(a):
    return np.clip((a.astype(np.float32) - 64.0) / 128.0, 0.0, 1.0)


def teacher_on_grid(path, hw):
    t = tifffile.imread(path)
    if t.dtype == np.uint16:
        t = (t >> 8).astype(np.uint8)
    return cv2.resize(t, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)


def shift_img(a, dy, dx):
    """out(y, x) = a(y + dy, x + dx), zero fill."""
    out = np.zeros_like(a); Hh, Ww = a.shape
    ys, yd = (slice(dy, Hh), slice(0, Hh - dy)) if dy >= 0 else (slice(0, Hh + dy), slice(-dy, Hh))
    xs, xd = (slice(dx, Ww), slice(0, Ww - dx)) if dx >= 0 else (slice(0, Ww + dx), slice(-dx, Ww))
    out[yd, xd] = a[ys, xs]
    return out


def ncc_peak(pred, teacher, m):
    tpl = pred[m:-m, m:-m].astype(np.float32)
    r = cv2.matchTemplate(teacher.astype(np.float32), tpl, cv2.TM_CCOEFF_NORMED)
    iy, ix = np.unravel_index(np.argmax(r), r.shape)
    return int(iy - m), int(ix - m), float(r[iy, ix]), float(r[m, m])


def hp(a, valid, sig=20 / 3.9):
    from scipy.ndimage import gaussian_filter
    a = a.astype(np.float32) * valid; w = gaussian_filter(valid.astype(np.float32), sig)
    return np.where(valid, a - gaussian_filter(a, sig) / np.maximum(w, 1e-3), 0).astype(np.float32)


def ncc_at(pred, teacher, m, dy, dx):
    tpl = pred[m:-m, m:-m].astype(np.float32)
    win = teacher[m + dy:teacher.shape[0] - m + dy, m + dx:teacher.shape[1] - m + dx].astype(np.float32)
    return float(cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)[0, 0])


def auc_vs_teacher(p, t, valid):
    lo, hi = NB.freeze_thresholds(t, valid)
    ink, noink = NB.classes(t, valid, lo, hi)
    sel = ink | noink
    return float(NB.roc_auc(ink[sel], p[sel])), int(ink.sum()), int(noink.sum())


def sep(p, valid):
    v = p[valid]; hi, lo = v[v > 0.5], v[v <= 0.5]
    return float(hi.mean() - lo.mean()) if hi.size and lo.size else float('nan')


def main():
    ap = argparse.ArgumentParser()
    for k in ('name', 'vol', 'teacher', 'pred', 'out', 'png'):
        ap.add_argument(k)
    ap.add_argument('--max-shift', type=int, default=30)
    ap.add_argument('--shift', default=None, help='dy,dx to apply (CT registration); default: mask shift')
    a = ap.parse_args()
    valid = np.load(os.path.splitext(a.vol)[0] + '_valid.npy')
    t = teacher_on_grid(a.teacher, valid.shape)
    P = {d: u8prob(tifffile.imread(a.pred + s)) for d, s in (('forward', '.tif'), ('reverse', '_reverse.tif'))}
    tv = valid & (t > 0)
    rec = dict(name=a.name, shape=list(valid.shape), valid_frac=float(valid.mean()), teacher_valid_frac=float(tv.mean()))
    for d, p in P.items():
        auc, npos, nneg = auc_vs_teacher(p, t, tv)
        rec[d] = dict(auc=auc, sep=sep(p, valid))
    rec['n_ink_cls'], rec['n_noink_cls'] = npos, nneg
    true = 'forward' if rec['forward']['auc'] >= rec['reverse']['auc'] else 'reverse'
    rule = 'forward' if rec['forward']['sep'] >= rec['reverse']['sep'] else 'reverse'
    rec['true_dir'], rec['rule_dir'] = true, rule
    pdy, pdx, pncc, ncc0 = ncc_peak(P[true], t, a.max_shift)
    rec.update(pred_shift=[pdy, pdx], pred_ncc_peak=pncc, ncc_at_zero=ncc0)
    mdy, mdx, mncc, _ = ncc_peak(valid.astype(np.float32), (t > 0).astype(np.float32), a.max_shift)
    rec.update(mask_shift=[mdy, mdx], mask_ncc=mncc)
    hdy, hdx, hncc, _ = ncc_peak(hp(P[true], valid), hp(t.astype(np.float32) / 255, t > 0), a.max_shift)
    rec.update(hp_shift=[hdy, hdx], hp_ncc=hncc)
    dy, dx = (mdy, mdx) if a.shift is None else (int(v) for v in a.shift.split(','))
    rec['shift_source'] = 'mask' if a.shift is None else 'ct_align'
    ncc = ncc_at(P[true], t, a.max_shift, dy, dx)
    rec.update(shift=[dy, dx], ncc=ncc)
    ts = shift_img(t, dy, dx)
    rec['auc_true_after_shift'] = auc_vs_teacher(P[true], ts, valid & (ts > 0))[0]
    rec['excluded_by_alignment'] = bool(max(abs(dy), abs(dx)) > 20 or ncc < 0.2)
    json.dump(rec, open(a.out, 'w'), indent=1)
    # picture: teacher (shifted) | prediction (true dir) | prediction (other dir), 1/4 scale
    s = 4; o = 'reverse' if true == 'forward' else 'forward'
    tiles = [ts.astype(np.float32) / 255.0, P[true], P[o]]
    tiles = [cv2.resize(x * valid, (valid.shape[1] // s, valid.shape[0] // s), interpolation=cv2.INTER_AREA) for x in tiles]
    img = np.concatenate([np.pad(x, 4, constant_values=1.0) for x in tiles], 1)
    cv2.imwrite(a.png, (np.clip(img, 0, 1) * 255).astype(np.uint8))
    print(f"{a.name}: true {true} (AUC fwd {rec['forward']['auc']:.4f} rev {rec['reverse']['auc']:.4f}; rule {rule}) "
          f"mask shift {dy:+d},{dx:+d} (NCC mask {mncc:.3f}) pred NCC there {ncc:.3f} (at 0 {ncc0:.3f}, pred peak {pdy:+d},{pdx:+d} {pncc:.3f}) "
          f"hp peak {hdy:+d},{hdx:+d} {hncc:.3f}  AUC after shift {rec['auc_true_after_shift']:.4f}"
          + ('  EXCLUDED' if rec['excluded_by_alignment'] else ''))


if __name__ == '__main__':
    main()
