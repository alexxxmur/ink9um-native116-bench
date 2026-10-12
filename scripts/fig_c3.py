"""FL4-C: side-by-side pictures for the owner (C3 result), same crop, same contrast rule per panel.

    fig_c3.py OUT_DIR [--step 12]

Panels per control: raw CT (sheet layer) | dense_native (before) | treatment X | control X | key
(c343: letter boxes; 0009B: manual labels; h0343P: teacher map). Model panels = mean of the 4 depth windows,
true direction, each panel stretched to its own 1-99 % percentiles inside the valid area.
"""
import argparse, json, os, sys
import numpy as np, cv2, tifffile, zarr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench as B  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--step', default='12')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
H = os.path.expanduser


def stretch(m, v):
    lo, hi = np.percentile(m[v], [1, 99]); return np.clip((m - lo) / (hi - lo + 1e-9), 0, 1) * v


def label(img, text):
    img = cv2.copyMakeBorder(img, 34, 0, 0, 0, cv2.BORDER_CONSTANT, value=255)
    cv2.putText(img, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2, cv2.LINE_AA)
    return img


for c, crop, scale in (('c343', None, 0.5), ('0009B', None, 0.5), ('h0343P', None, 0.5)):
    shape = tifffile.TiffFile(B.glob.glob(f'{B.mapdir(c, "orig", "r0")}__r0__w*.tif')[0]).pages[0].shape
    valid, pos, neg, key, kvalid, _ = B.truth(c, shape)
    td = B.TRUE_DIR[c]
    M = B.comps(c, 'orig', td, shape, (f't{a.step}k', f'c{a.step}k'))
    z = zarr.open(B.src(c, 'orig'), mode='r'); z = z if hasattr(z, 'shape') else z['0']
    ct = np.asarray(z[z.shape[0] // 2]).astype(np.float32)
    if c == 'c343':
        boxes = json.load(open(H('~/work/fl3/n343/inputs/concat_w047-w048_R5B2_z9500-11000_letter_boxes.json')))['letters']
        kimg = np.zeros(shape, np.float32)
        for l in boxes:
            x0, y0, x1, y1 = (int(round(v)) for v in l['box']); cv2.rectangle(kimg, (x0, y0), (x1, y1), 1.0, 3)
        kname = 'letter boxes (Nieuwlaar)'
    elif c == '0009B':
        kimg = (pos.astype(np.float32) + 0.35 * (neg.astype(np.float32))); kname = 'manual labels (grey = labelled bg)'
    else:
        kimg = key / 255.0; kname = 'teacher 2.2 um (key)'
    panels = [(stretch(ct, valid), 'raw CT, sheet layer'), (stretch(M['DN'], valid), 'dense_native (before)'),
              (stretch(M[f'X:t{a.step}k'], valid), f'TREATMENT {a.step}k (+116 keV)'),
              (stretch(M[f'X:c{a.step}k'], valid), f'CONTROL {a.step}k (no 116 keV)'), (np.clip(kimg, 0, 1), kname)]
    tiles = []
    for p, t in panels:
        im = (cv2.resize(p, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)
        tiles.append(label(cv2.copyMakeBorder(im, 0, 0, 0, 6, cv2.BORDER_CONSTANT, value=255), t))
    # c343 is long: stack panels vertically side by side in rows of 1 when width > height
    img = np.concatenate(tiles, 1) if shape[0] >= shape[1] else np.concatenate(tiles, 0)
    cv2.imwrite(f'{a.out}/{c}_compare.png', img)
    print(c, img.shape)
