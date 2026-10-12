"""FL4-A job lists for stack_infer.py (protocol §3, §5).

    plan.py PHASE [CTRL ...] > jobs.tsv        PHASE: main | F | null | depth

Controls and their inputs (all zarr groups with array '0', uint8 Z,Y,X):
  c343   66-layer render of Nieuwlaar's segment (8.64 um, reading side = villa 'reverse')
  title, w024   PHerc0139 native 113 keV surface volumes (28 layers, 9.362 um, reading side = 'forward')
  w043   PHerc0139 w043 native, cropped to the labelled band (28 layers, 9.362 um, 'forward')
  w00, ag144, ag174   PHerc0841 native 9.366 um crops (28 layers, 'forward')
Windows (villa [ls, le), centre-cropped to 17 layers by villa):
  66 layers: d in {-3, -1, +1, +3} -> [25+d, 42+d); R0 = d 0 = [25, 42)   (FL3 convention, layer 33+d centre)
  28 layers: Nieuwlaar's published z-ensemble windows 0:20 3:23 5:25 8:27; R0 = the published baseline window
             ([0, 28) on 0139 = native_bench default, [4, 25) on 0841 = ARM Z production window, [0, 28) on w043)
Hecate: zc = 33.5 (66) / 13.5 (28), spacing 8.64 / 9.362 / 9.366.  v8-in: 24 layers from ls in {19, 21, 23} (c343) /
ls 2 = central (28), tile stride 32 (see the budget note in jobs()).  Variant volumes ('shuf', 'slr') live next to the originals (nulls.py).
"""
import os, sys

H = os.path.expanduser
D = H('~/data/fl4a'); W = H('~/work/fl4a/maps')
CTRL = {
    'c343': (H('~/work/fl3/s0a/concat_w047-w048_R5B2_z9500-11000_66.zarr'), 66, 8.64, 'reverse'),
    'title': (f'{D}/bench0139/title/zarr113', 28, 9.362, 'forward'),
    'w024': (f'{D}/bench0139/w024/zarr113', 28, 9.362, 'forward'),
    'w043': (f'{D}/w043/ct28.zarr', 28, 9.362, 'forward'),
    'w00': (f'{D}/p0841/w00/ct28.zarr', 28, 9.366, 'forward'),
    'ag144': (f'{D}/p0841/ag144/ct28.zarr', 28, 9.366, 'forward'),
    'ag174': (f'{D}/p0841/ag174/ct28.zarr', 28, 9.366, 'forward'),
}
NULL_CTRL = ['c343', 'w043', 'w00', 'ag144', 'ag174']
A4_CTRL = []
FAMILY = ['soup42', 'soup43', 'dn', 'kl']


def windows(n):
    if n == 66:
        return {f'd{d:+d}': (25 + d, 42 + d) for d in (-3, -1, 1, 3)}
    return {'z0-20': (0, 20), 'z3-23': (3, 23), 'z5-25': (5, 25), 'z8-27': (8, 27)}


def r0_window(c, n):
    return (25, 42) if n == 66 else ((4, 25) if c in ('w00', 'ag144', 'ag174') else (0, 28))


def vol(c, variant):
    p, *_ = CTRL[c]
    if variant == 'orig': return p
    if c in A4_CTRL and variant == 'shuf': return p.replace('_66.zarr', '_66_shuf.zarr')   # FL3 shuffle_layers seed 0
    return f'{D}/nullvol/{c}_{variant}.zarr'


def jobs(phase, ctrls):
    out = []
    for c in ctrls:
        p, n, um, _ = CTRL[c]
        variants = ['orig'] if phase in ('main', 'F', 'depth', 'a4null') else ['shuf', 'slr']
        for v in variants:
            src = vol(c, v); o = f'{W}/{c}/{v}/{c}'
            if phase in ('main', 'null'):
                ls, le = r0_window(c, n); out.append(('villa', src, 'r0', ls, le, f'{o}__r0__w{ls}-{le}'))
                for m in FAMILY:
                    for wn, (ls, le) in windows(n).items():
                        out.append(('villa', src, m, ls, le, f'{o}__{m}__{wn}'))
                zc = 33.5 if n == 66 else 13.5
                out.append(('hecate', src, '-', zc, um, f'{o}__hecate__zc{zc}'))
                # budget (log 2026-10-11): v8-in runs ~31 tiles/s on the 5070 Ti; stride 32 instead of his 21
                # everywhere, central window only except c343 (3 windows); no v8 on w024; stride-21 check on w043
                v8w = (19, 21, 23) if n == 66 else (2,)
                if v != 'orig':
                    v8w = v8w[1:2] if n == 66 else v8w
                if c != 'w024':
                    for ls in v8w:
                        out.append(('v8', src, '-', ls, 32, f'{o}__v8__l{ls}'))
                if c == 'w043' and v == 'orig':
                    out.append(('v8', src, '-', 2, '-', f'{o}__v8s21__l2'))
            elif phase == 'F' and c != 'w024':          # budget: w024 (47 Mpx) only in 'main' (log 2026-10-11)
                for m in FAMILY:
                    for wn, (ls, le) in windows(n).items():
                        out.append(('villaF', src, m, f'{ls}:{le}', um, f'{o}__F{m}__{wn}'))
            elif phase == 'a4null' and n == 66:
                # A4 null: the whole stack at d = +-12 (far from the sheet), one window per family model
                for d in (-12, 12):
                    for m in FAMILY:
                        out.append(('villa', src, m, 25 + d, 42 + d, f'{o}__{m}__far{d:+d}'))
                    out.append(('hecate', src, '-', 32.5 + 1 + d, um, f'{o}__hecate__far{d:+d}'))
                    out.append(('v8', src, '-', 21 + d, 32, f'{o}__v8__far{d:+d}'))
            elif phase == 'depth' and n == 66:
                for m in ('dn', 'soup42'):
                    for d in (-10, -8, -2, 2, 8, 10):
                        out.append(('villa', src, m, 25 + d, 42 + d, f'{o}__{m}__d{d:+d}'))
    return out


if __name__ == '__main__':
    phase = sys.argv[1]
    ctrls = sys.argv[2:] or (NULL_CTRL if phase == 'null' else [c for c in CTRL if c not in A4_CTRL])
    for j in jobs(phase, ctrls):
        print('\t'.join(str(x) for x in j))
