"""FL4-C C3 checkpoint choice (rule fixed in the log before training).

    c3_select.py BENCH_C3.json   -> prints "<treatment tag> <control tag> <step>", e.g. "t12k c12k 012000"

Treatment step s* = argmax over s in {4k, 8k, 12k, 16k} of the mean of AUC(X:t<s>) on 0009B and box-AUC(X:t<s>) on
c343 (true direction); ties -> the later step. The control is compared at the same step.
"""
import json, sys

b = json.load(open(sys.argv[1]))['controls']
best = None
for s in ('04', '08', '12', '16'):
    k = f'X:t{s}k'
    v = (b['0009B']['maps'][k]['auc'] + b['c343']['maps'][k]['auc']) / 2
    if best is None or v >= best[0]:
        best = (v, s)
print(f't{best[1]}k c{best[1]}k 0{best[1]}000')
