"""Copy part of the public open-data bucket to a local directory (anonymous, threaded, resumable).

    s3_get.py S3_PREFIX LOCAL_DIR [--only SUBPATH ...]
--only keeps keys whose path relative to the prefix starts with one of SUBPATH (metadata files at the
top level, .zgroup/.zattrs/zarr.json, are always kept). Existing files of the right size are skipped.
"""
import argparse, os, concurrent.futures as cf
import s3fs

ap = argparse.ArgumentParser(); ap.add_argument('prefix'); ap.add_argument('dest'); ap.add_argument('--only', nargs='*', default=None)
ap.add_argument('--threads', type=int, default=32)
a = ap.parse_args()
fs = s3fs.S3FileSystem(anon=True)
pre = a.prefix.rstrip('/').replace('s3://', '')
keys = fs.find(pre, detail=True)
todo = []
for k, info in keys.items():
    rel = k[len(pre) + 1:]
    top = '/' not in rel
    if a.only and not top and not any(rel.startswith(o) for o in a.only):
        continue
    dst = os.path.join(a.dest, rel)
    if os.path.exists(dst) and os.path.getsize(dst) == info['size']:
        continue
    todo.append((k, dst, info['size']))
print(f'{len(todo)} files, {sum(t[2] for t in todo) / 1e9:.2f} GB to fetch', flush=True)

def get(t):
    k, dst, _ = t
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for i in range(6):
        try:
            fs.get(k, dst + '.part'); os.replace(dst + '.part', dst); return
        except Exception as e:
            if i == 5: raise
with cf.ThreadPoolExecutor(a.threads) as ex:
    for n, _ in enumerate(ex.map(get, todo)):
        if n % 500 == 0: print(n, flush=True)
print('done', flush=True)
