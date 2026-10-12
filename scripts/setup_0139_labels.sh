#!/bin/bash
# FL4-C C0.4: Nieuwlaar's native-113 keV dense labels for the 10 PHerc0139 replay segments, with HIS scripts and HIS
# shipped t* (training/theta_calibration.json: w040/w041 own, others median 47). Workspace copy so that his
# W/data/... paths land in ~/data/fl4c; segment dirs are symlinks (zarr113 -> volumes/native113/<w>.zarr, teacher tif).
set -euo pipefail
R=$HOME/data/fl4c; N=$R/nieuwlaar/training; SRC=$HOME/data/fl4a/dense-native/training
mkdir -p $N/data/segments; cp -r $SRC/scripts $N/; cp $SRC/theta_calibration.json $N/data/theta_calibration.json
[ -e $N/data/labels ] || ln -s $R/labels $N/data/labels
# zarr 3.4 (env fl3) has no Group.create_dataset: same array (v2 format, chunks, blosc zstd-3 bitshuffle) via create_array
python3 - $N/scripts/native_common.py <<'PY'
import sys; p = sys.argv[1]; s = open(p).read()
old = """    g = zarr.open_group(dst_path, mode="w")
    arr = g.create_dataset("0", shape=(d, h, w), chunks=NATIVE_CHUNKS, dtype=np.uint8,
                           fill_value=0, compressor=NATIVE_COMPRESSOR, overwrite=True)"""
new = """    g = zarr.open_group(dst_path, mode="w", zarr_format=2)  # FL4-C: zarr 3 API, same v2 array
    arr = g.create_array("0", shape=(d, h, w), chunks=NATIVE_CHUNKS, dtype=np.uint8,
                         fill_value=0, compressors=NATIVE_COMPRESSOR, overwrite=True)"""
assert old in s or new in s; open(p, 'w').write(s.replace(old, new)); print('patched native_common.py')
PY
python3 - "$R" "$N" <<'PY'
import json, os, sys
R, N = sys.argv[1], sys.argv[2]
segs = []
for line in open(f'{R}/teacher/sources.tsv'):
    w, s3 = line.split()
    sd = s3.split('/segments/')[1].split('/')[0]
    d = f'{N}/data/segments/{sd}'; os.makedirs(d, exist_ok=True)
    for src, dst in ((f'{R}/volumes/native113/{w}.zarr', f'{d}/zarr113'), (f'{R}/teacher/{w}.tif', f'{d}/{os.path.basename(s3)}')):
        if not os.path.lexists(dst): os.symlink(src, dst)
    segs.append({'wname': w, 'segment_dir': sd})
json.dump({'segments': segs}, open(f'{N}/data/split.json', 'w'), indent=1)
print(len(segs), 'segments')
PY
cd $N/scripts
NATIVE113_SEGMENTS=$N/data/segments NATIVE113_SPLIT=$N/data/split.json \
NATIVE9_LABELS=$R/labels/native9-scrollprizeorg-21slices NATIVE9_VOLUMES=$R/volumes/native113 \
  $HOME/envs/fl3/bin/python build_native_dense_labels.py --segs w030,w032,w033,w034,w040,w041,w043,w046,w049,w050 "$@"
cp $N/data/native_dense_report.json <repo>/labels/native_dense_0139_report.json
echo LABELS0139_DONE
