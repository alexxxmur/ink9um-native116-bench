#!/bin/bash
# FL4-C C0: native 116 keV (0343P, 0009B) + native 113 keV replay corpus (0139) inputs, ~27 GB
# (approved by owner 2026-10-12). Layout under ~/data/fl4c = Nieuwlaar's training/REPRODUCE.md.
# Only level 0 of every surface volume is fetched.
set -uo pipefail
R=$HOME/data/fl4c; S3=s3://vesuvius-challenge-open-data
PY=$HOME/envs/fl3/bin/python; G="$PY <repo>/scripts/s3_get.py"
DN=$HOME/data/fl4a/dense-native/training/scripts
mkdir -p $R/raw116 $R/teacher $R/volumes/native113 $R/labels/native9-scrollprizeorg-21slices $R/raw0009B
MINFREE_GB=150
guard(){ free=$(df -BG --output=avail $HOME | tail -1 | tr -dc 0-9); if [ "$free" -lt $MINFREE_GB ]; then echo "DISK GUARD: only ${free}G free, stopping"; exit 2; fi; echo "free ${free}G"; }
echo "START $(date)"; guard
# 1. PHerc0343P: 8 segments, native 8.64um/116keV level 0 + canonical 2.215um teacher map
for g in 20250511003658-tifxyz 20250902170435--5_b2 20250902170441--4_b2 20250902170447--3_b2 20250902171202--2_b2 \
         20250902171204--1_b2 20250904233748-0_b2 20250905172054-1_b2; do
  P=$S3/PHerc0343P/segments/$g
  $G $P/surface-volumes/8.64um-1.2m-116keV-volume-20250521134555.zarr $R/raw116/0343P_$g.zarr --only 0/ --threads 16
  id=${g%%-*}
  [ -s $R/teacher/0343P_$g.tif ] || $PY -c "import s3fs,sys; s3fs.S3FileSystem(anon=True).get(sys.argv[1],sys.argv[2])" \
    "$P/ink-detection/PHerc0343P-$id-2.215um-0.4m-111keV-volume-20260304131111-20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif" $R/teacher/0343P_$g.tif
done; guard
# 2. PHerc0009B 20250919125754: native level 0, 2.4um teacher, manual labels 20260918 (level 0)
P=$S3/PHerc0009B/segments/20250919125754-auto_grown_20250919055754487_inp_hr
$G $P/surface-volumes/8.64um-1.2m-116keV-volume-20250521125136.zarr $R/raw116/0009B_20250919125754.zarr --only 0/ --threads 16
[ -s $R/teacher/0009B_20250919125754.tif ] || $PY -c "import s3fs,sys; s3fs.S3FileSystem(anon=True).get(sys.argv[1],sys.argv[2])" \
  "$P/ink-detection/PHerc0009B-20250919125754-2.401um-0.35m-77keV-volume-20250820154339-20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif" $R/teacher/0009B_20250919125754.tif
for a in inklabels supervision validation; do
  $G $P/ink-labels/2.401um-volume-20250820154339/20260918/$a.zarr $R/raw0009B/$a.zarr --only 0/ --threads 16
done; guard
# 3. PHerc0139 native 113 keV replay corpus: 13 volumes + 78 keV teacher maps
while read w g; do
  P=$S3/PHerc0139/segments/$g
  $G $P/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr $R/volumes/native113/$w.zarr --only 0/ --threads 16
  if [ ! -s $R/teacher/$w.tif ]; then
    t=$($PY -c "import s3fs,sys; fs=s3fs.S3FileSystem(anon=True); print([x for x in fs.ls(sys.argv[1]+'/ink-detection') if '78keV' in x and 'autoresearch' in x and x.endswith('.tif')][0])" $P)
    $PY -c "import s3fs,sys; s3fs.S3FileSystem(anon=True).get(sys.argv[1],sys.argv[2])" $t $R/teacher/$w.tif; echo "$w $t" >> $R/teacher/sources.tsv
  fi
  guard
done <<LIST
w030 20250108000005-w030_2025010818
w032 20260203000000-w032_2026020303
w033 20260219000000-w033_2026021913
w034 20260303000000-w034_2026030317
w035 20260317000000-w035_2026031718
w039 20260302000000-w039_2026030210
w040 20250831000000-w040_2025083102
w041 20260108000000-w041_2026010816
w043 20260112000000-w043_2026011217
w044 20260115000000-w044_2026011522
w046 20260325000000-w046_20260325
w049 20260318000000-w049_20260318
w050 20260302000001-w050_2026030220
LIST
# 4. official native9 sparse labels (HF bucket, level 0)
$PY -c "
import sys; sys.path.insert(0,'$DN'); import fetch_aligned_labels as f
f.PREFIX='ink_9um/labels/native9-scrollprizeorg-21slices/'
sys.argv=['x','$R/labels/native9-scrollprizeorg-21slices','w035','w039','w040','w041','w044']; f.main()"
guard; du -sh $R/*; echo "C0_FETCH_DONE $(date)"
