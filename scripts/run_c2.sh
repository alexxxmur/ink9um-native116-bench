#!/bin/bash
# FL4-C C2 + C3 night queue (one GPU queue): treatment -> control -> maps of every checkpoint on the frozen
# benchmark -> measure -> checkpoint choice (rule fixed in the log before training) -> nulls + c343 depth maps for the
# chosen step of both arms -> final measure. Run as: systemd-run --user --unit=fl4c-c2 -p MemoryMax=30G bash run_c2.sh
set -u
W=~/work/fl4c; PY=python; S=<repo>/scripts
L=$W/c2.log; export TMPDIR=$W/tmp; cd $W
log(){ echo "$(date +%F' '%T) $*" >> $L; }
log "START; init.scope memory.events: $(tr '\n' ' ' < /sys/fs/cgroup/init.scope/memory.events)"
for arm in treatment control; do
  if [ -f runs/$arm/ckpt_016000.pth ]; then log "$arm already done"; continue; fi
  log "TRAIN $arm start"
  $PY -m vesuvius.ink_detection.training.train configs/train_$arm.json > runs/$arm.log 2>&1
  rc=$?
  if [ -f runs/$arm/ckpt_016000.pth ]; then log "TRAIN $arm OK (rc $rc; success = ckpt_016000.pth present)"
  else log "TRAIN $arm FAILED rc $rc"; tail -30 runs/$arm.log >> $L; exit 1; fi
done
# C3: every checkpoint of both arms, orig inputs, all controls
: > jobs/c3.tsv; MODELS=""
for arm in treatment control; do
  for s in 004000 008000 012000 016000; do
    t=${arm:0:1}${s:1:2}k; MODELS="$MODELS $t"
    $PY $S/fl4c/bench.py jobs $t=$W/runs/$arm/ckpt_$s.pth >> jobs/c3.tsv
  done
done
log "C3 maps: $(wc -l < jobs/c3.tsv) jobs, models$MODELS"
$PY $S/fl4/stack_infer.py jobs/c3.tsv --batch-size 32 --workers 4 2>&1 | grep --line-buffered -E "^[0-9]+/|Error|Traceback" | grep --line-buffered -v "rc=0" >> $L
$PY $S/fl4c/bench.py measure $W/bench_c3.json --models $MODELS > measure_c3.log 2>&1
log "C3 measure done"
CH=$($PY $S/fl4c/c3_select.py $W/bench_c3.json) || { log "c3_select failed"; exit 1; }
log "C3 choice: $CH"
T=$(echo $CH | cut -d' ' -f1); C=$(echo $CH | cut -d' ' -f2); ST=$(echo $CH | cut -d' ' -f3)
: > jobs/c3b.tsv
for x in "$T=treatment" "$C=control"; do
  tag=${x%%=*}; arm=${x#*=}
  $PY $S/fl4c/bench.py jobs $tag=$W/runs/$arm/ckpt_$ST.pth --nulls --depth --ctrls w00 ag144 ag174 c343 0009B h0343P | grep -v "/orig/" >> jobs/c3b.tsv
  $PY $S/fl4c/bench.py jobs $tag=$W/runs/$arm/ckpt_$ST.pth --depth --ctrls c343 | grep "__d[-+]" >> jobs/c3b.tsv
done
log "C3 nulls/depth: $(wc -l < jobs/c3b.tsv) jobs"
$PY $S/fl4/stack_infer.py jobs/c3b.tsv --batch-size 32 --workers 4 2>&1 | grep --line-buffered -E "^[0-9]+/|Error|Traceback" | grep --line-buffered -v "rc=0" >> $L
$PY $S/fl4c/bench.py measure $W/bench_c3_final.json --models $MODELS --nulls > measure_c3_final.log 2>&1
log "C2C3_DONE"
