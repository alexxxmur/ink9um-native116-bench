# ink_9um at native 8.64 µm / 116 keV: a frozen benchmark and a matched-control test of dense native pseudo-labels

**Question.** Half of the scrolls eligible for the First Letters prize were scanned at 8.64 µm / 116 keV / 1.2 m, but
none of the public ~9 µm ink models was trained on native data from that protocol (`ink_9um`: 2.4 µm volumes pooled to
9.6 µm plus five native 113 keV segments of PHerc0139; `dense_native`: native 113 keV; Hecate 9.6: distilled from 2.4 µm).
Does the 113 keV recipe that worked for Nieuwlaar (dense pseudo-labels on native scans, +4 AUC points on a held-out
segment against a matched control) transfer to 116 keV if one repeats it with native 116 keV data?

**Answer (one day of GPU, everything public, benchmark frozen before training).**

1. **The public models already read native 116 keV ink without any 116 keV training.** On PHerc0009B (8.64 µm / 116 keV,
   manual ink labels, in no model's training set) `dense_native` scores AUC 0.900 and the 5-model stack 0.913; the maps
   show the letters (`figures/0009B_compare.png`). There is no "116 keV domain gap" to close.
2. **Dense native 116 keV pseudo-labels (7 PHerc0343P segments, 2.2 µm canonical teacher) do not improve a different
   116 keV scroll.** On 0009B the fine-tuned model scores 0.903 against 0.904 for the matched control and 0.900 for the
   initialization; the four treatment and four control checkpoints overlap completely.
3. **The same training gives a large, consistent gain on the scroll it was trained on.** On our own render of
   Nieuwlaar's PHerc0343 w047–w048 segment (letter boxes as key) box-AUC rises from 0.654 to 0.711, all four treatment
   checkpoints above all four control checkpoints (which fall to 0.627–0.636). PHerc0343P are pieces of PHerc0343
   (separate scan, same inventory number). The gain is a same-scroll effect, not a protocol effect.

So: for 116 keV targets, spend effort on surfaces and on data from the target scroll itself, not on protocol-specific
fine-tuning.

## Benchmark

AUC in the true depth direction. `R0` = released `ink_9um` seed 42 step 75k; `DN` = `dense_native_016000`
(Nieuwlaar; the initialization of both arms); `STACK-A` = z-mean of soup42, soup43, dense_native, KLAVIS dense (each
the mean of four depth windows) and Hecate 9.6; `X` = the fine-tuned model, mean of the four depth windows.
Checkpoint 12k was selected by a rule fixed before training (mean of 0009B AUC and c343 box-AUC); the control is read
at the same step. Full numbers: `benchmark/BENCHMARK.csv`, `benchmark/bench_final.json`; the frozen definition:
`benchmark/benchmark_frozen.json` (2026-10-12 11:31 MSK, before the first training step).

| control | protocol | key | R0 | DN (init) | STACK-A | X treatment 4k / 8k / **12k** / 16k | X control 4k / 8k / **12k** / 16k |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **PHerc0009B** 20250919125754 | 8.64 µm / 116 keV | manual labels (20260918) | 0.781 | 0.900 | 0.913 | 0.901 / 0.901 / **0.903** / 0.902 | 0.902 / 0.903 / **0.904** / 0.905 |
| **c343** (PHerc0343 w047–w048, our 66-layer render) | 8.64 µm / 116 keV | Nieuwlaar's letter boxes, box-AUC | 0.632 | 0.654 | 0.678 | 0.704 / 0.696 / **0.711** / 0.706 | 0.636 / 0.632 / **0.627** / 0.631 |
| held-out PHerc0343P 20250511003658 | 8.64 µm / 116 keV | 2.2 µm teacher (upper bound) | 0.673 | 0.817 | 0.814 | 0.825 / 0.837 / **0.837** / 0.836 | 0.824 / 0.825 / **0.819** / 0.823 |
| PHerc0139 title | 9.362 µm / 113 keV | 78 keV teacher classes (Nieuwlaar's bench) | 0.814 | 0.960 | 0.937 | 0.964 / 0.969 / **0.968** / 0.967 | 0.975 / 0.971 / **0.970** / 0.971 |
| PHerc0841 w00 | 9.366 µm / 113 keV | labels 20260918 | 0.768 | 0.881 | 0.892 | 0.889 / 0.893 / **0.892** / 0.889 | 0.898 / 0.893 / **0.893** / 0.891 |
| PHerc0841 ag144 | 9.366 µm / 113 keV | labels 20260918 | 0.760 | 0.809 | 0.822 | 0.807 / 0.807 / **0.812** / 0.810 | 0.800 / 0.802 / **0.803** / 0.804 |
| PHerc0841 ag174 | 9.366 µm / 113 keV | labels 20260918 | 0.761 | 0.784 | 0.829 | 0.774 / 0.778 / **0.775** / 0.776 | 0.788 / 0.780 / **0.776** / 0.777 |

Letter-scale score (ShribyrLabs 48 µm high-pass correlation) on 0009B does not improve either: X 0.022 against DN 0.028
and STACK-A 0.058. Replacing DN by X inside the stack changes 0009B from 0.913 to 0.915 and c343 from 0.678 to 0.696.

Pre-registered gates (all three had to pass): (1) 0009B, treatment − control ≥ +0.03 and treatment − init ≥ +0.03 with
non-overlapping checkpoint ranges: **failed** (−0.001 / +0.003); (2) c343 box-AUC ≥ init + 0.03: passed (+0.058);
(3) 0841 median AUC not below control − 0.01: passed (0.812 against 0.803).

Figures (`figures/`): raw CT sheet layer | dense_native | treatment 12k | control 12k | key, for c343, 0009B and the
held-out 0343P segment, each panel stretched to its own 1–99 % percentiles.

## What was done

Everything follows Nieuwlaar's `ink9um-dense-native` recipe (`training/REPRODUCE.md` there) with the native 116 keV
data added as one more sampler key, and his frozen-benchmark / matched-control discipline.

**Data (all public, S3 open-data bucket and the HF `ink_9um` labels).**

- PHerc0343P: 8 segments with native 8.64 µm / 116 keV surface volumes and canonical 2.215 µm ink maps
  (`*new_canon_autoresearch_recipe*.tif`) as teacher. 7 train, `20250511003658-tifxyz` held out.
- PHerc0009B 20250919125754: native 8.64 µm / 116 keV volume, canonical 2.401 µm teacher, manual labels 20260918.
  Used only for threshold calibration (supervision region) and as the held-out test; never trained on.
- Replay corpus, 113 keV: Nieuwlaar's 10 native PHerc0139 segments with dense 78 keV-teacher labels (his scripts and
  thresholds, reproduced pixel-exact except ±757 px on w040/w041 from the 2026-08-19 label re-run) plus the 5 official
  native9 sparse label sets. His four benchmark segments (title, w024, w047, w053) excluded as in his configs.
- c343: our own 66-layer render (VC3D, 8.64 µm) of Nieuwlaar's public w047–w048 segment; key = his letter boxes.

**Labels for 116 keV** (`scripts/labels116.py`, reports in `labels/`).

- The S3 116 keV surface volumes have 31 layers (sheet at 15); layers [1, 29) are taken so the sheet sits at 14 like
  the 28-layer 113 keV volumes and the loader, windows and label format need no change (`scripts/prep116.py`).
- Teacher-to-native alignment: the prediction-based NCC peak is flat on 116 keV (the models barely see the letters
  before training), so the shift is taken from CT-to-CT registration of the 2.2 / 2.4 µm render of the same mesh
  against the native render (`scripts/ct_align.py`): [7, 7] px on all eight 0343P segments (NCC 0.59–0.63),
  [5, 8] px on 0009B (NCC 0.47). Nobody was excluded by alignment.
- Threshold by balanced accuracy against the 0009B manual labels: t\* = 75/255 with a flat plateau 48–130; 66/255
  (0.26) was fixed before training as the upper bound of the pre-registered range, inside the plateau on every region
  (`labels/theta_0009B.json`). Applied to all 0343P segments (they have no manual labels).
- Segment selection by the teacher map, before training: 2–40 % ink area and ≥ 10 letter-scale components; all 7
  passed (`labels/labels116_report.json`, previews in `labels/previews/`). Caveat: on the `_b2` segments the teacher
  shows letter fragments rather than connected lines; the rule does not distinguish that.

**Training** (`configs/`, `scripts/make_configs_fl4c.py`). villa `ink_detection` training (commit 0fb38c45,
`vesuvius.ink_detection.training.train`), Nieuwlaar's `train_dense_native.json` as the contract: patch 17×128×128,
z-window jitter ±2, robust-MAD normalisation, BCE label smoothing 0.5, AdamW 2e-5, wd 3e-5, warm-up 500, cosine,
fp16, grad clip 1, seed 42, batch 64. Differences from his run: init `dense_native_016000` (his best public model)
instead of KLAVIS; **16 000 steps**, checkpoints every 4 000; replay corpus = the native 113 keV set only (his 24-segment
pooled 2.4 µm corpus, ~250 GB of S3 reads, was not used). Treatment batch quota 44 native-113 / 20 native-116 of 64
(31 %, the share his native set had); control 64 / 0. The two configs differ in exactly the quota, one `datasets[]`
entry, `out_dir` and `description` (`configs/diff_treatment_vs_control.txt`). Both arms saw the identical set of
0139 patches (the patch cache of the treatment filtered to the 0139 entries). Sampler audits:
`benchmark/sampling_observed_*.json` (treatment 68.75 / 31.25 %, control 100 %). Validation: the corpus has no
validation masks, so a shared region (lower 15 % of w046 and w049, dense 0139, not in the benchmark) was excluded
from training in both arms (`labels/val_masks.json`).

**Cost.** RTX 5070 Ti 16 GB under WSL: 2.4–2.6 it/s at batch 64 (9.4 GB VRAM), 1 h 51 min per arm; ~20 GB of
downloads; benchmark maps ~1.5 h in total.

## Weights

`weights/` (GitHub release assets; 273 MB each; plain `state_dict`s with the same 508 keys as
`Nieuwlaar/ink9um-dense-native/weights/dense_native_016000.safetensors`, loadable the same way):

| file | arm | step | sha256 |
| --- | --- | --- | --- |
| `dense_native116_treatment_012000.safetensors` | native-113 + native-116 | 12 000 | `aae9c543a4d3dae2e639d277a1d7ae968bfb64bc123adb823caf241502004837` |
| `dense_native116_control_012000.safetensors` | native-113 only (matched control) | 12 000 | `237174c60c89e0bc2d7423451792a64c2b164ccab583a840e4efb478765f36f9` |

The treatment model is useful on PHerc0343 material; on everything else it equals the initialization. It is published
for the record, not as a recommendation.

## Reproduce

```bash
# 0. villa ink_detection (commit 0fb38c45 or later) in a Python env with torch; Nieuwlaar's repo for native_bench.py,
#    build_native_dense_labels.py and theta_calibration.json
scripts/fetch_c0.sh                      # ~20 GB: 0343P, 0009B, native 0139 volumes + teachers + native9 labels
python scripts/prep116.py                # 31 -> 28 layers for the 116 keV volumes
python scripts/ct_align.py ... ; python scripts/align116.py ...   # teacher shift per segment (reports in labels/)
python scripts/labels116.py calib        # t* on 0009B manual labels  -> labels/theta_0009B.json
python scripts/labels116.py build        # dense 116 keV labels for 0343P
scripts/setup_0139_labels.sh             # Nieuwlaar's native-113 dense labels (his scripts)
python scripts/val_masks.py; python scripts/make_configs_fl4c.py
python -m vesuvius.ink_detection.training.train configs/train_treatment.json
python -m vesuvius.ink_detection.training.train configs/train_control.json
python scripts/bench_prep.py             # benchmark inputs (0009B crop + labels on the native grid, held-out 0343P)
python scripts/bench.py jobs base > jobs.tsv; python scripts/stack_infer.py jobs.tsv     # baseline maps
python scripts/bench.py jobs t12k=<runs>/treatment/ckpt_012000.pth >> jobs.tsv; ...      # new-model maps
python scripts/bench.py measure out.json --models t04k t08k t12k t16k c04k c08k c12k c16k
python scripts/c3_select.py out.json     # checkpoint choice by the pre-registered rule
```

Paths at the top of the scripts (`<root>`, `<runs>`, model checkpoints, the villa checkout) must be edited first; the
scripts are the ones that ran, trimmed of our local paths. `scripts/measure.py` / `scripts/plan.py` hold the control
registry and metric code shared with our earlier calibration of the public models on 0139 / 0841 / 343
(Nieuwlaar's `native_bench` numbers are reproduced: `dense_native` on the title 0.9548, released seed-42 0.8136).

## Caveats

- One seed, one initialization, 16k steps, a single 116 keV scroll with manual labels. 0009B is a strong-ink segment
  (AUC 0.90 before any training, teacher-vs-manual 0.957), so a small gain would have been hard to see there; it was the
  only independent 116 keV test available.
- c343 and 0343P share an inventory number; the c343 gain is reported as a same-scroll effect, which is also the
  conservative reading.
- The 0343P teacher maps sit 5–8 px off the native canvas and were registered by CT-to-CT NCC; residual
  misregistration and the fragmentary `_b2` teachers may have weakened the labels. There is no second 116 keV scroll
  with manual labels to check this against.
- Held-out 0343P uses the teacher as key, so its numbers are an upper bound (Nieuwlaar's caveat §8 applies).

## Credits and licenses

- Recipe, benchmark code, label-building scripts, `dense_native_016000`, soup checkpoints: Erwin Nieuwlaar,
  `Nieuwlaar/ink9um-dense-native` (MIT). KLAVIS dense checkpoint: `domenicor046/ink9um-dense`.
- `ink_9um`, `hecate`, canonical 2 µm ink maps, surface volumes, labels: Vesuvius Challenge / Scroll Prize open data.
- Letter-scale score: ShribyrLabs `vesuvius-reports`. Depth-window and native-0841 conventions: AndreasHad04
  `villa-apple-silicon` (his 0009B native measurement, 0.815 for the released model, is reproduced here as R0 0.781 on
  our tighter crop with the released window).
- Code and weights in this repository: MIT. Author: alexxxmur, October 2026.
