"""FL4-C C2 configs: treatment (native113 + native116) and matched control (native113 only), plus smoke variants.

    make_configs_fl4c.py OUT_DIR [--smoke-iters 60]

Contract = Nieuwlaar's training/configs/train_dense_native.json (every non-dataset key), with the protocol C2 changes:
init dense_native_016000 (weights_only), 16 000 steps, save/val every 4000, dataloader_workers 6, batch 64, seed 42.
Corpus (data root ~/data/fl4c, his layout):
  0139nat  native9 official sparse labels (w035 w039 w040 w041 w044) + native-dense-0139-78keV (10, his recipe)
  0343Pnat native-dense-0343P-116keV (treatment only; segments selected in artifacts/fl4c/labels116_report.json)
Quotas of 64: treatment 0139nat 44 / 0343Pnat 20; control 0139nat 64. The arms differ in description, out_dir,
fixed_scroll_prior.target_batch_counts and the one extra datasets[] entry. Held out: title w024 w047 w053 (0139),
0343P 20250511003658 (benchmark), 0009B, c343 and everything else not listed here.
"""
import argparse, copy, json, os

H = os.path.expanduser
R = H('~/data/fl4c')
BASE = json.load(open(H('~/data/fl4a/dense-native/training/configs/train_dense_native.json')))
INIT = H('~/data/fl3/ckpt/dense_native-016000.rebuilt.pth')
RUNS = H('~/work/fl4c/runs')
NATIVE9 = ['w035', 'w039', 'w040', 'w041', 'w044']
NATIVE_DENSE = ['w030', 'w032', 'w033', 'w034', 'w040', 'w041', 'w043', 'w046', 'w049', 'w050']
HELD_OUT = ['title', 'w024', 'w047', 'w053', '0343P_20250511003658-tifxyz', '0009B']
REPORT = '<repo>/labels/labels116_report.json'


def entry(labels_sub, segs, vol_dir, scroll, family):
    return {
        'segments_path': f'{R}/labels/{labels_sub}', 'segments': list(segs),
        'surface_volume_paths': {s: f'{R}/volumes/{vol_dir}/{s}.zarr' for s in segs},
        'volume_scale': 0, 'source_family': family, 'sampling_scroll': scroll,
        'sampling_physical_segment_keys': {s: f'{scroll}:{s}' for s in segs},
        'sampling_representation_keys': {s: f'{family}:{s}' for s in segs},
    }


def datasets(treatment, available_only=False):
    out = [entry('native9-scrollprizeorg-21slices', NATIVE9, 'native113', '0139nat', 'native_9p362_level0'),
           entry('native-dense-0139-78keV', NATIVE_DENSE, 'native113', '0139nat', 'native_9p362_level0_dense78keV')]
    if treatment:
        rep = json.load(open(REPORT))
        segs = sorted(k for k, v in rep.items() if v['train'])
        out.append(entry('native-dense-0343P-116keV', segs, 'native116', '0343Pnat', 'native_8p64_level0_dense116keV'))
    if available_only:
        for d in out:
            keep = [s for s in d['segments'] if os.path.isdir(d['surface_volume_paths'][s])
                    and os.path.isdir(os.path.join(d['segments_path'], s))]
            d['segments'] = keep
            for k in ('surface_volume_paths', 'sampling_physical_segment_keys', 'sampling_representation_keys'):
                d[k] = {s: d[k][s] for s in keep}
        out = [d for d in out if d['segments']]
    return out


def make(name, treatment, iters=16000, save_every=4000, warmup=500, available_only=False, desc=''):
    c = copy.deepcopy(BASE)
    c.update(description=desc, checkpoint=INIT, weights_only=True, num_iterations=iters, save_every=save_every,
             val_every=save_every, warmup_steps=warmup, dataloader_workers=6, batch_size=64, seed=42,
             out_dir=f'{RUNS}/{name}')
    q = {'0139nat': 44, '0343Pnat': 20} if treatment else {'0139nat': 64}
    c['fixed_scroll_prior'] = {'seed': 42, 'target_batch_counts': q}
    c['datasets'] = datasets(treatment, available_only)
    present = {d['sampling_scroll'] for d in c['datasets']}
    assert set(q) <= present, (q, present)
    assert sum(q.values()) == c['batch_size']
    for d in c['datasets']:
        for s in d['segments']:
            assert not any(s == h or s.startswith(h) for h in HELD_OUT), s
    return c


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out_dir'); ap.add_argument('--smoke-iters', type=int, default=60)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    cfgs = {
        'train_treatment.json': make('treatment', True, desc=(
            'FL4-C TREATMENT: init dense_native_016000; native113 PHerc0139 (5 official native9 sparse + 10 dense '
            '78keV-teacher, key 0139nat=44/64) + native 8.64um/116keV PHerc0343P dense 2.215um-teacher pseudo-labels '
            '(key 0343Pnat=20/64, t=66/255 from 0009B manual labels, CT-registered teacher shift). 16k steps. '
            'Held out: title w024 w047 w053, 0343P 20250511003658, 0009B, c343.')),
        'train_control.json': make('control', False, desc=(
            'FL4-C MATCHED CONTROL: identical to train_treatment.json except the native116 dataset entry is absent '
            'and the quota is 0139nat=64/64.')),
        'smoke_treatment.json': make('smoke_treatment', True, iters=a.smoke_iters, save_every=a.smoke_iters // 2,
                                     warmup=10, available_only=True, desc='SMOKE of train_treatment.json'),
    }
    for n, c in cfgs.items():
        json.dump(c, open(os.path.join(a.out_dir, n), 'w'), indent=1)
        print(n, sum(len(d['segments']) for d in c['datasets']), 'representations',
              {d['sampling_scroll'] + '/' + d['source_family']: len(d['segments']) for d in c['datasets']},
              c['fixed_scroll_prior']['target_batch_counts'])


if __name__ == '__main__':
    main()
