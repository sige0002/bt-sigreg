"""Audit whether fixed-case control failures coincide with latent prediction error.

This is an offline expert-trajectory diagnostic.  It does not replay CEM and
therefore cannot by itself prove that a failure was caused by search or cost
ranking.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'lewm')]

import h5py
import numpy as np
import torch

from mylewm.training import preprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def failure_class(bt, official):
    if bt and official:
        return 'both_success'
    if not bt and not official:
        return 'both_failure'
    return 'bt_only_success' if bt else 'official_only_success'


def summarize(rows, key):
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return {'n': len(rows), 'mean': float(values.mean()),
            'median': float(np.median(values)), 'min': float(values.min()),
            'max': float(values.max())}


def model_statistics(model, dataset):
    if hasattr(model, 'training_action_mean'):
        return (model.training_action_mean.detach().cpu().numpy(),
                model.training_action_std.detach().cpu().numpy(), 'checkpoint_train_only')
    actions = dataset['action'][:].astype(np.float64)
    actions = actions[np.isfinite(actions).all(1)]
    return actions.mean(0), actions.std(0, ddof=0), 'upstream_all_data'


def audit_model(checkpoint, cases, dataset_path, manifest, device):
    model = torch.load(checkpoint, map_location=device, weights_only=False).eval()
    model.requires_grad_(False)
    rows = []
    with h5py.File(dataset_path) as dataset:
        mean, std, source = model_statistics(model, dataset)
        local = dict(manifest, action_mean=mean.tolist(), action_std=std.tolist())
        for index, case in enumerate(cases):
            offset = int(dataset['ep_offset'][case['episode']]) + int(case['start'])
            pixels = torch.from_numpy(dataset['pixels'][offset:offset + 36:5]).permute(0, 3, 1, 2).unsqueeze(0)
            actions = torch.from_numpy(dataset['action'][offset:offset + 35]).reshape(1, 7, 5, 2)
            if pixels.shape[1] != 8 or not torch.isfinite(actions).all():
                raise ValueError(f'invalid expert window at case {index}')
            x4, a3 = preprocess((pixels[:, :4], actions[:, :3]), local, device)
            with torch.inference_mode():
                encoded = model.encode({'pixels': x4, 'action': a3})
                predicted = model.predict(encoded['emb'][:, :3], encoded['act_emb'][:, :3])
                one = (predicted - encoded['emb'][:, 1:]).square().mean(-1)
                x3, _ = preprocess((pixels[:, :3], actions[:, :3]), local, device)
                normalized = ((actions.to(device).float() - torch.as_tensor(mean, device=device)) /
                              torch.as_tensor(std, device=device)).flatten(-2)
                rollout = model.rollout({'pixels': x3.unsqueeze(1)}, normalized.unsqueeze(1),
                                        history_size=3)['predicted_emb']
                target, _ = preprocess((pixels[:, -1:], actions[:, :1]), local, device)
                target_emb = model.encode({'pixels': target})['emb']
                five = (rollout[:, :, -1] - target_emb[:, None, -1]).square().mean()
            rows.append({'index': index, 'episode': case['episode'], 'start': case['start'],
                         'one_step_mse_mean': float(one.mean()),
                         'one_step_mse_each': one.squeeze(0).cpu().tolist(),
                         'five_step_terminal_mse': float(five)})
    return {'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
            'action_statistics_source': source, 'rows': rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bt-checkpoint', type=Path, required=True)
    p.add_argument('--official-checkpoint', type=Path, required=True)
    p.add_argument('--bt-result', type=Path, required=True)
    p.add_argument('--official-result', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = p.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to((ROOT / 'output').resolve()):
        p.error('--output must be a new JSON file under output/')
    paths = [args.bt_checkpoint, args.official_checkpoint, args.bt_result,
             args.official_result, args.manifest]
    if any(not path.is_file() for path in paths):
        p.error('all inputs must be existing trusted local files')
    manifest = json.loads(args.manifest.read_text())
    bt_result, official_result = (json.loads(args.bt_result.read_text()),
                                  json.loads(args.official_result.read_text()))
    expected_hashes = {'bt': sha(args.bt_checkpoint), 'official': sha(args.official_checkpoint)}
    if bt_result.get('checkpoint_sha256') != expected_hashes['bt']:
        raise ValueError('BT result/checkpoint hash mismatch')
    if official_result.get('checkpoint_sha256') != expected_hashes['official']:
        raise ValueError('official result/checkpoint hash mismatch')
    for result in (bt_result, official_result):
        if result.get('provenance', {}).get('manifest_sha256') != sha(args.manifest):
            raise ValueError('result/manifest hash mismatch')
    cases = [{'episode': e, 'start': s} for e, s in zip(bt_result['episodes'], bt_result['starts'], strict=True)]
    official_cases = [{'episode': e, 'start': s} for e, s in
                      zip(official_result['episodes'], official_result['starts'], strict=True)]
    if (cases != official_cases or len(cases) != len(bt_result['successes'])
            or len(cases) != len(official_result['successes'])):
        raise ValueError('results do not describe the same complete fixed cases')
    if cases != manifest['confirm'][:len(cases)]:
        raise ValueError('results do not match the fixed confirm prefix')
    models = {'bt': audit_model(args.bt_checkpoint.resolve(), cases, Path(manifest['dataset']), manifest, args.device),
              'official': audit_model(args.official_checkpoint.resolve(), cases, Path(manifest['dataset']), manifest, args.device)}
    for index, (bt_ok, official_ok) in enumerate(zip(bt_result['successes'], official_result['successes'])):
        category = failure_class(bt_ok, official_ok)
        models['bt']['rows'][index].update(control_success=bt_ok, paired_class=category)
        models['official']['rows'][index].update(control_success=official_ok, paired_class=category)
    summaries = {}
    for name, model in models.items():
        summaries[name] = {}
        for category in ('both_success', 'both_failure', 'bt_only_success', 'official_only_success'):
            selected = [row for row in model['rows'] if row['paired_class'] == category]
            summaries[name][category] = ({key: summarize(selected, key) for key in
                ('one_step_mse_mean', 'five_step_terminal_mse')} if selected else {'n': 0})
    report = {'schema': 'pusht_failure_audit_v1', 'scope':
              'Offline expert trajectories; association only, no CEM candidate-ranking or causal attribution.',
              'manifest_sha256': sha(args.manifest), 'cases': len(cases),
              'models': models, 'summaries': summaries,
              'interpretation': {'high_error_on_failures': 'consistent with a transition-model contribution',
                'low_error_on_failures': 'does not prove search/cost failure; expert and planned trajectories differ',
                'required_next_stage': 'instrument CEM candidates, predicted costs, selected actions, and realized outcomes'}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({'output': str(output), 'cases': len(cases), 'summaries': summaries}, indent=2))


if __name__ == '__main__':
    main()
