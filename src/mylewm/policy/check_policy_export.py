"""Replay real local LeRobot observations through ckpt and exported policy.

No robot connection or environment action execution. The recorded executed
actions update history; proposed CEM actions are compared, not sent anywhere.
"""
import argparse
import json
from pathlib import Path

import torch

from mylewm.data.verification import file_sha256, verify_training_data
from mylewm.data.contract import inference_manifest
from mylewm.data.lerobot_data import open_local
from mylewm.policy.runtime import load_checkpoint_controller


def check(checkpoint, policy_path, manifest_path, episode, ticks, output):
    from mylewm.policy.lerobot import BTSIGRegPolicy
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get('data_format') != 'lerobot':
        raise ValueError('This real-data check requires a prepared LeRobot manifest')
    identity = verify_training_data(manifest)
    policy = BTSIGRegPolicy.from_pretrained(policy_path)
    inference_manifest(policy.model, manifest)
    native = load_checkpoint_controller(checkpoint, policy.config.planner,
                                        legacy_contract=policy.config.input_contract)
    ds = open_local(Path(manifest['dataset']).parent.parent, manifest['camera_key'],
                    manifest['frameskip'], manifest.get('revision'), manifest.get('repo_id') or 'local/dataset')
    if not 0 <= episode < len(manifest['episode_lengths']):
        raise ValueError('Invalid episode')
    k = manifest['frameskip']
    start = 2 * k
    if ticks < k + 1 or start + ticks >= manifest['episode_lengths'][episode]:
        raise ValueError('Choose at least frameskip+1 ticks within a sufficiently long episode')
    offset = manifest['episode_offsets'][episode]
    camera = manifest['camera_key']
    # Select first actual frame from each official multi-frame query. No
    # synthesized pixels, repeated-frame priming, or random action histories.
    images = torch.stack([ds[offset + i * k][camera][0] for i in range(3)])
    table = ds.hf_dataset.with_format(None)
    actions = torch.tensor(table.select_columns(['action'])[offset:offset + manifest['episode_lengths'][episode]]['action'])
    history_actions = actions[:start].reshape(2, k, -1)
    goal = ds[offset + manifest['episode_lengths'][episode] - 1][camera][:1]
    times = [i * k / manifest['input_contract']['fps'] for i in range(3)]
    native.prime(images, history_actions, goal, times)
    policy.prime(images, history_actions, goal, times)
    # Use LeRobot's factory to reload the saved processors as a consumer would.
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    assert get_policy_class('bt_sigreg') is BTSIGRegPolicy
    pre, post = make_pre_post_processors(policy.config, pretrained_path=policy_path)
    output.mkdir(parents=True, exist_ok=False)
    status = output / 'status.json'
    status.write_text(json.dumps({'state': 'running'}))
    rows = []
    try:
        for i in range(start, start + ticks):
            frame = ds[offset + i]
            rgb = frame[camera][0]
            timestamp = float(frame['timestamp'])
            previous = actions[i - 1] if i > start else None
            expected = native.select_action(rgb[None], timestamp, previous)
            batch = {policy.config.camera_key: rgb, 'observation.timestamp': torch.tensor(timestamp)}
            if previous is not None:
                batch['observation.executed_action'] = previous
            processed = pre(batch)
            actual = post(policy.select_action(processed))[0]
            torch.testing.assert_close(actual, expected.cpu(), rtol=0, atol=0)
            if not torch.isfinite(actual).all():
                raise ValueError('Non-finite action')
            low = torch.tensor(policy.config.planner['action_low'])
            high = torch.tensor(policy.config.planner['action_high'])
            if not ((actual >= low) & (actual <= high)).all():
                raise ValueError('Action outside configured bounds')
            rows.append({'frame': i, 'timestamp': timestamp, 'action': actual.tolist(),
                         'cem_cost': native.last_cost})
            print(f'Real-data action agreement {len(rows)}/{ticks}: frame {i}', flush=True)
        result = {'state': 'succeeded', 'episode': episode, 'ticks': ticks,
                  'comparison': 'exact action equality across checkpoint and LeRobot policy, including processors',
                  'execution': 'offline recorded-history replay; no proposed action executed on a robot',
                  'checkpoint_sha256': file_sha256(checkpoint),
                  'manifest_sha256': file_sha256(manifest_path), 'data_identity': identity,
                  'policy_files': {p.name: file_sha256(p) for p in Path(policy_path).iterdir() if p.is_file()},
                  'rows': rows}
        (output / 'results.json').write_text(json.dumps(result, indent=2))
        status.write_text(json.dumps({'state': 'succeeded', 'ticks': ticks}))
        return result
    except Exception as exc:
        status.write_text(json.dumps({'state': 'failed', 'error': str(exc)}))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--episode', type=int, required=True)
    p.add_argument('--ticks', type=int, default=6)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if not args.execute:
        print('Dry-run: no checkpoint/data load, planning, or output creation')
        return
    check(args.checkpoint, args.policy, args.manifest, args.episode, args.ticks, args.output)


if __name__ == '__main__':
    main()
