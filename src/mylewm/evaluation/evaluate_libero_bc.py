"""Native LIBERO-10 evaluation of frozen-encoder flow BC (dry-run by default)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import h5py
import numpy as np
import torch

from mylewm.data.libero_bc_data import split_contract, validate_manifest
from mylewm.data.verification import file_sha256, verify_training_data
from mylewm.paths import ROOT, libero_paths
from mylewm.policy.libero_bc import BCController, load_bc_checkpoint
from mylewm.training.train_libero_bc import write_json


def pair(obs):
    return np.stack([obs['agentview_image'], obs['robot0_eye_in_hand_image']])


def array_hash(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def rollout(env, controller, obs, budget):
    """Only measured camera images enter the policy; success comes from LIBERO."""
    if budget < 1:
        raise ValueError('Positive environment budget required')
    if env.check_success():
        raise ValueError('Initial state already successful')
    frames, actions = [pair(obs)], []
    success = False
    begin = time.monotonic()
    policy_seconds = 0.
    for _ in range(budget):
        start = time.monotonic()
        action = controller.select_action(pair(obs))
        policy_seconds += time.monotonic() - start
        obs, _, done, _ = env.step(action)
        actions.append(np.asarray(action).copy())
        frames.append(pair(obs))
        success = bool(env.check_success())
        if success or done:
            break
    return {'success': success, 'steps': len(actions), 'elapsed': time.monotonic() - begin,
            'policy_seconds': policy_seconds, 'chunks': controller.chunks}, np.asarray(actions), frames


def save_video(path, frames, fps=20):
    import av
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=fps)
        stream.height, stream.width = frames[0].shape[1], frames[0].shape[2] * 2
        stream.pix_fmt = 'yuv420p'
        for cameras in frames:
            frame = av.VideoFrame.from_ndarray(np.concatenate(list(cameras), axis=1), format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def reference_images(m, task_name):
    """Held-out successful demo for display only, never supplied to the policy."""
    index = m['task_names'].index(task_name)
    with h5py.File(m['files'][index]['path'], 'r') as f:
        for row in m['test']:
            if row['task'] == index:
                demo = f[f'data/{row["demo"]}']
                if float(demo['rewards'][-1]) > 0:
                    return np.stack([demo[f'obs/{c}'][-1] for c in m['camera_order']]), row['demo']
    return None, None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True, help='step_N_bc.pt, not a world-model checkpoint')
    p.add_argument('--manifest', type=Path, default=ROOT / 'output/manifests/libero10/manifest.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--task-ids', type=int, nargs='+', default=list(range(10)))
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--budget', type=int, default=520)
    p.add_argument('--execute-actions', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--render-audit-dir', type=Path, default=ROOT / 'output/libero10/render_audit')
    p.add_argument('--verify-data', action='store_true')
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    args.checkpoint, args.manifest, args.output = [x.resolve() for x in (args.checkpoint, args.manifest, args.output)]
    if not args.checkpoint.is_file() or not args.checkpoint.name.endswith('_bc.pt'):
        raise ValueError('Supply a saved *_bc.pt policy')
    if args.output.exists() or args.output == ROOT / 'output' or not args.output.is_relative_to(ROOT / 'output'):
        raise ValueError('Use a fresh evaluation directory under output/')
    if min(args.episodes, args.budget, args.execute_actions) < 1 or min(args.offset, args.seed) < 0:
        raise ValueError('Invalid BC evaluation budget')
    if len(set(args.task_ids)) != len(args.task_ids) or not set(args.task_ids) <= set(range(10)):
        raise ValueError('Require unique native LIBERO-10 task IDs 0..9')
    m = json.loads(args.manifest.read_text())
    validate_manifest(m)
    identity = verify_training_data(m, full=args.verify_data)
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(controller='flow_matching_bc_v1', checkpoint_sha256=file_sha256(args.checkpoint),
                    manifest_sha256=file_sha256(args.manifest), data_identity=identity,
                    metric='native env.check_success()', goal='none; task ID supplied to policy',
                    initial_history='one current measured camera pair; ten settling actions',
                    control_fps=20, video_fps=20,
                    source_sha256={name: file_sha256(path) for name, path in {
                        'evaluation': Path(__file__),
                        'policy': Path(__file__).parents[1] / 'policy/libero_bc.py',
                        'data': Path(__file__).parents[1] / 'data/libero_bc_data.py'}.items()})
    if not args.execute:
        print(json.dumps(metadata, indent=2))
        print('Dry-run: no policy load, simulator, rendering, or output creation. Render audits checked with --execute.')
        return
    args.output.mkdir(parents=True)
    phase = 'loading_policy'
    write_json(args.output / 'config.json', metadata)
    write_json(args.output / 'status.json', {'state': 'running', 'phase': phase})
    try:
        torch.set_num_threads(4)
        policy, bundle = load_bc_checkpoint(args.checkpoint, args.device)
        if list(policy.task_names) != m['task_names'] or bundle['provenance']['split_contract'] != split_contract(m):
            raise ValueError('BC checkpoint and evaluation manifest/task mapping differ')
        if not 1 <= args.execute_actions <= policy.config.horizon:
            raise ValueError('Execution chunk exceeds learned horizon')
        for path in libero_paths():
            sys.path.insert(0, str(path))
        if os.environ.get('MUJOCO_GL') != 'osmesa':
            raise RuntimeError('Run via bash scripts/run_libero.sh for audited native OSMesa images')
        import mujoco
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        suite = benchmark.get_benchmark_dict()['libero_10']()
        audits, initial_states = {}, {}
        for task_id in args.task_ids:
            task = suite.get_task(task_id)
            if task.name not in policy.task_names:
                raise ValueError(f'Task absent from BC policy: {task.name}')
            audit_path = args.render_audit_dir / f'task_{task_id}' / 'report.json'
            audit = json.loads(audit_path.read_text())
            if not (audit['passed'] and audit['task'] == task.name and audit['mujoco'] == mujoco.__version__
                    and audit['render_backend'] == 'osmesa'):
                raise ValueError(f'Render audit missing or incompatible for {task.name}')
            audits[task.name] = file_sha256(audit_path)
            initial_states[task.name] = file_sha256(Path(get_libero_path('init_states')) / task.problem_folder / task.init_states_file)
        metadata.update(mujoco=mujoco.__version__, render_backend='osmesa', render_audit_sha256=audits,
                        init_states_sha256=initial_states, horizon=policy.config.horizon,
                        euler_steps=policy.config.euler_steps, policy_config=bundle['config'],
                        bc_step=bundle['step'],
                        training_provenance=bundle['provenance'])
        write_json(args.output / 'config.json', metadata)
        phase = 'environment_rollouts'
        results = []
        with (args.output / 'episodes.jsonl').open('x') as log:
            for task_id in args.task_ids:
                task = suite.get_task(task_id)
                states = torch.load(Path(get_libero_path('init_states')) / task.problem_folder / task.init_states_file,
                                    map_location='cpu', weights_only=False)
                if args.offset + args.episodes > len(states):
                    raise ValueError('Insufficient native initial states')
                reference, demo_name = reference_images(m, task.name)
                env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file),
                                         camera_heights=128, camera_widths=128, control_freq=20)
                try:
                    for init_id in range(args.offset, args.offset + args.episodes):
                        seed = args.seed + task_id * 10000 + init_id
                        random.seed(seed); np.random.seed(seed); env.seed(seed)
                        env.reset(); obs = env.set_init_state(states[init_id])
                        for _ in range(10):
                            obs, _, _, _ = env.step(np.zeros(7))
                        initial_state = np.asarray(env.get_sim_state()).copy()
                        controller = BCController(policy, task.name, seed, args.execute_actions)
                        print(f'BC task {task_id}, init {init_id}: starting (budget {args.budget})', flush=True)
                        row, actions, frames = rollout(env, controller, obs, args.budget)
                        stem = f'task{task_id}_init{init_id}'
                        video_path = Path('viewer/videos') / f'trial_{len(results)+1:03d}.mp4'
                        (args.output / video_path.parent).mkdir(parents=True, exist_ok=True)
                        save_video(args.output / video_path, frames)
                        arrays = {'actions': actions, 'initial_state': initial_state, 'initial': frames[0], 'final': frames[-1]}
                        if reference is not None:
                            arrays['goal'] = reference  # UI reference only, not a BC input.
                        np.savez_compressed(args.output / f'{stem}.npz', **arrays)
                        row.update(task_id=task_id, task=task.name, instruction=task.language, init_id=init_id,
                                   trial=len(results) + 1, video=str(video_path),
                                   initial_state_sha256=array_hash(initial_state), initial_image_sha256=array_hash(frames[0]),
                                   goal_demo=demo_name, goal_image_sha256=array_hash(reference) if reference is not None else None)
                        log.write(json.dumps(row) + '\n'); log.flush(); results.append(row)
                        write_json(args.output / 'status.json', {'state': 'running', 'phase': phase, 'episodes': len(results)})
                        print(json.dumps(row), flush=True)
                finally:
                    env.close()
        rates = {str(t): float(np.mean([r['success'] for r in results if r['task_id'] == t])) for t in args.task_ids}
        summary = {'task_rates': rates, 'macro_success': float(np.mean(list(rates.values()))), 'episodes': len(results)}
        write_json(args.output / 'summary.json', summary)
        lines = ['LIBERO BC: native task success; reference demo images are not policy inputs.',
                 '回目\ttask ID\tinit ID\t成功/失敗\t実行step\tタスク']
        lines += [f'{r["trial"]}回目\t{r["task_id"]}\t{r["init_id"]}\t{"成功" if r["success"] else "失敗"}\t{r["steps"]}\t{r["task"]}' for r in results]
        (args.output / 'results.txt').write_text('\n'.join(lines) + '\n')
        phase = 'visual_report'
        from mylewm.evaluation.build_libero_ui import build
        build(args.output, args.output / 'viewer')
        write_json(args.output / 'status.json', {'state': 'succeeded', **summary, 'visual_report': 'viewer/index.html'})
        print(json.dumps(summary), flush=True)
        print(f'Completed. Visual report: {args.output / "viewer/index.html"}', flush=True)
    except BaseException as e:
        write_json(args.output / 'status.json', {'state': 'failed', 'phase': phase, 'error': f'{type(e).__name__}: {e}'})
        raise


if __name__ == '__main__':
    main()
