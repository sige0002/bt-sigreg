"""Held-out dynamics and native candidate-ranking diagnostics, not benchmark success rates.

Uses the trusted object checkpoint and the existing planner without changing it.
Run native mode through scripts/run_libero.sh. Outputs must be fresh.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import h5py
import numpy as np
import torch

from mylewm.paths import ROOT, libero_paths
from mylewm.environments.libero_planner import CEM, image_tensor


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False, default=str))
    temporary.replace(path)


def windows(length):
    """Three shared anchors; all endpoints, including 32 actions, are observed."""
    last = length - 1 - 32
    if last < 8:
        raise ValueError('Demo is too short for real history and 32-action future')
    return [('early', 8), ('middle', (8 + last) // 2), ('late', last)]


@torch.no_grad()
def rollout(planner, history, past_actions, candidates):
    """Expose the latent predictions used by CEM.costs, preserving alignment."""
    z = history.expand(len(candidates), -1, -1)
    previous = past_actions.expand(len(candidates), -1, -1, -1)
    predictions = []
    for t in range(candidates.shape[1]):
        actions = torch.cat((previous, candidates[:, t:t+1]), dim=1)
        pred = planner.model.predict(z, planner.model.action_encoder(planner.normalize(actions)))[:, -1:]
        z = torch.cat((z[:, 1:], pred), dim=1)
        previous = actions[:, 1:]
        predictions.append(pred[:, 0])
    return torch.stack(predictions, dim=1)


class RecordedCEM(CEM):
    """Record the final mean sequence; inherited plan still executes unmodified."""
    def costs(self, history, past_actions, candidates, goal):
        costs = super().costs(history, past_actions, candidates, goal)
        elite = candidates[costs.topk(self.elites, largest=False).indices]
        self.final_sequence = (.1 * candidates[0] + .9 * elite.mean(0)).clone()
        return costs


def shuffled_actions(actions, generator, count=16):
    """Shuffle native time steps, preserving each complete seven-component action."""
    flat = actions.reshape(-1, 7)
    return torch.stack([flat[torch.randperm(len(flat), device=flat.device, generator=generator)]
                        .reshape_as(actions) for _ in range(count)])


def rank_correlation(x, y):
    from scipy.stats import spearmanr
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


def pair(obs):
    return np.stack([obs['agentview_image'], obs['robot0_eye_in_hand_image']])


def encode(model, images, device):
    return model.encode({'pixels': image_tensor(images, device)})['emb']


def reset_native_context(env, init_state, context_xml, seed):
    """Reset robot/gripper internals, then pin fixtures and gather real history."""
    random.seed(seed); np.random.seed(seed); env.seed(seed)
    # XML reset alone reuses the gripper object and its accumulated current_action.
    env.reset()
    env.reset_from_xml_string(context_xml)
    obs = env.set_init_state(init_state)
    frames = []
    for step in range(1, 14):
        obs, _, _, _ = env.step(np.zeros(7))
        if step in (5, 9, 13):
            frames.append(pair(obs))
    return obs, np.stack(frames), np.asarray(env.get_sim_state()).copy()


def offline(args, model, manifest):
    rows = []
    with (args.output / 'cases.jsonl').open('x') as log, torch.inference_mode():
        for task, name in enumerate(manifest['task_names']):
            cases = [c for c in manifest['test'] if c['task'] == task]
            with h5py.File(manifest['files'][task]['path'], 'r') as f:
                for ci, case in enumerate(cases):
                    d = f[f'data/{case["demo"]}']
                    donor = f'data/{cases[(ci + 1) % len(cases)]["demo"]}'
                    other = f[donor]
                    indices = sorted({i for _, t in windows(len(d['actions']))
                                      for i in (t-8, t-4, t, t+4, t+16, t+32)})
                    embeddings = {i: encode(model, np.stack([d[f'obs/{cam}'][i]
                                  for cam in manifest['camera_order']]), args.device) for i in indices}
                    for wi, (label, t) in enumerate(windows(len(d['actions']))):
                        history = torch.cat([embeddings[i] for i in (t-8, t-4, t)], dim=1)
                        past = torch.tensor(d['actions'][t-8:t], device=args.device).float().reshape(1, 2, 4, 7)
                        for h in (1, 4, 8):
                            seed = args.seed + task*10000 + ci*100 + wi*10 + h
                            planner = CEM(model, horizon=h, seed=seed)
                            correct = torch.tensor(d['actions'][t:t+4*h], device=args.device).float().reshape(h, 4, 7)
                            shuffled = shuffled_actions(correct, planner.generator)
                            donor_t = windows(len(other['actions']))[wi][1]
                            wrong = torch.tensor(other['actions'][donor_t:donor_t+4*h], device=args.device).float().reshape_as(correct)
                            candidates = torch.cat((correct[None], torch.zeros_like(correct)[None], wrong[None], shuffled))
                            predicted = rollout(planner, history, past, candidates)[:, -1]
                            target = embeddings[t+4*h].reshape(1, -1)
                            errors = (predicted-target).square().mean(-1)
                            torch.testing.assert_close(errors, planner.costs(history, past, candidates, target), atol=1e-6, rtol=1e-5)
                            persistence = (embeddings[t].reshape_as(target)-target).square().mean().item()
                            row = {'task_index':task, 'task':name, 'demo':case['demo'], 'donor':donor,
                                   'window':label, 't':t, 'horizon_actions':4*h, 'demo_final_success':bool(d['rewards'][-1]>0),
                                   'mse_correct':errors[0].item(), 'mse_zero':errors[1].item(),
                                   'mse_wrong_demo':errors[2].item(), 'mse_shuffle_mean':errors[3:].mean().item(),
                                   'mse_persistence':persistence,
                                   'correct_beats_shuffle_fraction':(errors[0]<errors[3:]).float().mean().item()}
                            if not torch.isfinite(errors).all():
                                raise FloatingPointError('Nonfinite prediction errors')
                            stem = f'task{task}_{case["demo"]}_{label}_h{h}'
                            np.savez_compressed(args.output / f'{stem}.npz', history=history.cpu().numpy(),
                                past_actions=past.cpu().numpy(), candidates=candidates.cpu().numpy(),
                                predicted=predicted.cpu().numpy(), target=target.cpu().numpy(), errors=errors.cpu().numpy())
                            rows.append(row); log.write(json.dumps(row)+'\n'); log.flush()
            print(json.dumps({'phase':'offline', 'task':name, 'cases_completed':len(rows)}), flush=True)
    groups = {}
    for h in (4, 16, 32):
        rs = [r for r in rows if r['horizon_actions']==h]
        base = np.mean([r['mse_persistence'] for r in rs])
        groups[str(h)] = {'cases':len(rs), 'mse_correct_mean':float(np.mean([r['mse_correct'] for r in rs])),
            'correct_over_persistence_ratio_of_means':float(np.mean([r['mse_correct'] for r in rs])/base),
            'correct_beats_persistence_fraction':float(np.mean([r['mse_correct']<r['mse_persistence'] for r in rs])),
            'correct_beats_zero_fraction':float(np.mean([r['mse_correct']<r['mse_zero'] for r in rs])),
            'correct_beats_wrong_demo_fraction':float(np.mean([r['mse_correct']<r['mse_wrong_demo'] for r in rs])),
            'correct_beats_shuffle_mean_fraction':float(np.mean([r['mse_correct']<r['mse_shuffle_mean'] for r in rs])),
            'shuffle_over_correct_ratio_of_means':float(np.mean([r['mse_shuffle_mean'] for r in rs])/np.mean([r['mse_correct'] for r in rs]))}
    if len(rows) != len(manifest['test'])*9:
        raise ValueError('Missing offline cases')
    return {'cases':len(rows), 'demos':len(manifest['test']), 'groups':groups}


def native(args, model, manifest):
    for p in libero_paths():
        sys.path.insert(0, str(p))
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import mujoco
    if os.environ.get('MUJOCO_GL') != 'osmesa':
        raise RuntimeError('Use the audited OSMesa wrapper')
    suite = benchmark.get_benchmark_dict()['libero_10']()
    rows, rankings = [], []
    with (args.output / 'cases.jsonl').open('x') as log, torch.inference_mode():
        for task_id in args.task_ids:
            task = suite.get_task(task_id)
            index = manifest['task_names'].index(task.name)
            audit = json.loads((args.render_audit_dir / f'task_{task_id}/report.json').read_text())
            if not (audit['passed'] and audit['task']==task.name and audit['mujoco']==mujoco.__version__ and audit['render_backend']=='osmesa'):
                raise ValueError('Missing or incompatible render audit')
            with h5py.File(manifest['files'][index]['path'], 'r') as f:
                cases = [c for c in manifest['test'] if c['task']==index]
                c = next(c for c in cases if f[f'data/{c["demo"]}/rewards'][-1]>0)
                goal_images = np.stack([f[f'data/{c["demo"]}/obs/{cam}'][-1] for cam in manifest['camera_order']])
            goal = encode(model, goal_images, args.device)
            init_file = Path(get_libero_path('init_states')) / task.problem_folder / task.init_states_file
            init_state = torch.load(init_file, weights_only=False)[args.init_id]
            seed = args.seed + task_id*10000 + args.init_id
            env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file),
                                     camera_heights=128, camera_widths=128)
            try:
                random.seed(seed); np.random.seed(seed); env.seed(seed)
                env.reset()
                # Fixed fixtures live in the model, outside flattened sim state.
                # Use the native XML reset API to preserve them across branches.
                context_xml = env.sim.model.get_xml()
                (args.output / f'task{task_id}_context.xml').write_text(context_xml)
                def restart():
                    return reset_native_context(env, init_state, context_xml, seed)

                initial_obs, hist_images, start_state = restart()
                if env.check_success():
                    raise ValueError('Initial context already successful')
                history = torch.cat([encode(model, rgb, args.device) for rgb in hist_images], dim=1)
                past = torch.zeros(1, 2, 4, 7, device=args.device)
                planner = RecordedCEM(model, horizon=8, samples=128, elites=16, iterations=5, seed=seed)
                first = planner.plan(history, past, goal)
                torch.testing.assert_close(first, planner.final_sequence[0].clamp(-1, 1), atol=0, rtol=0)
                generator = torch.Generator(device=args.device).manual_seed(seed+500)
                random_sequences = (.6*torch.randn((args.random_candidates, 8, 4, 7), generator=generator, device=args.device)).clamp(-1, 1)
                candidates = torch.cat((planner.final_sequence[None], torch.zeros(1, 8, 4, 7, device=args.device), random_sequences))
                names = ['cem', 'zero'] + [f'random_{i}' for i in range(args.random_candidates)]
                predicted = rollout(planner, history, past, candidates)
                costs = (predicted-goal.reshape(1, 1, -1)).square().mean(-1)
                torch.testing.assert_close(costs[:, -1], CEM.costs(planner, history, past, candidates, goal), atol=1e-6, rtol=1e-5)
                initial_cost = (history[:, -1]-goal.reshape(1, -1)).square().mean().item()
                for ci, label in enumerate(names):
                    started = time.monotonic()
                    obs, check_images, check_state = restart()
                    np.testing.assert_allclose(check_state, start_state, rtol=0, atol=1e-10)
                    np.testing.assert_array_equal(check_images, hist_images)
                    initial_predicates = [bool(env.env._eval_predicate(g)) for g in env.env.parsed_problem['goal_state']]
                    physical_keys = sorted(k for k, v in obs.items() if k.endswith(('_pos', '_qpos')) and np.asarray(v).ndim==1)
                    physical_initial = {k:np.asarray(obs[k]).tolist() for k in physical_keys}
                    frames, states, observed_z, flags, predicate_flags = [], [], [], [], []
                    actions = candidates[ci].reshape(32, 7).cpu().numpy()
                    for step, action in enumerate(actions, 1):
                        obs, _, done, _ = env.step(action)
                        success = bool(env.check_success())
                        if bool(done) != success:
                            raise ValueError('Native done and success disagree')
                        flags.append(success)
                        if step % 4 == 0:
                            frames.append(pair(obs)); states.append(np.asarray(env.get_sim_state()).copy())
                            observed_z.append(encode(model, frames[-1], args.device).reshape(-1))
                            predicate_flags.append([bool(env.env._eval_predicate(g)) for g in env.env.parsed_problem['goal_state']])
                    observed_z = torch.stack(observed_z)
                    actual_costs = (observed_z-goal.reshape(1, -1)).square().mean(-1)
                    prediction_errors = (predicted[ci]-observed_z).square().mean(-1)
                    persistence_errors = (history[:, -1]-observed_z).square().mean(-1)
                    if not torch.isfinite(actual_costs).all():
                        raise FloatingPointError('Nonfinite native cost')
                    row = {'task_id':task_id, 'task':task.name, 'init_id':args.init_id, 'candidate':label,
                           'goal_demo':c['demo'], 'initial_cost':initial_cost, 'predicted_costs':costs[ci].cpu().tolist(),
                           'actual_costs':actual_costs.cpu().tolist(), 'prediction_errors':prediction_errors.cpu().tolist(),
                           'persistence_errors':persistence_errors.cpu().tolist(), 'success_any':any(flags), 'success_final':flags[-1],
                           'initial_predicates':initial_predicates, 'predicates':predicate_flags,
                           'goal_predicates':env.env.parsed_problem['goal_state'],
                           'physical_initial':physical_initial,
                           'physical_final':{k:np.asarray(obs[k]).tolist() for k in physical_keys},
                           'initial_state_sha256':hashlib.sha256(start_state.tobytes()).hexdigest(),
                           'initial_images_sha256':hashlib.sha256(hist_images.tobytes()).hexdigest(),
                           'goal_image_sha256':hashlib.sha256(goal_images.tobytes()).hexdigest(),
                           'init_file_sha256':digest(init_file), 'render_audit_sha256':digest(args.render_audit_dir/f'task_{task_id}/report.json'),
                           'context_xml_sha256':hashlib.sha256(context_xml.encode()).hexdigest(),
                           'elapsed_s':time.monotonic()-started}
                    stem = f'task{task_id}_{label}'
                    np.savez_compressed(args.output/f'{stem}.npz', actions=actions, history_images=hist_images, initial_state=start_state,
                        goal=goal_images, frames=np.stack(frames), states=np.stack(states), success=np.asarray(flags),
                        predicted=predicted[ci].cpu().numpy(), observed=observed_z.cpu().numpy(), goal_z=goal.cpu().numpy())
                    rows.append(row); log.write(json.dumps(row)+'\n'); log.flush()
                    print(json.dumps({'phase':'native', 'task_id':task_id, 'candidate':label, 'elapsed_s':row['elapsed_s'],
                                      'predicted_cost':row['predicted_costs'][-1], 'actual_cost':row['actual_costs'][-1]}), flush=True)
                task_rows = rows[-len(names):]
                for chunk in (1, 4, 8):
                    observed = [r['actual_costs'][chunk-1] for r in task_rows]
                    estimated = [r['predicted_costs'][chunk-1] for r in task_rows]
                    rankings.append({'task_id':task_id, 'horizon_actions':4*chunk,
                        'spearman_all':rank_correlation(estimated, observed),
                        'spearman_random_only':rank_correlation(estimated[2:], observed[2:]),
                        'cem_predicted_beats_random_fraction':float(np.mean(np.array(estimated[2:])>estimated[0])),
                        'cem_actual_beats_random_fraction':float(np.mean(np.array(observed[2:])>observed[0])),
                        'cem_predicted_progress':initial_cost-estimated[0], 'cem_actual_progress':initial_cost-observed[0]})
                dump(args.output/'rankings.json', rankings)
            finally:
                env.close()
    if len(rows) != len(args.task_ids)*(2+args.random_candidates):
        raise ValueError('Missing native cases')
    return {'cases':len(rows), 'tasks':len(args.task_ids), 'native_steps':32*len(rows), 'rankings':rankings,
            'success_counts':{kind:sum(r['success_any'] for r in rows if r['candidate'].startswith(kind)) for kind in ('cem','zero','random')},
            'note':'32-action branching diagnostic, not full-episode benchmark success rate; fixed CEM 128x5.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['offline', 'native'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT/'output/manifests/libero10/manifest.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--task-ids', type=int, nargs='+', default=list(range(10)))
    parser.add_argument('--init-id', type=int, default=0)
    parser.add_argument('--random-candidates', type=int, default=8)
    parser.add_argument('--render-audit-dir', type=Path, default=ROOT/'output/libero10/eval_bt100k_20260914/render_audit')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to((ROOT/'output').resolve()) or args.output == ROOT/'output':
        parser.error('Output must be a fresh subdirectory of output/')
    if args.random_candidates < 2 or args.init_id < 0 or not set(args.task_ids)<=set(range(10)) or len(set(args.task_ids))!=len(args.task_ids):
        parser.error('Invalid native cases')
    if not args.checkpoint.name.endswith('_object.ckpt'):
        parser.error('Requires a trusted object checkpoint')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        dump(args.output/'status.json', {'state':'running', 'pid':os.getpid()})
        m = json.loads(args.manifest.read_text())
        if m['frameskip'] != 4 or m['history'] != 3:
            raise ValueError('Incompatible temporal contract')
        for item in m['files']:
            st = Path(item['path']).stat()
            if st.st_size != item['size'] or st.st_mtime_ns != item['mtime_ns']:
                raise ValueError('Dataset size/mtime changed')
        source_paths = [Path(__file__), ROOT/'src/mylewm/environments/libero_planner.py',
                        ROOT/'src/mylewm/algorithms/libero_model.py', ROOT/'lewm/jepa.py', ROOT/'lewm/module.py']
        official = ROOT/'external/libero'
        source_paths += [official/'libero/lifelong/metric.py', official/'libero/libero/envs/env_wrapper.py',
                         official/'libero/libero/envs/bddl_base_domain.py']
        config = {**vars(args), 'checkpoint_sha256':digest(args.checkpoint), 'manifest_sha256':digest(args.manifest),
                  'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in source_paths},
                  'data_verification':'size and mtime; no full hash scan',
                  'native_protocol':'official init state, fixed context XML reset per candidate, 5 zero settling steps, then 8 zero steps for real model history',
                  'cem':{'samples':128,'iterations':5,'elites':16,'horizon_chunks':8},
                  'torch':torch.__version__, 'test_seen_before':True}
        dump(args.output/'config.json', config)
        torch.set_num_threads(4); torch.manual_seed(args.seed)
        random.seed(args.seed); np.random.seed(args.seed)
        print('Loading trusted checkpoint', flush=True)
        model = torch.load(args.checkpoint, map_location=args.device, weights_only=False).eval()
        torch.testing.assert_close(model.training_action_mean.cpu(), torch.tensor(m['action_mean'], dtype=model.training_action_mean.dtype))
        torch.testing.assert_close(model.training_action_std.cpu(), torch.tensor(m['action_std'], dtype=model.training_action_std.dtype))
        summary = (offline if args.stage=='offline' else native)(args, model, m)
        if digest(args.checkpoint)!=config['checkpoint_sha256'] or digest(args.manifest)!=config['manifest_sha256']:
            raise ValueError('Checkpoint or manifest changed during evaluation')
        summary['elapsed_s'] = time.monotonic()-started
        dump(args.output/'summary.json', summary)
        dump(args.output/'status.json', {'state':'succeeded','pid':os.getpid(),'cases':summary['cases'],'elapsed_s':summary['elapsed_s']})
        print(json.dumps(summary), flush=True)
    except BaseException:
        dump(args.output/'status.json', {'state':'failed','pid':os.getpid(),'error':traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
