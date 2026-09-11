"""Train a task-conditioned flow BC head on a frozen LIBERO world-model ViT.

Dry-run by default. This is a separate downstream stage, not world-model training.
"""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from mylewm.data.libero_bc_data import LiberoBCClips, preprocess_images, split_contract, validate_manifest
from mylewm.data.verification import file_sha256, verify_training_data, training_budget
from mylewm.paths import ROOT
from mylewm.policy.libero_bc import BCConfig, LiberoBCPolicy
from mylewm.training.loop import StepBatches, atomic_save
from mylewm.training.state import capture_rng, restore_rng, tensor_state_hash, reconcile_metrics


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def encoder_provenance(checkpoint, manifest, world_model_manifest=None):
    """Require evidence of the encoder's split, including for relocated old runs."""
    sidecar = checkpoint.parent / 'config.json'
    parent = json.loads(sidecar.read_text())
    source = manifest if world_model_manifest is None else world_model_manifest
    if file_sha256(source) != parent['manifest_sha256']:
        raise ValueError('Encoder manifest hash mismatch: pass the original --world-model-manifest for a relocated run')
    old, current = [json.loads(p.read_text()) for p in (source, manifest)]
    if split_contract(old) != split_contract(current):
        raise ValueError('Encoder and BC data splits/preprocessing/statistics differ')
    for before, after in zip(old['files'], current['files'], strict=True):
        if before['path'] != after['path']:
            old_hash = old.get('data_fingerprints', {}).get(before['path'], {}).get('sha256')
            old_hash = old_hash or parent.get('data_identity', {}).get(before['path'], {}).get('sha256')
            new_hash = current.get('data_fingerprints', {}).get(after['path'], {}).get('sha256')
            if not old_hash or old_hash != new_hash:
                raise ValueError('Relocated encoder data requires matching recorded dataset SHA-256 evidence')
    if parent.get('mode') not in ('raw', 'tc', 'bt'):
        raise ValueError('Require a Raw/TC/BT world model, not an obsolete method')
    return {'world_model_checkpoint': str(checkpoint), 'world_model_sha256': file_sha256(checkpoint),
            'world_model_config_sha256': file_sha256(sidecar), 'world_model_mode': parent['mode'],
            'world_model_manifest_sha256': file_sha256(source), 'split_contract': split_contract(current)}


def make_policy(checkpoint, manifest, config):
    from mylewm.algorithms.libero_model import TwoViewJEPA
    model = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not isinstance(model, TwoViewJEPA):
        raise ValueError('Require a two-camera LIBERO world model; a PushT checkpoint is incompatible')
    for name in ('mean', 'std'):
        expected = torch.tensor(manifest['action_' + name], dtype=torch.float64)
        actual = getattr(model, 'training_action_' + name).detach().cpu().double()
        if actual.shape != (7,) or not torch.allclose(actual, expected, rtol=1e-6, atol=1e-8):
            raise ValueError('World-model action statistics differ from the training manifest')
    return LiberoBCPolicy(model.encoder, manifest['task_names'], model.training_action_mean,
                           model.training_action_std, config)


def source_identity():
    paths = ['policy/libero_bc.py', 'data/libero_bc_data.py', 'training/train_libero_bc.py',
             'training/state.py', 'training/loop.py', 'data/verification.py', 'algorithms/libero_model.py']
    base = Path(__file__).resolve().parents[1]
    return {p: file_sha256(base / p) for p in paths}


@torch.no_grad()
def validate(policy, loader, seed):
    policy.eval()
    generator = torch.Generator(device=policy.device).manual_seed(seed)
    total, count = 0., 0
    for pixels, actions, task_ids in loader:
        images = preprocess_images(pixels, policy.device)
        actions, task_ids = actions.to(policy.device), task_ids.to(policy.device)
        loss = policy.loss(images, actions, task_ids, generator)
        total += float(loss) * len(images)
        count += len(images)
    return total / count


def run(args):
    args.checkpoint, args.manifest, args.output = [p.resolve() for p in (args.checkpoint, args.manifest, args.output)]
    if not args.checkpoint.is_file() or not args.checkpoint.name.endswith('_object.ckpt'):
        raise ValueError('Supply a trusted *_object.ckpt world model')
    if not args.output.is_relative_to(ROOT / 'output') or args.output == ROOT / 'output':
        raise ValueError('Use a separate run directory under output/')
    if args.output.exists() != args.resume:
        raise ValueError('Use a fresh output directory, or --resume for an existing BC run')
    if min(args.steps, args.batch_size, args.save_every, args.val_every, args.threads) < 1 or args.workers < 0 or args.seed < 0:
        raise ValueError('Invalid BC training budget')
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError('Invalid optimizer settings')
    if args.stop_after is not None and not 1 <= args.stop_after <= args.steps:
        raise ValueError('--stop-after must be within the fixed total update budget')
    cfg = BCConfig(args.horizon, args.width, args.depth, args.heads, args.euler_steps)
    m = json.loads(args.manifest.read_text())
    validate_manifest(m)
    print('Checking BC data metadata and encoder provenance', flush=True)
    identity = verify_training_data(m, full=args.verify_data)
    provenance = encoder_provenance(args.checkpoint, args.manifest, args.world_model_manifest)
    ds, val_ds = LiberoBCClips(m, cfg.horizon), LiberoBCClips(m, cfg.horizon, 'validation')
    config = {'schema': 'libero_bc_training_v1', **provenance, 'manifest': str(args.manifest),
              'manifest_sha256': file_sha256(args.manifest), 'model': asdict(cfg),
              'task_names': m['task_names'], 'data_identity': identity, 'source_sha256': source_identity(),
              'steps': args.steps, 'batch_size': args.batch_size, 'workers': args.workers, 'threads': args.threads,
              'lr': args.lr, 'weight_decay': args.weight_decay, 'seed': args.seed, 'device': args.device,
              'save_every': args.save_every, 'val_every': args.val_every,
              'precision': 'float32', 'optimizer': 'AdamW constant LR; clip head gradient norm at 1',
              'deterministic': args.deterministic,
              'dependencies': {n: str(importlib.metadata.version(n)) for n in ('torch', 'transformers', 'numpy', 'h5py')},
              'recipe': 'TC-LeWM v3 Appendix A.1 based local flow BC; not official reproduction',
              'chunk_contract': '8 native actions by default; exclude incomplete tails; no augmentation',
              'budget': training_budget(len(ds), args.steps, args.batch_size)}
    if not args.execute:
        print(json.dumps(config, indent=2))
        print('Dry-run: no checkpoint load, training or output creation.')
        return config
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(args.threads)
    if args.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if not args.resume:
        args.output.mkdir(parents=True)
        write_json(args.output / 'config.json', config)
    elif json.loads((args.output / 'config.json').read_text()) != config:
        raise ValueError('BC resume configuration/source/data/encoder mismatch')
    write_json(args.output / 'status.json', {'state': 'running', 'phase': 'loading_encoder'})
    try:
        policy = make_policy(args.checkpoint, m, cfg).to(args.device)
        frozen_hash = tensor_state_hash(policy.encoder.state_dict())
        optimizer = torch.optim.AdamW(policy.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        start = 0
        if args.resume:
            state = torch.load(args.output / 'resume.pt', map_location='cpu', weights_only=False)
            if state['config'] != config or state['frozen_encoder_sha256'] != frozen_hash:
                raise ValueError('BC resume checkpoint identity mismatch')
            policy.load_state_dict(state['policy'], strict=True)
            optimizer.load_state_dict(state['optimizer'])
            start = state['step']
            if not 0 < start < args.steps:
                raise ValueError('BC run is already complete or has an invalid step')
            reconcile_metrics(args.output / 'metrics.jsonl', state)
            restore_rng(state['random_state'])
        end = args.steps if args.stop_after is None else args.stop_after
        if end <= start:
            raise ValueError('--stop-after must be after the saved step')
        loader = torch.utils.data.DataLoader(ds, batch_sampler=StepBatches(len(ds), args.batch_size, end, args.seed, start),
                    num_workers=args.workers, pin_memory=policy.device.type == 'cuda',
                    generator=torch.Generator().manual_seed(args.seed + 100))
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, num_workers=0,
                    generator=torch.Generator().manual_seed(args.seed + 101))
        metadata = {**provenance, 'manifest_sha256': file_sha256(args.manifest),
                    'frozen_encoder_sha256': frozen_hash, 'bc_source_sha256': source_identity(),
                    'training_config': config}
        print(json.dumps({'event': 'bc_start', 'step': start, 'until': end,
                          'frozen_encoder_parameters': sum(p.numel() for p in policy.encoder.parameters()),
                          'trainable_head_parameters': sum(p.numel() for p in policy.head.parameters())}), flush=True)
        begin = time.monotonic()
        with (args.output / 'metrics.jsonl').open('a') as log:
            for step, (pixels, actions, tasks) in enumerate(loader, start + 1):
                policy.train(); optimizer.zero_grad(set_to_none=True)
                images = preprocess_images(pixels, policy.device)
                loss = policy.loss(images, actions.to(policy.device), tasks.to(policy.device))
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite BC training loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(policy.head.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                row = {'step': step, 'flow_loss': float(loss.detach()), 'grad_norm': float(norm),
                       'elapsed_session': time.monotonic() - begin}
                if step % args.val_every == 0 or step == end:
                    row['validation_flow_loss'] = validate(policy, val_loader, args.seed + 200)
                if step % args.save_every == 0 or step == end:
                    if tensor_state_hash(policy.encoder.state_dict()) != frozen_hash:
                        raise RuntimeError('Frozen ViT changed during BC training')
                    atomic_save(policy.bundle(metadata, step), args.output / f'step_{step}_bc.pt')
                    atomic_save({'step': step, 'policy': policy.state_dict(), 'optimizer': optimizer.state_dict(),
                                 'config': config, 'frozen_encoder_sha256': frozen_hash,
                                 'random_state': capture_rng(), 'metrics_row': row}, args.output / 'resume.pt')
                log.write(json.dumps(row, allow_nan=False) + '\n'); log.flush()
                if step == start + 1 or step % 10 == 0 or step == end:
                    print(json.dumps(row), flush=True)
                if step % 10 == 0:
                    write_json(args.output / 'status.json', {'state': 'running', 'step': step})
        final = {'state': 'succeeded' if end == args.steps else 'paused', 'step': end,
                 'checkpoint': f'step_{end}_bc.pt', 'frozen_encoder_sha256': frozen_hash}
        if end == args.steps:
            write_json(args.output / 'completed.json', final)
        write_json(args.output / 'status.json', final)
        return final
    except BaseException as e:
        write_json(args.output / 'status.json', {'state': 'failed', 'error': f'{type(e).__name__}: {e}'})
        raise
    finally:
        ds.close(); val_ds.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True, help='Trusted Raw/TC/BT LIBERO *_object.ckpt')
    p.add_argument('--manifest', type=Path, default=ROOT / 'output/manifests/libero10/manifest.json')
    p.add_argument('--world-model-manifest', type=Path, help='Original manifest when the dataset was relocated')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=40000)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=3072)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--horizon', type=int, default=8)
    p.add_argument('--width', type=int, default=256)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--euler-steps', type=int, default=10)
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--val-every', type=int, default=1000)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--verify-data', action='store_true')
    p.add_argument('--deterministic', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after', type=int, help='Pause for a short check, preserving the fixed total budget')
    p.add_argument('--execute', action='store_true')
    return p


if __name__ == '__main__':
    run(parser().parse_args())
