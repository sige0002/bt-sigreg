"""README §7–11 trainer. Measured crossed groups, never fabricated branches."""
import argparse
import json
import os
import random
import time
import itertools
from pathlib import Path

import torch
from train_pusht import build_model
from multitask_jepa import MultiTaskJEPAObjective


def prepare_group(g, stats, device):
    ctx, future, ah, af = (g[k] for k in ('context', 'future', 'action_history', 'future_actions'))
    if ctx.ndim != 7 or future.ndim != 7 or af.ndim != 5:
        raise ValueError('images: [V,S,M,T,C,H,W]; actions: [S,M,T,block,A]')
    v, s, m, length = ctx.shape[:4]
    k, block, dim = af.shape[2:]
    if min(v, s, m) < 2 or future.shape[:3] != (v, s, m):
        raise ValueError('need >=2 views, >=2 states, >=2 actions per complete table')
    if ctx.dtype != torch.uint8 or future.dtype != torch.uint8 or ctx.shape[4] != 3:
        raise ValueError('images must be RGB uint8')
    if future.shape[3:] != (k, *ctx.shape[4:]) or ah.shape != (s, m, length-1, block, dim):
        raise ValueError('frame/action interval mismatch')
    if af.shape[:2] != (s, m) or not torch.isfinite(af).all() or not torch.isfinite(ah).all():
        raise ValueError('invalid action blocks')
    if not torch.equal(af, af[0:1].expand_as(af)):
        raise ValueError('each action column must execute identical controls across initial states')
    if not torch.equal(ctx, ctx[:, :, :1].expand_as(ctx)):
        raise ValueError('each initial-state row must have identical pre-branch observations')
    if not torch.equal(ah, ah[:, :1].expand_as(ah)):
        raise ValueError('each initial-state row must share pre-branch actions')
    if any(torch.equal(ctx[i], ctx[j]) and torch.equal(future[i], future[j])
           for i,j in itertools.combinations(range(v),2)):
        raise ValueError('duplicated visual condition is not a second view')
    if any(torch.equal(af[:, i], af[:, j]) for i,j in itertools.combinations(range(m),2)):
        raise ValueError('action columns must contain distinct executed control sequences')
    b = s*m
    mask = g['positive']
    if mask.dtype != torch.bool or mask.shape != (k, b, b) or not mask.diagonal(dim1=-2, dim2=-1).all():
        raise ValueError('positive must be explicit [K,S*M,S*M] boolean with true diagonal')
    mean, std = stats['action_mean'].to(device), stats['action_std'].to(device)
    if (mean.shape != (dim,) or std.shape != (dim,) or
        not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all()):
        raise ValueError('invalid shared action statistics')
    def actions(x):
        return ((x.to(device).float()-mean)/std).reshape(b, x.shape[2], block*dim)
    def pixels(x):
        x = x.to(device).float()/255
        mean = x.new_tensor([0.485, 0.456, 0.406]).reshape(1,1,3,1,1)
        std = x.new_tensor([0.229, 0.224, 0.225]).reshape(1,1,3,1,1)
        return (x-mean)/std
    history = actions(ah)
    contexts = {str(i): {'pixels': pixels(ctx[i].reshape(b, length, *ctx.shape[4:])),
                         'action': history} for i in range(v)}
    futures = {str(i): {'pixels': pixels(future[i].reshape(b, k, *future.shape[4:]))}
               for i in range(v)}
    return contexts, futures, actions(af), mask.to(device)


def atomic_save(value, path):
    tmp = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tables', type=Path, required=True)
    p.add_argument('--steps', type=int, default=50000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--horizons', type=int, nargs='+', default=[1,2,4])
    p.add_argument('--resume', type=Path)
    p.add_argument('--cache-dir', type=Path, default=Path('.cache/stable-wm'))
    p.add_argument('--output-dir', type=Path, default=Path('.cache/stable-wm/pusht/crossed_v1'))
    args = p.parse_args()
    if args.steps < 1 or args.save_every < 1:
        p.error('steps and save-every must be positive')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    manifest = torch.load(args.tables/'manifest.pt', weights_only=True)
    if manifest.get('schema') != 'measured_crossed_v1':
        raise ValueError('measured_crossed_v1 manifest required')
    train, val = manifest['train'], manifest['validation']
    if not train or not val:
        raise ValueError('nonempty train and validation groups required')
    if {i['family'] for i in train} & {i['family'] for i in val}:
        raise ValueError('physical experiment families overlap across train and validation')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(args.cache_dir/'hf_pusht/config.json').to(device)
    model.register_buffer('training_action_mean', manifest['action_mean'].clone().float().to(device))
    model.register_buffer('training_action_std', manifest['action_std'].clone().float().to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    objective = MultiTaskJEPAObjective(model)
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if state['manifest'] != str((args.tables/'manifest.pt').resolve()):
            raise ValueError('resume manifest differs')
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        start = state['step']
        torch.set_rng_state(state['rng'].cpu())
        random.setstate(state['python_rng'])
        if device == 'cuda':
            torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    def compute(item):
        g = torch.load(args.tables/item['file'], weights_only=True)
        c, f, a, mask = prepare_group(g, manifest, device)
        if next(iter(c.values()))['pixels'].shape[-2:] != (224,224):
            raise ValueError('current PushT model/evaluation requires 224x224 images')
        if a.shape[-1] != model.action_encoder.patch_embed.in_channels:
            raise ValueError('action block dimension differs from model; padding is forbidden')
        return objective.loss_from_observations(c, f, tuple(args.horizons), mask,
                                               future_actions=a)[0]
    started = time.monotonic()
    with (args.output_dir/'metrics.jsonl').open('a') as log:
        for step in range(start, args.steps):
            model.train()
            opt.zero_grad(set_to_none=True)
            loss = compute(random.choice(train))
            if not torch.isfinite(loss):
                raise FloatingPointError(f'nonfinite loss at {step}')
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
            opt.step()
            row = {'step':step+1, 'loss':float(loss.detach()), 'grad_norm':float(grad_norm),
                   'elapsed':time.monotonic()-started}
            if (step+1) % args.save_every == 0 or step+1 == args.steps:
                model.eval()
                with torch.no_grad():
                    row['validation_loss'] = sum(float(compute(item)) for item in val)/len(val)
                state = {'step':step+1, 'model':model.state_dict(), 'optimizer':opt.state_dict(),
                         'rng':torch.get_rng_state(), 'python_rng':random.getstate(),
                         'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else [],
                         'manifest':str((args.tables/'manifest.pt').resolve()), 'args':vars(args)}
                atomic_save(state, args.output_dir/'resume.ckpt')
                atomic_save(model, args.output_dir/'model_object.ckpt')
            log.write(json.dumps(row)+'\n')
            log.flush()
            if (step+1) % 10 == 0 or 'validation_loss' in row:
                print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
