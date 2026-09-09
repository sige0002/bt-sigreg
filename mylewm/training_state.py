"""Update-indexed schedules and resumable random state for controlled comparisons."""
from contextlib import contextmanager
import math
import random
import json
import time
import hashlib
from pathlib import Path

import numpy as np
import torch


def add_schedule_arguments(parser):
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--min-lr', type=float, default=0.)
    parser.add_argument('--max-lr', '--lr', dest='lr', type=float, default=5e-5)
    parser.add_argument('--diagnostics-every',type=int,default=5000)
    parser.add_argument('--deterministic',action='store_true')


def tensor_state_hash(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class UpdateSchedule:
    """LR used by update s (1-based), not the LR of the following update.

    Warmup uses max_lr*s/warmup_steps. The warmup boundary is max_lr;
    the final update is min_lr. With no warmup, update 1 is max_lr.
    A one-update, no-warmup smoke run uses max_lr.
    """
    def __init__(self, total_steps, warmup_steps, max_lr, min_lr):
        if total_steps < 1 or not 0 <= warmup_steps < total_steps:
            raise ValueError('Require total_steps >= 1 and 0 <= warmup_steps < total_steps')
        if not all(math.isfinite(x) for x in (max_lr, min_lr)) or not 0 <= min_lr <= max_lr:
            raise ValueError('Require finite 0 <= min_lr <= max_lr')
        self.config = dict(total_steps=total_steps, warmup_steps=warmup_steps,
                           max_lr=max_lr, min_lr=min_lr, kind='linear_warmup_cosine_v1')
        self.completed_steps = 0

    def lr_at(self, step):
        c = self.config
        if not 1 <= step <= c['total_steps']:
            raise ValueError('Update outside fixed schedule')
        if step <= c['warmup_steps']:
            return c['max_lr'] * step / c['warmup_steps']
        boundary = max(1, c['warmup_steps'])
        if c['total_steps'] == boundary:
            return c['max_lr']
        progress = (step - boundary) / (c['total_steps'] - boundary)
        return c['min_lr'] + (c['max_lr'] - c['min_lr']) * (1 + math.cos(math.pi * progress)) / 2

    def apply(self, optimizer, step):
        if step != self.completed_steps + 1:
            raise ValueError('Schedule must advance once per optimizer update')
        lr = self.lr_at(step)
        for group in optimizer.param_groups:
            group['lr'] = lr
        return [group['lr'] for group in optimizer.param_groups]

    def mark_completed(self, step):
        if step != self.completed_steps + 1:
            raise ValueError('Nonsequential optimizer update')
        self.completed_steps = step

    def state_dict(self):
        return {'config': self.config, 'completed_steps': self.completed_steps}

    def load_state_dict(self, state):
        if state['config'] != self.config:
            raise ValueError('resume mismatch: scheduler')
        if not 0 <= state['completed_steps'] <= self.config['total_steps']:
            raise ValueError('Invalid scheduler position')
        self.completed_steps = state['completed_steps']


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


@contextmanager
def isolated_rng(seed):
    state = capture_rng()
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        yield
    finally:
        restore_rng(state)


def check_resume_config(previous, current):
    # Paths/operational flags do not affect training. All other fields, including
    # adapter code, loss coefficients, device/precision and validation frequency do.
    ignored = {'resume', 'output', 'command'}
    for key in sorted((set(previous) | set(current)) - ignored):
        if previous.get(key) != current.get(key):
            raise ValueError(f'resume mismatch: {key}')


def reconcile_metrics(path, state):
    """Archive a crash tail; keep one authoritative row per committed update."""
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    step = state['step']
    committed = [row for row in rows if row['step'] <= step]
    if not committed or committed[-1]['step'] < step:
        committed.append(state['metrics_row'])
    if [row['step'] for row in committed] != list(range(1, step + 1)):
        raise ValueError('Resume metrics missing or duplicate committed updates')
    if committed != rows:
        if path.exists():
            path.rename(path.with_name(f'metrics_before_resume_{time.time_ns()}.jsonl'))
        path.write_text(''.join(json.dumps(row)+'\n' for row in committed))


class IndependentRegularizer(torch.nn.Module):
    """Regularizer draws cannot shift the model's dropout stream across methods."""
    def __init__(self,inner,seed):
        super().__init__()
        self.inner=inner
        self.register_buffer('random_seed',torch.tensor(seed,dtype=torch.int64))
        self.register_buffer('draw_index',torch.tensor(0,dtype=torch.int64))

    def forward(self,*args,**kwargs):
        with isolated_rng((int(self.random_seed)+int(self.draw_index)) % (2**32)):
            value=self.inner(*args,**kwargs)
        self.draw_index.add_(1)
        return value
