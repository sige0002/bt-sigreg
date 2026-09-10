"""Exercise the actual training loop, including validation and DataLoader restart."""
import argparse
import json
import random

import numpy as np
import pytest
import torch

from mylewm.training import loop as training
from mylewm.training.state import UpdateSchedule, check_resume_config, reconcile_metrics, IndependentRegularizer,tensor_state_hash


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.BatchNorm1d(8),
                                       torch.nn.Dropout(.3), torch.nn.Linear(8, 3))


class TinyRegularizer(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.register_buffer('directions', torch.randn(3))
    def forward(self,x):
        return self.directions.square().mean()+torch.rand_like(x).mean()*.01


class TinyClips(torch.utils.data.Dataset):
    def __init__(self, manifest, validation=False):
        self.n = 8 if validation else 29
    def __len__(self):
        return self.n
    def __getitem__(self, index):
        x = torch.tensor([index / 29, index % 3, 1.], dtype=torch.float32)
        return x, x * .2


def test_schedule_boundaries_and_group_lrs():
    schedule = UpdateSchedule(50000, 500, 5e-5, 0.)
    assert schedule.lr_at(1) == pytest.approx(1e-7, rel=1e-15)
    assert schedule.lr_at(500) == 5e-5
    assert schedule.lr_at(501) < 5e-5
    assert schedule.lr_at(50000) == 0.
    assert UpdateSchedule(10, 0, 1., .1).lr_at(1) == 1.
    assert UpdateSchedule(10, 0, 1., .1).lr_at(10) == .1
    assert UpdateSchedule(1, 0, 1., 0.).lr_at(1) == 1.
    opt = torch.optim.AdamW([{'params': [torch.nn.Parameter(torch.ones(1))]},
                             {'params': [torch.nn.Parameter(torch.zeros(1))]}])
    assert schedule.apply(opt, 1) == pytest.approx([1e-7, 1e-7], rel=1e-15)
    schedule.mark_completed(1)
    with pytest.raises(ValueError):
        schedule.apply(opt, 1)


@pytest.mark.parametrize('key', ['steps', 'warmup_steps', 'lr', 'min_lr', 'gaussian_weight',
                                'adapter_sources', 'source_sha256',
                                'manifest_sha256', 'seed', 'precision'])
def test_resume_rejects_contract_changes(key):
    with pytest.raises(ValueError, match=key):
        check_resume_config({key: 'old'}, {key: 'new'})


@pytest.mark.parametrize('logged_steps', [[1], [1, 2, 3, 4]])
def test_crash_tail_is_archived_and_checkpoint_row_recovered(tmp_path, logged_steps):
    path=tmp_path/'metrics.jsonl'
    original=''.join(json.dumps({'step':s})+'\n' for s in logged_steps)
    path.write_text(original)
    reconcile_metrics(path, {'step':2, 'metrics_row':{'step':2}})
    assert [json.loads(x)['step'] for x in path.read_text().splitlines()]==[1,2]
    assert next(tmp_path.glob('metrics_before_resume_*.jsonl')).read_text()==original


def assert_tree_equal(a, b, atol=0.):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0., atol=atol)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key], atol)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y, atol)
    else:
        assert a == b


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('workers', [0, 2])
def test_real_loop_uninterrupted_equals_resumed(tmp_path, monkeypatch, device, workers):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    if device == 'cpu':
        monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    # CUBLAS workspace must be set before CUDA initialization in the test process.
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    monkeypatch.setattr(training, 'GaussianSIGReg', TinyRegularizer)
    adapter = training.TrainingAdapter(TinyClips,
        lambda batch, m, d: tuple(x.to(d) for x in batch), TinyModel)
    dataset = tmp_path / 'data.identity'
    dataset.write_bytes(b'fixed synthetic dataset')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'dataset': str(dataset), 'dataset_size': dataset.stat().st_size,
        'dataset_mtime_ns': dataset.stat().st_mtime_ns, 'action_mean': [0.], 'action_std': [1.]}))
    seen = []
    calls = 0
    interrupt = False

    def objective(model, x, a, reg, *args):
        nonlocal calls
        if model.training:
            calls += 1
            if interrupt and calls == 3:
                raise InterruptedError('simulated interruption after saved update 2')
            seen.append(x.detach().cpu().clone())
        # Exercise all RNG families, dropout, BN and regularizer buffers.
        noise = torch.rand_like(x) * .01 + random.random() * .01 + np.random.rand() * .01
        loss = (model.net(x) - a - noise).square().mean() + reg(x) * .001
        return loss, {'prediction': loss.detach()}

    monkeypatch.setattr(training, 'one_step_objective', objective)
    args = argparse.Namespace(steps=6, warmup_steps=2, lr=.003, min_lr=0., seed=3072,
        manifest=manifest, mode='raw', batch_size=4, workers=workers,
        gaussian_weight=.09, save_every=2, resume=False,
        output=tmp_path/'full',deterministic=True)
    try:
        training.train(args, adapter=adapter)
        full_batches = seen.copy()
        full = torch.load(args.output/'resume.pt', map_location='cpu', weights_only=False)
        full_rows = [json.loads(line) for line in (args.output/'metrics.jsonl').read_text().splitlines()]
        assert json.loads((args.output/'completed.json').read_text()) == {
            'step': 6, 'state': 'completed', 'recipe': 'controlled_comparison'}
        seen.clear(); calls = 0; interrupt = True
        args.output = tmp_path/'resumed'
        with pytest.raises(InterruptedError):
            training.train(args, adapter=adapter)
        interrupt = False
        args.resume = True
        training.train(args, adapter=adapter)
        resumed = torch.load(args.output/'resume.pt', map_location='cpu', weights_only=False)
        resumed_rows = [json.loads(line) for line in (args.output/'metrics.jsonl').read_text().splitlines()]
        assert_tree_equal(full_batches, seen)
        for key in ('model', 'optimizer', 'regularizer', 'scheduler', 'random_state', 'next_batch_index'):
            assert_tree_equal(full[key], resumed[key])
        for row in full_rows + resumed_rows:
            row.pop('elapsed_session')
            row.pop('peak_gpu_allocated_bytes')
        assert full_rows == resumed_rows
        assert [r['learning_rates'] for r in full_rows][-1] == [0.]
        assert int(full['regularizer']['draw_index'])==6
    finally:
        torch.use_deterministic_algorithms(deterministic)


def test_regularizer_draw_count_does_not_shift_model_rng():
    class Draws(torch.nn.Module):
        def __init__(self,n):super().__init__();self.n=n
        def forward(self,x):return torch.randn(self.n).sum()+x.sum()
    streams=[]
    for n in (5,1000):
        torch.manual_seed(17)
        regularizer=IndependentRegularizer(Draws(n),400)
        stream=[]
        for _ in range(3):
            regularizer(torch.zeros(1))
            stream.append(torch.randn(10))
        streams.append(torch.stack(stream))
        assert int(regularizer.draw_index)==3
    torch.testing.assert_close(*streams,atol=0,rtol=0)


def test_tensor_initialization_hash_detects_weight_change():
    a={'weight':torch.ones(2,3),'tracked':torch.tensor(0)}
    b={key:value.clone() for key,value in a.items()}
    assert tensor_state_hash(a)==tensor_state_hash(b)
    b['weight'][0,0]+=1
    assert tensor_state_hash(a)!=tensor_state_hash(b)
