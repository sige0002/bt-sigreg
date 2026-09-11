import copy
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch import nn
from transformers import ViTConfig, ViTModel

from mylewm.algorithms.libero_model import TwoViewJEPA
from mylewm.data.libero_bc_data import CAMERAS, LiberoBCClips, preprocess_images, validate_manifest
from mylewm.data.verification import file_sha256
from mylewm.policy.libero_bc import BCConfig, BCController, LiberoBCPolicy, load_bc_checkpoint
from mylewm.training import train_libero_bc as trainer
from mylewm.training.state import tensor_state_hash


@pytest.fixture
def bc_data(tmp_path):
    files = []
    names = ['task_b', 'task_a']  # Deliberately not the native environment's order.
    for task, name in enumerate(names):
        path = tmp_path / f'{name}_demo.hdf5'
        with h5py.File(path, 'w') as f:
            for demo in range(3):
                group = f.create_group(f'data/demo_{demo}')
                frames = np.arange(12, dtype=np.uint8)[:, None, None, None] * np.ones((12, 32, 32, 3), dtype=np.uint8)
                for cam in CAMERAS:
                    group.create_dataset('obs/' + cam, data=frames + task * 20)
                group.create_dataset('actions', data=np.arange(12, dtype=np.float32)[:, None].repeat(7, axis=1) / 20)
                group.create_dataset('rewards', data=np.ones(12))
        st = path.stat()
        files.append({'path': str(path), 'size': st.st_size, 'mtime_ns': st.st_mtime_ns})
    index = tmp_path / 'files.json'
    index.write_text(json.dumps(files))
    m = {'schema': 'rbg_libero10_v1', 'files': files, 'dataset': str(index),
         'dataset_size': index.stat().st_size, 'dataset_mtime_ns': index.stat().st_mtime_ns,
         'task_names': names, 'camera_order': CAMERAS, 'image_size': 224, 'frameskip': 4, 'history': 3,
         'image_convention': 'native opengl; do not flip training images independently of simulator',
         'action_mean': [0.] * 7, 'action_std': [1.] * 7}
    for demo, split in enumerate(('train_demos', 'validation', 'test')):
        m[split] = [{'task': task, 'demo': f'demo_{demo}', 'length': 12, 'start': 1} for task in range(2)]
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(m))
    world = TwoViewJEPA.__new__(TwoViewJEPA)
    nn.Module.__init__(world)
    world.encoder = ViTModel(ViTConfig(image_size=224, patch_size=14, hidden_size=24, num_hidden_layers=1,
                                     num_attention_heads=3, intermediate_size=48, hidden_dropout_prob=.5),
                             add_pooling_layer=False)
    world.register_buffer('training_action_mean', torch.zeros(7, dtype=torch.float64))
    world.register_buffer('training_action_std', torch.ones(7, dtype=torch.float64))
    parent = tmp_path / 'world_model'
    parent.mkdir()
    checkpoint = parent / 'step_1_object.ckpt'
    torch.save(world, checkpoint)
    (parent / 'config.json').write_text(json.dumps({'mode': 'bt', 'manifest_sha256': file_sha256(manifest)}))
    return m, manifest, checkpoint


@pytest.fixture
def policy(bc_data):
    torch.set_num_threads(1)
    m, _, checkpoint = bc_data
    return trainer.make_policy(checkpoint, m, BCConfig(horizon=8, width=24, depth=1, heads=3))


def test_bc_native_chunks_and_split_boundaries(bc_data):
    m, _, _ = bc_data
    ds = LiberoBCClips(m)
    assert len(ds) == 10  # 12-8+1 per demo, two training demos.
    pixels, actions, task = ds[4]
    assert task == 0 and torch.all(pixels == 4)
    torch.testing.assert_close(actions[:, 0], torch.arange(4, 12) / 20)
    pixels, actions, task = ds[5]
    assert task == 1 and torch.all(pixels == 20)
    with pytest.raises(IndexError):
        ds[10]
    val = LiberoBCClips(m, split='validation')
    assert len(val) == 2 and torch.all(val[0][0] == 1)
    ds.close(); val.close()
    bad = copy.deepcopy(m)
    bad['test'][0] = bad['train_demos'][0]
    with pytest.raises(ValueError, match='leakage'):
        validate_manifest(bad)
    with pytest.raises(ValueError, match='boundary'):
        LiberoBCClips(m, horizon=12, split='validation')


def test_bc_preprocess_matches_world_model(bc_data):
    from mylewm.training.train_libero import preprocess
    m, _, _ = bc_data
    pixels = torch.randint(256, (2, 2, 3, 128, 128), dtype=torch.uint8)
    actual = preprocess_images(pixels, 'cpu')
    expected, _ = preprocess((pixels[:, None], torch.zeros(2, 3, 4, 7)), m, 'cpu')
    torch.testing.assert_close(actual, expected[:, 0], rtol=0, atol=0)


def test_bc_frozen_encoder_tokens_and_head_gradients(policy):
    before = tensor_state_hash(policy.encoder.state_dict())
    policy.train()
    assert not policy.encoder.training and not any(p.requires_grad for p in policy.encoder.parameters())
    images = torch.randn(2, 2, 3, 224, 224)
    visual = policy.features(images)
    assert visual.shape == (2, 34, 24) and not visual.requires_grad
    with torch.no_grad():
        raw = policy.encoder(images.flatten(0, 1)).last_hidden_state
    torch.testing.assert_close(visual[:, 0], raw[::2, 0])
    torch.testing.assert_close(visual[:, 17], raw[1::2, 0])
    expected = raw[0, 1:].reshape(16, 16, 24)[:4, :4].mean((0, 1))
    torch.testing.assert_close(visual[0, 1], expected)
    optimizer = torch.optim.AdamW(policy.head.parameters(), lr=.001)
    head_before = tensor_state_hash(policy.head.state_dict())
    policy.loss(images, torch.rand(2, 8, 7), torch.tensor([0, 1])).backward()
    assert all(p.grad is None for p in policy.encoder.parameters())
    assert policy.head.task.weight.grad.abs().sum() > 0
    assert policy.head.visual.weight.grad.abs().sum() > 0
    optimizer.step()
    assert before == tensor_state_hash(policy.encoder.state_dict())
    assert head_before != tensor_state_hash(policy.head.state_dict())


def test_bc_sampling_conditions_and_checkpoint(policy, tmp_path):
    images = torch.randn(1, 2, 3, 224, 224)
    def sample(p, task=0):
        return p.sample(images, torch.tensor([task]), torch.Generator().manual_seed(9))
    a = sample(policy)
    assert a.shape == (1, 8, 7) and a.abs().max() <= 1
    assert not torch.equal(a, sample(policy, 1))
    path = tmp_path / 'step_1_bc.pt'
    torch.save(policy.bundle({'test': True}, 1), path)
    restored, bundle = load_bc_checkpoint(path)
    torch.testing.assert_close(a, sample(restored), rtol=0, atol=0)
    assert restored.task_names == policy.task_names and bundle['step'] == 1
    with pytest.raises(ValueError, match='task IDs'):
        sample(policy, 2)


def test_bc_flow_target_and_euler_direction(policy):
    # An oracle velocity field reaches a known action target after integration.
    class Oracle(nn.Module):
        def forward(self, x, time, visual, task):
            return (.25 - x) / (1 - time[:, None, None])
    policy.head = Oracle()
    images = torch.zeros(1, 2, 3, 224, 224)
    actions = torch.full((1, 8, 7), .25)
    loss = policy.loss(images, actions, torch.tensor([0]), torch.Generator().manual_seed(9))
    assert float(loss) < 1e-10
    out = policy.sample(images, torch.tensor([0]), torch.Generator().manual_seed(10))
    torch.testing.assert_close(out, actions)


def test_bc_controller_maps_names_and_refreshes_observation(policy):
    calls = []
    def sample(images, ids, generator):
        calls.append((images.clone(), ids.clone()))
        return torch.arange(8, dtype=torch.float32)[None, :, None].repeat(1, 1, 7)
    policy.sample = sample
    control = BCController(policy, 'task_a', execute_actions=2)
    for value in (10, 20, 30):
        control.select_action(np.full((2, 128, 128, 3), value, np.uint8))
    assert len(calls) == 2 and all(int(c[1][0]) == 1 for c in calls)
    assert not torch.equal(calls[0][0], calls[1][0])
    with pytest.raises(ValueError, match='absent'):
        BCController(policy, 'unknown')


def test_bc_training_dry_run_and_exact_resume(bc_data, monkeypatch, tmp_path):
    _, manifest, checkpoint = bc_data
    monkeypatch.setattr(trainer, 'ROOT', tmp_path)
    def args(name, extra=()):
        return trainer.parser().parse_args(['--checkpoint', str(checkpoint), '--manifest', str(manifest),
                    '--output', str(tmp_path / 'output' / name), '--device', 'cpu', '--threads', '1',
                    '--steps', '4', '--batch-size', '2', '--workers', '0', '--width', '24', '--depth', '1',
                    '--heads', '3', '--val-every', '2', '--save-every', '2', *extra])
    trainer.run(args('dry'))
    assert not (tmp_path / 'output').exists()
    trainer.run(args('full', ['--execute']))
    trainer.run(args('resumed', ['--execute', '--stop-after', '2']))
    paused = tmp_path / 'output/resumed'
    assert json.loads((paused / 'status.json').read_text())['state'] == 'paused'
    assert not (paused / 'completed.json').exists()
    trainer.run(args('resumed', ['--execute', '--resume']))
    a, b = [torch.load(tmp_path / 'output' / name / 'resume.pt', weights_only=False) for name in ('full', 'resumed')]
    assert tensor_state_hash(a['policy']) == tensor_state_hash(b['policy'])
    for i in a['optimizer']['state']:
        for k, value in a['optimizer']['state'][i].items():
            torch.testing.assert_close(value, b['optimizer']['state'][i][k], rtol=0, atol=0)
    assert torch.equal(a['random_state']['torch'], b['random_state']['torch'])
    rows = [json.loads(s) for s in (paused / 'metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows] == [1, 2, 3, 4]
    saved, _ = load_bc_checkpoint(paused / 'step_4_bc.pt')
    assert tensor_state_hash(saved.encoder.state_dict()) == a['frozen_encoder_sha256']


def test_bc_encoder_provenance_and_failure_status(bc_data, monkeypatch, tmp_path):
    m, manifest, checkpoint = bc_data
    wrong = tmp_path / 'wrong_manifest.json'
    wrong.write_text(json.dumps(dict(m, task_names=list(reversed(m['task_names'])))))
    with pytest.raises(ValueError, match='manifest hash'):
        trainer.encoder_provenance(checkpoint, wrong)
    model = torch.load(checkpoint, weights_only=False)
    model.training_action_std[0] = 2
    torch.save(model, checkpoint)
    monkeypatch.setattr(trainer, 'ROOT', tmp_path)
    out = tmp_path / 'output/failed'
    args = trainer.parser().parse_args(['--checkpoint', str(checkpoint), '--manifest', str(manifest),
                        '--output', str(out), '--steps', '1', '--device', 'cpu', '--execute'])
    with pytest.raises(ValueError, match='statistics'):
        trainer.run(args)
    assert json.loads((out / 'status.json').read_text())['state'] == 'failed'
    assert not (out / 'completed.json').exists()


def test_bc_rollout_native_success_and_budget(policy):
    from mylewm.evaluation.evaluate_libero_bc import rollout
    class Env:
        def __init__(self, success_step):
            self.steps, self.success_step = 0, success_step
        def check_success(self):
            return self.steps == self.success_step
        def obs(self):
            return {k: np.full((128, 128, 3), self.steps, np.uint8) for k in ('agentview_image', 'robot0_eye_in_hand_image')}
        def step(self, action):
            self.steps += 1
            return self.obs(), 999., False, {}  # Reward alone must never imply success.
    for success_step, budget, expected in ((3, 10, True), (8, 2, False)):
        env = Env(success_step)
        ctrl = BCController(policy, 'task_a', execute_actions=2)
        row, actions, frames = rollout(env, ctrl, env.obs(), budget)
        assert row['success'] is expected
        assert len(actions) == min(success_step, budget) and len(frames) == len(actions) + 1
        assert row['chunks'] == (len(actions) + 1) // 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA BC forward/backward and checkpoint inference')
def test_bc_cuda_update_and_inference(policy, tmp_path):
    policy.to('cuda')
    images = torch.randn(1, 2, 3, 224, 224, device='cuda')
    tasks = torch.tensor([0], device='cuda')
    loss = policy.loss(images, torch.zeros(1, 8, 7, device='cuda'), tasks)
    loss.backward()
    assert all(p.grad is None for p in policy.encoder.parameters())
    out = policy.sample(images, tasks, torch.Generator(device='cuda').manual_seed(1))
    assert torch.isfinite(out).all() and out.abs().max() <= 1
