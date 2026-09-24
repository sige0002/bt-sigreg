"""Cube contracts: dimensions, leakage, native grouping, three-way parity/resume."""
import copy
import json
from argparse import Namespace

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from mylewm.data.cube_data import prepare, inspect_schema, cube_contract
from mylewm.paths import ROOT
from mylewm.training import train_cube as cube
from mylewm.training import train as shared
from mylewm.training.state import tensor_state_hash
from test_library_training import TinyData, TinyModel, recipe, fit
from test_training_state import assert_tree_equal


@pytest.fixture
def native_cube(tmp_path):
    path = tmp_path/'cube.h5'
    n, length = 40, 45
    rows = n*length
    with h5py.File(path, 'w') as f:
        f['ep_len'] = np.full(n, length, dtype=np.int64)
        f['ep_offset'] = np.arange(n, dtype=np.int64)*length
        f.create_dataset('pixels', shape=(rows, 224, 224, 3), dtype='u1',
                         chunks=(1, 224, 224, 3), compression='lzf', fillvalue=0)
        actions = np.random.default_rng(2).uniform(-1, 1, (rows, 5)).astype('f4')
        actions[length-1::length] = np.nan
        f['action'] = actions
        f['qpos'] = np.zeros((rows, 16))
        f['qvel'] = np.zeros((rows, 15))
        f['privileged_block_0_pos'] = np.zeros((rows, 3))
        f['privileged_block_0_quat'] = np.tile([1., 0., 0., 0.], (rows, 1))
    manifest = prepare(path, tmp_path/'manifest.json', 3072, 2, 2)
    return path, manifest


def test_prepare_isolation_training_only_statistics_and_native_temporal_grouping(native_cube):
    path, manifest = native_cube
    cube.validate_manifest(manifest)
    groups = [set(manifest[k]) for k in ('train_episodes', 'validation_episodes', 'test_episodes')]
    assert all(not (groups[i] & groups[j]) for i in range(3) for j in range(i))
    with h5py.File(path) as f:
        physical = np.concatenate([f['action'][e*45:e*45+44] for e in manifest['train_episodes']]).astype('f8')
    np.testing.assert_allclose(manifest['action_mean'], physical.mean(0), atol=1e-14)
    np.testing.assert_allclose(manifest['action_std'], physical.std(0, ddof=1), atol=1e-14)
    train, val = shared.datasets(manifest)
    assert len(train) == manifest['train_clips'] == 32*26
    assert len(val) == 2
    assert all(train.dataset.clip_indices[i][0] in groups[0] for i in train.indices)
    clip = train[0]
    assert clip['pixels'].shape == (4, 3, 224, 224)
    assert clip['action'].shape == (4, 25)
    ep, start = train.dataset.clip_indices[train.indices[0]]
    with h5py.File(path) as f:
        raw = f['action'][ep*45+start:ep*45+start+20].astype('f8')
    expected = ((raw-manifest['action_mean'])/manifest['action_std']).astype('f4').reshape(4,25)
    np.testing.assert_allclose(clip['action'].numpy(), expected, rtol=0, atol=0)
    assert {x['episode'] for x in manifest['confirm']} <= groups[2]


def test_bad_schema_and_wrong_domain_rejected(native_cube):
    path, manifest = native_cube
    bad = copy.deepcopy(manifest)
    bad['input_contract']['domain'] = 'pusht'
    with pytest.raises(ValueError): cube.validate_manifest(bad)
    with h5py.File(path, 'r+') as f: f['action'][0, 0] = np.nan
    with pytest.raises(ValueError, match='Nonfinite'): inspect_schema(path)
    with pytest.raises(FileExistsError): prepare(path, path.parent/'manifest.json', 3072, 2, 2)


def test_multi_cube_variant_rejected(native_cube):
    path, _ = native_cube
    with h5py.File(path, 'r+') as f:
        f['privileged_block_1_pos'] = np.zeros((len(f['action']), 3))
    with pytest.raises(ValueError, match='single-cube'):
        inspect_schema(path)


def test_explicit_three_way_model_identity_and_official_action_bottleneck():
    hashes = []
    for family in ('raw', 'cayley', 'norm_preserving'):
        config = cube.load_model_config(ROOT/f'mylewm/configs/cube_{family}.yaml')
        torch.manual_seed(3072)
        model = cube.build_model(config)
        assert model.action_encoder.patch_embed.in_channels == 25
        assert model.action_encoder.patch_embed.out_channels == 10
        hashes.append(tensor_state_hash(model.state_dict()))
        bad = copy.deepcopy(config)
        bad['model']['action_encoder']['input_dim'] = 10
        with pytest.raises(ValueError): cube.build_model(bad)
        del model
    assert len(set(hashes)) == 1


class CubeTinyData(TinyData):
    def __getitem__(self, i):
        batch = super().__getitem__(i)
        batch['action'] = torch.cat([batch['action'], batch['action'], batch['action'][:, :5]], -1)
        return batch


def tiny_module(family):
    torch.manual_seed(12)
    model = TinyModel()
    model.action_encoder = torch.nn.Linear(25, 6)
    config = cube.expected_config(family)
    config['regularizer'].update(dim=6, projections=16)
    if family == 'cayley': config['regularizer']['hidden'] = 6
    batches = shared.EpochBatches(CubeTinyData(), 4, 6, 23)
    return shared.TrainingModule(model, cube.build_regularizer(config, 44),
                                  dict(recipe(), family=family), batches)


def test_identity_regularizers_share_loss_model_gradient_and_future_teacher():
    reference = None
    batch = next(iter(DataLoader(CubeTinyData(), batch_size=4)))
    for family in ('raw', 'cayley', 'norm_preserving'):
        m = tiny_module(family)
        m.log_dict = lambda *a, **k: None
        inputs = copy.deepcopy(batch)
        inputs['pixels'].requires_grad_()
        torch.manual_seed(42)
        out = m(inputs, stage='fit')
        out['loss'].backward()
        values = (out['loss'].detach().clone(), [p.grad.detach().clone() for p in m.model.parameters()])
        if reference is None: reference = copy.deepcopy(values)
        else: assert_tree_equal(reference, values)
        assert inputs['pixels'].grad[:, -1].abs().sum() > 0


def test_terminal_sentinel_never_poison_gradients(native_cube):
    _, manifest = native_cube
    train, _ = shared.datasets(manifest)
    sample = train[-1]
    assert torch.isnan(sample['action'][-1]).any()
    assert torch.isfinite(sample['action'][:3]).all()
    batch = {k: torch.stack([v, v]) for k, v in sample.items()}
    for family in ('raw', 'cayley', 'norm_preserving'):
        module = tiny_module(family)
        module.log_dict = lambda *a, **k: None
        output = module(copy.deepcopy(batch), stage='fit')
        assert torch.isfinite(output['loss'])
        output['loss'].backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters())


@pytest.mark.parametrize('family', ['raw', 'cayley', 'norm_preserving'])
def test_cube_strict_resume_and_transport_free_export(tmp_path, family):
    # fit's tiny loader uses 10 action dimensions; adapt only the test model input.
    def make():
        m = tiny_module(family)
        torch.manual_seed(91)
        m.model.action_encoder = torch.nn.Linear(10, 6)
        return m
    full = make()
    rows_full, a = fit(tmp_path/'full', full)
    rows_part, _ = fit(tmp_path/'part', make(), stop=True)
    resumed = make()
    rows_resumed, b = fit(tmp_path/'resumed', resumed, resume=tmp_path/'part/final.ckpt')
    assert rows_full == rows_part + rows_resumed
    assert_tree_equal(full.state_dict(), resumed.state_dict())
    assert_tree_equal(a['optimizer_states'], b['optimizer_states'])
    assert_tree_equal(a['lr_schedulers'], b['lr_schedulers'])
    model = torch.load(tmp_path/'full/step_6_object.ckpt', weights_only=False)
    assert not hasattr(model, 'sigreg')
    changed = make()
    changed.recipe['family'] = 'wrong'
    with pytest.raises(ValueError, match='Recipe mismatch'):
        changed.on_load_checkpoint(torch.load(tmp_path/'part/final.ckpt', weights_only=False))
