"""Real LeRobot writer/reader fixtures and cross-format checkpoint contracts."""
import argparse
import copy
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from mylewm import train
from mylewm.data_contract import verify_training_data
from mylewm.input_contract import inference_manifest, validate_contract
from mylewm.prepare_dataset import prepare
from mylewm.tests.test_library_training import TinyModel
from mylewm.tests.test_training_state import assert_tree_equal
from mylewm.tools.infer_trajectories import inference_dataset, predict


@pytest.fixture
def contract():
    return json.loads((Path(__file__).parents[1] / 'configs/lerobot_pusht_input.json').read_text())


@pytest.fixture
def paired_sources(tmp_path, contract):
    pytest.importorskip('lerobot')
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root = tmp_path / 'lerobot'
    ds = LeRobotDataset.create('local/fixture', root=root, fps=10, use_videos=False,
        features={'observation.image': {'dtype': 'image', 'shape': (8, 8, 3), 'names': ['h', 'w', 'c']},
                  'action': {'dtype': 'float32', 'shape': (2,), 'names': ['x', 'y']}})
    pixels, actions = [], []
    for ep in range(3):
        for frame in range(25):
            # Distinct per-episode/per-frame RGB content catches index mistakes.
            pixel = np.full((8, 8, 3), [ep * 60 + frame, frame * 3, 200 - frame], dtype=np.uint8)
            action = np.array([ep * 100 + frame, 2 * frame - ep], dtype=np.float32)
            pixels.append(pixel)
            actions.append(action)
            ds.add_frame({'observation.image': pixel, 'action': action, 'task': 'unused task'})
        ds.save_episode()
    ds.finalize()
    hdf = tmp_path / 'same.h5'
    with h5py.File(hdf, 'w') as f:
        f['pixels'], f['action'] = np.stack(pixels), np.stack(actions)
        f['ep_len'], f['ep_offset'] = [25] * 3, [0, 25, 50]
    hm = prepare(hdf, tmp_path / 'hdf.json', 'hdf5', contract)
    lm = prepare(root, tmp_path / 'lr.json', 'lerobot', contract, 'observation.image')
    return hm, lm


def test_same_frames_actions_splits_and_training_statistics(paired_sources):
    hm, lm = paired_sources
    assert hm['action_mean'] == lm['action_mean']
    assert hm['action_std'] == lm['action_std']
    assert hm['train_episodes'] == lm['train_episodes']
    h, l = train.clip_dataset(hm), train.clip_dataset(lm)
    assert len(h) == len(l) == 18
    for i in [0, 5, 6, 11, 12, 17]:
        assert_tree_equal(h[i], l[i])
    ep = hm['train_episodes'][0]
    # Train-only mean excludes terminal action (frame 24).
    np.testing.assert_array_equal(hm['action_mean'], [ep * 100 + 11.5, 23 - ep])
    assert verify_training_data(lm)['verification'] == 'size_mtime'
    verify_training_data(lm, full=True)


def test_cross_format_uses_checkpoint_stats_and_predictions(paired_sources):
    hm, lm = paired_sources
    model = TinyModel().eval()
    model.input_contract = hm['input_contract']
    for name in ('mean', 'std'):
        model.register_buffer('training_action_' + name, torch.tensor(hm['action_' + name], dtype=torch.float64))
    lm['action_mean'] = [9999., 9999.]
    lm['action_std'] = [.001, .001]
    hd, ld = inference_dataset(model, hm), inference_dataset(model, lm)
    hb = next(iter(torch.utils.data.DataLoader(hd, batch_size=2)))
    lb = next(iter(torch.utils.data.DataLoader(ld, batch_size=2)))
    assert_tree_equal(hb, lb)
    assert_tree_equal(predict(model, hb), predict(model, lb))


@pytest.mark.parametrize('field,value', [('fps', 20), ('camera', 'wrist'),
    ('action_units', ['m', 'm']), ('action_names', ['target_y', 'target_x']),
    ('action_convention', 'relative'), ('frameskip', 2), ('domain', 'different_robot')])
def test_reject_semantic_mismatch(contract, field, value):
    model = TinyModel()
    model.input_contract = contract
    other = copy.deepcopy(contract)
    other[field] = value
    with pytest.raises(ValueError, match='Incompatible'):
        inference_manifest(model, {'input_contract': other})


def test_legacy_contract_needs_explicit_training_conditions(contract):
    model = TinyModel()
    model.register_buffer('training_action_mean', torch.zeros(2))
    model.register_buffer('training_action_std', torch.ones(2))
    source = {'input_contract': contract}
    with pytest.raises(ValueError, match='Legacy checkpoint'):
        inference_manifest(model, source)
    assert inference_manifest(model, source, contract)['action_std'] == [1., 1.]
    with pytest.raises(ValueError):
        validate_contract(dict(contract, fps=0))


def test_general_action_dimension_and_sampling(tmp_path, contract):
    c = dict(contract, action_names=[f'joint_{i}' for i in range(7)],
             action_units=['rad'] * 7, action_convention='absolute_joint_position', frameskip=2)
    path = tmp_path / 'robot.h5'
    with h5py.File(path, 'w') as f:
        f['pixels'] = np.zeros((36, 8, 8, 3), dtype=np.uint8)
        f['action'] = np.arange(36 * 7, dtype=np.float32).reshape(36, 7)
        f['ep_len'], f['ep_offset'] = [12] * 3, [0, 12, 24]
    m = prepare(path, tmp_path / 'robot.json', 'hdf5', c)
    ds, _ = train.datasets(m)
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=2)))
    assert batch['action'].shape == (2, 4, 14)
    model = train.build_model(input_dim=14).eval()
    pred, target = predict(model, batch)
    assert pred.shape == target.shape == (2, 3, 192)


@pytest.mark.parametrize('column', ['timestamp', 'episode_index', 'frame_index'])
def test_invalid_lerobot_alignment_refused(paired_sources, tmp_path, column):
    import pyarrow as pa
    import pyarrow.parquet as pq
    _, lm = paired_sources
    root = Path(lm['dataset']).parent.parent
    path = next((root / 'data').rglob('*.parquet'))
    table = pq.read_table(path)
    data = table[column].to_pylist()
    data[1] = 999
    table = table.set_column(table.schema.get_field_index(column), column,
                             pa.array(data, type=table[column].type))
    pq.write_table(table, path)
    with pytest.raises(ValueError, match='alignment'):
        prepare(root, tmp_path / 'bad.json', 'lerobot', lm['input_contract'], 'observation.image')


@pytest.mark.parametrize('workers', [0, 2])
def test_lerobot_workers_deterministic(paired_sources, workers):
    _, lm = paired_sources
    ds = train.clip_dataset(lm)
    expected = next(iter(torch.utils.data.DataLoader(ds, batch_size=3, num_workers=0)))
    actual = next(iter(torch.utils.data.DataLoader(ds, batch_size=3, num_workers=workers)))
    assert_tree_equal(expected, actual)


def test_inference_prepare_does_not_fit_statistics(paired_sources, tmp_path):
    _, lm = paired_sources
    m = prepare(Path(lm['dataset']).parent.parent, tmp_path / 'inference.json', 'lerobot',
                lm['input_contract'], 'observation.image', purpose='inference')
    assert 'action_mean' not in m and m['train_episodes'] == []
    assert m['test_episodes'] == [0, 1, 2]


def test_missing_and_added_files_refused_without_download(paired_sources, monkeypatch):
    _, lm = paired_sources
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', lambda *a, **kw: pytest.fail('implicit download'))
    root = Path(lm['dataset']).parent.parent
    extra = root / 'data/extra.parquet'
    extra.write_bytes(b'extra')
    with pytest.raises(ValueError, match='inventory'):
        train.clip_dataset(lm)
    extra.unlink()
    (root / 'meta/stats.json').unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        train.clip_dataset(lm)


@pytest.mark.parametrize('data_format', ['hdf5', 'lerobot'])
def test_train_export_resume_and_cross_format_inference(paired_sources, tmp_path, monkeypatch, data_format):
    hm, lm = paired_sources
    m = hm if data_format == 'hdf5' else lm
    manifest = tmp_path / 'train_manifest.json'
    manifest.write_text(json.dumps(m))
    monkeypatch.setattr(train, 'build_model', TinyModel)
    branch = train.GaussianBranch
    monkeypatch.setattr(train, 'GaussianBranch', lambda mode, seed, **kw:
                        branch(mode, seed, dim=6, hidden=6, projections=16))
    args = argparse.Namespace(manifest=manifest, output=tmp_path / 'full', mode='bt',
        steps=4, warmup_steps=1, batch_size=2, workers=0, save_every=2, val_every=2,
        lr=.001, seed=31, accelerator='cpu', precision='32-true', resume=None, execute=True,
        bt_depth=2, bt_hidden=6, bt_kappa=.2)
    train.run(args)
    full = torch.load(args.output / 'last.ckpt', weights_only=False)
    checkpoint = args.output / 'step_2.ckpt'
    exported = torch.load(args.output / 'step_4_object.ckpt', weights_only=False).eval()
    assert exported.input_contract == m['input_contract']
    other = lm if data_format == 'hdf5' else hm
    ds = inference_dataset(exported, other)
    pred, target = predict(exported, next(iter(torch.utils.data.DataLoader(ds, batch_size=2))))
    assert pred.shape == target.shape == (2, 3, 6)
    assert torch.isfinite(pred).all()
    args.output, args.resume = tmp_path / 'resumed', checkpoint
    train.run(args)
    resumed = torch.load(args.output / 'last.ckpt', weights_only=False)
    for key in ('state_dict', 'optimizer_states', 'lr_schedulers'):
        assert_tree_equal(full[key], resumed[key])
