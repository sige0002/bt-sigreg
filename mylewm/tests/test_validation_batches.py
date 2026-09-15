"""Singleton-tail regression using native LIBERO clips and the shared loss."""
import argparse
import json

import h5py
import numpy as np
import pytest
import torch

from mylewm.training import loop as training
from mylewm.training.train_libero import LiberoClips
from test_bt_sigreg import TinyBTModel
from test_training_state import assert_tree_equal


@pytest.mark.parametrize('size,batch_size,expected', [
    (50, 49, [48, 2]), (50, 7, [7] * 6 + [6, 2]),
    (50, 128, [50]), (50, 16, [16, 16, 16, 2]),
    (6, 3, [3, 3]), (4, 3, [2, 2]),
    (5, 2, [2, 3]), (3, 2, [3]), (2, 2, [2]),
])
def test_validation_batches_keep_every_case_once_in_order(size, batch_size, expected):
    sampler = training.ValidationBatches(size, batch_size)
    batches = list(sampler)
    assert [len(batch) for batch in batches] == expected
    assert len(sampler) == len(batches)
    assert [index for batch in batches for index in batch] == list(range(size))
    assert list(sampler) == batches


@pytest.fixture
def libero_manifest(tmp_path):
    path = tmp_path / 'task.hdf5'
    with h5py.File(path, 'w') as f:
        for index in range(51):
            for camera, offset in [('agentview_rgb', 0), ('eye_in_hand_rgb', 80)]:
                pixels = (np.arange(13, dtype=np.uint8) + index + offset)[:, None, None, None]
                f[f'data/demo_{index}/obs/{camera}'] = np.broadcast_to(pixels, (13, 2, 2, 3))
            f[f'data/demo_{index}/actions'] = np.full((13, 7), index / 100, dtype=np.float32)
    files = [{'path': str(path), 'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}]
    index_path = tmp_path / 'files.json'
    index_path.write_text(json.dumps(files))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({
        'dataset': str(index_path), 'dataset_size': index_path.stat().st_size,
        'dataset_mtime_ns': index_path.stat().st_mtime_ns, 'files': files,
        'train_demos': [{'task': 0, 'demo': 'demo_50', 'length': 13}],
        'validation': [{'task': 0, 'demo': f'demo_{i}', 'length': 13, 'start': 0}
                       for i in range(50)],
        'camera_order': ['agentview_rgb', 'eye_in_hand_rgb'],
        'action_mean': [0.] * 7, 'action_std': [1.] * 7,
    }))
    return manifest


def tiny_preprocess(batch, manifest, device):
    # Preserve both cameras and all temporal/action indices; replace only the
    # expensive image backbone input with RGB averages for the tiny test model.
    pixels, actions = (value.to(device) for value in batch)
    return pixels.float().mean((-2, -1)) / 255, actions.flatten(-2)


def arguments(manifest, output, mode='bt'):
    return argparse.Namespace(
        steps=4, warmup_steps=1, lr=.003, min_lr=0., seed=17,
        manifest=manifest, mode=mode, batch_size=49, workers=0,
        bt_depth=2, bt_hidden=8, bt_kappa=.2, gaussian_weight=.09,
        diagnostics_every=0, save_every=2, resume=False, output=output,
    )


@pytest.mark.parametrize('mode', ['raw', 'tc', 'bt'])
def test_libero_validation_singleton_tail_completes_and_bt_resumes(
        tmp_path, monkeypatch, libero_manifest, mode):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    visited = []

    class ObservedClips(LiberoClips):
        def __getitem__(self, index):
            if self.validation:
                visited.append(index)
            return super().__getitem__(index)

    adapter = training.TrainingAdapter(ObservedClips, tiny_preprocess, lambda: TinyBTModel(2))
    args = arguments(libero_manifest, tmp_path / 'full', mode)
    objective = training.one_step_objective
    validation_sizes = []
    train_calls = 0
    interrupt = False

    def observe(model, pixels, actions, *pos):
        nonlocal train_calls
        if model.training:
            train_calls += 1
            if interrupt and train_calls == 3:
                raise InterruptedError('saved update 2')
        else:
            validation_sizes.append(len(pixels))
        return objective(model, pixels, actions, *pos)

    monkeypatch.setattr(training, 'one_step_objective', observe)
    training.train(args, adapter=adapter)
    assert validation_sizes == [48, 2, 48, 2]
    assert visited == list(range(50)) * 2
    assert json.loads((args.output / 'completed.json').read_text())['step'] == 4
    assert json.loads((args.output / 'config.json').read_text())['validation_batching'] == 'sequential_min_two_v1'
    rows = [json.loads(line) for line in (args.output / 'metrics.jsonl').read_text().splitlines()]
    assert all(np.isfinite(list(row['validation'].values())).all() for row in rows if 'validation' in row)

    if mode != 'bt':
        return
    full = torch.load(args.output / 'resume.pt', map_location='cpu', weights_only=False)
    visited.clear()
    validation_sizes.clear()
    train_calls = 0
    interrupt = True
    args.output = tmp_path / 'resumed'
    with pytest.raises(InterruptedError, match='saved update 2'):
        training.train(args, adapter=adapter)
    interrupt = False
    args.resume = True
    training.train(args, adapter=adapter)
    resumed = torch.load(args.output / 'resume.pt', map_location='cpu', weights_only=False)
    assert validation_sizes == [48, 2, 48, 2]
    assert visited == list(range(50)) * 2
    for key in ('model', 'regularizer', 'optimizer', 'scheduler', 'random_state', 'next_batch_index'):
        assert_tree_equal(full[key], resumed[key])
    assert full['metrics_row']['validation'] == resumed['metrics_row']['validation']


@pytest.mark.parametrize('size', [0, 1])
def test_reject_insufficient_validation_before_model_or_output(
        tmp_path, monkeypatch, libero_manifest, size):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    manifest = json.loads(libero_manifest.read_text())
    manifest['validation'] = manifest['validation'][:size]
    libero_manifest.write_text(json.dumps(manifest))

    def forbidden():
        raise AssertionError('Model must not be initialized')

    adapter = training.TrainingAdapter(LiberoClips, tiny_preprocess, forbidden)
    args = arguments(libero_manifest, tmp_path / 'invalid')
    with pytest.raises(ValueError, match='at least two validation clips'):
        training.train(args, adapter=adapter)
    assert not args.output.exists()


@pytest.mark.parametrize('batch_size', [0, 1])
def test_reject_invalid_training_batch_before_loading_data(batch_size):
    args = argparse.Namespace(mode='bt', batch_size=batch_size)
    with pytest.raises(ValueError, match='--batch-size >= 2'):
        training.train(args)
