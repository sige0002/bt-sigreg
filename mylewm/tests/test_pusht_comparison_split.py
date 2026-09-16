import copy

import h5py
import numpy as np
import pytest

from mylewm.data.prepare_pusht_comparison import prepare, training_action_rows
from mylewm.training.train import datasets, split_datasets


def test_public_clip_split_and_training_statistics(tmp_path):
    path = tmp_path / 'data.h5'
    n = 220
    actions = np.stack([np.arange(n), np.arange(n) ** 2], axis=1).astype(float)
    with h5py.File(path, 'w') as f:
        f.create_dataset('pixels', shape=(n, 224, 224, 3), dtype='u1', chunks=(1, 224, 224, 3))
        f['action'] = actions
        f['ep_len'] = [110, 110]
        f['ep_offset'] = [0, 110]
    manifest = prepare(path, tmp_path / 'manifest.json', seed=3072)
    train, val = datasets(manifest)
    assert len(train) == 164 and len(val) == 18  # 182 clips, fraction remainder to train
    assert not set(train.indices) & set(val.indices)
    assert set(train.indices) | set(val.indices) == set(range(182))
    expected_rows = set()
    for i in train.indices:
        ep, start = train.dataset.clip_indices[i]
        begin = int(train.dataset.offsets[ep]) + start
        expected_rows.update(range(begin, begin + 20))
    selected = actions[sorted(expected_rows)]
    np.testing.assert_allclose(manifest['action_mean'], selected.mean(0))
    np.testing.assert_allclose(manifest['action_std'], selected.std(0, ddof=1))
    bad = copy.deepcopy(manifest)
    bad['train_indices_sha256'] = 'changed'
    with pytest.raises(ValueError, match='changed'):
        split_datasets(train.dataset, bad)
    train.dataset.clip_indices.reverse()
    with pytest.raises(ValueError, match='changed'):
        split_datasets(train.dataset, manifest)
    with pytest.raises(FileExistsError):
        prepare(path, tmp_path / 'manifest.json')


def test_statistics_exclude_held_only_rows():
    class Clips:
        offsets = np.array([0, 10])
        clip_indices = [(0, 0), (0, 2), (1, 1)]
        span = 3
    actual = training_action_rows(Clips(), [0, 2], 20)
    assert np.flatnonzero(actual).tolist() == [0, 1, 2, 11, 12, 13]
