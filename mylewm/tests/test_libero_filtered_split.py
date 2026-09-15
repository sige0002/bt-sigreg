import json

import h5py
import numpy as np
import pytest

from mylewm.training.train_libero import prepare, split_counts
from mylewm.data.libero_bc_data import validate_manifest


def test_filtered_split_keeps_every_demo_and_train_only_statistics(tmp_path):
    folder = tmp_path / 'data'
    folder.mkdir()
    for task in range(10):
        with h5py.File(folder / f'task{task}_demo.hdf5', 'w') as f:
            for demo in range(20 + task):
                # Noncontiguous original IDs survive upstream filtering.
                a = np.arange(140).reshape(20, 7) / 1000 + demo / 100
                f.create_dataset(f'data/demo_{demo*2}/actions', data=a)
    target = tmp_path / 'manifest/manifest.json'
    prepare(folder, target, 'ratio_80_10_10')
    m = json.loads(target.read_text())
    validate_manifest(m)
    for task in range(10):
        sets = [{r['demo'] for r in m[k] if r['task'] == task}
                for k in ('train_demos', 'validation', 'test')]
        assert len(set.union(*sets)) == 20 + task
        assert all(sets) and not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    actions = []
    for row in m['train_demos']:
        with h5py.File(m['files'][row['task']]['path']) as f:
            actions.append(f[f'data/{row["demo"]}/actions'][:])
    np.testing.assert_allclose(m['action_mean'], np.concatenate(actions).mean(0))
    assert split_counts(50, 'ratio_80_10_10') == (40, 5)
    with pytest.raises(ValueError):
        split_counts(9, 'ratio_80_10_10')
