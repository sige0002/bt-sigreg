import h5py
import numpy as np
from omegaconf import OmegaConf

from lewm.eval import get_dataset


def test_actual_loader_outside_cache(tmp_path):
    path = tmp_path / 'external data.h5'
    with h5py.File(path, 'w') as f:
        f['ep_len'] = [41]
        f['ep_offset'] = [0]
        f['action'] = np.zeros((41, 2), dtype=np.float32)
    cfg = OmegaConf.create({'eval': {'dataset_path': str(path)},
                            'dataset': {'keys_to_cache': ['action']}})
    dataset = get_dataset(cfg, 'ignored_cache_name')
    assert dataset.h5_path == path
    assert dataset.get_col_data('action').shape == (41, 2)
