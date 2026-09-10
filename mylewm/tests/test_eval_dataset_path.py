import h5py
import numpy as np
from omegaconf import OmegaConf

from lewm.eval import get_dataset


def test_metadata_provenance_does_not_hash_dataset(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from mylewm import evaluation_contract as contract
    path = tmp_path / 'data.h5'
    path.write_bytes(b'data')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'dataset':str(path), 'dataset_size':4}))
    checkpoint = tmp_path / 'model.ckpt'
    checkpoint.write_bytes(b'checkpoint')
    original = contract.file_sha256
    def guarded(target):
        assert Path(target) != path, 'Unexpected full dataset read'
        return original(target)
    monkeypatch.setattr(contract, 'file_sha256', guarded)
    result = contract.provenance(path, manifest, checkpoint, Path(__file__).resolve().parents[2], verify_data=False)
    assert result['dataset_sha256'] is None
    assert result['dataset_verification'] == 'metadata_only'


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
    assert dataset.get_row_data([0, 40])['episode_idx'].tolist() == [0, 0]
    assert dataset.get_row_data([0, 40])['step_idx'].tolist() == [0, 40]


def test_existing_episode_column_is_not_hidden(tmp_path):
    path = tmp_path / 'data.h5'
    with h5py.File(path, 'w') as f:
        f['ep_len'] = [2, 2]
        f['ep_offset'] = [0, 2]
        f['action'] = np.zeros((4, 2))
        f['ep_idx'] = [0, 0, 1, 1]
        f['step_idx'] = [0, 1, 0, 1]
    cfg = OmegaConf.create({'eval': {'dataset_path': str(path)}, 'dataset': {'keys_to_cache': ['action']}})
    ds = get_dataset(cfg, 'unused')
    assert 'ep_idx' in ds.column_names
    assert ds.get_row_data([1, 2])['ep_idx'].tolist() == [0, 1]


def test_unsupported_schema_has_actionable_error(tmp_path):
    import pytest
    from mylewm.pusht_eval_data import validate_schema
    path = tmp_path / 'unsupported.h5'
    with h5py.File(path, 'w') as f:
        f['other'] = [1]
    with pytest.raises(ValueError, match='ep_len.*available keys.*other'):
        validate_schema(path)
