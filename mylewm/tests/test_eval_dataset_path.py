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
