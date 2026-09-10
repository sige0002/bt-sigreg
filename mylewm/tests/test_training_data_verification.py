import os

import pytest

from mylewm.data import verification as dc


def manifest_for(path):
    stat = path.stat()
    return {'dataset': str(path), 'dataset_size': stat.st_size,
            'dataset_mtime_ns': stat.st_mtime_ns}


def test_fast_launch_never_hashes_and_preserves_prepare_evidence(tmp_path, monkeypatch):
    path = tmp_path / 'data.h5'
    path.write_bytes(b'original')
    manifest = manifest_for(path)
    manifest['data_fingerprints'] = dc.data_fingerprints(manifest)
    def forbidden(*args):
        raise AssertionError('Full data scan on fast launch')
    monkeypatch.setattr(dc, 'file_sha256', forbidden)
    result = dc.verify_training_data(manifest)
    assert result['prepared_sha256'] == manifest['data_fingerprints']
    path.write_bytes(b'longer changed data')
    with pytest.raises(ValueError, match='changed'):
        dc.verify_training_data(manifest)


def test_strict_detects_same_metadata_tampering(tmp_path):
    path = tmp_path / 'data.h5'
    path.write_bytes(b'original')
    manifest = manifest_for(path)
    manifest['data_fingerprints'] = dc.data_fingerprints(manifest)
    assert dc.verify_training_data(manifest, full=True) == manifest['data_fingerprints']
    old = path.stat()
    path.write_bytes(b'modified')
    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
    dc.verify_training_data(manifest)
    with pytest.raises(ValueError, match='SHA-256'):
        dc.verify_training_data(manifest, full=True)


def test_legacy_manifest_and_libero_files(tmp_path, monkeypatch):
    index = tmp_path / 'files.json'
    index.write_bytes(b'index')
    demo = tmp_path / 'demo.hdf5'
    demo.write_bytes(b'demo')
    manifest = manifest_for(index)
    manifest['files'] = [{'path': str(demo), 'size': demo.stat().st_size,
                          'mtime_ns': demo.stat().st_mtime_ns}]
    assert dc.verify_training_data(manifest)['prepared_sha256'] is None
    assert len(dc.verify_training_data(manifest, full=True)) == 2
    demo.unlink()
    with pytest.raises(FileNotFoundError):
        dc.verify_training_data(manifest)
