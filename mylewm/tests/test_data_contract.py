import os
import h5py
import numpy as np
import pytest

from mylewm.data.verification import data_fingerprints, training_budget
from mylewm.training.loop import Clips


def test_clip_budget_from_synthetic_episode_lengths(tmp_path):
    path=tmp_path/'tiny.h5'
    with h5py.File(path,'w') as f:
        f['ep_len']=[41,50,80]
        f['ep_offset']=[0,41,91]
    clips=Clips({'dataset':str(path),'train_episodes':[0,2]})
    assert len(clips)==(41-15)+(80-15)==91
    budget=training_budget(len(clips),50000,8)
    assert budget['updates_per_full_epoch_drop_last']==11
    assert budget['updates_for_100_full_epochs_drop_last']==1100
    assert budget['presented_clips']==400000
    assert budget['presentation_equivalent_epochs']==400000/91


def test_content_hash_detects_change_with_preserved_size_and_mtime(tmp_path):
    path=tmp_path/'tiny.h5'
    path.write_bytes(b'original')
    manifest={'dataset':str(path)}
    first=data_fingerprints(manifest)
    old=path.stat()
    path.write_bytes(b'modified')
    os.utime(path,ns=(old.st_atime_ns,old.st_mtime_ns))
    second=data_fingerprints(manifest)
    assert path.stat().st_size==old.st_size
    assert path.stat().st_mtime_ns==old.st_mtime_ns
    assert first!=second


def test_libero_identity_hashes_all_hdf5_files(tmp_path):
    index=tmp_path/'index.json'
    index.write_bytes(b'index')
    demo=tmp_path/'demo.hdf5'
    demo.write_bytes(b'demo')
    assert len(data_fingerprints({'dataset':str(index),'files':[{'path':str(demo)}]}))==2


@pytest.mark.parametrize('clips,updates,batch', [(0,1,1),(1,0,1),(1,1,0)])
def test_invalid_budget_rejected(clips,updates,batch):
    with pytest.raises(ValueError): training_budget(clips,updates,batch)
