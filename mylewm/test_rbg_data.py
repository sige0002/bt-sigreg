"""Temporal and split contracts using tiny, exactly indexed synthetic files."""
import h5py
import numpy as np
import torch
from mylewm.train_rbg import Clips
from mylewm.train_rbg_libero import LiberoClips


def test_pusht_last_window_does_not_cross_episode(tmp_path):
    p=tmp_path/'tiny.h5'
    with h5py.File(p,'w') as f:
        f['ep_len']=[20,20]; f['ep_offset']=[0,20]
        f['pixels']=np.arange(40,dtype=np.uint8)[:,None,None,None]*np.ones((40,2,2,3),dtype=np.uint8)
        f['action']=np.repeat(np.arange(40)[:,None],2,1).astype(np.float32)
    ds=Clips({'dataset':str(p),'train_episodes':[0,1]})
    assert len(ds)==10
    x,a=ds[4]
    torch.testing.assert_close(x[:,0,0,0],torch.tensor([4,9,14,19],dtype=torch.uint8))
    torch.testing.assert_close(a.flatten(0,1)[:,0],torch.arange(4,19).float())
    x,_=ds[5]
    assert x[0,0,0,0]==20


def test_libero_preserves_two_distinct_views_and_native_actions(tmp_path):
    p=tmp_path/'task.hdf5'
    with h5py.File(p,'w') as f:
        for cam,add in [('agentview_rgb',0),('eye_in_hand_rgb',100)]:
            f[f'data/demo_0/obs/{cam}']=(np.arange(20,dtype=np.uint8)+add)[:,None,None,None]*np.ones((20,2,2,3),dtype=np.uint8)
        f['data/demo_0/actions']=np.repeat(np.arange(20)[:,None],7,1).astype(np.float32)
    m={'files':[{'path':str(p),'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}],
       'train_demos':[{'task':0,'demo':'demo_0','length':20}],
       'camera_order':['agentview_rgb','eye_in_hand_rgb']}
    ds=LiberoClips(m)
    assert len(ds)==8
    x,a=ds[7]
    torch.testing.assert_close(x[:,0,0,0,0],torch.tensor([7,11,15,19],dtype=torch.uint8))
    torch.testing.assert_close(x[:,1,0,0,0],torch.tensor([107,111,115,119],dtype=torch.uint8))
    assert a.shape==(3,4,7)
    torch.testing.assert_close(a.flatten(0,1)[:,0],torch.arange(7,19).float())
