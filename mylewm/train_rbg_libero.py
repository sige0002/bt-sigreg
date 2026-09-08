"""LIBERO-10 data/model adapters for the identical RBG comparison trainer."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'lewm'))
import h5py
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from mylewm import train_rbg as common


def prepare(folder,path):
    if path.exists(): raise FileExistsError(path)
    files=sorted(folder.glob('*.hdf5'))
    if len(files)!=10: raise ValueError('Require exactly ten official task files')
    rng=np.random.default_rng(20260906)
    train=[]; validation=[]; test=[]; metadata=[]; actions=[]
    for task,p in enumerate(files):
        metadata.append({'path':str(p.resolve()),'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns})
        with h5py.File(p) as f:
            names=sorted(f['data'],key=lambda s:int(s.split('_')[-1]))
            names=np.array(names)[rng.permutation(len(names))]
            for j,name in enumerate(names):
                n=len(f[f'data/{name}/actions'])
                if n<=12: raise ValueError('Demonstration too short')
                entry={'task':task,'demo':str(name),'length':n}
                if j<40:
                    train.append(entry)
                    a=f[f'data/{name}/actions'][:]
                    if not np.isfinite(a).all(): raise ValueError('Nonfinite action')
                    actions.append(a)
                else:
                    entry['start']=int(rng.integers(0,n-12))
                    (validation if j<45 else test).append(entry)
    a=np.concatenate(actions).astype(np.float64)
    if min(a.std(0,ddof=1))<=0: raise ValueError('Degenerate action')
    path.parent.mkdir(parents=True,exist_ok=True)
    index=path.with_name('files.json')
    if index.exists(): raise FileExistsError(index)
    index.write_text(json.dumps(metadata,indent=2))
    m={'schema':'rbg_libero10_v1','dataset':str(index.resolve()),
       'dataset_size':index.stat().st_size,'dataset_mtime_ns':index.stat().st_mtime_ns,
       'files':metadata,'train_demos':train,'validation':validation,'test':test,
       'action_mean':a.mean(0).tolist(),'action_std':a.std(0,ddof=1).tolist(),
       'frameskip':4,'history':3,'image_size':224,'camera_order':['agentview_rgb','eye_in_hand_rgb'],
       'image_convention':'native opengl; do not flip training images independently of simulator',
       'task_names':[p.stem.removesuffix('_demo') for p in files],
       'initialization':'random','split_seed':20260906}
    path.write_text(json.dumps(m,indent=2))
    print(json.dumps({'train_demos':len(train),'validation_demos':len(validation),'test_demos':len(test)}),flush=True)


class LiberoClips(torch.utils.data.Dataset):
    def __init__(self,m,validation=False):
        self.m,self.validation,self.files=m,validation,{}
        for item in m['files']:
            p=Path(item['path'])
            if p.stat().st_size!=item['size'] or p.stat().st_mtime_ns!=item['mtime_ns']:
                raise ValueError(f'Dataset changed: {p}')
        self.demos=m['validation'] if validation else m['train_demos']
        self.ends=np.cumsum([x['length']-12 for x in self.demos])
    def __len__(self):
        return len(self.demos) if self.validation else int(self.ends[-1])
    def __getitem__(self,i):
        if self.validation: entry=self.demos[i]; start=entry['start']
        else:
            j=int(np.searchsorted(self.ends,i,side='right'))
            entry=self.demos[j]; start=int(i-(self.ends[j-1] if j else 0))
        task=entry['task']
        if task not in self.files: self.files[task]=h5py.File(self.m['files'][task]['path'],'r')
        demo=self.files[task][f'data/{entry["demo"]}']
        x=np.stack([demo[f'obs/{cam}'][start:start+13:4] for cam in self.m['camera_order']],axis=1)
        a=demo['actions'][start:start+12]
        if x.shape[0]!=4 or a.shape!=(12,7): raise ValueError('Temporal contract violated')
        return torch.from_numpy(x).permute(0,1,4,2,3),torch.from_numpy(a).reshape(3,4,7)


def preprocess(batch,m,device):
    pixels,actions=[x.to(device,non_blocking=True) for x in batch]
    b,t,v,c,h,w=pixels.shape
    x=F.interpolate(pixels.flatten(0,2).float()/255,size=(224,224),mode='bilinear',align_corners=False,antialias=True)
    x=x.reshape(b,t,v,3,224,224)
    mean=x.new_tensor([.485,.456,.406]).view(1,1,1,3,1,1)
    std=x.new_tensor([.229,.224,.225]).view(1,1,1,3,1,1)
    a=(actions.float()-x.new_tensor(m['action_mean']))/x.new_tensor(m['action_std'])
    return (x-mean)/std,a.flatten(-2)


def build_model():
    cfg=OmegaConf.create({'embed_dim':192,'history_size':3,'img_size':224,
                         'model':OmegaConf.load(ROOT/'lewm/config/train/model/lewm.yaml')})
    cfg.model._target_='mylewm.libero_model.TwoViewJEPA'
    cfg.model.action_encoder.input_dim=28
    cfg.model.projector.input_dim=384
    return hydra.utils.instantiate(cfg.model)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['prepare','train'])
    p.add_argument('--dataset',type=Path,default=ROOT/'.cache/libero-datasets/libero_10')
    p.add_argument('--manifest',type=Path,default=ROOT/'.cache/stable-wm/libero10/rbg_v0/manifest.json')
    p.add_argument('--output',type=Path)
    p.add_argument('--mode',choices=['raw','rbg','tc','bt'],default='rbg')
    common.add_bt_arguments(p)
    p.add_argument('--steps','--total-steps',dest='steps',type=int,default=50000)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--blocks',type=int,default=4)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--save-every',type=int,default=1000)
    p.add_argument('--seed',type=int,default=3072)
    common.add_schedule_arguments(p)
    p.add_argument('--gaussian-weight',type=float,default=.09)
    p.add_argument('--cross-weight',type=float,default=.01)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.command=='prepare': prepare(args.dataset,args.manifest)
    else:
        if args.output is None: p.error('--output required')
        args.adapter_sources={str(f.relative_to(ROOT)):common.digest(f) for f in
            [Path(__file__),ROOT/'mylewm/libero_model.py']}
        # Explicit process-local adapters; common training loop and objective are unchanged.
        common.Clips=LiberoClips
        common.preprocess=preprocess
        common.build_model=build_model
        common.train(args)


if __name__=='__main__': main()
