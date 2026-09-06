"""Reproducible from-scratch PushT training for RBG/Raw/TC comparisons."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lewm'))
sys.path.insert(0, str(ROOT))
import h5py
import hdf5plugin  # noqa: F401
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from mylewm.rbg import BlockSIGReg, one_step_objective


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(dataset, path):
    if path.exists():
        raise FileExistsError(path)
    with h5py.File(dataset) as f:
        lengths, offsets = f['ep_len'][:], f['ep_offset'][:]
        eligible = np.flatnonzero(lengths >= 41)
        rng = np.random.default_rng(20260906)
        ids = rng.permutation(eligible)
        n = len(ids)
        train, val, test = ids[:int(.8*n)], ids[int(.8*n):int(.9*n)], ids[int(.9*n):]
        actions = []
        for e in train:
            a = f['action'][int(offsets[e]):int(offsets[e]+lengths[e])-1]
            actions.append(a[np.isfinite(a).all(1)])
        a = np.concatenate(actions).astype(np.float64)
        def cases(es, count):
            return [{'episode':int(e), 'start':int(rng.integers(0,int(lengths[e])-40))}
                    for e in es[:count]]
        manifest = {'schema':'rbg_pusht_v1', 'dataset':str(dataset.resolve()),
                    'dataset_size':dataset.stat().st_size,
                    'dataset_mtime_ns':dataset.stat().st_mtime_ns,
                    'train_episodes':train.tolist(), 'validation_episodes':val.tolist(),
                    'test_episodes':test.tolist(), 'validation':cases(val,256),
                    'pilot':cases(val[256:],50), 'confirm':cases(test,200),
                    'action_mean':a.mean(0).tolist(), 'action_std':a.std(0,ddof=1).tolist(),
                    'frameskip':5, 'history':3, 'split_seed':20260906,
                    'initialization':'random; official checkpoint not used',
                    'note':'Official published model may have seen held-out episodes.'}
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'train':len(train),'val':len(val),'test':len(test)}),flush=True)


class Clips(torch.utils.data.Dataset):
    def __init__(self, manifest, validation=False):
        self.manifest, self.validation, self.file = manifest, validation, None
        with h5py.File(manifest['dataset']) as f:
            self.offsets, lengths = f['ep_offset'][:], f['ep_len'][:]
        self.episodes = np.array(manifest['train_episodes'])
        # Exclude terminal action; final image is start+15.
        self.ends = np.cumsum(lengths[self.episodes]-15)
    def __len__(self):
        return len(self.manifest['validation']) if self.validation else int(self.ends[-1])
    def __getitem__(self, i):
        if self.file is None:
            self.file = h5py.File(self.manifest['dataset'],'r')
        if self.validation:
            case = self.manifest['validation'][i]
            ep, start = case['episode'],case['start']
        else:
            j = int(np.searchsorted(self.ends,i,side='right'))
            ep = self.episodes[j]
            start = i - (int(self.ends[j-1]) if j else 0)
        off = int(self.offsets[ep])+int(start)
        pixels = self.file['pixels'][off:off+16:5]
        actions = self.file['action'][off:off+15]
        if len(pixels)!=4 or not np.isfinite(actions).all():
            raise ValueError(f'invalid transition episode={ep} start={start}')
        return torch.from_numpy(pixels).permute(0,3,1,2),torch.from_numpy(actions).reshape(3,5,2)


class StepBatches:
    """Stateless per-update sampling permits exact resume without replaying I/O."""
    def __init__(self,n,batch,steps,seed,start=0):
        self.n,self.batch,self.steps,self.seed,self.start=n,batch,steps,seed,start
    def __len__(self):
        return self.steps-self.start
    def __iter__(self):
        for step in range(self.start,self.steps):
            yield np.random.default_rng(np.random.SeedSequence([self.seed,step])).integers(
                0,self.n,size=self.batch).tolist()


def preprocess(batch,m,device):
    pixels, actions = [x.to(device,non_blocking=True) for x in batch]
    x = pixels.float()/255
    mean=x.new_tensor([.485,.456,.406]).view(1,1,3,1,1)
    std=x.new_tensor([.229,.224,.225]).view(1,1,3,1,1)
    a=(actions.float()-x.new_tensor(m['action_mean']))/x.new_tensor(m['action_std'])
    return (x-mean)/std,a.flatten(-2)


def build_model():
    cfg=OmegaConf.create({'embed_dim':192,'history_size':3,'img_size':224,
                         'model':OmegaConf.load(ROOT/'lewm/config/train/model/lewm.yaml')})
    cfg.model.action_encoder.input_dim=10
    return hydra.utils.instantiate(cfg.model)


def atomic_save(value,path):
    tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,tmp)
    os.replace(tmp,path)


def train(args):
    torch.set_num_threads(4)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    m=json.loads(args.manifest.read_text())
    dataset=Path(m['dataset'])
    if dataset.stat().st_size!=m['dataset_size'] or dataset.stat().st_mtime_ns!=m['dataset_mtime_ns']:
        raise ValueError('Dataset changed since split preparation')
    if min(m['action_std'])<=0:
        raise ValueError('invalid action scale')
    model=build_model().to(device)
    model.register_buffer('training_action_mean',torch.tensor(m['action_mean'],device=device,dtype=torch.float64))
    model.register_buffer('training_action_std',torch.tensor(m['action_std'],device=device,dtype=torch.float64))
    reg=BlockSIGReg(blocks=args.blocks if args.mode=='rbg' else 1).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.001)
    config={**vars(args),'manifest_sha256':digest(args.manifest),
            'parameters':sum(p.numel() for p in model.parameters()),
            'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in
                [Path(__file__),ROOT/'mylewm/rbg.py',ROOT/'lewm/jepa.py',ROOT/'lewm/module.py']}}
    config=json.loads(json.dumps(config,default=str))
    start_step=0
    if args.resume:
        state=torch.load(args.output/'resume.pt',map_location=device,weights_only=False)
        previous=json.loads((args.output/'config.json').read_text())
        for key in ('seed','mode','blocks','batch_size','lr','steps','manifest_sha256','source_sha256'):
            if previous[key]!=config[key]: raise ValueError(f'resume mismatch: {key}')
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        reg.load_state_dict(state['regularizer'])
        start_step=state['step']
        torch.set_rng_state(state['rng'].cpu())
        if device=='cuda': torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'config.json').write_text(json.dumps(config,indent=2))
    ds=Clips(m)
    loader=torch.utils.data.DataLoader(ds,batch_sampler=StepBatches(len(ds),args.batch_size,args.steps,args.seed,start_step),
        num_workers=args.workers,pin_memory=device=='cuda',persistent_workers=args.workers>0)
    val=torch.utils.data.DataLoader(Clips(m,True),batch_size=args.batch_size,num_workers=0)
    begin=time.monotonic()
    print(json.dumps({'event':'start','parameters':config['parameters'],'mode':args.mode,
                      'train_clips':len(ds),'steps':args.steps,'device':device}),flush=True)
    cross_weight=args.cross_weight if args.mode=='rbg' else 0.
    with (args.output/'metrics.jsonl').open('a') as log:
        for step,batch in enumerate(loader,start_step+1):
            model.train(); opt.zero_grad(set_to_none=True)
            x,a=preprocess(batch,m,device)
            with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):
                loss,parts=one_step_objective(model,x,a,reg,args.gaussian_weight,cross_weight,args.mode)
            if not torch.isfinite(loss): raise FloatingPointError(f'loss at {step}')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            row={'step':step,'loss':float(loss.detach()),'grad_norm':float(norm),
                 'elapsed_session':time.monotonic()-begin,**{k:float(v.detach()) for k,v in parts.items()}}
            if step%args.save_every==0 or step==args.steps:
                model.eval(); values=[]
                with torch.random.fork_rng(devices=[0] if device=='cuda' else []),torch.no_grad():
                    torch.manual_seed(999)
                    for vb in val:
                        vx,va=preprocess(vb,m,device)
                        with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):
                            _,vp=one_step_objective(model,vx,va,reg,args.gaussian_weight,cross_weight,args.mode)
                        values.append({k:float(v) for k,v in vp.items()})
                row['validation']={k:float(np.mean([v[k] for v in values])) for k in values[0]}
                atomic_save(model,args.output/f'step_{step}_object.ckpt')
                atomic_save({'step':step,'model':model.state_dict(),'regularizer':reg.state_dict(),
                    'optimizer':opt.state_dict(),'rng':torch.get_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else []},args.output/'resume.pt')
            log.write(json.dumps(row)+'\n'); log.flush()
            if step==1 or step%10==0 or step==args.steps: print(json.dumps(row),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['prepare','train'])
    p.add_argument('--dataset',type=Path,default=ROOT/'.cache/stable-wm/datasets/pusht_expert_train.h5')
    p.add_argument('--manifest',type=Path,default=ROOT/'.cache/stable-wm/pusht/rbg_v0/manifest.json')
    p.add_argument('--output',type=Path)
    p.add_argument('--mode',choices=['rbg','raw','tc'],default='rbg')
    p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--blocks',type=int,default=4)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--save-every',type=int,default=1000)
    p.add_argument('--seed',type=int,default=3072)
    p.add_argument('--lr',type=float,default=5e-5)
    p.add_argument('--gaussian-weight',type=float,default=.09)
    p.add_argument('--cross-weight',type=float,default=.01)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.command=='prepare': prepare(args.dataset,args.manifest)
    else:
        if args.output is None: p.error('--output is required for train')
        train(args)


if __name__=='__main__': main()
