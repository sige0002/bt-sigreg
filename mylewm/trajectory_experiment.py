"""Ordinary-trajectory LeWM + open-loop prediction, no crossed tables."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from module import SIGReg


def sha(path):
    return hashlib.file_digest(open(path, 'rb'), 'sha256').hexdigest()


def prepare(dataset, output):
    if output.exists():
        raise FileExistsError(output)
    with h5py.File(dataset,'r') as f:
        lengths, offsets = f['ep_len'][:], f['ep_offset'][:]
        rng = np.random.default_rng(20260906)
        eligible = np.flatnonzero(lengths > 40)
        reserved = rng.choice(eligible, 378, replace=False)
        def cases(ids):
            return [{'episode':int(e), 'start':int(rng.integers(0,lengths[e]-26))} for e in ids]
        pilot, confirm = cases(reserved[:50]), cases(reserved[50:250])
        train = np.setdiff1d(eligible, reserved)
        val = reserved[250:]
        # Same full-dataset action scaling used by the published checkpoint's evaluator.
        # No task reward or target-state labels are used for training.
        actions = f['action'][:]
        actions = actions[np.isfinite(actions).all(1)].astype(np.float64)
        value = {'schema':'trajectory_v1','dataset':str(dataset.resolve()),
                 'dataset_size':dataset.stat().st_size,'seed':20260906,
                 'train_episodes':train.tolist(),'validation':cases(val),
                 'pilot':pilot,'confirm':confirm,'action_mean':actions.mean(0).tolist(),
                 'action_std':actions.std(0).tolist(),'frameskip':5,'history':3,'horizons':[1,2,4],
                 'init_checkpoint':str(Path('.cache/stable-wm/pusht/lewm_object.ckpt').resolve()),
                 'initial_sha256':sha('.cache/stable-wm/pusht/lewm_object.ckpt'),
                 'noninferiority_margin':0.0,'alpha':0.05,
                 'note':'Fine-tuning holdout only; official checkpoint may have trained on these episodes.'}
        # Validation needs a full 31-frame window.
        for item in value['validation']:
            item['start'] = min(item['start'],int(lengths[item['episode']])-31)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(value,indent=2))
    print(json.dumps({'manifest':str(output),'train_episodes':len(train),'pilot':50,'confirm':200,'validation':128}))


class Trajectories(torch.utils.data.Dataset):
    def __init__(self, manifest, validation=False):
        self.manifest = manifest
        self.file = None
        with h5py.File(manifest['dataset'],'r') as f:
            self.offsets = f['ep_offset'][:]
            self.lengths = f['ep_len'][:]
        self.validation = validation
        if validation:
            self.cases = manifest['validation']
        else:
            self.episodes = np.asarray(manifest['train_episodes'])
            self.ends = np.cumsum(self.lengths[self.episodes]-30)

    def __len__(self):
        return len(self.cases) if self.validation else int(self.ends[-1])

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.manifest['dataset'],'r')
        if self.validation:
            ep, start = self.cases[index]['episode'],self.cases[index]['start']
        else:
            e = int(np.searchsorted(self.ends,index,side='right'))
            ep = self.episodes[e]
            start = int(index-(self.ends[e-1] if e else 0))
        offset = int(self.offsets[ep])+start
        pixels = self.file['pixels'][offset:offset+31:5]
        actions = self.file['action'][offset:offset+30]
        if len(pixels) != 7 or not np.isfinite(actions).all():
            raise ValueError('invalid trajectory interval')
        return torch.from_numpy(pixels).permute(0,3,1,2),torch.from_numpy(actions).reshape(6,5,2)


def preprocess(batch, manifest, device):
    pixels, actions = [x.to(device) for x in batch]
    x = pixels.float()/255
    mean = x.new_tensor([.485,.456,.406]).view(1,1,3,1,1)
    std = x.new_tensor([.229,.224,.225]).view(1,1,3,1,1)
    a = (actions.float()-x.new_tensor(manifest['action_mean']))/x.new_tensor(manifest['action_std'])
    return (x-mean)/std,a.flatten(-2)


def objective(model, pixels, actions, regularizer, multi_weight=.1):
    # Future observations are encoded independently and never passed into F.
    z = model.encode({'pixels':pixels[:,:3]})['emb']
    y = model.encode({'pixels':pixels[:,3:]})['emb']
    u = model.action_encoder(actions)
    pred = model.predict(z,u[:,:3])
    one = (pred-torch.cat((z[:,1:],y[:,:1]),1)).square().mean()
    rolling = z
    terms = []
    for h in range(4):
        q = pred[:,-1:] if h == 0 else model.predict(rolling[:,-3:],u[:,h:h+3])[:,-1:]
        rolling = torch.cat((rolling,q),1)
        if h in (0,1,3):
            terms.append((q-y[:,h:h+1]).square().mean())
    multi = torch.stack(terms).mean()
    sig = regularizer(torch.cat((z,y),1).transpose(0,1).float())
    return one + .09*sig + multi_weight*multi, {'one_step':one,'multi_step':multi,'sigreg':sig}


def save(value,path):
    tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,tmp)
    os.replace(tmp,path)


def train(args):
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    manifest=json.loads(args.manifest.read_text())
    device='cuda'
    model=torch.load(manifest['init_checkpoint'],map_location=device,weights_only=False)
    model.requires_grad_(True)
    for name,key in [('training_action_mean','action_mean'),('training_action_std','action_std')]:
        model.register_buffer(name,torch.tensor(manifest[key],device=device,dtype=torch.float64))
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-3)
    sig=SIGReg().to(device)
    gen=torch.Generator().manual_seed(args.seed)
    ds=Trajectories(manifest)
    sampler=torch.utils.data.RandomSampler(ds,replacement=True,num_samples=args.steps*args.batch_size,generator=gen)
    loader=torch.utils.data.DataLoader(ds,batch_size=args.batch_size,sampler=sampler,num_workers=2,pin_memory=True)
    val=torch.utils.data.DataLoader(Trajectories(manifest,True),batch_size=args.batch_size,num_workers=0)
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'metrics.jsonl').exists():
        raise FileExistsError('Use a fresh output directory; never overwrite a completed experiment')
    (args.output/'config.json').write_text(json.dumps({**vars(args),'manifest_sha256':sha(args.manifest)},default=str,indent=2))
    start=time.monotonic()
    with (args.output/'metrics.jsonl').open('w') as log:
        for step,batch in enumerate(loader,1):
            model.train()
            opt.zero_grad(set_to_none=True)
            x,a=preprocess(batch,manifest,device)
            loss,parts=objective(model,x,a,sig,args.multi_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'nonfinite loss at {step}')
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            row={'step':step,'loss':float(loss.detach()),'grad_norm':float(grad),'elapsed':time.monotonic()-start,
                 **{k:float(v.detach()) for k,v in parts.items()}}
            if step % args.save_every == 0 or step == args.steps:
                model.eval()
                values=[]
                with torch.no_grad():
                    for vb in val:
                        vx,va=preprocess(vb,manifest,device)
                        _,parts=objective(model,vx,va,sig,args.multi_weight)
                        values.append({k:float(v) for k,v in parts.items()})
                row['validation']={k:float(np.mean([v[k] for v in values])) for k in values[0]}
                save(model,args.output/f'step_{step}_object.ckpt')
                save({'step':step,'model':model.state_dict(),'optimizer':opt.state_dict(),
                      'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                      'manifest_sha256':sha(args.manifest)},args.output/'optimizer.ckpt')
            log.write(json.dumps(row)+'\n'); log.flush()
            if step%10 == 0 or step == 1:
                print(json.dumps(row),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['prepare','train'])
    p.add_argument('--manifest',type=Path,default=Path('.cache/stable-wm/pusht/trajectory_v1/manifest.json'))
    p.add_argument('--dataset',type=Path,default=Path('.cache/stable-wm/datasets/pusht_expert_train.h5'))
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--save-every',type=int,default=1000)
    p.add_argument('--lr',type=float,default=1e-6)
    p.add_argument('--multi-weight',type=float,default=.1)
    p.add_argument('--seed',type=int,default=3072)
    p.add_argument('--output',type=Path,default=Path('.cache/stable-wm/pusht/trajectory_v1/train'))
    args=p.parse_args()
    if args.command=='prepare': prepare(args.dataset,args.manifest)
    else: train(args)


if __name__=='__main__': main()
