"""Episode-disjoint frozen PushT state/block probes; no model or training writes."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'lewm'))
import h5py
import numpy as np
import torch
from mylewm.train_rbg import preprocess,digest
from mylewm.representation_probes import diagnose


def cases(manifest,lengths,train_count,eval_count):
    rng=np.random.default_rng(8811)
    result=[]
    for partition,count in [('train_episodes',train_count),('validation_episodes',eval_count),
                            ('test_episodes',eval_count)]:
        if count>len(manifest[partition]): raise ValueError('Too few distinct episodes')
        episodes=rng.choice(manifest[partition],count,replace=False)
        result.append([{'episode':int(ep),'start':int(rng.integers(0,lengths[ep]-15))} for ep in episodes])
    sets=[{c['episode'] for c in part} for part in result]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError('Probe episode leakage')
    return result


def state_targets(s):
    # Native PushT: agent xy, block xy/angle, agent velocity xy.
    if s.shape[-1]!=7: raise ValueError('Unexpected native PushT state')
    return np.concatenate([s[...,:4],np.sin(s[...,4:5]),np.cos(s[...,4:5]),s[...,5:]],axis=-1)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'.cache/stable-wm/pusht/rbg_v0/manifest.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train-episodes',type=int,default=256)
    p.add_argument('--eval-episodes',type=int,default=128)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if min(args.train_episodes,args.eval_episodes)<2 or args.batch_size<1: p.error('Invalid probe budget')
    torch.set_num_threads(4)
    m=json.loads(args.manifest.read_text())
    model=torch.load(args.checkpoint,map_location=args.device,weights_only=False).eval()
    model.requires_grad_(False)
    zs=[];targets=[]
    with h5py.File(m['dataset']) as f:
        selected=cases(m,f['ep_len'][:],args.train_episodes,args.eval_episodes)
        offsets=f['ep_offset'][:]
        for partition in selected:
            features=[];labels=[]
            for begin in range(0,len(partition),args.batch_size):
                images=[]
                for case in partition[begin:begin+args.batch_size]:
                    off=int(offsets[case['episode']])+case['start']
                    images.append(f['pixels'][off:off+16:5])
                    labels.append(state_targets(f['state'][off:off+16:5]))
                pixels=torch.from_numpy(np.stack(images)).permute(0,1,4,2,3)
                dummy=torch.zeros(len(images),3,5,2)
                x,_=preprocess((pixels,dummy),m,args.device)
                with torch.no_grad():features.append(model.encode({'pixels':x})['emb'].cpu().numpy())
            zs.append(np.concatenate(features));targets.append(np.stack(labels))
    report={'checkpoint_sha256':digest(args.checkpoint),'manifest_sha256':digest(args.manifest),
            'probe_source_sha256':{str(path.relative_to(ROOT)):digest(path) for path in
                                  [Path(__file__),ROOT/'mylewm/representation_probes.py']},
            'protocol':vars(args),'cases':selected,
            'state_coordinates':['agent_x','agent_y','block_x','block_y','sin_angle','cos_angle','agent_vx','agent_vy'],
            'results':diagnose(zs,targets)}
    with args.output.open('x') as f:json.dump(report,f,default=str,indent=2)
    print(json.dumps({'output':str(args.output),'all_block_readouts':report['results']['state_readouts']['all']}),flush=True)


if __name__=='__main__':main()
