"""Episode-disjoint LIBERO frozen-feature probes using recorded real camera pairs."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'lewm'))
import h5py
import numpy as np
import torch
from mylewm.train_rbg import digest
from mylewm.train_rbg_libero import preprocess,LiberoClips
from mylewm.representation_probes import diagnose,task_probe


def cases(manifest,train_count,eval_count):
    rng=np.random.default_rng(8811);result=[]
    for key,count in [('train_demos',train_count),('validation',eval_count),('test',eval_count)]:
        selected=[]
        for task in range(10):
            options=[v for v in manifest[key] if v['task']==task]
            if count>len(options):raise ValueError('Insufficient distinct task demonstrations')
            for i in rng.choice(len(options),count,replace=False):
                entry=dict(options[i]);entry['start']=int(rng.integers(0,entry['length']-12))
                selected.append(entry)
        result.append(selected)
    keys=[{(c['task'],c['demo']) for c in part} for part in result]
    if any(keys[i]&keys[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError('Probe demonstration leakage')
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'.cache/stable-wm/libero10/rbg_v0/manifest.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train-per-task',type=int,default=40)
    p.add_argument('--eval-per-task',type=int,default=5)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if min(args.train_per_task,args.eval_per_task,args.batch_size)<1:p.error('Invalid probe budget')
    torch.set_num_threads(4)
    m=json.loads(args.manifest.read_text())
    LiberoClips(m)  # Verify source dataset metadata before directly reading labels.
    selected=cases(m,args.train_per_task,args.eval_per_task)
    model=torch.load(args.checkpoint,map_location=args.device,weights_only=False).eval()
    model.requires_grad_(False)
    zs=[];targets=[];labels=[]
    files=[h5py.File(item['path'],'r') for item in m['files']]
    try:
        for partition in selected:
            features=[];states=[]
            for begin in range(0,len(partition),args.batch_size):
                images=[]
                for case in partition[begin:begin+args.batch_size]:
                    demo=files[case['task']][f'data/{case["demo"]}'];start=case['start']
                    images.append(np.stack([demo[f'obs/{cam}'][start:start+13:4]
                                            for cam in m['camera_order']],axis=1))
                    # Robot coordinates have common semantics across tasks. Native
                    # simulator state vectors include task-dependent object layouts.
                    states.append(np.concatenate([demo[f'obs/{key}'][start:start+13:4]
                        for key in ('ee_pos','joint_states','gripper_states')],axis=-1))
                pixels=torch.from_numpy(np.stack(images)).permute(0,1,2,5,3,4)
                x,_=preprocess((pixels,torch.zeros(len(images),3,4,7)),m,args.device)
                with torch.no_grad():features.append(model.encode({'pixels':x})['emb'].cpu().numpy())
            zs.append(np.concatenate(features));targets.append(np.stack(states))
            labels.append(np.array([case['task'] for case in partition]))
    finally:
        for file in files:file.close()
    result=diagnose(zs,targets)
    means=[z.mean(1) for z in zs]
    result['task_identity_readout']={name:task_probe(
        [x if block is None else np.split(x,4,axis=-1)[block] for x in means],labels)
        for name,block in [('all',None),*[(f'block_{i}',i) for i in range(4)]]}
    result['per_task_variance']={m['task_names'][t]:{
        'within_window':float(np.square(zs[2][labels[2]==t]-zs[2][labels[2]==t].mean(1,keepdims=True)).mean()),
        'between_window_means':float(means[2][labels[2]==t].var(0).mean())} for t in range(10)}
    report={'checkpoint_sha256':digest(args.checkpoint),'manifest_sha256':digest(args.manifest),
            'probe_source_sha256':{str(path.relative_to(ROOT)):digest(path) for path in
                                  [Path(__file__),ROOT/'mylewm/representation_probes.py']},
            'protocol':vars(args),'cases':selected,'results':result,
            'state_coordinates':['eef_x','eef_y','eef_z',*[f'joint_{i}' for i in range(7)],'gripper_0','gripper_1'],
            'limitation':'Robot state and task identity only; no claim of static object-property identification.'}
    with args.output.open('x') as f:json.dump(report,f,default=str,indent=2)
    print(json.dumps({'output':str(args.output),'task_readout':result['task_identity_readout']['all']}),flush=True)


if __name__=='__main__':main()
