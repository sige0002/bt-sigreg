"""Shared Raw/TC/BT training loop and PushT split/reference adapter.

New PushT runs use train.py. LIBERO uses this loop via train_libero.py.
Historical runs require their original source revision for strict resume.
"""
import argparse
import importlib.metadata
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
from mylewm.objectives import GaussianSIGReg, one_step_objective
from mylewm.bt_sigreg import BTSIGReg, add_bt_arguments
from mylewm.data_contract import file_sha256, data_fingerprints, training_budget
from mylewm.training_diagnostics import encoder_gradient_norms,action_diagnostics,transport_statistics
from mylewm.training_state import (UpdateSchedule, add_schedule_arguments, capture_rng,
    restore_rng, isolated_rng, check_resume_config, reconcile_metrics, IndependentRegularizer,tensor_state_hash)


def digest(path):
    return file_sha256(path)


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
    if args.mode not in ('raw', 'tc', 'bt'):
        raise ValueError(f'Unsupported mode: {args.mode}')
    schedule=UpdateSchedule(args.steps,args.warmup_steps,args.lr,args.min_lr)
    if getattr(args,'deterministic',False):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    m=json.loads(args.manifest.read_text())
    dataset=Path(m['dataset'])
    if dataset.stat().st_size!=m['dataset_size'] or dataset.stat().st_mtime_ns!=m['dataset_mtime_ns']:
        raise ValueError('Dataset changed since split preparation')
    if min(m['action_std'])<=0:
        raise ValueError('invalid action scale')
    print(json.dumps({'event':'hashing_data_for_training_contract'}),flush=True)
    data_identity=data_fingerprints(m)
    model=build_model().to(device)
    initial_model_sha256=tensor_state_hash(model.state_dict())
    model.register_buffer('training_action_mean',torch.tensor(m['action_mean'],device=device,dtype=torch.float64))
    model.register_buffer('training_action_std',torch.tensor(m['action_std'],device=device,dtype=torch.float64))
    with isolated_rng(args.seed+10000):
        inner = (BTSIGReg(depth=args.bt_depth, kappa=args.bt_kappa, hidden=args.bt_hidden)
                 if args.mode == 'bt' else GaussianSIGReg())
        reg=IndependentRegularizer(inner,args.seed+10000).to(device)
    transport_parameters = list(reg.parameters())
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.001)
    if transport_parameters:
        opt.add_param_group({'params': transport_parameters})
    config={**vars(args),'schema':'controlled_training_v2',
            'schedule':schedule.config,'device':device,
            'precision':'bf16_autocast' if device=='cuda' else 'float32',
            'torch_version':str(torch.__version__),
            'dependency_versions':{name:importlib.metadata.version(name) for name in
                ('stable-pretraining','stable-worldmodel','transformers','h5py','numpy','hydra-core')},
            'numerical_backend':{'cudnn_benchmark':torch.backends.cudnn.benchmark,
                'cudnn_tf32':torch.backends.cudnn.allow_tf32,'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
                'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG')},
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'manifest_sha256':digest(args.manifest),
            'initial_model_sha256':initial_model_sha256,
            'data_identity':data_identity,'recipe':'controlled_comparison',
            'parameters':sum(p.numel() for p in model.parameters()),
            'training_only_parameters':sum(p.numel() for p in transport_parameters),
            'transport':inner.transport.config() if args.mode=='bt' else None,
            'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in
                [Path(__file__),ROOT/'mylewm/bt_sigreg.py',ROOT/'mylewm/training_state.py',ROOT/'mylewm/training_diagnostics.py',ROOT/'mylewm/data_contract.py',ROOT/'mylewm/objectives.py',
                 ROOT/'lewm/jepa.py',ROOT/'lewm/module.py',ROOT/'lewm/config/train/model/lewm.yaml']}}
    config=json.loads(json.dumps(config,default=str))
    start_step=0
    if args.resume:
        state=torch.load(args.output/'resume.pt',map_location=device,weights_only=False)
        previous=json.loads((args.output/'config.json').read_text())
        check_resume_config(previous,config)
        if digest(args.output/'initialization.pt')!=state['initialization_sha256']:
            raise ValueError('resume mismatch: initialization')
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        reg.load_state_dict(state['regularizer'])
        start_step=state['step']
        schedule.load_state_dict(state['scheduler'])
        if schedule.completed_steps!=start_step or state['next_batch_index']!=start_step:
            raise ValueError('resume mismatch: update/data position')
        reconcile_metrics(args.output/'metrics.jsonl',state)
        restore_rng(state['random_state'])
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'config.json').write_text(json.dumps(config,indent=2))
        atomic_save({'model':model.state_dict(),'regularizer':reg.state_dict(),
                     'random_state':capture_rng()},args.output/'initialization.pt')
    initialization_sha256=digest(args.output/'initialization.pt')
    ds=Clips(m)
    budget=training_budget(len(ds),args.steps,args.batch_size)
    budget_path=args.output/'budget.json'
    if args.resume:
        if json.loads(budget_path.read_text())!=budget:
            raise ValueError('resume mismatch: training budget')
    else:
        budget_path.write_text(json.dumps(budget,indent=2))
    loader=torch.utils.data.DataLoader(ds,batch_sampler=StepBatches(len(ds),args.batch_size,args.steps,args.seed,start_step),
        num_workers=args.workers,pin_memory=device=='cuda',persistent_workers=args.workers>0,
        generator=torch.Generator().manual_seed(args.seed+100))
    val=torch.utils.data.DataLoader(Clips(m,True),batch_size=args.batch_size,num_workers=0,
        generator=torch.Generator().manual_seed(args.seed+101))
    begin=time.monotonic()
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    print(json.dumps({'event':'start','parameters':config['parameters'],'mode':args.mode,
                      'train_clips':len(ds),'steps':args.steps,'device':device}),flush=True)
    with (args.output/'metrics.jsonl').open('a') as log:
        for step,batch in enumerate(loader,start_step+1):
            learning_rates=schedule.apply(opt,step)
            model.train(); reg.train(); opt.zero_grad(set_to_none=True)
            x,a=preprocess(batch,m,device)
            diagnostic_every=getattr(args,'diagnostics_every',0)
            transport_diagnostics = {}
            hook = None
            if args.mode == 'bt' and diagnostic_every and (step == 1 or step % diagnostic_every == 0):
                hook = inner.transport.register_forward_hook(
                    lambda module, inputs, output: transport_diagnostics.update(transport_statistics(inputs[0], output)))
            try:
                with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):
                    loss,parts=one_step_objective(model,x,a,reg,args.gaussian_weight,args.mode)
            finally:
                if hook is not None: hook.remove()
            if not torch.isfinite(loss): raise FloatingPointError(f'loss at {step}')
            gradient_diagnostics=None
            if diagnostic_every and (step==1 or step%diagnostic_every==0):
                gradient_diagnostics=encoder_gradient_norms(model,parts,args.gaussian_weight)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            transport_norm = (torch.nn.utils.clip_grad_norm_(transport_parameters,1.,error_if_nonfinite=True)
                              if transport_parameters else None)
            opt.step()
            schedule.mark_completed(step)
            row={'step':step,'loss':float(loss.detach()),'grad_norm':float(norm),
                 'transport_grad_norm':float(transport_norm) if transport_norm is not None else None,
                 'learning_rates':learning_rates,
                 'presented_clips':step*args.batch_size,
                 'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
                 'elapsed_session':time.monotonic()-begin,**{k:float(v.detach()) for k,v in parts.items()}}
            if gradient_diagnostics is not None:
                row['weighted_encoder_projector_gradient_norms']=gradient_diagnostics
            if transport_diagnostics:
                row['transport_diagnostics_pre_update']=transport_diagnostics
            if step%args.save_every==0 or step==args.steps:
                model.eval(); reg.eval(); values=[]
                regularizer_position=reg.draw_index.clone()
                with isolated_rng(999),torch.no_grad():
                    for vb in val:
                        vx,va=preprocess(vb,m,device)
                        with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):
                            _,vp=one_step_objective(model,vx,va,reg,args.gaussian_weight,args.mode)
                        values.append({k:float(v) for k,v in vp.items()})
                        if diagnostic_every and len(values)==1:
                            row['validation_first_batch_action_diagnostics']=action_diagnostics(model,vx,va)
                reg.draw_index.copy_(regularizer_position)
                row['validation']={k:float(np.mean([v[k] for v in values])) for k in values[0]}
                atomic_save(model,args.output/f'step_{step}_object.ckpt')
                atomic_save({'step':step,'model':model.state_dict(),'regularizer':reg.state_dict(),
                    'optimizer':opt.state_dict(),'scheduler':schedule.state_dict(),
                    'random_state':capture_rng(),'next_batch_index':step,
                    'metrics_row':row,
                    'initialization_sha256':initialization_sha256},args.output/'resume.pt')
            log.write(json.dumps(row)+'\n'); log.flush()
            if step==1 or step%10==0 or step==args.steps: print(json.dumps(row),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['prepare','train'])
    p.add_argument('--dataset',type=Path,default=ROOT/'.cache/stable-wm/datasets/pusht_expert_train.h5')
    p.add_argument('--manifest',type=Path,default=ROOT/'output/manifests/pusht/manifest.json')
    p.add_argument('--output',type=Path)
    p.add_argument('--mode',choices=['raw','tc','bt'],default='bt')
    add_bt_arguments(p)
    p.add_argument('--steps','--total-steps',dest='steps',type=int,default=50000)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--save-every',type=int,default=1000)
    p.add_argument('--seed',type=int,default=3072)
    add_schedule_arguments(p)
    p.add_argument('--gaussian-weight',type=float,default=.09)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.command=='prepare': prepare(args.dataset,args.manifest)
    else:
        if args.output is None: p.error('--output is required for train')
        train(args)


if __name__=='__main__': main()
