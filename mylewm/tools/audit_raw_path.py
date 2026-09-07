"""Stagewise real-data/official-weight audit. Does not train or save a model."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'lewm'))
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from mylewm.train_rbg import preprocess, digest
from mylewm.rbg import BlockSIGReg, one_step_objective
from mylewm.training_state import isolated_rng


def difference(left,right,atol=1e-6,rtol=1e-5):
    left=left.detach().float(); right=right.detach().float()
    if left.shape!=right.shape:
        return {'passed':False,'left_shape':list(left.shape),'right_shape':list(right.shape)}
    delta=(left-right).abs()
    return {'passed':bool(torch.allclose(left,right,atol=atol,rtol=rtol)),
            'max_abs':float(delta.max()),'rms':float(delta.square().mean().sqrt()),
            'atol':atol,'rtol':rtol,'shape':list(left.shape)}


def optimizer_contract(opt,model):
    names={id(p):name for name,p in model.named_parameters()}
    return [{**{k:v for k,v in group.items() if k!='params'},
              'parameter_names':[names[id(p)] for p in group['params']]} for group in opt.param_groups]


def run_branch(source,x,a,branch,training,device,precision,official_sigreg_fp32=False):
    from module import SIGReg
    from train import lejepa_forward
    from stable_pretraining.optim import create_optimizer
    model=copy.deepcopy(source).to(device).train(training)
    class FP32SIGReg(SIGReg):
        def forward(self,z):
            with torch.autocast(device_type=z.device.type,enabled=False):
                return super().forward(z.float())
    official_reg=FP32SIGReg if official_sigreg_fp32 else SIGReg
    reg=(official_reg() if branch=='official' else BlockSIGReg(blocks=1)).to(device)
    opt=(create_optimizer(model.parameters(), {'type':'AdamW','lr':1e-7,'weight_decay':.001},
                          named_params=model.named_parameters()) if branch=='official'
         else torch.optim.AdamW(model.parameters(),lr=1e-7,weight_decay=.001))
    tensors={}
    hooks=[]
    for name in ('projector','action_encoder','pred_proj'):
        def capture(module,inputs,output,key=name): tensors[key]=output
        hooks.append(getattr(model,name).register_forward_hook(capture))
    with isolated_rng(418),torch.autocast(device_type=device,dtype=torch.bfloat16,
                                         enabled=precision=='bf16'):
        if branch=='official':
            wrapper=SimpleNamespace(model=model,sigreg=reg,log_dict=lambda *args,**kwargs:None)
            cfg=SimpleNamespace(history_size=3,num_preds=1,
                                loss=SimpleNamespace(sigreg=SimpleNamespace(weight=.09)))
            out=lejepa_forward(wrapper,{'pixels':x,'action':a},'train',cfg)
            loss=out['loss']; prediction=out['pred_loss']; gaussian=out['sigreg_loss']
        else:
            loss,parts=one_step_objective(model,x,a[:,:3],reg,.09,0.,'raw')
            prediction=parts['prediction']; gaussian=parts['gaussian']
    teacher=torch.autograd.grad(prediction,tensors['projector'],retain_graph=True)[0]
    snapshots={'encode':tensors['projector'].detach().cpu(),
               'action_embedding':tensors['action_encoder'][:,:3].detach().cpu(),
               'predict':tensors['pred_proj'].detach().cpu(),
               'prediction_loss':prediction.detach().cpu(),
               'raw_sigreg':gaussian.detach().cpu(),'loss':loss.detach().cpu()}
    loss.backward()
    gradients={name:p.grad.detach().cpu().clone() for name,p in model.named_parameters() if p.grad is not None}
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    opt.step()
    weights={name:p.detach().cpu().clone() for name,p in model.named_parameters()}
    bn={name:p.detach().cpu().clone() for name,p in model.named_buffers() if 'running_' in name}
    for hook in hooks: hook.remove()
    return {'tensors':snapshots,'gradients':gradients,'updated_parameters':weights,'bn_buffers':bn,
            'optimizer':optimizer_contract(opt,model),'grad_norm':float(norm),
            'future_teacher_gradient_norm':float(teacher.reshape(x.shape[0],4,-1)[:,-1].norm())}


def audit(args):
    from utils import get_img_preprocessor
    from mylewm.train_rbg import Clips
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(args.deterministic)
    m=json.loads(args.manifest.read_text())
    ds=Clips(m)
    # Episode start zero is legal in both 16-frame and 20-frame adapters.
    pixels=[]; actions=[]; cases=[]
    with h5py.File(m['dataset']) as f:
        for ep in m['train_episodes'][:args.batch_size]:
            off=int(f['ep_offset'][ep])
            pixels.append(torch.from_numpy(f['pixels'][off:off+16:5]).permute(0,3,1,2))
            actions.append(torch.from_numpy(f['action'][off:off+20]).reshape(4,5,2))
            cases.append({'episode':ep,'start':0})
        all_actions=torch.from_numpy(f['action'][:])
    pixels=torch.stack(pixels); actions=torch.stack(actions)
    own_x,own_a=preprocess((pixels,actions[:,:3]),m,args.device)
    image_transform=get_img_preprocessor('pixels','pixels',224)
    official_x=torch.stack([image_transform({'pixels':x})['pixels'] for x in pixels]).to(args.device)
    finite=all_actions[~torch.isnan(all_actions).any(1)]
    official_mean=finite.mean(0); official_std=finite.std(0)
    official_a=torch.nan_to_num((actions.float()-official_mean)/official_std).flatten(-2).to(args.device)
    canonical_a=((actions.float()-torch.tensor(m['action_mean']))/torch.tensor(m['action_std'])).flatten(-2).to(args.device)
    model=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    report={'schema':'raw_path_audit_v1','checkpoint_sha256':digest(args.checkpoint),
            'manifest_sha256':digest(args.manifest),'cases':cases,'device':args.device,
            'precision':args.precision,'data_loader_status':'explicit installed-HDF5 adaptation; native Lance unresolved',
            'official_sigreg_fp32_control':args.official_sigreg_fp32,
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in
                (Path(__file__),ROOT/'lewm/train.py',ROOT/'lewm/jepa.py',ROOT/'lewm/module.py',
                 ROOT/'mylewm/rbg.py',ROOT/'mylewm/train_rbg.py')},
            'preprocessing':{'pixels':difference(official_x,own_x),
                'actions_own_statistics':difference(official_a[:,:3],own_a),
                'action_roundtrip':difference(own_a.reshape_as(actions[:,:3]).cpu()*torch.tensor(m['action_std'])+
                                               torch.tensor(m['action_mean']),actions[:,:3]),
                'raw_pixels_dtype':str(pixels.dtype),'raw_pixels_range':[int(pixels.min()),int(pixels.max())],
                'frame_offsets':[0,5,10,15],'input_action_offsets':list(range(15)),
                'official_action_mean':official_mean.tolist(),'official_action_std':official_std.tolist(),
                'difference_classification':'all-data vs train-only action statistics are intentional recipe differences'},
            'stages':{},'notes':['Stage comparisons use identical train-only-normalized inputs to isolate the algorithm.',
                'Official branch executes lewm/train.py:lejepa_forward; candidate executes one_step_objective.',
                'Both use actual update LR=1e-7. No claim of matching historical checkpoint training.',
                'SIGReg float32 under bf16 is an intentional candidate difference; bf16 equality is not asserted.',
                'FP32 tolerances: outputs/gradients atol=1e-6 rtol=1e-5; parameters atol=1e-7 rtol=1e-6.']}
    for training in (False,True):
        print(json.dumps({'event':'raw_audit','training':training}),flush=True)
        left=run_branch(model,own_x,canonical_a,'official',training,args.device,args.precision,args.official_sigreg_fp32)
        right=run_branch(model,own_x,canonical_a,'candidate',training,args.device,args.precision)
        comparisons={k:difference(left['tensors'][k],right['tensors'][k]) for k in left['tensors']}
        for category in ('gradients','updated_parameters','bn_buffers'):
            options={'atol':1e-7,'rtol':1e-6} if category=='updated_parameters' else {}
            comparisons[category]={k:difference(v,right[category][k],**options) for k,v in left[category].items()}
        report['stages']['train' if training else 'eval']={
            'comparisons':comparisons,'optimizer_groups_equal':left['optimizer']==right['optimizer'],
            'optimizer':left['optimizer'],'gradient_keys_equal':left['gradients'].keys()==right['gradients'].keys(),
            'teacher_gradient_norms':[left['future_teacher_gradient_norm'],right['future_teacher_gradient_norm']],
            'first_forward_mismatch':next((k for k in left['tensors'] if not comparisons[k]['passed']),None),
            'gradient_mismatches':sum(not v['passed'] for v in comparisons['gradients'].values()),
            'updated_parameter_mismatches':sum(not v['passed'] for v in comparisons['updated_parameters'].values())}
    report['causal_audit']={}
    for training in (False,True):
        outputs=[]
        for change_future in (False,True):
            branch=copy.deepcopy(model).to(args.device).train(training)
            x=own_x.clone()
            if change_future:x[:,-1]=x[:,-1].flip(-1)
            with torch.no_grad(),isolated_rng(941),torch.autocast(device_type=args.device,
                                      dtype=torch.bfloat16,enabled=args.precision=='bf16'):
                z=branch.encode({'pixels':x})['emb']
                pred=branch.predict(z[:,:3],branch.action_encoder(canonical_a[:,:3]))
                outputs.append((z[:,:3],pred))
        report['causal_audit']['train' if training else 'eval']={
            'earlier_encoded_frames':difference(outputs[0][0],outputs[1][0]),
            'earlier_predictions':difference(outputs[0][1],outputs[1][1]),
            'interpretation':'A train-mode difference can arise from projector BN across flattened batch/time; no future image was passed to predictor.'}
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,default=ROOT/'.cache/stable-wm/pusht/lewm_object.ckpt')
    p.add_argument('--manifest',type=Path,default=ROOT/'.cache/stable-wm/pusht/rbg_v0/manifest.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--precision',choices=['fp32','bf16'],default='fp32')
    p.add_argument('--official-sigreg-fp32',action='store_true',
                   help='Diagnostic control: run official SIGReg statistics in float32 too')
    p.add_argument('--deterministic',action='store_true')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.batch_size<2:p.error('batch size >= 2 required')
    report=audit(args)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps({'output':str(args.output),'preprocessing':report['preprocessing'],
        'stages':{k:{n:v[n] for n in ('first_forward_mismatch','gradient_mismatches',
                 'updated_parameter_mismatches','optimizer_groups_equal','teacher_gradient_norms')}
                  for k,v in report['stages'].items()},'causal_audit':report['causal_audit']}),flush=True)


if __name__=='__main__':main()
