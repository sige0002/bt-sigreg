"""Train-only, two-stage BN-statistic audit. Never modifies learned parameters.

Produces a separately named checkpoint, never silently replaces the raw one.
No validation/test observations enter running-stat estimation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'lewm'))
import torch
from mylewm import train_rbg as common
from mylewm.training_state import isolated_rng


def parameter_hash(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def buffer_hashes(model):
    return {name:hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
            for name,value in model.named_buffers()}


def calibrate(model,ds,manifest,preprocess,batches,batch_size,device,seed=99117):
    if batches<1 or batch_size<2: raise ValueError('Require batches >=1 and batch_size >=2')
    before=parameter_hash(model); buffers_before=buffer_hashes(model)
    allowed={f'{name}.{suffix}' for name,layer in model.named_modules()
             if name.split('.')[0] in ('projector','pred_proj')
             and isinstance(layer,torch.nn.modules.batchnorm._BatchNorm)
             for suffix in ('running_mean','running_var','num_batches_tracked')}
    indices=list(common.StepBatches(len(ds),batch_size,batches,seed))
    index_hash=hashlib.sha256(json.dumps(indices,separators=(',',':')).encode()).hexdigest()
    with isolated_rng(seed):
        for stage in ('projector','pred_proj'):
            model.eval()
            layers=[layer for layer in getattr(model,stage).modules()
                    if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm)]
            momenta=[layer.momentum for layer in layers]
            try:
                for layer in layers:
                    layer.reset_running_stats(); layer.momentum=None; layer.train()
                loader=torch.utils.data.DataLoader(ds,batch_sampler=indices,num_workers=0,
                            generator=torch.Generator().manual_seed(seed))
                with torch.no_grad():
                    for batch in loader:
                        x,a=preprocess(batch,manifest,device)
                        z=model.encode({'pixels':x})['emb']
                        if stage=='pred_proj': model.predict(z[:,:3],model.action_encoder(a))
            finally:
                for layer,momentum in zip(layers,momenta): layer.momentum=momentum
                model.eval()
    after=parameter_hash(model); buffers_after=buffer_hashes(model)
    if before!=after: raise AssertionError('Calibration modified learned parameters')
    changed={name for name in buffers_before if buffers_before[name]!=buffers_after[name]}
    if buffers_before.keys()!=buffers_after.keys() or not changed<=allowed:
        raise AssertionError('Calibration modified non-BN buffers')
    return {'parameter_sha256_before':before,'parameter_sha256_after':after,
            'buffer_sha256_before':buffers_before,'buffer_sha256_after':buffers_after,
            'changed_buffers':sorted(changed),'allowed_buffers':sorted(allowed),
            'calibration_split':'training only','seed':seed,'batches_per_stage':batches,
            'batch_size':batch_size,'stages':['projector','pred_proj'],
            'same_batch_order_both_stages':True,'clip_indices_sha256':index_hash,
            'precision':'float32; autocast disabled by caller',
            'columns':['pixels','action'] if 'files' not in manifest else [*manifest['camera_order'],'actions']}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'.cache/stable-wm/pusht/rbg_v0/manifest.json')
    p.add_argument('--benchmark',choices=['pusht','libero10'],default='pusht')
    p.add_argument('--batches',type=int,default=16)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--device',default='cpu')
    p.add_argument('--seed',type=int,default=99117)
    args=p.parse_args()
    if args.output.exists() or args.output.with_suffix('.json').exists(): raise FileExistsError(args.output)
    torch.set_num_threads(4)
    if args.benchmark=='libero10':
        from mylewm import train_rbg_libero as lib
        dataset_cls,preprocess=lib.LiberoClips,lib.preprocess
    else: dataset_cls,preprocess=common.Clips,common.preprocess
    m=json.loads(args.manifest.read_text())
    ds=dataset_cls(m,False)
    model=torch.load(args.checkpoint,map_location=args.device,weights_only=False).eval()
    calibration=calibrate(model,ds,m,preprocess,args.batches,args.batch_size,args.device,args.seed)
    common.atomic_save(model,args.output)
    report={**vars(args),**calibration,
            'source_checkpoint_sha256':common.digest(args.checkpoint),
            'manifest_sha256':common.digest(args.manifest)}
    args.output.with_suffix('.json').write_text(json.dumps(report,default=str,indent=2))
    print(json.dumps(report,default=str),flush=True)


if __name__=='__main__': main()
