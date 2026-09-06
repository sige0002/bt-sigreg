"""Train-only, two-stage BN-statistic audit. Never modifies learned parameters.

Produces a separately named checkpoint, never silently replaces the raw one.
No validation/test observations enter running-stat estimation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'lewm'))
import torch
from mylewm import train_rbg as common


def parameter_hash(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'.cache/stable-wm/pusht/rbg_v0/manifest.json')
    p.add_argument('--benchmark',choices=['pusht','libero10'],default='pusht')
    p.add_argument('--batches',type=int,default=16)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    torch.set_num_threads(4)
    if args.benchmark=='libero10':
        from mylewm import train_rbg_libero as lib
        dataset_cls,preprocess=lib.LiberoClips,lib.preprocess
    else: dataset_cls,preprocess=common.Clips,common.preprocess
    m=json.loads(args.manifest.read_text())
    ds=dataset_cls(m,False)
    model=torch.load(args.checkpoint,map_location=args.device,weights_only=False).eval()
    before=parameter_hash(model)
    for stage in ('projector','pred_proj'):
        model.eval()
        layers=[layer for layer in getattr(model,stage).modules()
                if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm)]
        momenta=[layer.momentum for layer in layers]
        for layer in layers:
            layer.reset_running_stats(); layer.momentum=None; layer.train()
        loader=torch.utils.data.DataLoader(ds,batch_sampler=common.StepBatches(
            len(ds),args.batch_size,args.batches,99117),num_workers=0)
        with torch.no_grad():
            for batch in loader:
                x,a=preprocess(batch,m,args.device)
                z=model.encode({'pixels':x})['emb']
                if stage=='pred_proj': model.predict(z[:,:3],model.action_encoder(a))
        for layer,momentum in zip(layers,momenta): layer.momentum=momentum
    model.eval()
    after=parameter_hash(model)
    if before!=after: raise AssertionError('Calibration modified learned parameters')
    common.atomic_save(model,args.output)
    report={**vars(args),'parameter_sha256_before':before,'parameter_sha256_after':after,
            'source_checkpoint_sha256':common.digest(args.checkpoint),
            'manifest_sha256':common.digest(args.manifest),
            'calibration_split':'training only; same stateless batches in both stages; seed 99117'}
    args.output.with_suffix('.json').write_text(json.dumps(report,default=str,indent=2))
    print(json.dumps(report,default=str),flush=True)


if __name__=='__main__': main()
