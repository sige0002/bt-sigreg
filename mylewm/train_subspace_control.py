"""Budget-matched Sub-JEPA control using the inspected official implementation.

The default uses 16 subspaces x 64 projections = 1024 total, NOT the paper's
16 x 1024 setting. Both counts are explicit in every run's config.
"""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'external/sub-jepa'))
import torch
from subjepa import MultiSubspaceSIGReg
from mylewm import train_rbg as common
from mylewm.rbg import one_step_objective


class SubspaceAdapter(torch.nn.Module):
    def __init__(self,dim=192,subspaces=16,per_subspace_projections=64):
        super().__init__()
        self.reg=MultiSubspaceSIGReg(embed_dim=dim,num_subspaces=subspaces,
            num_proj=per_subspace_projections,init_mode='orthogonal_frozen')
    def forward(self,z):
        with torch.autocast(device_type=z.device.type,enabled=False):
            loss=self.reg(z.float().transpose(0,1))
        return loss,z.sum()*0


def objective(model,pixels,actions,regularizer,gaussian_weight,cross_weight,mode):
    return one_step_objective(model,pixels,actions,regularizer,gaussian_weight,0.,'raw')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--benchmark',choices=['pusht','libero10'],default='pusht')
    p.add_argument('--manifest',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--blocks',type=int,default=16)
    p.add_argument('--per-subspace-projections',type=int,default=64)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--save-every',type=int,default=1000)
    p.add_argument('--seed',type=int,default=3072)
    p.add_argument('--lr',type=float,default=5e-5)
    p.add_argument('--gaussian-weight',type=float,default=.09)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    args.mode='sub'; args.cross_weight=0.
    if args.manifest is None:
        args.manifest=ROOT/f'.cache/stable-wm/{args.benchmark}/rbg_v0/manifest.json'
    sources=[Path(__file__),ROOT/'external/sub-jepa/subjepa.py']
    if args.benchmark=='libero10':
        from mylewm import train_rbg_libero as lib
        common.Clips=lib.LiberoClips; common.preprocess=lib.preprocess; common.build_model=lib.build_model
        sources.extend([ROOT/'mylewm/train_rbg_libero.py',ROOT/'mylewm/libero_model.py'])
    args.adapter_sources={str(f.relative_to(ROOT)):common.digest(f) for f in sources}
    common.BlockSIGReg=lambda **kwargs:SubspaceAdapter(subspaces=args.blocks,
        per_subspace_projections=args.per_subspace_projections)
    common.one_step_objective=objective
    common.train(args)


if __name__=='__main__': main()
