"""Mechanism controls, separate from the running RBG v0 experiment.

global_mix=0, cross_weight=0 removes the covariance penalty.
global_mix=.5 mixes block/global Gaussian losses without doubling lambda.
The latter is budget-matched, not a proof of equal gradient strength.
"""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from mylewm.rbg import BlockSIGReg
from mylewm import train_rbg as common


class MixedGaussian(torch.nn.Module):
    def __init__(self,dim=192,blocks=4,global_mix=.5,projections=1024):
        super().__init__()
        if not 0<=global_mix<=1:raise ValueError('global_mix must be in [0,1]')
        if 0<global_mix<1 and projections%2:raise ValueError('Mixed budget must be even')
        self.mix=global_mix
        per=projections//2 if 0<global_mix<1 else projections
        self.block=BlockSIGReg(dim=dim,blocks=blocks,projections=per)
        self.global_reg=BlockSIGReg(dim=dim,blocks=1,projections=per)
    def forward(self,z):
        if self.mix==0:return self.block(z)
        if self.mix==1:
            gaussian,_=self.global_reg(z)
            with torch.autocast(device_type=z.device.type,enabled=False):
                cross=self.block.cross_covariance(z.float())
            return gaussian,cross
        block,cross=self.block(z)
        global_loss,_=self.global_reg(z)
        return (1-self.mix)*block+self.mix*global_loss,cross


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--benchmark',choices=['pusht','libero10'],default='pusht')
    p.add_argument('--manifest',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--global-mix',type=float,default=.5)
    p.add_argument('--cross-weight',type=float,default=.01)
    p.add_argument('--gaussian-weight',type=float,default=.09)
    p.add_argument('--blocks',type=int,default=4)
    p.add_argument('--steps','--total-steps',dest='steps',type=int,default=50000)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--save-every',type=int,default=1000)
    p.add_argument('--seed',type=int,default=3072)
    common.add_schedule_arguments(p)
    args=p.parse_args()
    if not 0<=args.global_mix<=1 or args.cross_weight<0 or args.gaussian_weight<0:
        p.error('Invalid regularizer coefficients')
    args.mode='rbg';args.resume=False
    if args.manifest is None:args.manifest=ROOT/f'.cache/stable-wm/{args.benchmark}/rbg_v0/manifest.json'
    sources=[Path(__file__)]
    if args.benchmark=='libero10':
        from mylewm import train_rbg_libero as lib
        common.Clips=lib.LiberoClips;common.preprocess=lib.preprocess;common.build_model=lib.build_model
        sources.extend([ROOT/'mylewm/train_rbg_libero.py',ROOT/'mylewm/libero_model.py'])
    args.adapter_sources={str(f.relative_to(ROOT)):common.digest(f) for f in sources}
    common.BlockSIGReg=lambda **kwargs:MixedGaussian(blocks=args.blocks,global_mix=args.global_mix)
    common.train(args)


if __name__=='__main__':main()
