"""Reviewable first-budget plan. This command never launches training."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]


def plan(seed=3072,steps=50000,methods=('raw','rbg')):
    if steps<=500 or not methods or not set(methods)<={'raw','rbg'} or len(set(methods))!=len(methods):
        raise ValueError('Require >500 updates and distinct Raw/RBG methods')
    run=ROOT/f'.cache/stable-wm/pusht/controlled_v2_s{seed}_steps{steps}'
    initial=run/'shared_initialization.pt'
    train=[]
    for mode in methods:
        train.append([sys.executable,str(ROOT/'mylewm/train_rbg.py'),'train','--mode',mode,
            '--steps',str(steps),'--batch-size','128','--warmup-steps','500','--max-lr','5e-5',
            '--min-lr','0','--seed',str(seed),'--save-every','5000',
            '--deterministic',
            '--initialization',str(initial),'--output',str(run/mode)])
    return {'schema':'controlled_comparison_plan_v2','dry_run':True,'benchmark':'pusht',
        'seed':seed,'total_steps':steps,'methods':list(methods),'output':str(run),
        'official_first':[sys.executable,str(ROOT/'mylewm/tools/evaluate_official_pusht.py'),'--execute'],
        'shared_initialization':[sys.executable,str(ROOT/'mylewm/tools/create_shared_initialization.py'),
                                  '--seed',str(seed),'--output',str(initial)],
        'training':train,
        'prerequisites':['Issue #1-4 audit conclusions reviewed','Issue #5 same-case official reference complete',
                         'User explicitly resumes long training after current stop instruction'],
        'evaluation_rules':{'main':'saved BN eval and calibrated BN eval both reported; no per-model test selection',
             'calibration':{'split':'train','seed':99117,'stages':2,'batches_per_stage':32,'batch_size':128},
             'regression_cases':200,'held_out_final_claim':'new unused cases must be fixed before final claims',
             'planner':'same physical search, history, goal, horizon, CEM samples and iterations'},
        'not_implemented':['automatic training/evaluation pipeline for the new plan',
                            'long-run representation/control diagnostic aggregation'],
        'not_scheduled':['TC','Sub-JEPA','LIBERO','additional training seeds']}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=3072)
    p.add_argument('--steps',type=int,default=50000)
    p.add_argument('--methods',nargs='+',choices=['raw','rbg'],default=['raw','rbg'])
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    report=plan(args.seed,args.steps,args.methods)
    if args.output:
        with args.output.open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
