"""Replay recorded physical actions to verify evaluator state/masks/successes."""
import argparse
import json
import os
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np
import torch
import stable_worldmodel as swm
from mylewm.evaluation_contract import array_hash


class ReplayPolicy(swm.policy.BasePolicy):
    def __init__(self,result):
        super().__init__()
        self.result=result;self.index=0
    def get_action(self,info,**kwargs):
        if self.index==0:
            actual=[{key:array_hash(info[key][i]) for key in expected}
                    for i,expected in enumerate(self.result['initial_runtime_hashes'])]
            if actual!=self.result['initial_runtime_hashes']:
                raise ValueError('Replay initial/goal state or image mismatch')
        actions=np.asarray(self.result['physical_actions'][self.index]['actions'])
        self.index+=1
        return actions


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--result',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result=json.loads(args.result.read_text())
    cfg=result['config']
    torch.set_num_threads(4);torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    world=swm.World(**cfg['world'],image_shape=(cfg['eval']['img_size'],cfg['eval']['img_size']))
    dataset=swm.data.HDF5Dataset(cfg['eval']['dataset_name'],
        cache_dir=ROOT/'.cache/stable-wm',keys_to_cache=cfg['dataset']['keys_to_cache'])
    policy=ReplayPolicy(result)
    world.set_policy(policy)
    original_step=world.envs.step
    masks_verified=[]
    def step(actions,mask=None):
        actual=np.ones(world.num_envs,dtype=bool) if mask is None else np.asarray(mask)
        expected=result['physical_actions'][policy.index-1]['mask']
        if not np.array_equal(actual,expected):raise ValueError('Replay environment mask mismatch')
        masks_verified.append(True)
        return original_step(actions,mask=mask)
    world.envs.step=step
    world.reset(seed=cfg['seed'])
    metrics=world.evaluate(dataset=dataset,start_steps=result['starts'],
        goal_offset=cfg['eval']['goal_offset_steps'],eval_budget=cfg['eval']['eval_budget'],
        episodes_idx=result['episodes'],callables=cfg['eval']['callables'],video=args.output)
    successes=np.asarray(metrics['episode_successes'],dtype=bool).tolist()
    if successes!=result['successes'] or len(masks_verified)!=len(result['physical_actions']):
        raise ValueError('Replay outcome/length mismatch')
    report={'source_result':str(args.result),'initial_states_and_goals_match':True,
            'all_execution_masks_match':True,'all_successes_match':True,
            'cases':len(successes),'successes':sum(successes),'steps':len(masks_verified),
            'video_provenance':'re-rendered physical action replay, not original CEM execution recording'}
    with (args.output/'replay_verification.json').open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
