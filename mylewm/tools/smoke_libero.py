"""Check native LIBERO-10 simulator, fixed states, images and action contract."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('LIBERO_CONFIG_PATH',str(ROOT/'.cache/libero-config'))
os.environ.setdefault('MUJOCO_GL','egl')
sys.path.insert(0,str(ROOT/'external/libero'))
import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--task-id',type=int,default=0)
    args=parser.parse_args()
    suite=benchmark.get_benchmark_dict()['libero_10']()
    task=suite.get_task(args.task_id)
    env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/
        task.problem_folder/task.bddl_file),camera_heights=128,camera_widths=128)
    try:
        env.seed(0)
        env.reset()
        # Upstream older PyTorch assumed weights_only=False for init-state lists.
        path=Path(get_libero_path('init_states'))/task.problem_folder/task.init_states_file
        states=torch.load(path,weights_only=False)
        obs=env.set_init_state(states[0])
        for _ in range(10): obs,reward,done,info=env.step(np.zeros(7))
        print(json.dumps({'task':task.name,'task_count':suite.n_tasks,
            'observations':{k:list(v.shape) for k,v in obs.items() if hasattr(v,'shape')},
            'reward':float(reward),'done':bool(done),'success':bool(env.check_success())}),flush=True)
    finally:
        env.close()


if __name__=='__main__': main()
