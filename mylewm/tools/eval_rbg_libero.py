"""Native LIBERO-10 fixed-init CEM evaluation, not the TC paper's BC track."""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'lewm',ROOT/'external/libero',ROOT/'.cache/libero-runtime'):
    sys.path.insert(0,str(p))
os.environ.setdefault('LIBERO_CONFIG_PATH',str(ROOT/'.cache/libero-config'))
os.environ.setdefault('MUJOCO_GL','egl')
import h5py
import mujoco
import numpy as np
import torch
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from mylewm.libero_planner import CEM,image_tensor


def pair(obs):
    return np.stack([obs['agentview_image'],obs['robot0_eye_in_hand_image']])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'.cache/stable-wm/libero10/rbg_v0/manifest.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--task-ids',type=int,nargs='+',default=list(range(10)))
    p.add_argument('--episodes',type=int,default=50)
    p.add_argument('--offset',type=int,default=0)
    p.add_argument('--budget',type=int,default=520)
    p.add_argument('--horizon',type=int,default=8)
    p.add_argument('--samples',type=int,default=128)
    p.add_argument('--iterations',type=int,default=5)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda')
    p.add_argument('--render-audit-dir',type=Path,default=ROOT/'.cache/stable-wm/libero10/rbg_v0/render_audit')
    args=p.parse_args()
    if os.environ['MUJOCO_GL']!='osmesa':
        raise RuntimeError('EGL imagery failed local audit. Use bash mylewm/run_libero.sh for verified OSMesa rendering.')
    if args.output.exists(): raise FileExistsError(args.output)
    if args.offset<0 or args.episodes<1 or args.budget<1 or len(set(args.task_ids))!=len(args.task_ids):
        raise ValueError('Invalid fixed cases')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed); np.random.seed(args.seed)
    m=json.loads(args.manifest.read_text())
    suite=benchmark.get_benchmark_dict()['libero_10']()
    audits={}
    for task_id in args.task_ids:
        task_name=suite.get_task(task_id).name
        task_index=m['task_names'].index(task_name)
        path=args.render_audit_dir/f'task_{task_index}'/'report.json'
        audit=json.loads(path.read_text())
        if not (audit['passed'] and audit['task']==task_name and
                audit['mujoco']==mujoco.__version__ and audit['render_backend']==os.environ['MUJOCO_GL']):
            raise RuntimeError(f'Render audit missing or incompatible for {task_name}')
        audits[task_name]=hashlib.sha256(path.read_bytes()).hexdigest()
    model=torch.load(args.checkpoint,map_location=args.device,weights_only=False).eval()
    args.output.mkdir(parents=True)
    metadata={**vars(args),'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              'mujoco':mujoco.__version__,'render_backend':os.environ['MUJOCO_GL'],
              'render_audit_sha256':audits,'controller':'CEM; terminal latent MSE; receding one 4-action chunk',
              'initial_history':'repeat initial measured observation three times; raw previous actions zero',
              'goal':'last image of first held-out successful demonstration for each task; fixed across methods',
              'metric':'native env.check_success(); not demonstration state distance or TC BC score'}
    (args.output/'config.json').write_text(json.dumps(metadata,default=str,indent=2))
    results=[]
    with (args.output/'episodes.jsonl').open('x') as log:
        for task_id in args.task_ids:
            task=suite.get_task(task_id)
            index=m['task_names'].index(task.name)
            cases=[x for x in m['test'] if x['task']==index]
            goal_images=None
            with h5py.File(m['files'][index]['path']) as f:
                for case in cases:
                    demo=f[f'data/{case["demo"]}']
                    if float(demo['rewards'][-1])>0:
                        goal_images=np.stack([demo[f'obs/{cam}'][-1] for cam in m['camera_order']])
                        break
            if goal_images is None: raise ValueError(f'No successful fixed goal for {task.name}')
            with torch.no_grad():
                goal=model.encode({'pixels':image_tensor(goal_images,args.device)})['emb']
            states=torch.load(Path(get_libero_path('init_states'))/task.problem_folder/task.init_states_file,
                              weights_only=False)
            if args.offset+args.episodes>len(states): raise ValueError('Insufficient native initial states')
            random.seed(args.seed+task_id*10000); np.random.seed(args.seed+task_id*10000)
            env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/
                 task.problem_folder/task.bddl_file),camera_heights=128,camera_widths=128)
            try:
                for init_id in range(args.offset,args.offset+args.episodes):
                    random.seed(args.seed+task_id*10000+init_id)
                    np.random.seed(args.seed+task_id*10000+init_id)
                    env.seed(args.seed+task_id*10000+init_id)
                    env.reset(); obs=env.set_init_state(states[init_id])
                    for _ in range(10): obs,_,_,_=env.step(np.zeros(7))
                    if env.check_success(): raise ValueError('Initial state already successful')
                    initial_images=pair(obs)
                    initial_state=np.asarray(env.get_sim_state()).copy()
                    with torch.no_grad():
                        z=model.encode({'pixels':image_tensor(pair(obs),args.device)})['emb']
                    history=z.repeat(1,3,1)
                    previous=torch.zeros(1,2,4,7,device=args.device)
                    planner=CEM(model,args.horizon,args.samples,min(16,args.samples),args.iterations,
                                args.seed+task_id*10000+init_id)
                    success=False; count=0; start=time.monotonic(); actions_taken=[]
                    while count<args.budget and not success:
                        chunk=planner.plan(history,previous,goal)
                        for action in chunk.cpu().numpy():
                            obs,reward,done,info=env.step(action)
                            actions_taken.append(action.tolist()); count+=1
                            success=bool(env.check_success())
                            if success or count>=args.budget: break
                        if not success and count<args.budget:
                            with torch.no_grad():
                                z=model.encode({'pixels':image_tensor(pair(obs),args.device)})['emb']
                            history=torch.cat((history[:,1:],z),dim=1)
                            previous=torch.cat((previous[:,1:],chunk[None,None]),dim=1)
                    row={'task_id':task_id,'task':task.name,'init_id':init_id,'success':success,
                         'steps':count,'elapsed':time.monotonic()-start,'goal_demo':case['demo'],
                         'initial_image_sha256':hashlib.sha256(initial_images.tobytes()).hexdigest(),
                         'initial_state_sha256':hashlib.sha256(initial_state.tobytes()).hexdigest(),
                         'goal_image_sha256':hashlib.sha256(goal_images.tobytes()).hexdigest()}
                    log.write(json.dumps(row)+'\n'); log.flush(); results.append(row)
                    np.savez_compressed(args.output/f'task{task_id}_init{init_id}.npz',
                        actions=np.asarray(actions_taken),initial_state=initial_state,
                        initial=initial_images,goal=goal_images,final=pair(obs))
                    print(json.dumps(row),flush=True)
            finally: env.close()
    task_rates={str(t):float(np.mean([r['success'] for r in results if r['task_id']==t])) for t in args.task_ids}
    (args.output/'summary.json').write_text(json.dumps({'task_rates':task_rates,
        'macro_success':float(np.mean(list(task_rates.values()))),'episodes':len(results)},indent=2))


if __name__=='__main__': main()
