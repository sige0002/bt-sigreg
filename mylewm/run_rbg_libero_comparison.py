"""First-seed LIBERO comparison after the PushT GPU queue has completed."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mylewm.compare_libero import compare
from mylewm.run_rbg_pusht_comparison import complete


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--wait-pid',type=int)
    p.add_argument('--seed',type=int,default=3072)
    p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--episodes',type=int,default=50)
    args=p.parse_args()
    if not 1<=args.episodes<=50 or args.steps<1: p.error('Invalid experiment budget')
    deadline=time.monotonic()+48*3600
    while args.wait_pid:
        try: os.kill(args.wait_pid,0)
        except ProcessLookupError: break
        if time.monotonic()>deadline: raise TimeoutError('PushT queue did not finish in 48h')
        time.sleep(10)
    pusht=ROOT/f'.cache/stable-wm/pusht/rbg_v0/comparison_s{args.seed}.json'
    prior=json.loads(pusht.read_text())
    if not all(k in prior['results'] for k in ('rbg','raw','tc','sub','official')):
        raise RuntimeError('PushT prerequisite comparison is incomplete')
    run=ROOT/'.cache/stable-wm/libero10/rbg_v0'
    env={**os.environ,'PYTHONPATH':os.pathsep.join([str(ROOT),str(ROOT/'lewm')]),'PYTHONUNBUFFERED':'1'}
    def execute(command,name):
        print(json.dumps({'event':name,'command':command}),flush=True)
        with (run/f'{name}_s{args.seed}.log').open('x') as log:
            subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    results={}
    for mode in ('rbg','raw','tc','sub'):
        directory=run/f'{mode}_s{args.seed}'
        if not complete(directory,args.steps):
            if directory.exists(): raise RuntimeError(f'Incomplete run requires audit: {directory}')
            if mode=='sub':
                command=[sys.executable,str(ROOT/'mylewm/train_subspace_control.py'),'--benchmark','libero10']
            else:
                command=[sys.executable,str(ROOT/'mylewm/train_rbg_libero.py'),'train','--mode',mode]
            execute(command+['--steps',str(args.steps),'--batch-size','128','--seed',str(args.seed),
                             '--output',str(directory)],mode+'_train')
        raw=directory/f'step_{args.steps}_object.ckpt'
        calibrated=directory/f'step_{args.steps}_bn_calibrated_object.ckpt'
        if not calibrated.exists():
            execute([sys.executable,str(ROOT/'mylewm/calibrate_rbg_bn.py'),
                     '--checkpoint',str(raw),'--output',str(calibrated),'--benchmark','libero10',
                     '--manifest',str(run/'manifest.json'),'--device','cuda','--batches','32',
                     '--batch-size','128'],mode+'_calibration')
        for label,checkpoint in ((mode,raw),(mode+'_bn',calibrated)):
            output=run/f'{label}_s{args.seed}_eval'
            if not (output/'summary.json').exists():
                if output.exists(): raise RuntimeError(f'Incomplete evaluation requires audit: {output}')
                execute(['bash',str(ROOT/'mylewm/run_libero.sh'),str(ROOT/'mylewm/eval_rbg_libero.py'),
                         '--checkpoint',str(checkpoint),'--output',str(output),
                         '--episodes',str(args.episodes)],label+'_eval')
            results[label]=str(output)
    comparisons={label:compare(results[label],results['rbg']) for label in ('raw','tc','sub')}
    comparisons.update({label+'_vs_rbg_bn':compare(results[label],results['rbg_bn'])
                        for label in ('raw_bn','tc_bn','sub_bn')})
    comparisons['rbg_calibration_effect']=compare(results['rbg'],results['rbg_bn'])
    destination=run/f'comparison_s{args.seed}.json'
    with destination.open('x') as f:
        json.dump({'stage':'first-seed 10k-step comparison; not TC paper BC reproduction',
                   'results':results,'comparisons':comparisons},f,indent=2)


if __name__=='__main__': main()
