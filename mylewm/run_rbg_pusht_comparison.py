"""Sequential first-seed comparison; never overlap GPU training and evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mylewm.compare_paired import compare


def complete(path,steps):
    metrics=path/'metrics.jsonl'
    if not metrics.exists(): return False
    lines=metrics.read_text().splitlines()
    return bool(lines and json.loads(lines[-1])['step']==steps and
                (path/f'step_{steps}_object.ckpt').exists())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--wait-pid',type=int)
    p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--seed',type=int,default=3072)
    p.add_argument('--cases',type=int,default=200)
    args=p.parse_args()
    if args.cases%50: p.error('cases must be a multiple of 50')
    run=ROOT/'.cache/stable-wm/pusht/rbg_v0'
    env={**os.environ,'PYTHONPATH':os.pathsep.join([str(ROOT),str(ROOT/'lewm'),str(ROOT/'mylewm')]),
         'STABLEWM_HOME':str(ROOT/'.cache/stable-wm'),'PYTHONUNBUFFERED':'1'}
    if args.wait_pid:
        deadline=time.monotonic()+24*3600
        while True:
            try: os.kill(args.wait_pid,0)
            except ProcessLookupError: break
            if time.monotonic()>deadline: raise TimeoutError('Initial training did not finish in 24h')
            time.sleep(10)
        if not complete(run/f'rbg_s{args.seed}',args.steps):
            raise RuntimeError('Initial training exited without a complete checkpoint')
    results={}
    for mode in ('rbg','raw','tc','sub','official'):
        directory=run/f'{mode}_s{args.seed}'
        if mode!='official' and not complete(directory,args.steps):
            if directory.exists(): raise RuntimeError(f'Incomplete run requires audit: {directory}')
            if mode=='sub':
                command=[sys.executable,str(ROOT/'mylewm/train_subspace_control.py'),'--benchmark','pusht']
            else:
                command=[sys.executable,str(ROOT/'mylewm/train_rbg.py'),'train','--mode',mode]
            command+=['--steps',str(args.steps),'--batch-size','128','--seed',str(args.seed),
                      '--output',str(directory)]
            print(json.dumps({'event':'train_start','mode':mode}),flush=True)
            with (run/f'{mode}_s{args.seed}_train.log').open('x') as log:
                subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        policy='pusht/lewm' if mode=='official' else f'pusht/rbg_v0/{mode}_s{args.seed}/step_{args.steps}'
        policies={mode:policy}
        if mode!='official':
            calibrated=directory/f'step_{args.steps}_bn_calibrated_object.ckpt'
            if not calibrated.exists():
                command=[sys.executable,str(ROOT/'mylewm/calibrate_rbg_bn.py'),
                    '--checkpoint',str(directory/f'step_{args.steps}_object.ckpt'),
                    '--output',str(calibrated),'--manifest',str(run/'manifest.json'),
                    '--device','cuda','--batch-size','128','--batches','32']
                with (run/f'{mode}_s{args.seed}_calibration.log').open('x') as log:
                    subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            policies[mode+'_bn']=policy+'_bn_calibrated'
        for label,eval_policy in policies.items():
            paths=[]
            for offset in range(0,args.cases,50):
                filename=f'rbg_v0_s{args.seed}_{label}_confirm_{offset}.txt'
                result=ROOT/'.cache/stable-wm'/Path(eval_policy).parent/(filename+'.json')
                if not result.exists():
                    command=[sys.executable,str(ROOT/'lewm/eval.py'),f'policy={eval_policy}',
                        'eval.num_eval=50',f'+eval.manifest={run}/manifest.json',
                        '+eval.partition=confirm',f'+eval.offset={offset}',f'output.filename={filename}']
                    print(json.dumps({'event':'eval_start','mode':label,'offset':offset}),flush=True)
                    with (run/f'{label}_s{args.seed}_eval_{offset}.log').open('x') as log:
                        subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                paths.append(str(result))
            results[label]=paths
    reports={name:compare(results[name],results['rbg']) for name in ('raw','tc','sub','official')}
    reports.update({name+'_vs_rbg_bn':compare(results[name],results['rbg_bn'])
                    for name in ('raw_bn','tc_bn','sub_bn','official')})
    reports['rbg_calibration_effect']=compare(results['rbg'],results['rbg_bn'])
    destination=run/f'comparison_s{args.seed}.json'
    if destination.exists(): raise FileExistsError(destination)
    destination.write_text(json.dumps({'stage':'first-seed fixed-budget; not final multi-seed evidence',
        'results':results,'comparisons':reports},indent=2))
    print(json.dumps(reports),flush=True)


if __name__=='__main__': main()
