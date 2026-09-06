"""Run fixed pilot/confirmation cases for baseline and one fixed candidate."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from compare_paired import compare


def main():
    root=Path(__file__).resolve().parents[1]
    run=root/'.cache/stable-wm/pusht/trajectory_v1'
    env={**os.environ,'PYTHONPATH':str(root/'lewm')+os.pathsep+str(root/'mylewm'),
         'STABLEWM_HOME':str(root/'.cache/stable-wm'),'PYTHONUNBUFFERED':'1'}
    candidate=run/'train/step_1000_object.ckpt'
    deadline=time.monotonic()+1800
    while not candidate.exists():
        if time.monotonic()>deadline: raise TimeoutError('Training checkpoint did not arrive within 30 minutes')
        time.sleep(10)
    # Checkpoint publication happens just before the training process exits.
    time.sleep(5)
    policies={'baseline':'pusht/lewm','candidate':'pusht/trajectory_v1/train/step_1000'}
    for partition,count in [('pilot',50),('confirm',200)]:
        paths={k:[] for k in policies}
        for offset in range(0,count,50):
            for name,policy in policies.items():
                filename=f'trajectory_{partition}_{offset}_{name}.txt'
                result=root/'.cache/stable-wm'/Path(policy).parent/(filename+'.json')
                paths[name].append(str(result))
                # Do not rerun fixed cases if a completed result already exists.
                if result.exists(): continue
                cmd=[sys.executable,str(root/'lewm/eval.py'),f'policy={policy}',
                     'eval.num_eval=50','eval.eval_budget=50',
                     f'+eval.manifest={run}/manifest.json',f'+eval.partition={partition}',
                     f'+eval.offset={offset}',f'output.filename={filename}']
                print(f'START {partition} offset={offset} {name}',flush=True)
                with (run/f'{partition}_{offset}_{name}.log').open('w') as log:
                    subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                data=json.loads(result.read_text())
                print(f'FINISH {name}: {sum(data["successes"])}/50',flush=True)
        report=compare(paths['baseline'],paths['candidate'])
        (run/f'{partition}_comparison.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({'partition':partition,**report}),flush=True)


if __name__=='__main__': main()
