"""Complete predeclared independent seeds sequentially, after first-seed audit."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def result_path(benchmark,seed):
    return ROOT/f'.cache/stable-wm/{benchmark}/rbg_v0/comparison_s{seed}.json'


def require_comparison(benchmark,seed):
    report=json.loads(result_path(benchmark,seed).read_text())
    required={'rbg','raw','tc','sub','rbg_bn','raw_bn','tc_bn','sub_bn'}
    if benchmark=='pusht': required.add('official')
    if not required<=report['results'].keys() or not report['comparisons']:
        raise RuntimeError(f'Incomplete {benchmark} comparison seed {seed}')
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--wait-pid',type=int,required=True)
    p.add_argument('--first-seed',type=int,default=3072)
    p.add_argument('--seeds',type=int,nargs='+',default=[3073,3074])
    args=p.parse_args()
    if args.first_seed in args.seeds or len(set(args.seeds))!=len(args.seeds):
        p.error('Independent seed identifiers required')
    deadline=time.monotonic()+7*24*3600
    while True:
        try: os.kill(args.wait_pid,0)
        except ProcessLookupError: break
        if time.monotonic()>deadline: raise TimeoutError('First seed queue exceeded seven days')
        time.sleep(10)
    for benchmark in ('pusht','libero10'): require_comparison(benchmark,args.first_seed)
    for seed in args.seeds:
        for benchmark,script in [('pusht','run_rbg_pusht_comparison.py'),
                                 ('libero10','run_rbg_libero_comparison.py')]:
            if result_path(benchmark,seed).exists():
                require_comparison(benchmark,seed)
                continue
            log_path=ROOT/f'.cache/stable-wm/{benchmark}/rbg_v0/replicate_queue_s{seed}.log'
            command=[sys.executable,str(ROOT/'mylewm'/script),'--seed',str(seed)]
            print(json.dumps({'event':'replicate_start','benchmark':benchmark,'seed':seed}),flush=True)
            with log_path.open('x') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            require_comparison(benchmark,seed)
    destination=ROOT/'.cache/stable-wm/rbg_v0_replication_index.json'
    with destination.open('x') as f:
        json.dump({'seeds':[args.first_seed,*args.seeds],
                   'results':{benchmark:[str(result_path(benchmark,s))
                       for s in [args.first_seed,*args.seeds]] for benchmark in ('pusht','libero10')},
                   'status':'Comparisons generated; research diagnostics and final audit still required'},f,indent=2)


if __name__=='__main__': main()
