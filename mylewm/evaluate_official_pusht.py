"""Independent official-only 200-case regression evaluation; no training queue."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def commands(manifest,tag,cases=200):
    if cases<50 or cases%50:raise ValueError('Cases must be a positive multiple of 50')
    if not tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in tag):
        raise ValueError('Tag must be a plain experiment identifier')
    return [[sys.executable,str(ROOT/'lewm/eval.py'),'policy=pusht/lewm',
             'eval.num_eval=50',f'+eval.manifest={manifest}','+eval.partition=confirm',
             f'+eval.offset={offset}','+eval.audit_provenance=true','+eval.shared_physical_search=true',
             f'output.filename={tag}_official_confirm_{offset}.txt']
            for offset in range(0,cases,50)]


def archive_videos(result_path):
    """Move SWM's shared env_N.mp4 outputs before the next batch overwrites them."""
    destination=result_path.with_suffix('.videos')
    if destination.exists():return
    result=json.loads(result_path.read_text())
    sources=[result_path.parent/f'env_{i}.mp4' for i in range(len(result['episodes']))]
    finished=result_path.stat().st_mtime
    if not all(p.exists() and finished-result['elapsed']-2 <= p.stat().st_mtime <= finished+1 for p in sources):
        return  # old cached numeric results may no longer have their original videos
    destination.mkdir()
    for source in sources:shutil.move(str(source),str(destination/source.name))
    with (destination/'cases.json').open('x') as stream:
        json.dump({'episodes':result['episodes'],'starts':result['starts'],
                   'source_result':str(result_path)},stream,indent=2)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,default=ROOT/'.cache/stable-wm/pusht/rbg_v0/manifest.json')
    p.add_argument('--tag',default='controlled_v2')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--cases',type=int,default=200)
    args=p.parse_args()
    plan=commands(args.manifest,args.tag,args.cases)
    if not args.execute:
        print(json.dumps({'dry_run':True,'training_commands':0,'commands':plan},indent=2));return
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    from mylewm.evaluation_contract import provenance,protocol,validate_result
    root=ROOT/'.cache/stable-wm'
    identity=provenance(root/'datasets/pusht_expert_train.h5',args.manifest,root/'pusht/lewm_object.ckpt',ROOT)
    cases=json.loads(args.manifest.read_text())['confirm']
    if len(cases)<args.cases:raise ValueError('Insufficient fixed regression cases')
    env={**os.environ,'STABLEWM_HOME':str(root),
         'PYTHONPATH':os.pathsep.join([str(ROOT),str(ROOT/'lewm')]),'PYTHONUNBUFFERED':'1'}
    paths=[]
    for index,command in enumerate(plan):
        filename=command[-1].split('=',1)[1]
        path=root/'pusht'/(filename+'.json')
        with initialize_config_dir(version_base=None,config_dir=str(ROOT/'lewm/config/eval')):
            cfg=compose(config_name='pusht',overrides=command[2:])
            cfg.world.max_episode_steps=2*cfg.eval.eval_budget
            expected_protocol=protocol(OmegaConf.to_container(cfg,resolve=True))
        selected=cases[index*50:(index+1)*50]
        if not path.exists():
            log_path=root/'pusht'/(filename+'.log')
            print(json.dumps({'event':'official_eval_start','offset':index*50,'log':str(log_path)}),flush=True)
            with log_path.open('x') as stream:
                subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        validate_result(json.loads(path.read_text()),selected,identity,expected_protocol)
        archive_videos(path)
        paths.append(path)
    results=[json.loads(path.read_text()) for path in paths]
    successes=sum(sum(result['successes']) for result in results)
    report={'schema':'official_pusht_regression_v2','cases':args.cases,'successes':successes,
            'success_rate':successes/args.cases,'provenance':identity,'results':[str(p) for p in paths],
            'scope':'existing regression cases; no Raw/RBG performance comparison until trained separately'}
    destination=root/'pusht'/f'{args.tag}_official_summary.json'
    if destination.exists():
        if json.loads(destination.read_text())!=report:raise ValueError('Summary already exists with different results')
    else:
        with destination.open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
