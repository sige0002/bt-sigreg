"""Fixed-case paired comparison; exact conservative one-sided 95% bound."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta
from mylewm.evaluation_contract import protocol as evaluation_protocol


def compare(baseline, candidate, alpha=.05):
    protocol = None
    initial_states = {}
    def collect(paths):
        nonlocal protocol
        result={}
        checkpoint = None
        for path in paths:
            data=json.loads(Path(path).read_text())
            cfg=data['config']
            # policy/output names may differ; the actual experiment must not.
            signature=evaluation_protocol(cfg)
            if 'provenance' in data:
                signature['provenance']={k:v for k,v in data['provenance'].items() if k!='checkpoint_sha256'}
                signature['planner_action_mean']=data['planner_action_mean']
                signature['planner_action_std']=data['planner_action_std']
                if len(data.get('initial_runtime_hashes',[]))!=len(data['episodes']) or not data.get('physical_actions'):
                    raise ValueError('Incomplete execution audit')
                if data['checkpoint_sha256']!=data['provenance']['checkpoint_sha256']:
                    raise ValueError('Checkpoint provenance mismatch')
            if protocol is None: protocol=signature
            elif signature!=protocol: raise ValueError('evaluation protocols differ')
            if checkpoint is None: checkpoint=data['checkpoint_sha256']
            elif checkpoint!=data['checkpoint_sha256']: raise ValueError('mixed checkpoints within one method')
            for index,(ep,start,success) in enumerate(zip(data['episodes'],data['starts'],data['successes'],strict=True)):
                key=(ep,start)
                if key in result: raise ValueError('duplicate evaluation case')
                if type(success) is not bool:raise ValueError('Nonboolean success flag')
                if 'provenance' in data:
                    current=data['initial_runtime_hashes'][index]
                    if key in initial_states and initial_states[key]!=current:
                        raise ValueError('Paired initial/goal observations differ')
                    initial_states[key]=current
                result[key]=bool(success)
        return result
    b,c=collect(baseline),collect(candidate)
    if not b or b.keys()!=c.keys(): raise ValueError('paired evaluation cases must match')
    n=len(b)
    wins=sum(c[k] and not b[k] for k in b)
    losses=sum(b[k] and not c[k] for k in b)
    # Bonferroni bounds on P(candidate-only success) and P(baseline-only
    # success). Does not assume independent successes within each pair.
    win_lower=0. if wins==0 else float(beta.ppf(alpha/2,wins,n-wins+1))
    loss_upper=1. if losses==n else float(beta.ppf(1-alpha/2,losses+1,n-losses))
    lower=win_lower-loss_upper
    return {'n':n,'baseline_successes':sum(b.values()),'candidate_successes':sum(c.values()),
            'baseline_rate':sum(b.values())/n,'candidate_rate':sum(c.values())/n,
            'candidate_only_success':wins,'baseline_only_success':losses,
            'difference':(wins-losses)/n,'one_sided_95_lower_bound':lower,
            'observed_no_degradation':wins>=losses,
            'strict_noninferiority_established':lower>=0,
            'scope':'Fixed trained checkpoints and this evaluation distribution, not all tasks or training seeds.'}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',nargs='+',required=True)
    p.add_argument('--candidate',nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=compare(a.baseline,a.candidate)
    a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
