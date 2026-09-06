"""Fixed-case paired comparison; exact conservative one-sided 95% bound."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta


def compare(baseline, candidate, alpha=.05):
    protocol = None
    def collect(paths):
        nonlocal protocol
        result={}
        checkpoint = None
        for path in paths:
            data=json.loads(Path(path).read_text())
            cfg=data['config']
            # policy/output names may differ; the actual experiment must not.
            solver={k:v for k,v in cfg['solver'].items() if k!='model'}
            signature={k:cfg[k] for k in ('seed','world','plan_config','dataset')}
            signature['solver']=solver
            signature['eval']={k:cfg['eval'][k] for k in
                               ('goal_offset_steps','eval_budget','img_size','dataset_name','callables')}
            if protocol is None: protocol=signature
            elif signature!=protocol: raise ValueError('evaluation protocols differ')
            if checkpoint is None: checkpoint=data['checkpoint_sha256']
            elif checkpoint!=data['checkpoint_sha256']: raise ValueError('mixed checkpoints within one method')
            for ep,start,success in zip(data['episodes'],data['starts'],data['successes'],strict=True):
                key=(ep,start)
                if key in result: raise ValueError('duplicate evaluation case')
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
