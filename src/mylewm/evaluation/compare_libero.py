"""Paired native-success comparison with protocol and initial-state checks."""
import json
from pathlib import Path

from scipy.stats import beta


def compare(baseline, candidate):
    baseline, candidate = Path(baseline), Path(candidate)
    configs = [json.loads((p/'config.json').read_text()) for p in (baseline,candidate)]
    if configs[0]['controller'] != configs[1]['controller']:
        raise ValueError('Incompatible evaluation protocol: controller (BC and CEM are separate tracks)')
    keys = ('manifest_sha256','mujoco','render_backend','render_audit_sha256',
            'controller','initial_history','goal','metric','seed','budget','horizon',
            'task_ids','episodes','offset')
    if configs[0]['controller'] == 'flow_matching_bc_v1':
        keys += ('execute_actions', 'euler_steps', 'policy_config', 'bc_step', 'init_states_sha256', 'control_fps', 'source_sha256')
        # Allow the world-model source/method to differ, but hold BC training fixed.
        training_keys = ('manifest_sha256', 'model', 'task_names', 'steps', 'batch_size', 'workers',
                         'lr', 'weight_decay', 'seed', 'precision', 'optimizer', 'deterministic', 'source_sha256')
        training = [c['training_provenance']['training_config'] for c in configs]
        for key in training_keys:
            if training[0][key] != training[1][key]:
                raise ValueError(f'Incompatible BC training protocol: {key}')
    else:
        keys += ('samples','iterations')
    for key in keys:
        if configs[0][key] != configs[1][key]:
            raise ValueError(f'Incompatible evaluation protocol: {key}')
    rows=[]
    for directory,config in zip((baseline,candidate),configs):
        data={}
        for line in (directory/'episodes.jsonl').read_text().splitlines():
            row=json.loads(line)
            key=(row['task_id'],row['init_id'])
            if key in data: raise ValueError('Duplicate evaluation case')
            if type(row['success']) is not bool: raise ValueError('Success must be boolean')
            data[key]=row
        expected={(t,i) for t in config['task_ids'] for i in
                  range(config['offset'],config['offset']+config['episodes'])}
        if data.keys()!=expected: raise ValueError('Incomplete evaluation')
        rows.append(data)
    a,b=rows
    if not a or a.keys()!=b.keys(): raise ValueError('Cases do not match')
    for key in a:
        for field in ('task','initial_state_sha256','initial_image_sha256','goal_image_sha256','goal_demo'):
            if a[key][field]!=b[key][field]: raise ValueError(f'Case mismatch: {key} {field}')
    def rates(keys):
        keys=list(keys); n=len(keys)
        wins=sum(b[k]['success'] and not a[k]['success'] for k in keys)
        losses=sum(a[k]['success'] and not b[k]['success'] for k in keys)
        lower=(0. if wins==0 else float(beta.ppf(.025,wins,n-wins+1))) - (
            1. if losses==n else float(beta.ppf(.975,losses+1,n-losses)))
        return {'n':n,'baseline_rate':sum(a[k]['success'] for k in keys)/n,
                'candidate_rate':sum(b[k]['success'] for k in keys)/n,
                'candidate_only_success':wins,'baseline_only_success':losses,
                'difference':(wins-losses)/n,'one_sided_95_lower_bound':lower,
                'strict_noninferiority_established':lower>=0}
    return {'pooled':rates(a),'per_task':{str(t):rates(k for k in a if k[0]==t)
            for t in configs[0]['task_ids']},
            'scope':'Fixed checkpoints, balanced fixed cases; intervals are descriptive, not adjusted across tasks or comparisons.'}
