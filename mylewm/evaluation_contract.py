"""Small provenance records for paired evaluation and strict result-cache checks."""
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path

from mylewm.data_contract import file_sha256


def array_hash(value):
    import numpy as np
    x=np.ascontiguousarray(value)
    h=hashlib.sha256()
    h.update(str(x.dtype).encode()); h.update(str(x.shape).encode()); h.update(x.tobytes())
    return h.hexdigest()


def protocol(config):
    value={key:config[key] for key in ('seed','world','plan_config','dataset')}
    value['solver']={key:v for key,v in config['solver'].items() if key!='model'}
    value['eval']={key:config['eval'][key] for key in
                  ('goal_offset_steps','eval_budget','img_size','dataset_name','callables')}
    value['eval']['shared_physical_search']=config['eval'].get('shared_physical_search',False)
    return value


def provenance(dataset,manifest,checkpoint,root):
    import stable_worldmodel as swm
    import stable_pretraining as spt
    sources=[root/'lewm/eval.py',root/'lewm/jepa.py',root/'lewm/module.py',
             root/'mylewm/evaluation_contract.py',root/'mylewm/planning_action_adapter.py',
             Path(inspect.getfile(swm.World)),Path(inspect.getfile(swm.solver.CEMSolver)),
             Path(inspect.getfile(swm.policy.WorldModelPolicy))]
    return {'schema':'paired_eval_v2','dataset_sha256':file_sha256(dataset),
            'manifest_sha256':file_sha256(manifest),'checkpoint_sha256':file_sha256(checkpoint),
            'source_sha256':{str(p):file_sha256(p) for p in sources},
            'versions':{name:importlib.metadata.version(name) for name in
                         ('torch','torchvision','stable-pretraining','stable-worldmodel','gymnasium','pymunk')},
            'success_definition':'dataset goal reached under configured environment termination; not fixed T 95% coverage',
            'case_use':'existing 200-case regression set, not untouched final test',
            'training_overlap':'official checkpoint may have seen these dataset episodes'}


def validate_result(data,expected_cases,identity,expected_protocol):
    if data.get('provenance')!=identity: raise ValueError('Evaluation provenance/cache mismatch')
    if protocol(data['config'])!=expected_protocol: raise ValueError('Evaluation protocol/cache mismatch')
    cases=list(zip(data['episodes'],data['starts'],strict=True))
    expected=[(c['episode'],c['start']) for c in expected_cases]
    if cases!=expected or len(set(cases))!=len(cases): raise ValueError('Evaluation case mismatch/duplicate')
    if len(data['successes'])!=len(cases) or any(type(x) is not bool for x in data['successes']):
        raise ValueError('Incomplete/nonboolean successes')
    if data['checkpoint_sha256']!=identity['checkpoint_sha256']:raise ValueError('Checkpoint mismatch')
    if not data.get('physical_actions') or len(data.get('initial_runtime_hashes',[]))!=len(cases):
        raise ValueError('Missing physical actions or initial state audit')
