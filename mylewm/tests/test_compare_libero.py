import json
import pytest
from mylewm.evaluation.compare_libero import compare


def fixture_run(path,successes):
    path.mkdir()
    config={k:'same' for k in ('manifest_sha256','mujoco','render_backend','render_audit_sha256',
        'controller','initial_history','goal','metric')}
    config.update(seed=42,budget=520,horizon=8,samples=128,iterations=5,
                  task_ids=[0,1],episodes=2,offset=0)
    (path/'config.json').write_text(json.dumps(config))
    rows=[]
    for t in range(2):
        for i in range(2):
            rows.append({'task_id':t,'init_id':i,'task':str(t),'success':successes[2*t+i],
                'initial_state_sha256':f'{t}_{i}','initial_image_sha256':f'{t}_{i}',
                'goal_image_sha256':str(t),'goal_demo':'demo_0'})
    (path/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return rows


def test_paired_success_counts_and_noninferiority_not_assumed(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b'
    fixture_run(a,[True,False,False,True]);fixture_run(b,[True,True,False,False])
    report=compare(a,b)
    assert report['pooled']['candidate_only_success']==1
    assert report['pooled']['baseline_only_success']==1
    assert report['pooled']['difference']==0
    assert not report['pooled']['strict_noninferiority_established']
    assert report['per_task']['0']['difference']==.5
    assert report['per_task']['1']['difference']==-.5


@pytest.mark.parametrize('fault',['partial','duplicate','initial_state','goal','protocol'])
def test_reject_noncomparable_evaluation(tmp_path,fault):
    a,b=tmp_path/'a',tmp_path/'b'
    fixture_run(a,[True]*4); rows=fixture_run(b,[True]*4)
    if fault=='partial':rows.pop()
    elif fault=='duplicate':rows.append(rows[0])
    elif fault=='initial_state':rows[0]['initial_state_sha256']='wrong'
    elif fault=='goal':rows[0]['goal_image_sha256']='wrong'
    else:
        c=json.loads((b/'config.json').read_text());c['budget']=521
        (b/'config.json').write_text(json.dumps(c))
    (b/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError):compare(a,b)


def test_bc_comparison_holds_head_training_fixed_and_rejects_cem(tmp_path):
    a, b = tmp_path/'a', tmp_path/'b'
    fixture_run(a, [True]*4); fixture_run(b, [True]*4)
    base = json.loads((a/'config.json').read_text())
    bc = dict(base, controller='flow_matching_bc_v1', execute_actions=8, euler_steps=10,
              policy_config={'width': 256}, bc_step=40000, init_states_sha256='same',
              control_fps=20, source_sha256='same')
    keys = ('manifest_sha256', 'model', 'task_names', 'steps', 'batch_size', 'workers',
            'lr', 'weight_decay', 'seed', 'precision', 'optimizer', 'deterministic', 'source_sha256')
    bc['training_provenance'] = {'training_config': {k: 'same' for k in keys}}
    (a/'config.json').write_text(json.dumps(bc))
    with pytest.raises(ValueError, match='controller'):
        compare(a, b)
    (b/'config.json').write_text(json.dumps(bc))
    assert compare(a, b)['pooled']['difference'] == 0
    bc['bc_step'] = 20000
    (b/'config.json').write_text(json.dumps(bc))
    with pytest.raises(ValueError, match='bc_step'):
        compare(a, b)
    bc['bc_step'] = 40000
    bc['training_provenance']['training_config']['steps'] = 20000
    (b/'config.json').write_text(json.dumps(bc))
    with pytest.raises(ValueError, match='BC training'):
        compare(a, b)
