import json
import pytest
from mylewm.compare_libero import compare


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
