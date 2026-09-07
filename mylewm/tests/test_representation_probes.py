import numpy as np
import pytest
from mylewm.representation_probes import regression_probe,task_probe
from mylewm.tools.probe_rbg_pusht import cases,state_targets


def test_nonlinear_predictability_distinguished_from_linear():
    rng=np.random.default_rng(46)
    xs=[rng.normal(size=(n,1)) for n in (512,128,128)]
    ys=[x*x-1 for x in xs]
    linear=regression_probe(xs,ys)
    nonlinear=regression_probe(xs,ys,nonlinear=True)
    assert linear['test_r2_mean']<.2
    assert nonlinear['test_r2_mean']>.8


def test_training_statistics_do_not_use_heldout_targets():
    rng=np.random.default_rng(4)
    xs=[rng.normal(size=(32,2)) for _ in range(3)]
    ys=[x.copy() for x in xs]
    a=regression_probe(xs,ys)
    ys[2]+=1000
    b=regression_probe(xs,ys)
    assert a['alpha']==b['alpha']
    assert a['training_target_mean']==b['training_target_mean']
    assert a['training_target_std']==b['training_target_std']


def test_episode_disjoint_cases_and_periodic_angle():
    m={'train_episodes':list(range(10)),'validation_episodes':list(range(10,20)),
       'test_episodes':list(range(20,30))}
    chosen=cases(m,np.full(30,40),4,3)
    assert chosen==cases(m,np.full(30,40),4,3)
    assert all(0<=c['start']<25 for part in chosen for c in part)
    m['test_episodes']=m['train_episodes']
    with pytest.raises(ValueError):cases(m,np.full(30,40),10,10)
    state=np.zeros((2,7));state[1,4]=2*np.pi
    np.testing.assert_allclose(state_targets(state)[0],state_targets(state)[1],atol=1e-12)


def test_task_probe_and_libero_balanced_demo_split():
    from mylewm.tools.probe_rbg_libero import cases as libero_cases
    m={key:[{'task':t,'demo':f'demo_{i}','length':30}
            for t in range(10) for i in range(start,start+3)]
       for key,start in [('train_demos',0),('validation',3),('test',6)]}
    selected=libero_cases(m,2,1)
    assert [len(p) for p in selected]==[20,10,10]
    assert all(0<=c['start']<18 for p in selected for c in p)
    labels=[np.tile(np.arange(3),10) for _ in range(3)]
    xs=[np.eye(3)[y] for y in labels]
    assert task_probe(xs,labels)['test_accuracy']==1.
    labels[2]=labels[2]+10
    with pytest.raises(ValueError):task_probe(xs,labels)
