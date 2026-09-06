import torch
from trajectory_experiment import objective,preprocess
from test_crossed_architecture import RecordingModel


def test_open_loop_uses_only_context_and_reuses_predictions():
    model=RecordingModel()
    x=torch.randn(4,7,3,2,2,requires_grad=True)
    a=torch.randn(4,6,2,requires_grad=True)
    reg=lambda z:z.square().mean()
    loss,parts=objective(model,x,a,reg)
    predictions=[p.clone() for p in model.inputs]
    assert len(predictions)==4
    assert torch.equal(predictions[1][:,-1], (model.inputs[0]+model.action_encoder(a)[:,:3])[:,-1])
    loss.backward()
    assert x.grad[:,3:].abs().sum()>0 and a.grad.abs().sum()>0
    model.inputs.clear()
    x2=x.detach().clone(); x2[:,3:]+=100
    objective(model,x2,a,reg)
    assert all(torch.equal(p,q) for p,q in zip(predictions,model.inputs))


def test_action_frameskip_scaling_order():
    pixels=torch.zeros(1,7,3,2,2,dtype=torch.uint8)
    actions=torch.arange(60).reshape(1,6,5,2)
    x,a=preprocess((pixels,actions),{'action_mean':[0,1],'action_std':[2,2]},'cpu')
    assert a.shape==(1,6,10)
    assert torch.equal(a[0,0],torch.tensor([0,0,1,1,2,2,3,3,4,4]).float())
    assert torch.isfinite(x).all()


def test_paired_equal_scores_do_not_prove_noninferiority(tmp_path):
    import json
    from compare_paired import compare
    cfg={'seed':42,'world':{},'plan_config':{},'dataset':{},'solver':{},
         'eval':dict(goal_offset_steps=25,eval_budget=50,img_size=224,dataset_name='pusht',callables=[])}
    base={'episodes':list(range(50)),'starts':[0]*50,'successes':[True]*42+[False]*8,
          'checkpoint_sha256':'base','config':cfg}
    a=tmp_path/'a.json'; b=tmp_path/'b.json'
    a.write_text(json.dumps(base))
    base['checkpoint_sha256']='candidate'
    b.write_text(json.dumps(base))
    result=compare([a],[b])
    assert result['observed_no_degradation']
    assert not result['strict_noninferiority_established']
    assert result['one_sided_95_lower_bound']<0
    base['config']['seed']=7
    b.write_text(json.dumps(base))
    import pytest
    with pytest.raises(ValueError,match='protocols differ'):
        compare([a],[b])
