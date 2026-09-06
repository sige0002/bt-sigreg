import pytest
import torch
from multitask_jepa import MultiTaskJEPAObjective
from multitask_lewm import normalize_latent


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(3))
        self.action_encoder = torch.nn.Linear(2, 3, bias=False)
        self.inputs = []
        self.outputs = []

    def encode(self, info):
        return {'emb': info['pixels'].mean((-1,-2)) @ self.weight}

    def predict(self, z, a):
        self.inputs.append(z.detach().clone())
        out = z + a
        self.outputs.append(normalize_latent(out[:, -1:]).detach().clone())
        return out


def example():
    c = {v: {'pixels': torch.randn(4, 3, 3, 2, 2), 'action': torch.randn(4,2,2)} for v in ('a','b')}
    f = {v: {'pixels': torch.randn(4,4,3,2,2, requires_grad=True)} for v in c}
    return c, f, torch.randn(4,4,2, requires_grad=True)


def test_future_is_not_predictor_input_and_predictions_are_reused():
    torch.manual_seed(0)
    model = RecordingModel()
    obj = MultiTaskJEPAObjective(model)
    c, f, a = example()
    loss, _ = obj.loss_from_observations(c, f, (1,2,4), future_actions=a)
    original = [x.clone() for x in model.outputs]
    for offset in (0,4):
        for h in range(1,4):
            assert torch.allclose(model.inputs[offset+h][:,-1:], model.outputs[offset+h-1])
    loss.backward()
    assert all(x['pixels'].grad is not None and x['pixels'].grad.abs().sum() > 0 for x in f.values())
    assert a.grad is not None and a.grad.abs().sum() > 0
    assert model.action_encoder.weight.grad.abs().sum() > 0
    model.inputs.clear(); model.outputs.clear()
    different = {v: {'pixels': torch.randn_like(x['pixels'])*10} for v,x in f.items()}
    obj.loss_from_observations(c, different, (1,2,4), future_actions=a)
    assert all(torch.equal(x,y) for x,y in zip(original,model.outputs))
    model.outputs.clear()
    obj.loss_from_observations(c, f, (1,2,4), future_actions=-a)
    assert not torch.allclose(original[-1],model.outputs[-1])


def test_future_horizons_are_one_based_and_old_call_is_rejected():
    model = RecordingModel(); obj = MultiTaskJEPAObjective(model)
    c,f,a = example()
    with pytest.raises(ValueError, match='explicit future_actions'):
        obj.loss_from_observations(c,f)
    with pytest.raises(ValueError, match='one-based'):
        obj.loss_from_observations(c,f,(0,),future_actions=a)
    loss,_ = obj.loss_from_observations(c,f,(1,),future_actions=a)
    loss.backward()
    for view in f.values():
        grad = view['pixels'].grad
        assert grad[:,0].abs().sum() > 0
        assert grad[:,1:].abs().sum() == 0


def test_horizon_specific_positive_sets():
    model = RecordingModel(); obj = MultiTaskJEPAObjective(model)
    c,f,a = example()
    masks = torch.eye(4,dtype=torch.bool).repeat(4,1,1)
    masks[1] = True
    _, metrics = obj.loss_from_observations(c,f,(1,2),masks,future_actions=a)
    assert all(value['cross'] == 0 for value in metrics[2].values())
    assert all(value['cross'] > 0 for value in metrics[1].values())


def test_measured_group_validation_and_action_block_order():
    from train_crossed import prepare_group
    ctx = torch.randint(0,255,(2,2,1,3,3,2,2),dtype=torch.uint8).expand(-1,-1,2,-1,-1,-1,-1).clone()
    af = torch.arange(2*4*5*2).float().reshape(1,2,4,5,2).expand(2,-1,-1,-1,-1).clone()
    g = {'context':ctx, 'future':torch.randint(0,255,(2,2,2,4,3,2,2),dtype=torch.uint8),
         'action_history':torch.zeros(2,2,2,5,2),'future_actions':af,
         'positive':torch.eye(4,dtype=torch.bool).repeat(4,1,1)}
    stats = {'action_mean':torch.zeros(2),'action_std':torch.ones(2)}
    _,_,a,_ = prepare_group(g,stats,'cpu')
    assert torch.equal(a[0,0],torch.arange(10).float())
    assert a.shape == (4,4,10)
    g['future_actions'][1,0,0,0,0] += 1
    with pytest.raises(ValueError,match='identical controls'):
        prepare_group(g,stats,'cpu')
    g['future_actions'][1,0,0,0,0] -= 1
    g['context'][1] = g['context'][0]
    g['future'][1] = g['future'][0]
    with pytest.raises(ValueError,match='duplicated'):
        prepare_group(g,stats,'cpu')


def test_actual_jepa_updates_actions_and_normalizes_cem_rollout(tmp_path):
    from pathlib import Path
    from train_pusht import build_model
    config = Path('.cache/stable-wm/hf_pusht/config.json')
    if not config.exists() or not torch.cuda.is_available():
        pytest.skip('local official config and CUDA required for integration test')
    torch.manual_seed(7)
    model = build_model(config).cuda()
    obj = MultiTaskJEPAObjective(model)
    c = {v: {'pixels':torch.randn(4,3,3,224,224,device='cuda'),
              'action':torch.randn(4,2,10,device='cuda')} for v in ('a','b')}
    f = {v: {'pixels':torch.randn(4,4,3,224,224,device='cuda')} for v in c}
    a = torch.randn(4,4,10,device='cuda')
    opt = torch.optim.AdamW(model.parameters(),lr=1e-5)
    # Official AdaLN-zero blocks intentionally block action-encoder gradients
    # on the first update. Verify the path opens after the gate update.
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        loss,_ = obj.loss_from_observations(c,f,(1,2,4),future_actions=a)
        loss.backward()
        if step == 1:
            for component in (model.encoder,model.action_encoder,model.predictor):
                assert sum(p.grad.abs().sum() for p in component.parameters() if p.grad is not None) > 0
        opt.step()
    model.eval()
    with torch.no_grad():
        out = model.rollout({'pixels':c['a']['pixels'][:,None]},
                            torch.randn(4,1,7,10,device='cuda'))['predicted_emb']
        assert torch.allclose(out.norm(dim=-1),torch.ones_like(out[...,0]),atol=1e-6)
        z = model.encode({'pixels':c['a']['pixels']})['emb']
        act = torch.randn(4,3,10,device='cuda')
        p1 = model.predict(z,model.action_encoder(act))
        p2 = model.predict(z,model.action_encoder(-act))
        assert not torch.equal(p1,p2)
        torch.save(model,tmp_path/'model.ckpt')
        restored = torch.load(tmp_path/'model.ckpt',weights_only=False)
        assert torch.equal(p1,restored.predict(z,restored.action_encoder(act)))
