import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'lewm'))
from module import SIGReg
from mylewm.algorithms.objectives import GaussianSIGReg, one_step_objective


def test_raw_loss_and_gradient_exact():
    torch.manual_seed(13)
    x = torch.randn(4, 16, 12, requires_grad=True)
    old, new = SIGReg(num_proj=32), GaussianSIGReg(12, 32)
    torch.manual_seed(31)
    a = old(x)
    ga, = torch.autograd.grad(a, x)
    torch.manual_seed(31)
    b = new(x)
    gb, = torch.autograd.grad(b, x)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(ga, gb, rtol=0, atol=0)



def test_no_trainable_parameters_and_finite_gradient():
    reg = GaussianSIGReg()
    assert sum(p.numel() for p in reg.parameters()) == 0
    x = torch.randn(4, 8, 192, requires_grad=True)
    a = reg(x)
    a.backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize('kwargs', [{'dim': 0}, {'projections': 0}, {'knots': 1}])
def test_reject_invalid(kwargs):
    with pytest.raises(ValueError):
        GaussianSIGReg(**kwargs)


def test_zero_collapse_stationary_not_claimed_fixed():
    x = torch.zeros(4, 16, 8, requires_grad=True)
    a = GaussianSIGReg(8, 32)(x)
    a.backward()
    assert a > 0 and x.grad.abs().max() == 0


def test_targets_are_next_frame_and_future_not_predictor_input():
    class Toy(torch.nn.Module):
        def encode(self, batch):
            return {'emb': batch['pixels']}
        def action_encoder(self, a):
            return a
        def predict(self, z, a):
            self.seen = z.detach().clone()
            return z + a
    model = Toy()
    x = torch.randn(8, 4, 4, requires_grad=True)
    a = (x[:, 1:] - x[:, :-1]).detach()
    _, parts = one_step_objective(model, x, a, GaussianSIGReg(4, 16))
    assert parts['prediction'] < 1e-12
    torch.testing.assert_close(model.seen, x[:, :3])


def test_stateless_sampler_resumes_and_changes_batches():
    from mylewm.training.loop import StepBatches
    full = list(StepBatches(1000, 16, 12, 31))
    resumed = list(StepBatches(1000, 16, 12, 31, start=7))
    assert full[7:] == resumed
    assert full[0] != full[1]



def test_official_adaln_zero_opens_action_gradient_after_update():
    from module import ARPredictor, Embedder
    torch.manual_seed(5)
    encoder=Embedder(input_dim=2,emb_dim=8)
    predictor=ARPredictor(num_frames=3,input_dim=8,hidden_dim=8,output_dim=8,
                          depth=2,heads=2,mlp_dim=32,dim_head=4,dropout=0.,emb_dropout=0.)
    optimizer=torch.optim.AdamW(list(encoder.parameters())+list(predictor.parameters()),lr=1e-3)
    z,a,target=torch.randn(8,3,8),torch.randn(8,3,2),torch.randn(8,3,8)
    norms=[]
    for _ in range(2):
        optimizer.zero_grad()
        (predictor(z,encoder(a))-target).square().mean().backward()
        norms.append(sum(float(p.grad.abs().sum()) for p in encoder.parameters()))
        optimizer.step()
    assert norms[0]==0 and norms[1]>0

@pytest.mark.parametrize('mode', ['raw', 'tc', 'bt'])
def test_objective_uses_full_latent_or_temporal_residual(mode):
    class Toy(torch.nn.Module):
        def encode(self, batch): return {'emb': batch['pixels']}
        def action_encoder(self, a): return a
        def predict(self, z, a): return z + a
    seen = []
    def regularizer(z):
        seen.append(z)
        return z.square().mean()
    x, a = torch.randn(3, 4, 8), torch.randn(3, 3, 8)
    loss, parts = one_step_objective(Toy(), x, a, regularizer, mode=mode)
    expected = x.transpose(0, 1)
    if mode == 'tc': expected = expected - expected.mean(0, keepdim=True)
    torch.testing.assert_close(seen[0], expected)
    torch.testing.assert_close(loss, parts['prediction'] + .09 * parts['gaussian'])
    assert 'cross' not in parts


def test_retired_mode_is_rejected_before_model_execution():
    with pytest.raises(ValueError, match='Unsupported mode'):
        one_step_objective(None, None, None, None, mode='rbg')
