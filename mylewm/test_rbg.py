import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lewm'))
from module import SIGReg
from mylewm.rbg import BlockSIGReg, one_step_objective


def test_raw_loss_and_gradient_exact():
    torch.manual_seed(13)
    x = torch.randn(4, 16, 12, requires_grad=True)
    old, new = SIGReg(num_proj=32), BlockSIGReg(12, 1, 32)
    torch.manual_seed(31)
    a = old(x)
    ga, = torch.autograd.grad(a, x)
    torch.manual_seed(31)
    b, c = new(x)
    gb, = torch.autograd.grad(b, x)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(ga, gb, rtol=0, atol=0)
    assert c == 0


def test_cross_covariance_formula_and_copy():
    torch.manual_seed(1)
    x = torch.randn(3, 32, 4)
    z = torch.cat([x, x], -1).requires_grad_()
    reg = BlockSIGReg(8, 2, 32)
    _, cross = reg(z)
    y = x - x.mean(1, keepdim=True)
    expected = ((y.transpose(1, 2) @ y / 31).square()).mean()
    torch.testing.assert_close(cross, expected)
    assert cross > .1
    cross.backward()
    assert z.grad.abs().sum() > 0


def test_no_trainable_parameters_and_finite_gradient():
    reg = BlockSIGReg()
    assert sum(p.numel() for p in reg.parameters()) == 0
    x = torch.randn(4, 8, 192, requires_grad=True)
    a, b = reg(x)
    (a + b).backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize('kwargs', [{'blocks': 5}, {'projections': 3}, {'knots': 1}])
def test_reject_invalid(kwargs):
    with pytest.raises(ValueError):
        BlockSIGReg(**kwargs)


def test_zero_collapse_stationary_not_claimed_fixed():
    x = torch.zeros(4, 16, 8, requires_grad=True)
    a, b = BlockSIGReg(8, 2, 32)(x)
    (a+b).backward()
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
    _, parts = one_step_objective(model, x, a, BlockSIGReg(4, 2, 16))
    assert parts['prediction'] < 1e-12
    torch.testing.assert_close(model.seen, x[:, :3])


def test_stateless_sampler_resumes_and_changes_batches():
    from mylewm.train_rbg import StepBatches
    full = list(StepBatches(1000, 16, 12, 31))
    resumed = list(StepBatches(1000, 16, 12, 31, start=7))
    assert full[7:] == resumed
    assert full[0] != full[1]


def test_gaussian_marginals_allow_nongaussian_relation():
    torch.manual_seed(71)
    u = torch.randn(100000)
    sign = torch.randint(2, (100000,)) * 2 - 1
    v = sign * u
    # Cross covariance vanishes but dependence remains: |U| == |V|.
    assert abs((u*v).mean()) < .03
    assert abs(v.square().mean()-1) < .03
    assert abs(v.pow(4).mean()-3) < .1
    mixed = (u+v) / (2**.5)
    assert mixed.pow(4).mean() > 5.5


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
