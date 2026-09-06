import pytest
import torch
from mylewm.rbg import BlockSIGReg
from mylewm.train_rbg_ablation import MixedGaussian


@pytest.mark.parametrize('mix,blocks',[(0.,4),(1.,1)])
def test_boundary_gaussian_values_and_gradients(mix,blocks):
    z=torch.randn(4,16,16,requires_grad=True)
    mixed=MixedGaussian(dim=16,blocks=4,global_mix=mix,projections=64)
    reference=BlockSIGReg(dim=16,blocks=blocks,projections=64)
    torch.manual_seed(43);a,cross=mixed(z)
    torch.manual_seed(43);b,_=reference(z)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    ga=torch.autograd.grad(a,z,retain_graph=True)[0]
    gb=torch.autograd.grad(b,z)[0]
    torch.testing.assert_close(ga,gb,atol=0,rtol=0)
    assert cross>0 and not list(mixed.parameters())


def test_mixed_projection_budget_and_convex_loss():
    reg=MixedGaussian(dim=16,blocks=4,global_mix=.5,projections=64)
    assert reg.block.projections*4+reg.global_reg.projections==64
    z=torch.randn(4,16,16)
    torch.manual_seed(22);actual,cross=reg(z)
    torch.manual_seed(22);a,c=reg.block(z);b,_=reg.global_reg(z)
    torch.testing.assert_close(actual,.5*(a+b),atol=0,rtol=0)
    torch.testing.assert_close(cross,c,atol=0,rtol=0)
