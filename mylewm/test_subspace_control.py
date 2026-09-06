import pytest
import torch
from pathlib import Path


@pytest.mark.skipif(not Path('external/sub-jepa/subjepa.py').exists(),reason='Official Sub-JEPA checkout required')
def test_official_subspace_adapter_axes_and_gradients():
    from mylewm.train_subspace_control import SubspaceAdapter
    reg=SubspaceAdapter(dim=12,subspaces=3,per_subspace_projections=16)
    assert not list(reg.parameters())
    p=reg.reg.projection_matrices
    torch.testing.assert_close(p@p.transpose(-1,-2),torch.eye(4).expand(3,-1,-1),atol=1e-6,rtol=1e-6)
    z=torch.randn(4,8,12,requires_grad=True)
    torch.manual_seed(10)
    a,_=reg(z)
    torch.manual_seed(10)
    b=reg.reg(z.transpose(0,1))
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    a.backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum()>0
