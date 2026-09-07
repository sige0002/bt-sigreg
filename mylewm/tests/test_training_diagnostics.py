from types import SimpleNamespace
import torch
from mylewm.training_diagnostics import encoder_gradient_norms,action_diagnostics


def test_gradient_diagnostics_include_coefficients_without_accumulating_gradients():
    encoder=torch.nn.Linear(2,2,bias=False)
    model=SimpleNamespace(encoder=encoder,projector=torch.nn.Identity())
    z=encoder(torch.tensor([[1.,2.],[3.,-1.]]))
    parts={'prediction':(z-1).square().mean(),'gaussian':z.square().mean(),
           'cross':(z[:,0]*z[:,1]).mean()}
    reference={name:float(torch.autograd.grad(term,encoder.weight,retain_graph=True)[0].norm())
               for name,term in parts.items()}
    norms=encoder_gradient_norms(model,parts,.09,.01)
    for name,weight in [('prediction',1.),('gaussian',.09),('cross',.01)]:
        torch.testing.assert_close(torch.tensor(norms[name]),torch.tensor(reference[name]*weight))
    assert encoder.weight.grad is None
    (parts['prediction']+.09*parts['gaussian']+.01*parts['cross']).backward()
    assert encoder.weight.grad is not None


def test_action_diagnostics_distinguish_physical_zero_from_normalized_zero():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('training_action_mean',torch.tensor([1.,2.]))
            self.register_buffer('training_action_std',torch.tensor([2.,4.]))
        def encode(self,b):return {'emb':b['pixels']}
        def action_encoder(self,a):return a
        def predict(self,z,a):return z+a
    report=action_diagnostics(Model().eval(),torch.zeros(2,4,2),torch.ones(2,3,2))
    assert report['zero_normalized_action']['prediction_mse']==0
    assert report['zero_physical_action']['prediction_mse']==.25
