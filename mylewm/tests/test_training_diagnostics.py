from types import SimpleNamespace
import torch
from mylewm.training.diagnostics import encoder_gradient_norms,action_diagnostics,transport_statistics


def test_monitor_shell_handles_missing_and_partial_logs(tmp_path):
    import json
    import subprocess
    from pathlib import Path
    script=Path(__file__).resolve().parents[2]/'scripts/monitor_training.sh'
    command=['bash',str(script),'--once','--run',str(tmp_path)]
    result=subprocess.run(command,capture_output=True,text=True,timeout=10)
    assert result.returncode==0 and 'Waiting for complete metrics' in result.stdout
    (tmp_path/'config.json').write_text(json.dumps({'steps':100}))
    row={'step':10,'loss':.5,'prediction':.2,'gaussian':3.,'elapsed_session':10.,
         'validation':{'prediction':.3,'gaussian':4.}}
    (tmp_path/'metrics.jsonl').write_text(json.dumps(row)+'\ninvalid\n'+ '{"step":11')
    result=subprocess.run(command,capture_output=True,text=True,timeout=10)
    assert result.returncode==0
    assert 'Step: 10 / 100 (10.00%)' in result.stdout
    assert 'Loss: 0.5' in result.stdout and 'Validation @ 10' in result.stdout
    assert 'skipped 1 malformed' in result.stdout
    result=subprocess.run(command+['--interval','0'],capture_output=True,text=True,timeout=10)
    assert result.returncode==2


def test_transport_statistics_do_not_change_gradients_or_rng():
    z=torch.randn(4,8,12,requires_grad=True)
    u=1.1*z
    state=torch.get_rng_state().clone()
    report=transport_statistics(z,u)
    assert torch.equal(state,torch.get_rng_state()) and z.grad is None
    assert abs(report['u_variance_trace']/report['z_variance_trace']-1.21)<1e-5
    assert abs(report['sampled_distance_ratio_min']-1.1)<1e-6
    assert abs(report['sampled_distance_ratio_max']-1.1)<1e-6
    assert report['sampled_distance_pairs']==32
    u.sum().backward()
    torch.testing.assert_close(z.grad,torch.full_like(z,1.1))
    collapsed=transport_statistics(torch.zeros(4,8,12),torch.zeros(4,8,12))
    assert collapsed['sampled_distance_ratio_min'] is None
    assert collapsed['sampled_distance_pairs']==0


def test_gradient_diagnostics_include_coefficients_without_accumulating_gradients():
    encoder=torch.nn.Linear(2,2,bias=False)
    model=SimpleNamespace(encoder=encoder,projector=torch.nn.Identity())
    z=encoder(torch.tensor([[1.,2.],[3.,-1.]]))
    parts={'prediction':(z-1).square().mean(),'gaussian':z.square().mean()}
    reference={name:float(torch.autograd.grad(term,encoder.weight,retain_graph=True)[0].norm())
               for name,term in parts.items()}
    norms=encoder_gradient_norms(model,parts,.09)
    for name,weight in [('prediction',1.),('gaussian',.09)]:
        torch.testing.assert_close(torch.tensor(norms[name]),torch.tensor(reference[name]*weight))
    assert encoder.weight.grad is None
    (parts['prediction']+.09*parts['gaussian']).backward()
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
