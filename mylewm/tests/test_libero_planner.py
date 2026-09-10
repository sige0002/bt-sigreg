import numpy as np
import torch
from mylewm.environments.libero_planner import CEM, image_tensor


class Integrator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy=torch.nn.Parameter(torch.zeros(1))
        self.register_buffer('training_action_mean',torch.zeros(7))
        self.register_buffer('training_action_std',torch.ones(7))
    def action_encoder(self,a): return a
    def predict(self,z,a):
        return z+a.reshape(*a.shape[:2],4,7)[...,0].mean(-1,keepdim=True)


def test_planner_future_actions_and_history_alignment():
    planner=CEM(Integrator(),horizon=2,samples=4,elites=2,iterations=1)
    candidates=torch.zeros(2,2,4,7)
    candidates[1,0,:,0]=1
    costs=planner.costs(torch.zeros(1,3,1),torch.zeros(1,2,4,7),candidates,torch.ones(1,1,1))
    torch.testing.assert_close(costs,torch.tensor([1.,0.]))


def test_cem_reproducible_bounded_actions():
    p=CEM(Integrator(),horizon=2,samples=8,elites=2,iterations=2,seed=31)
    q=CEM(Integrator(),horizon=2,samples=8,elites=2,iterations=2,seed=31)
    args=(torch.zeros(1,3,1),torch.zeros(1,2,4,7),torch.ones(1,1,1))
    a=p.plan(*args); b=q.plan(*args)
    torch.testing.assert_close(a,b)
    assert a.shape==(4,7) and a.abs().max()<=1


def test_image_transform_matches_training():
    from mylewm.training.train_libero import preprocess
    rgb=np.random.default_rng(1).integers(0,256,(2,128,128,3),dtype=np.uint8)
    pixels=torch.from_numpy(rgb).permute(0,3,1,2)[None,None]
    x,_=preprocess((pixels,torch.zeros(1,3,4,7)),{'action_mean':[0]*7,'action_std':[1]*7},'cpu')
    torch.testing.assert_close(x,image_tensor(rgb,'cpu'),rtol=0,atol=0)
