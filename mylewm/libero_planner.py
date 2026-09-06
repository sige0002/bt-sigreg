"""Shared CEM controller; no learned policy and no privileged state input."""
import torch
import torch.nn.functional as F


def image_tensor(images, device):
    """Native OpenGL RGB pair -> normalized (1,1,2,3,224,224)."""
    x=torch.as_tensor(images.copy(),device=device).permute(0,3,1,2).float()/255
    x=F.interpolate(x,size=(224,224),mode='bilinear',align_corners=False,antialias=True)
    mean=x.new_tensor([.485,.456,.406]).view(1,3,1,1)
    std=x.new_tensor([.229,.224,.225]).view(1,3,1,1)
    return ((x-mean)/std)[None,None]


class CEM:
    def __init__(self,model,horizon=8,samples=128,elites=16,iterations=5,seed=0):
        if not 1<elites<=samples or horizon<1 or iterations<1:
            raise ValueError('Invalid CEM budget')
        self.model=model
        self.device=next(model.parameters()).device
        self.horizon,self.samples,self.elites,self.iterations=horizon,samples,elites,iterations
        self.generator=torch.Generator(device=self.device).manual_seed(seed)
        self.action_mean=model.training_action_mean.float().to(self.device)
        self.action_std=model.training_action_std.float().to(self.device)
        if self.action_mean.numel()!=7 or torch.any(self.action_std<=0):
            raise ValueError('LIBERO requires seven native action dimensions')
        self.mean=torch.zeros(horizon,4,7,device=self.device)

    def normalize(self,raw):
        return ((raw-self.action_mean)/self.action_std).flatten(-2)

    @torch.no_grad()
    def costs(self,history,past_actions,candidates,goal):
        n=candidates.shape[0]
        z=history.expand(n,-1,-1)
        previous=past_actions.expand(n,-1,-1,-1)
        for t in range(candidates.shape[1]):
            actions=torch.cat((previous,candidates[:,t:t+1]),dim=1)
            pred=self.model.predict(z,self.model.action_encoder(self.normalize(actions)))[:,-1:]
            z=torch.cat((z[:,1:],pred),dim=1)
            previous=actions[:,1:]
        return (z[:,-1]-goal.reshape(1,-1)).square().mean(-1)

    @torch.no_grad()
    def plan(self,history,past_actions,goal):
        mean=self.mean
        std=torch.full_like(mean,.6)
        for _ in range(self.iterations):
            candidates=(mean+std*torch.randn((self.samples,*mean.shape),
                         device=self.device,generator=self.generator)).clamp(-1,1)
            candidates[0]=mean
            costs=self.costs(history,past_actions,candidates,goal)
            if not torch.isfinite(costs).all(): raise FloatingPointError('Nonfinite planning costs')
            elite=candidates[costs.topk(self.elites,largest=False).indices]
            mean=.1*mean+.9*elite.mean(0)
            std=(.1*std+.9*elite.std(0,unbiased=False)).clamp_min(.03)
        self.mean=torch.cat((mean[1:],torch.zeros_like(mean[:1])),dim=0)
        return mean[0].clamp(-1,1)
