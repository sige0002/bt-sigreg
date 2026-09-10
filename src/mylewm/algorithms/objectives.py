"""Full-dimensional Gaussian regularization and native one-step Raw/TC/BT loss."""
import torch
from torch import nn


class GaussianSIGReg(nn.Module):
    """FP32 SIGReg for the shared trainer; preserves its buffer and RNG contract."""
    def __init__(self, dim=192, projections=1024, knots=17):
        super().__init__()
        if dim < 1 or projections < 1 or knots < 2:
            raise ValueError('Positive dimension/projections and knots >= 2 required')
        self.dim, self.projections = dim, projections
        t = torch.linspace(0, 3, knots)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt)
        weights[[0, -1]] = dt
        self.register_buffer('t', t)
        self.register_buffer('phi', torch.exp(-t.square() / 2))
        self.register_buffer('weights', weights * self.phi)

    def forward(self, z):
        if z.ndim != 3 or z.shape[-1] != self.dim or z.shape[1] < 2:
            raise ValueError('expected (time, batch >= 2, dim)')
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            a = torch.randn(self.dim, self.projections, device=z.device)
            a = a / a.norm(dim=0)
            xt = (z @ a).unsqueeze(-1) * self.t
            err = (xt.cos().mean(-3) - self.phi).square() + xt.sin().mean(-3).square()
            return ((err @ self.weights) * z.shape[1]).mean()


def one_step_objective(model, pixels, actions, regularizer, gaussian_weight=.09,
                       mode='raw'):
    """Four observations, three aligned action chunks; causal 3->3 prediction."""
    if mode not in ('raw', 'tc', 'bt'):
        raise ValueError(f'Unsupported mode: {mode}')
    if pixels.shape[1] != 4 or actions.shape[1] != 3:
        raise ValueError('require four frames and three action chunks')
    z = model.encode({'pixels': pixels})['emb']
    u = model.action_encoder(actions)
    pred = model.predict(z[:, :3], u)
    one = (pred - z[:, 1:]).square().mean()
    source = z.transpose(0, 1)
    if mode == 'tc':
        source = source - source.mean(0, keepdim=True)
    gaussian = regularizer(source)
    total = one + gaussian_weight * gaussian
    return total, {'prediction': one, 'gaussian': gaussian,
                   'latent_std': z.float().std(dim=0).mean()}
