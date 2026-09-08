"""Training-only RBG v0 regularizer. No changes to inference networks."""
import torch
from torch import nn


class BlockSIGReg(nn.Module):
    def __init__(self, dim=192, blocks=4, projections=1024, knots=17):
        super().__init__()
        if dim < 1 or blocks < 1 or dim % blocks:
            raise ValueError('dim must be positive and divisible by blocks')
        if projections < blocks or projections % blocks or knots < 2:
            raise ValueError('projections must divide equally; knots >= 2')
        self.dim, self.blocks = dim, blocks
        self.projections = projections // blocks
        t = torch.linspace(0, 3, knots)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt)
        weights[[0, -1]] = dt
        self.register_buffer('t', t)
        self.register_buffer('phi', torch.exp(-t.square() / 2))
        self.register_buffer('weights', weights * self.phi)

    def forward(self, z, directions=None):
        if z.ndim != 3 or z.shape[-1] != self.dim or z.shape[1] < 2:
            raise ValueError('expected (time, batch >= 2, dim)')
        # Statistics stay float32 even when the network uses autocast.
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            chunks = z.chunk(self.blocks, dim=-1)
            terms = []
            for k, x in enumerate(chunks):
                if directions is None:
                    a = torch.randn(x.shape[-1], self.projections, device=x.device)
                    a = a / a.norm(dim=0)
                else:
                    a = directions[k].to(x)
                    if a.shape != (x.shape[-1], self.projections):
                        raise ValueError('invalid projection shape')
                xt = (x @ a).unsqueeze(-1) * self.t
                err = (xt.cos().mean(-3) - self.phi).square() + xt.sin().mean(-3).square()
                terms.append(((err @ self.weights) * x.shape[1]).mean())
            gaussian = torch.stack(terms).mean()
            cross = self.cross_covariance(z)
        return gaussian, cross

    def cross_covariance(self, z):
        chunks = z.chunk(self.blocks, dim=-1)
        centered = [x - x.mean(1, keepdim=True) for x in chunks]
        terms = []
        for i in range(self.blocks):
            for j in range(i + 1, self.blocks):
                cov = centered[i].transpose(1, 2) @ centered[j] / (z.shape[1] - 1)
                terms.append(cov.square().mean())
        return torch.stack(terms).mean() if terms else z.sum() * 0


def one_step_objective(model, pixels, actions, regularizer, gaussian_weight=.09,
                       cross_weight=.01, mode='rbg'):
    """Four observations, three aligned action chunks; causal 3->3 prediction."""
    if pixels.shape[1] != 4 or actions.shape[1] != 3:
        raise ValueError('require four frames and three action chunks')
    z = model.encode({'pixels': pixels})['emb']
    u = model.action_encoder(actions)
    pred = model.predict(z[:, :3], u)
    one = (pred - z[:, 1:]).square().mean()
    source = z.transpose(0, 1)
    if mode == 'tc':
        source = source - source.mean(0, keepdim=True)
    elif mode not in ('raw', 'rbg', 'bt'):
        raise ValueError(mode)
    if mode == 'bt' and cross_weight != 0:
        raise ValueError('BT uses prediction + Gaussian only; cross_weight must be zero')
    gaussian, cross = regularizer(source)
    total = one + gaussian_weight * gaussian + cross_weight * cross
    return total, {'prediction': one, 'gaussian': gaussian, 'cross': cross,
                   'latent_std': z.float().std(dim=0).mean()}
