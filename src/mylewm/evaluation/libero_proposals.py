"""Diagnostic-only action proposal controls; production CEM is unchanged."""
import h5py
import numpy as np
import torch
from mylewm.environments.libero_planner import CEM


def fit_temporal_correlation(manifest):
    """Fit global lag-one correlation using training episodes only."""
    mean, std = np.asarray(manifest['action_mean']), np.asarray(manifest['action_std'])
    cross, left, right = (np.zeros(7) for _ in range(3))
    pairs = 0
    for task, item in enumerate(manifest['files']):
        with h5py.File(item['path'], 'r') as f:
            for case in manifest['train_demos']:
                if case['task'] != task:
                    continue
                x = (np.asarray(f[f'data/{case["demo"]}/actions'], dtype=np.float64)-mean)/std
                cross += (x[:-1]*x[1:]).sum(0)
                left += (x[:-1]**2).sum(0)
                right += (x[1:]**2).sum(0)
                pairs += len(x)-1
    rho = np.clip(cross/np.maximum(np.sqrt(left*right), 1e-12), 0, .99)
    return {'rho':rho.tolist(), 'adjacent_pairs':pairs, 'training_demos':len(manifest['train_demos']),
            'fit':'training episodes only; global normalized lag-one correlation clipped to [0, .99]'}


def correlated_noise(shape, generator, device, rho):
    x = torch.randn(shape, generator=generator, device=device)
    flat = x.reshape(shape[0], -1, 7)
    rho = torch.as_tensor(rho, device=device, dtype=x.dtype)
    scale = (1-rho.square()).sqrt()
    for t in range(1, flat.shape[1]):
        flat[:, t] = rho*flat[:, t-1] + scale*flat[:, t]
    return x


class ProposalCEM(CEM):
    def __init__(self, model, *, scaled=False, rho=None, unscored=False, **kwargs):
        super().__init__(model, **kwargs)
        self.scaled, self.rho, self.unscored = scaled, rho, unscored
        self.center = self.action_mean if scaled else torch.zeros_like(self.action_mean)
        self.initial_std = self.action_std if scaled else torch.full_like(self.action_std, .6)
        self.mean = self.center.expand_as(self.mean).clone()

    @torch.no_grad()
    def plan(self, history, past_actions, goal):
        mean = self.mean
        std = self.initial_std.expand_as(mean).clone()
        floor = .03*self.initial_std/.6
        for _ in range(self.iterations):
            noise = correlated_noise((self.samples, *mean.shape), self.generator, self.device,
                                     self.rho if self.rho is not None else [0.]*7)
            candidates = (mean+std*noise).clamp(-1, 1)
            candidates[0] = mean
            if self.unscored:
                # Same proposal and averaging budget; selection never uses the model.
                elite = candidates[:self.elites]
            else:
                costs = self.costs(history, past_actions, candidates, goal)
                if not torch.isfinite(costs).all():
                    raise FloatingPointError('Nonfinite planning costs')
                elite = candidates[costs.topk(self.elites, largest=False).indices]
            mean = .1*mean+.9*elite.mean(0)
            std = torch.maximum(.1*std+.9*elite.std(0, unbiased=False), floor)
        self.final_sequence = mean.clone()
        self.mean = torch.cat((mean[1:], self.center.expand_as(mean[:1])), dim=0)
        return mean[0].clamp(-1, 1)
