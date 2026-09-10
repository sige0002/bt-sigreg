"""Shared checkpoint/LeRobot CEM runtime in physical action coordinates."""
from collections import deque
import copy
import math

import torch

from mylewm.data.contract import model_contract, inference_manifest


def validate_planner(value, dim):
    c = copy.deepcopy(value)
    required = {'horizon', 'samples', 'elites', 'iterations', 'seed', 'action_low', 'action_high'}
    if set(c) != required:
        raise ValueError(f'Planner requires exactly {sorted(required)}')
    for key in ('horizon', 'samples', 'elites', 'iterations', 'seed'):
        if type(c[key]) is not int or c[key] < (0 if key == 'seed' else 1):
            raise ValueError(f'Invalid planner {key}')
    if not 2 <= c['elites'] <= c['samples']:
        raise ValueError('Require 2 <= elites <= samples')
    lo, hi = torch.tensor(c['action_low']), torch.tensor(c['action_high'])
    if lo.shape != (dim,) or hi.shape != (dim,) or not torch.isfinite(lo).all() or not torch.isfinite(hi).all() or not (lo < hi).all():
        raise ValueError('Require finite per-component physical action bounds')
    return c


class WorldModelController:
    """Single robot, explicit real history, one physical command per input tick.

    prime() supplies three actual images and the two executed action chunks
    between them. select_action() is then called once per physical control tick.
    No fabricated history, implicit unit conversion, or robot I/O.
    """
    def __init__(self, model, planner, legacy_contract=None):
        self.model = model.eval()
        self.contract = model_contract(model, legacy_contract)
        stats = inference_manifest(model, {'input_contract': self.contract}, legacy_contract)
        self.planner = validate_planner(planner, len(self.contract['action_names']))
        self.k = self.contract['frameskip']
        self.dim = len(self.contract['action_names'])
        from mylewm.training.train import Preprocess
        self.preprocess = Preprocess(dict(stats, frameskip=self.k))
        self.reset()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def reset(self):
        self.generator = torch.Generator(device=self.device).manual_seed(self.planner['seed'])
        self.history = self.past_actions = self.goal = None
        self.queue = deque()
        self.executed = []
        self.last_action = None
        self.last_timestamp = None
        self.initial_timestamp = None
        self.last_cost = None
        self.prime_image = None

    def images(self, value):
        x = torch.as_tensor(value).detach().to('cpu')
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError('Images must have shape (T,3,H,W)')
        if x.dtype != torch.uint8:
            if not x.is_floating_point() or not torch.isfinite(x).all() or (x < 0).any() or (x > 1).any():
                raise ValueError('RGB must be uint8 or unnormalized float in [0,1]')
            x = (x * 255).round().to(torch.uint8)
        return self.preprocess.image({'pixels': x})['pixels'].contiguous().to(self.device)

    @torch.inference_mode()
    def prime(self, images, past_actions, goal, timestamps):
        self.reset()
        times = torch.as_tensor(timestamps, dtype=torch.float64)
        if times.shape != (3,) or not torch.isfinite(times).all() or not torch.allclose(
                times[1:] - times[:-1], torch.full((2,), self.k / self.contract['fps'], dtype=torch.float64), atol=1e-4, rtol=0):
            raise ValueError('Prime requires three actual timestamps spaced frameskip/fps apart')
        x = self.images(images)
        g = self.images(goal)
        a = torch.as_tensor(past_actions, device=self.device, dtype=torch.float32)
        if x.shape[0] != 3 or g.shape[0] != 1 or a.shape != (2, self.k, self.dim) or not torch.isfinite(a).all():
            raise ValueError('Prime requires 3 images, 2 executed action chunks and 1 goal image')
        self.history = self.model.encode({'pixels': x[None]})['emb'].detach()
        self.prime_image = x[-1:].clone()
        self.goal = self.model.encode({'pixels': g[None]})['emb'][:, -1].detach()
        self.past_actions = a[None].clone()
        self.initial_timestamp = float(times[-1])

    @torch.inference_mode()
    def costs(self, candidates):
        n = len(candidates)
        z = self.history.expand(n, -1, -1)
        past = self.past_actions.expand(n, -1, -1, -1)
        mean = self.model.training_action_mean.to(device=self.device, dtype=torch.float64)
        std = self.model.training_action_std.to(device=self.device, dtype=torch.float64)
        for t in range(candidates.shape[1]):
            actions = torch.cat((past, candidates[:, t:t + 1]), 1)
            normalized = ((actions.double() - mean) / std).float().flatten(-2)
            next_z = self.model.predict(z, self.model.action_encoder(normalized))[:, -1:]
            z = torch.cat((z[:, 1:], next_z), 1)
            past = actions[:, 1:]
        return (z[:, -1] - self.goal).square().mean(-1)

    @torch.inference_mode()
    def plan(self):
        if self.history is None:
            raise ValueError('Call prime with actual history before planning')
        p = self.planner
        low = torch.tensor(p['action_low'], device=self.device)
        high = torch.tensor(p['action_high'], device=self.device)
        shape = (p['horizon'], self.k, self.dim)
        mean = ((low + high) / 2).expand(shape).clone()
        std = ((high - low) / 2).expand(shape).clone()
        for _ in range(p['iterations']):
            candidates = mean + std * torch.randn((p['samples'], *shape), device=self.device, generator=self.generator)
            candidates = candidates.clamp(low, high)
            candidates[0] = mean
            costs = self.costs(candidates)
            if not torch.isfinite(costs).all():
                raise ValueError('Non-finite CEM cost')
            elite = candidates[costs.topk(p['elites'], largest=False).indices]
            mean = elite.mean(0)
            std = elite.std(0, unbiased=False).clamp_min((high - low) * .01)
        # Return an evaluated candidate; the elite mean need not have low cost.
        best = costs.argmin()
        self.last_cost = float(costs[best])
        return candidates[best, 0].clone()

    @torch.inference_mode()
    def select_action(self, image, timestamp, executed_action=None):
        if self.history is None:
            raise ValueError('Call prime with actual history before selecting actions')
        if not math.isfinite(timestamp):
            raise ValueError('Invalid observation timestamp')
        expected = self.initial_timestamp if self.last_timestamp is None else self.last_timestamp + 1 / self.contract['fps']
        if abs(timestamp - expected) > 1e-4:
            raise ValueError('Observation timing differs from the training FPS; reset and prime again')
        if self.last_action is None and not torch.equal(self.images(image), self.prime_image):
            raise ValueError('First observation must match the final priming image')
        if self.last_action is not None:
            if executed_action is None:
                raise ValueError('Report the actually executed previous action')
            executed = torch.as_tensor(executed_action, device=self.device, dtype=torch.float32)
            if executed.shape != (self.dim,) or not torch.isfinite(executed).all():
                raise ValueError('Invalid executed action')
            self.executed.append(executed.clone())
        elif executed_action is not None:
            raise ValueError('No previous action to acknowledge after prime')
        if not self.queue:
            if self.last_action is not None:
                if len(self.executed) != self.k:
                    raise ValueError('Incomplete executed action chunk')
                x = self.images(image)
                if x.shape[0] != 1:
                    raise ValueError('Expected one current image')
                z = self.model.encode({'pixels': x[None]})['emb']
                self.history = torch.cat((self.history[:, 1:], z), 1)
                chunk = torch.stack(self.executed)[None, None]
                self.past_actions = torch.cat((self.past_actions[:, 1:], chunk), 1)
                self.executed = []
            self.queue.extend(self.plan().unbind(0))
        self.last_timestamp = timestamp
        self.last_action = self.queue.popleft().clone()
        return self.last_action.clone()


def load_world_model(checkpoint, device='cpu', legacy_contract=None):
    # Existing object exports contain Python objects: trusted local inputs only.
    from mylewm.training import train  # register the existing LeWM import paths
    model = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(model, dict) and 'recipe' in model and 'state_dict' in model:
        contract = model['recipe'].get('input_contract') or legacy_contract
        if contract is None:
            raise ValueError('Training checkpoint lacks input contract; supply verified training conditions')
        from mylewm.data.contract import validate_contract
        contract = validate_contract(contract)
        state = {key.removeprefix('model.'): value for key, value in model['state_dict'].items()
                 if key.startswith('model.')}
        with torch.random.fork_rng(devices=[]):
            model = train.build_model(len(contract['action_names']) * contract['frameskip'])
        for name in ('mean', 'std'):
            model.register_buffer('training_action_' + name, torch.zeros(len(contract['action_names']), dtype=torch.float64))
        model.load_state_dict(state, strict=True)
        model.input_contract = contract
        model.to(device)
    if not isinstance(model, torch.nn.Module):
        raise ValueError('Require a trusted object export or new-recipe Lightning checkpoint')
    model_contract(model, legacy_contract)
    return model.eval()


def load_checkpoint_controller(checkpoint, planner, device='cpu', legacy_contract=None):
    model = load_world_model(checkpoint, device, legacy_contract)
    return WorldModelController(model, planner, legacy_contract)
