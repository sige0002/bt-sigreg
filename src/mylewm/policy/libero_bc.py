"""Frozen two-view ViT + task-conditioned flow-matching behavior cloning.

Based on TC-LeWM v3 Appendix A.1; local implementation, not official code.
The world-model predictor, projector and SIGReg transport are not used here.
"""
import math
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from mylewm.data.libero_bc_data import preprocess_images


@dataclass(frozen=True)
class BCConfig:
    horizon: int = 8
    width: int = 256
    depth: int = 4
    heads: int = 8
    euler_steps: int = 10

    def __post_init__(self):
        if any(type(x) is not int or x < 1 for x in asdict(self).values()):
            raise ValueError('BC dimensions and Euler steps must be positive integers')
        if self.width % self.heads or self.width < 4 or self.width % 2:
            raise ValueError('Require even BC width divisible by attention heads')


class FlowBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(width, elementwise_affine=False) for _ in range(3)])
        self.self_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))

    def forward(self, x, visual, condition):
        parts = self.modulation(condition).chunk(6, dim=-1)
        def norm(i):
            return self.norms[i](x) * (1 + parts[2 * i + 1][:, None]) + parts[2 * i][:, None]
        q = norm(0)
        x = x + self.self_attention(q, q, q, need_weights=False)[0]
        x = x + self.cross_attention(norm(1), visual, visual, need_weights=False)[0]
        return x + self.mlp(norm(2))


class FlowHead(nn.Module):
    def __init__(self, feature_dim, tasks, config):
        super().__init__()
        self.config = config
        d = config.width
        self.action = nn.Linear(7, d)
        self.visual = nn.Linear(feature_dim, d)
        self.action_position = nn.Parameter(torch.randn(1, config.horizon, d) * .02)
        self.visual_position = nn.Parameter(torch.randn(1, 34, d) * .02)
        self.task = nn.Embedding(tasks, d)
        self.time = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([FlowBlock(d, config.heads) for _ in range(config.depth)])
        self.output = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 7))

    def forward(self, noisy_actions, time, visual, task_ids):
        half = self.config.width // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=time.device) / max(half - 1, 1))
        angles = time[:, None] * frequencies[None] * 1000
        condition = self.time(torch.cat((angles.cos(), angles.sin()), -1)) + self.task(task_ids)
        x = self.action(noisy_actions) + self.action_position
        visual = self.visual(visual) + self.visual_position
        for block in self.blocks:
            x = block(x, visual, condition)
        return self.output(x)


class LiberoBCPolicy(nn.Module):
    def __init__(self, encoder, task_names, action_mean, action_std, config=BCConfig()):
        super().__init__()
        if not task_names or len(set(task_names)) != len(task_names):
            raise ValueError('Require unique task names')
        self.config, self.task_names = config, tuple(task_names)
        self.encoder = encoder.requires_grad_(False).eval()
        self.head = FlowHead(encoder.config.hidden_size, len(task_names), config)
        self.register_buffer('action_mean', torch.as_tensor(action_mean, dtype=torch.float32).clone())
        self.register_buffer('action_std', torch.as_tensor(action_std, dtype=torch.float32).clone())
        if self.action_mean.shape != (7,) or self.action_std.shape != (7,):
            raise ValueError('BC requires seven action dimensions')
        if not torch.isfinite(self.action_mean).all() or not torch.isfinite(self.action_std).all() or (self.action_std <= 0).any():
            raise ValueError('Invalid action normalization')

    @property
    def device(self):
        return self.action_mean.device

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()  # Never enable dropout or update the frozen backbone.
        return self

    @torch.no_grad()
    def features(self, images):
        if images.ndim != 5 or images.shape[1:3] != (2, 3) or images.shape[-2:] != (224, 224):
            raise ValueError('Expected normalized (batch,2,3,224,224) images')
        b = images.shape[0]
        tokens = self.encoder(images.flatten(0, 1), interpolate_pos_encoding=True).last_hidden_state
        patches = tokens[:, 1:]
        side = math.isqrt(patches.shape[1])
        if side * side != patches.shape[1] or side < 4:
            raise ValueError('Require a square ViT patch grid of at least 4x4')
        grid = patches.transpose(1, 2).reshape(b * 2, -1, side, side)
        pooled = F.adaptive_avg_pool2d(grid, (4, 4)).flatten(2).transpose(1, 2)
        return torch.cat((tokens[:, :1], pooled), 1).reshape(b, 34, -1)

    def validate_tasks(self, ids, batch):
        if ids.shape != (batch,) or ids.dtype != torch.long or (ids < 0).any() or (ids >= len(self.task_names)).any():
            raise ValueError('Invalid BC task IDs; map native task names to checkpoint order')

    def loss(self, images, actions, task_ids, generator=None):
        self.validate_tasks(task_ids, len(images))
        if actions.shape != (len(images), self.config.horizon, 7) or not torch.isfinite(actions).all():
            raise ValueError('Invalid BC action targets')
        visual = self.features(images)
        target = (actions - self.action_mean) / self.action_std
        noise = torch.randn(target.shape, device=target.device, generator=generator)
        time = torch.rand(len(target), device=target.device, generator=generator)
        noisy = (1 - time[:, None, None]) * noise + time[:, None, None] * target
        velocity = self.head(noisy, time, visual, task_ids)
        return F.mse_loss(velocity.float(), target - noise)

    @torch.no_grad()
    def sample(self, images, task_ids, generator=None):
        self.validate_tasks(task_ids, len(images))
        visual = self.features(images)
        x = torch.randn((len(images), self.config.horizon, 7), device=images.device, generator=generator)
        for step in range(self.config.euler_steps):
            time = x.new_full((len(x),), step / self.config.euler_steps)
            x = x + self.head(x, time, visual, task_ids) / self.config.euler_steps
        actions = x * self.action_std + self.action_mean
        if not torch.isfinite(actions).all():
            raise FloatingPointError('Nonfinite BC actions')
        return actions.clamp(-1, 1)

    def bundle(self, provenance, step):
        return {'schema': 'libero_flow_bc_v1', 'config': asdict(self.config),
                'encoder_config': self.encoder.config.to_dict(),
                'encoder_pooler': self.encoder.pooler is not None,
                'encoder_mask_token': self.encoder.embeddings.mask_token is not None,
                'task_names': list(self.task_names), 'state_dict': self.state_dict(),
                'provenance': provenance, 'step': step}


def load_bc_checkpoint(path, device='cpu'):
    from transformers import ViTConfig, ViTModel
    bundle = torch.load(path, map_location='cpu', weights_only=True)
    if bundle.get('schema') != 'libero_flow_bc_v1':
        raise ValueError('Expected a LIBERO BC checkpoint, not a world-model checkpoint')
    config = ViTConfig.from_dict(bundle['encoder_config'])
    encoder = ViTModel(config, add_pooling_layer=bundle['encoder_pooler'],
                       use_mask_token=bundle['encoder_mask_token'])
    state = bundle['state_dict']
    policy = LiberoBCPolicy(encoder, bundle['task_names'], state['action_mean'],
                            state['action_std'], BCConfig(**bundle['config']))
    policy.load_state_dict(state, strict=True)
    return policy.to(device).eval(), bundle


class BCController:
    """Task-name mapped action queue; fresh observations at every chunk boundary."""
    def __init__(self, policy, task_name, seed=42, execute_actions=None):
        self.policy = policy.eval()
        try:
            self.task_id = policy.task_names.index(task_name)
        except ValueError as e:
            raise ValueError(f'Task absent from BC checkpoint: {task_name}') from e
        self.execute_actions = policy.config.horizon if execute_actions is None else execute_actions
        if not 1 <= self.execute_actions <= policy.config.horizon:
            raise ValueError('Execution chunk must fit the learned action horizon')
        self.generator = torch.Generator(device=policy.device).manual_seed(seed)
        self.queue = deque()
        self.chunks = 0

    def select_action(self, camera_pair):
        if not self.queue:
            pixels = torch.as_tensor(np.asarray(camera_pair).copy()).permute(0, 3, 1, 2)[None]
            images = preprocess_images(pixels, self.policy.device)
            tasks = torch.tensor([self.task_id], device=self.policy.device)
            actions = self.policy.sample(images, tasks, self.generator)[0, :self.execute_actions]
            self.queue.extend(actions.cpu().numpy())
            self.chunks += 1
        return self.queue.popleft().copy()
