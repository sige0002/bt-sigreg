"""Adapter for applying the myLeWM objective to the upstream JEPA model."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from multitask_lewm import cross_task_loss, normalize_latent


class MultiTaskJEPAObjective(nn.Module):
    """Shared-E/A/F objective over task/view batches.

    ``model`` is the official LeWM ``JEPA`` instance.  Inputs are dictionaries
    mapping view names to the same ``{'pixels': [B,T,C,H,W], 'action': ...}``
    structure.  Future targets are encoded observations at each horizon.
    """

    def __init__(self, model: nn.Module, temperature: float = 0.1, cross_weight: float = 1.0, target_detach: bool = False):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.cross_weight = cross_weight
        self.target_detach = target_detach

    def forward(self, queries: dict[str, Tensor], targets: dict[str, Tensor], positive: Tensor | None = None):
        terms = []
        metrics = {}
        for source, q in queries.items():
            for teacher, y in targets.items():
                if source == teacher:
                    continue
                if q.ndim == 3:
                    if y.shape != q.shape:
                        raise ValueError("query/target sequence shapes must match")
                    per_horizon = []
                    horizon_metrics = {}
                    for h in range(q.shape[1]):
                        loss, values = cross_task_loss(q[:, h], y[:, h], positive=positive, temperature=self.temperature, cross_weight=self.cross_weight, target_detach=self.target_detach)
                        per_horizon.append(loss)
                        horizon_metrics[h] = values
                    loss = torch.stack(per_horizon).mean()
                    metrics[f"{source}_to_{teacher}"] = horizon_metrics
                else:
                    loss, values = cross_task_loss(q, y, positive=positive, temperature=self.temperature, cross_weight=self.cross_weight, target_detach=self.target_detach)
                    metrics[f"{source}_to_{teacher}"] = values
                terms.append(loss)
        if not terms:
            raise ValueError("at least two distinct views are required")
        return torch.stack(terms).mean(), metrics

    def loss_from_observations(
        self,
        contexts: dict[str, dict[str, Tensor]],
        futures: dict[str, dict[str, Tensor]],
        horizons: tuple[int, ...] = (1,),
        positive: Tensor | None = None,
        *,
        future_actions: Tensor | None = None,
        history_size: int = 3,
    ):
        """One physical correspondence table, README §11.

        Context pixels: [B,L,C,H,W]; context action: [B,L-1,A].
        future_actions: [B,K,A], beginning at the final context frame.
        Future pixels: [B,K,C,H,W], strictly AFTER the final context frame.
        Horizons are one-based rollout distances, not context indices.
        positive may be [B,B] or [K,B,B]. Separate groups must call this
        method separately so their candidates are never mixed.
        """
        if future_actions is None:
            raise ValueError('explicit future_actions required; old same-frame training is invalid')
        if len(contexts) < 2 or set(contexts) != set(futures):
            raise ValueError('matching context/future sets with >=2 visual conditions required')
        if history_size < 1 or not horizons or min(horizons) < 1:
            raise ValueError('history_size and one-based horizons must be positive')
        if future_actions.ndim != 3 or max(horizons) > future_actions.shape[1]:
            raise ValueError('not enough future action blocks')
        queries: dict[str, Tensor] = {}
        targets: dict[str, Tensor] = {}
        for view, info in contexts.items():
            z = normalize_latent(self.model.encode({'pixels': info['pixels']})['emb'])
            actions = info['action']
            if actions.shape != (z.shape[0], z.shape[1] - 1, future_actions.shape[-1]):
                raise ValueError('context actions must cover exactly L-1 frame intervals')
            predictions = []
            for h in range(max(horizons)):
                actions = torch.cat((actions, future_actions[:, h:h+1]), dim=1)
                act_emb = self.model.action_encoder(actions[:, -history_size:])
                pred = normalize_latent(self.model.predict(z[:, -history_size:], act_emb)[:, -1:])
                z = torch.cat((z, pred), dim=1)
                predictions.append(pred[:, 0])
            queries[view] = torch.stack([predictions[h-1] for h in horizons], dim=1)
        for view, info in futures.items():
            if info['pixels'].shape[1] < max(horizons):
                raise ValueError('not enough strictly future frames')
            selected = info['pixels'][:, [h-1 for h in horizons]]
            targets[view] = normalize_latent(self.model.encode({'pixels': selected})['emb'])
        terms, metrics = [], {}
        for index, h in enumerate(horizons):
            mask = positive[h-1] if positive is not None and positive.ndim == 3 else positive
            loss, values = self({v: q[:, index] for v, q in queries.items()},
                               {v: y[:, index] for v, y in targets.items()}, mask)
            terms.append(loss)
            metrics[h] = values
        return torch.stack(terms).mean(), metrics

    @staticmethod
    def encode_future(model: nn.Module, info: dict[str, Tensor], horizons: tuple[int, ...]):
        """Encode selected future frames with the shared official encoder."""
        pixels = info["pixels"]
        result = {}
        for h in horizons:
            if h < 0 or h >= pixels.shape[1]:
                raise ValueError(f"horizon {h} outside sequence length {pixels.shape[1]}")
            encoded = model.encode({"pixels": pixels[:, h : h + 1]})["emb"][:, 0]
            result[h] = normalize_latent(encoded)
        return result
