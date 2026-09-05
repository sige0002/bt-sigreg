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

    def __init__(self, model: nn.Module, temperature: float = 0.1, cross_weight: float = 1.0, target_detach: bool = True):
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
    ):
        """Run the official LeWM E/A/F and compute the multi-task loss.

        Each view dictionary contains ``pixels`` of shape ``[B,T,C,H,W]`` and
        ``action`` of shape ``[B,T,A]``.  ``contexts[v]`` supplies the history
        and actions used by the predictor; ``futures[v]`` supplies the observed
        future sequence.  The predictor output at each context time is matched
        to the corresponding future frame.  Horizon selection is applied after
        the model output, keeping all view/task branches on the shared model.
        """
        queries: dict[str, Tensor] = {}
        targets: dict[str, Tensor] = {}
        for view, info in contexts.items():
            encoded = self.model.encode({k: v for k, v in info.items()})
            if "action" not in encoded:
                raise ValueError("context must contain action")
            pred = self.model.predict(encoded["emb"], encoded["act_emb"])
            queries[view] = pred[:, list(horizons)]
        for view, info in futures.items():
            encoded = self.model.encode({"pixels": info["pixels"]})
            targets[view] = encoded["emb"][:, list(horizons)]
        return self(queries, targets, positive=positive)

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
