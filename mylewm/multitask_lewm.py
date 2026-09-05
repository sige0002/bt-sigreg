"""Minimal multi-task objective for LeWM.

This module is deliberately independent of the upstream checkout.  It accepts
predicted and observed future embeddings, so it can be used with the official
``JEPA`` encoder/predictor without modifying that implementation.
"""

from __future__ import annotations

import torch
from torch import Tensor


def normalize_latent(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalize the last dimension while keeping exact zero vectors zero."""
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def cross_task_loss(
    query: Tensor,
    target: Tensor,
    positive: Tensor | None = None,
    temperature: float = 0.1,
    prediction_weight: float = 1.0,
    cross_weight: float = 1.0,
    normalize: bool = True,
    diagonal_positive: bool = True,
    target_detach: bool = False,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute matched future prediction plus all-candidate identification.

    Args:
        query: predicted future embeddings, ``[B, D]``.
        target: observed future embeddings, ``[B, D]``; row ``b`` matches query
            row ``b``.
        positive: optional ``[B, B]`` mask.  ``positive[b, c]`` marks target
            ``c`` as a valid physical-equivalence positive for query ``b``.
            The diagonal is added automatically.
        temperature: positive scalar softmax temperature.
        normalize: apply unit-length normalization before both terms.

    Returns ``(total, metrics)``.  ``metrics`` are detached scalar tensors.
    """
    if query.ndim != 2 or target.ndim != 2 or query.shape != target.shape:
        raise ValueError(f"query and target must both be [B,D], got {query.shape} and {target.shape}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if prediction_weight < 0 or cross_weight < 0:
        raise ValueError("loss weights must be non-negative")
    b = query.shape[0]
    if b == 0:
        raise ValueError("empty candidate set")
    q = normalize_latent(query) if normalize else query
    y = target.detach() if target_detach else target
    y = normalize_latent(y) if normalize else y
    distances = (q[:, None, :] - y[None, :, :]).square().sum(-1)
    logits = -distances / temperature
    positives = torch.eye(b, dtype=torch.bool, device=query.device) if diagonal_positive else torch.zeros((b, b), dtype=torch.bool, device=query.device)
    if positive is not None:
        if positive.shape != (b, b):
            raise ValueError(f"positive must be [{b},{b}], got {positive.shape}")
        positives |= positive.to(device=query.device, dtype=torch.bool)
    if not positives.any(dim=1).all():
        raise ValueError("each query must have at least one positive target")
    numerator = torch.logsumexp(logits.masked_fill(~positives, -torch.inf), dim=1)
    cross = -(numerator - torch.logsumexp(logits, dim=1)).mean()
    prediction = (q - y).square().sum(-1).mean()
    total = prediction_weight * prediction + cross_weight * cross
    return total, {"prediction": prediction.detach(), "cross": cross.detach()}


def loss_over_views(
    queries: dict[str, Tensor],
    targets: dict[str, Tensor],
    positive: Tensor | None = None,
    **kwargs,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Average the objective over ordered distinct observation-view pairs.

    Tensors are ``[B, D]`` for one horizon.  The caller can invoke this once
    per horizon and average horizons externally, or pass a flattened horizon.
    """
    terms = []
    stats: dict[str, Tensor] = {}
    views = list(queries)
    if len(views) < 2 or set(views) != set(targets):
        raise ValueError("queries and targets must contain the same >=2 views")
    for source in views:
        for teacher in views:
            if source == teacher:
                continue
            term, values = cross_task_loss(
                queries[source], targets[teacher], positive=positive, **kwargs
            )
            terms.append(term)
            for name, value in values.items():
                stats[f"{source}_to_{teacher}/{name}"] = value
    total = torch.stack(terms).mean()
    stats["total"] = total.detach()
    return total, stats
