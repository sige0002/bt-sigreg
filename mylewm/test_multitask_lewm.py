import torch

from multitask_lewm import cross_task_loss, loss_over_views, normalize_latent
from multitask_jepa import MultiTaskJEPAObjective


class DummyJEPA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def encode(self, info):
        x = info["pixels"].float().mean(dim=tuple(range(2, info["pixels"].ndim)))
        info["emb"] = (x * self.scale).unsqueeze(-1).repeat(1, 1, 4)
        if "action" in info:
            info["act_emb"] = info["action"].float()[..., :1].repeat(1, 1, 2)
        return info

    def predict(self, emb, act_emb):
        return emb + act_emb[..., :1]


def test_cross_loss_backprop_and_multi_positive():
    torch.manual_seed(0)
    q = torch.randn(4, 8, requires_grad=True)
    y = torch.randn(4, 8, requires_grad=True)
    positive = torch.eye(4, dtype=torch.bool)
    positive[0, 1] = positive[1, 0] = True
    loss, metrics = cross_task_loss(q, y, positive=positive, temperature=0.2)
    assert torch.isfinite(loss)
    assert set(metrics) == {"prediction", "cross"}
    loss.backward()
    assert q.grad is not None and y.grad is not None


def test_zero_vector_and_view_average():
    z = normalize_latent(torch.zeros(2, 3))
    assert torch.equal(z, torch.zeros_like(z))
    q = {"bright": torch.randn(3, 5, requires_grad=True), "dark": torch.randn(3, 5, requires_grad=True)}
    y = {"bright": torch.randn(3, 5, requires_grad=True), "dark": torch.randn(3, 5, requires_grad=True)}
    loss, stats = loss_over_views(q, y, temperature=0.3)
    assert torch.isfinite(loss) and "total" in stats
    loss.backward()


def test_explicit_positive_and_target_detach():
    q = torch.randn(2, 4, requires_grad=True)
    y = torch.randn(2, 4, requires_grad=True)
    mask = torch.tensor([[True, False], [False, True]])
    loss, _ = cross_task_loss(q, y, positive=mask, diagonal_positive=False, target_detach=True)
    loss.backward()
    assert y.grad is None


def test_official_jepa_adapter_runs_sequence():
    model = DummyJEPA()
    objective = MultiTaskJEPAObjective(model, target_detach=True)
    contexts = {v: {"pixels": torch.randn(2, 3, 1, 2, 2), "action": torch.randn(2, 3, 2)} for v in ("a", "b")}
    futures = {v: {"pixels": torch.randn(2, 3, 1, 2, 2)} for v in ("a", "b")}
    loss, _ = objective.loss_from_observations(contexts, futures, horizons=(1, 2))
    assert torch.isfinite(loss)
    loss.backward()
