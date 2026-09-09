import json
from types import SimpleNamespace

import gymnasium as gym
import torch

from mylewm.cem_audit import AuditedCEMSolver


class Cost(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.weight = torch.nn.Parameter(torch.zeros(()))
    def get_cost(self, info, candidates):
        return candidates.square().sum((-1, -2))


def test_audited_solver_preserves_outputs_and_records(monkeypatch, tmp_path):
    path = tmp_path/'audit.jsonl'
    monkeypatch.setenv('BT_CEM_AUDIT_PATH', str(path))
    kwargs = dict(model=Cost(), batch_size=1, num_samples=4, n_steps=2, topk=2,
                  device='cpu', seed=7)
    audited = AuditedCEMSolver(**kwargs)
    plain = __import__('stable_worldmodel').solver.CEMSolver(**kwargs)
    config = SimpleNamespace(horizon=2, action_block=1)
    space = gym.spaces.Box(-1, 1, shape=(1, 2), dtype=float)
    for solver in (audited, plain): solver.configure(action_space=space, n_envs=1, config=config)
    info = {'goal': torch.ones(1, 3, 1), 'pixels': torch.zeros(1, 3, 1)}
    actual, expected = audited(info), plain(info)
    torch.testing.assert_close(actual['actions'], expected['actions'])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1 and len(rows[0]['iterations']) == 2
    assert rows[0]['replan'] == 0 and rows[0]['selected_predicted_cost'] >= 0
