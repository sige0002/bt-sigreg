import json
from types import SimpleNamespace

import gymnasium as gym
import torch
import weakref

from mylewm.evaluation.cem_audit import AuditedCEMSolver


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


def test_multiple_cases_release_mutated_model_inputs_and_score_original_history(monkeypatch, tmp_path):
    class MutatingCost(Cost):
        previous = None

        def get_cost(self, info, candidates):
            # Model scoring must not keep previous expanded image storage alive.
            assert self.previous is None or self.previous() is None
            history = info['action'].mean()
            info['pixels'] = info['pixels'].clone()
            self.previous = weakref.ref(info['pixels'])
            info['action'] = candidates[:, :, :1]
            return (candidates-history).square().sum((-1, -2))

    monkeypatch.setenv('BT_CEM_AUDIT_PATH', str(tmp_path/'audit.jsonl'))
    kwargs = dict(batch_size=1, num_samples=4, n_steps=3, topk=2, device='cpu', seed=7)
    audited = AuditedCEMSolver(model=MutatingCost(), **kwargs)
    plain = __import__('stable_worldmodel').solver.CEMSolver(model=MutatingCost(), **kwargs)
    config = SimpleNamespace(horizon=2, action_block=1)
    space = gym.spaces.Box(-1, 1, shape=(3, 2), dtype=float)
    for solver in (audited, plain):
        solver.configure(action_space=space, n_envs=3, config=config)
    info = {'goal': torch.arange(3.).reshape(3, 1, 1),
            'pixels': torch.zeros(3, 1, 3, 16, 16), 'action': torch.full((3, 1, 2), .25)}
    actual, expected = audited(info), plain(info)
    torch.testing.assert_close(actual['actions'], expected['actions'])
    rows = [json.loads(line) for line in (tmp_path/'audit.jsonl').read_text().splitlines()]
    assert len(rows) == 3
    for i, row in enumerate(rows):
        assert abs(row['selected_predicted_cost'] - float((actual['actions'][i]-.25).square().sum())) < 1e-5
        assert row['iterations'][0]['candidate_before_normalization']['count'] > 0
