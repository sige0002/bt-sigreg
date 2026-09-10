import sys

import gymnasium as gym
import numpy as np
import pytest
import torch

from mylewm.paths import ROOT
from mylewm.evaluation.cached_cem import CachedCEMSolver
from stable_worldmodel.policy import PlanConfig
from stable_worldmodel.solver import CEMSolver


def test_cache_is_scoped_and_restored(monkeypatch):
    sys.path.insert(0, str(ROOT / 'lewm'))
    from jepa import JEPA
    model = JEPA(torch.nn.Identity(), torch.nn.Identity(), torch.nn.Identity()).eval()
    calls = []

    def encode(info):
        calls.append(1)
        info['emb'] = info['pixels'] * 2
        return info

    def cost(info, candidates):
        return model.encode(dict(pixels=info['pixels']))['emb']

    model.encode = encode
    model.get_cost = cost
    solver = CachedCEMSolver(model, n_steps=2)

    def solve(self, info_dict, init_action=None):
        for value in (1., 2.):
            for _ in range(2):
                info = {'pixels': torch.full((1, 3, 1), value)}
                result = self.model.get_cost(info, None)
                assert result.shape == (1, 1, 1)
                assert result.item() == value * 2
        raise RuntimeError('injected failure')

    monkeypatch.setattr(CEMSolver, 'solve', solve)
    for _ in range(2):
        with pytest.raises(RuntimeError, match='injected failure'):
            solver.solve({})
        assert model.encode is encode
        assert model.get_cost is cost
    assert len(calls) == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA checkpoint equivalence')
def test_cached_search_matches_checkpoint():
    checkpoint = ROOT / 'output/pusht/bt_compiled_100k_s3072/step_10000_object.ckpt'
    if not checkpoint.exists():
        pytest.skip('Local trusted 10k checkpoint unavailable')
    sys.path.insert(0, str(ROOT / 'lewm'))
    model = torch.load(checkpoint, map_location='cuda', weights_only=False).eval()
    gen = torch.Generator().manual_seed(42)
    info = {'pixels': torch.randn(2, 1, 3, 224, 224, generator=gen),
            'goal': torch.randn(2, 1, 3, 224, 224, generator=gen),
            'action': torch.zeros(2, 1, 10)}
    outputs = []
    import time
    for cls in (CEMSolver, CachedCEMSolver):
        solver = cls(model=model, device='cuda', num_samples=300, n_steps=30, topk=30, seed=42)
        solver.configure(action_space=gym.spaces.Box(-1, 1, (2, 2)), n_envs=2,
                         config=PlanConfig(horizon=5, receding_horizon=5, action_block=5))
        start = time.perf_counter()
        outputs.append(solver(info))
        print(cls.__name__, 'wall_seconds', time.perf_counter() - start)
    torch.testing.assert_close(outputs[0]['actions'], outputs[1]['actions'], rtol=0, atol=0)
    np.testing.assert_array_equal(outputs[0]['costs'], outputs[1]['costs'])
