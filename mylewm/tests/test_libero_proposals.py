import h5py
import numpy as np
import torch
from mylewm.environments.libero_planner import CEM
from mylewm.evaluation.libero_proposals import ProposalCEM, correlated_noise, fit_temporal_correlation
from test_libero_planner import Integrator


def test_baseline_matches_production_across_replans():
    p, q = [cls(Integrator(), samples=16, elites=4, iterations=3, seed=4) for cls in (CEM, ProposalCEM)]
    args = (torch.zeros(1,3,1), torch.zeros(1,2,4,7), torch.ones(1,1,1))
    for _ in range(3):
        torch.testing.assert_close(p.plan(*args), q.plan(*args), rtol=0, atol=0)
        torch.testing.assert_close(p.mean, q.mean, rtol=0, atol=0)


def test_smoothing_preserves_marginal_scale():
    x = correlated_noise((4096,8,4,7), torch.Generator().manual_seed(4), 'cpu', [.8]*7).flatten(1,2)
    assert abs(x.std().item()-1) < .02
    assert .75 < (x[:,:-1]*x[:,1:]).mean().item() < .85


def test_fit_excludes_heldout_and_episode_boundaries(tmp_path):
    p = tmp_path/'data.hdf5'
    with h5py.File(p,'w') as f:
        f['data/a/actions'] = np.ones((3,7))
        f['data/b/actions'] = -np.ones((3,7))
        f['data/heldout/actions'] = np.tile([[10.]*7, [-10.]*7],(10,1))
    m = {'action_mean':[0.]*7,'action_std':[1.]*7,'files':[{'path':str(p)}],
         'train_demos':[{'task':0,'demo':'a'},{'task':0,'demo':'b'}]}
    result = fit_temporal_correlation(m)
    assert result['adjacent_pairs'] == 4
    assert result['rho'] == [.99]*7


def test_unscored_never_calls_world_model():
    p = ProposalCEM(Integrator(), scaled=True, rho=[.8]*7, unscored=True)
    def fail(*args):
        raise AssertionError('Unscored control called model')
    p.costs = fail
    assert p.plan(None,None,None).abs().max() <= 1
