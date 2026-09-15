import numpy as np
import pytest
import torch
from mylewm.environments.libero_planner import CEM
from mylewm.evaluation.diagnose_libero_model import RecordedCEM, rollout, shuffled_actions, windows, rank_correlation


class Integrator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer('training_action_mean', torch.zeros(7))
        self.register_buffer('training_action_std', torch.ones(7))
    def action_encoder(self, a):
        return a
    def predict(self, z, a):
        return z + a.reshape(*a.shape[:2], 4, 7)[..., 0].sum(-1, keepdim=True)


def test_rollout_uses_correct_future_actions_and_exposes_all_endpoints():
    p = CEM(Integrator(), horizon=2, samples=4, elites=2)
    history = torch.tensor([[[1.], [3.], [5.]]])
    past = torch.ones(1, 2, 4, 7) * 100
    actions = torch.zeros(2, 2, 4, 7)
    actions[0, 0, :, 0] = 2
    actions[0, 1, :, 0] = 3
    predictions = rollout(p, history, past, actions)
    torch.testing.assert_close(predictions[:, :, 0], torch.tensor([[13., 25.], [5., 5.]]))
    target = torch.tensor([[[25.]]])
    torch.testing.assert_close((predictions[:, -1]-25).square().mean(-1), p.costs(history, past, actions, target))


def test_recording_does_not_change_planner_or_rng():
    model = Integrator()
    p = CEM(model, horizon=2, samples=8, elites=2, iterations=3, seed=99)
    q = RecordedCEM(model, horizon=2, samples=8, elites=2, iterations=3, seed=99)
    args = (torch.zeros(1, 3, 1), torch.zeros(1, 2, 4, 7), torch.ones(1, 1, 1))
    for _ in range(2):
        a = p.plan(*args); b = q.plan(*args)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(b, q.final_sequence[0], rtol=0, atol=0)
        torch.testing.assert_close(p.mean, q.mean, rtol=0, atol=0)


def test_shuffle_preserves_complete_actions_not_independent_components():
    actions = torch.arange(56).reshape(2, 4, 7).float()
    shuffled = shuffled_actions(actions, torch.Generator().manual_seed(1), count=4)
    for s in shuffled:
        np.testing.assert_array_equal(sorted(map(tuple, s.reshape(-1, 7).numpy())), sorted(map(tuple, actions.reshape(-1, 7).numpy())))
    assert not torch.equal(shuffled[0], actions)


def test_windows_keep_real_history_and_future_in_same_demo():
    for n in (41, 100, 235):
        for _, t in windows(n):
            assert t-8 >= 0 and t+32 < n
    with pytest.raises(ValueError):
        windows(40)


def test_rank_ties_and_constant_costs():
    assert rank_correlation([1, 2, 3], [3, 2, 1]) == -1
    assert rank_correlation([1, 1, 1], [1, 2, 3]) is None
    assert rank_correlation([1, 1, 2], [3, 3, 4]) == 1


def test_native_branch_reset_clears_robot_then_pins_xml_before_gathering_history():
    from mylewm.evaluation.diagnose_libero_model import reset_native_context
    class Env:
        def __init__(self): self.calls=[]; self.steps=0
        def seed(self, value): self.calls.append(('seed',value))
        def reset(self): self.calls.append(('reset',)); self.steps=0
        def reset_from_xml_string(self, xml): self.calls.append(('xml',xml))
        def set_init_state(self, state): self.calls.append(('state',state))
        def step(self, action):
            np.testing.assert_array_equal(action,np.zeros(7))
            self.steps+=1
            rgb=np.full((2,2,3),self.steps,dtype=np.uint8)
            return {'agentview_image':rgb,'robot0_eye_in_hand_image':rgb},0,False,{}
        def get_sim_state(self):return np.array([self.steps])
    env=Env()
    for _ in range(2):
        _,frames,state=reset_native_context(env,'fixed_state','fixed_xml',42)
        assert env.calls[-4:]==[('seed',42),('reset',),('xml','fixed_xml'),('state','fixed_state')]
        np.testing.assert_array_equal(frames[:,0,0,0,0],[5,9,13])
        np.testing.assert_array_equal(state,[13])


def test_extra_cem_branch_is_not_counted_as_random_and_order_does_not_matter():
    from mylewm.evaluation.diagnose_libero_model import native_ranking
    def row(name, predicted, actual):
        return {'candidate': name, 'predicted_costs': [predicted], 'actual_costs': [actual],
                'task_id': 2, 'initial_cost': 10}
    rows = [row('cem', 2, 3), row('zero', 0, 0), row('random_0', 1, 4), row('random_1', 3, 2)]
    original = native_ranking(rows, 1)
    extended = native_ranking([row('cem_large', -100, -100), *reversed(rows)], 1)
    assert original['cem_predicted_beats_random_fraction'] == .5
    assert original['cem_actual_beats_random_fraction'] == .5
    for key in original.keys() - {'spearman_all'}:
        assert original[key] == extended[key]
