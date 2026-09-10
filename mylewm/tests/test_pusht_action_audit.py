import numpy as np
import stable_worldmodel as swm

from mylewm.evaluation.pusht_action_audit import EnvironmentAudit, action_summary


def test_nonzero_commands_move_multiple_real_pusht_environments_and_preserve_frames():
    world = swm.World(env_name='swm/PushT-v1', num_envs=3, max_episode_steps=50, image_shape=(224, 224))
    try:
        world.reset(seed=42)
        audit = EnvironmentAudit(world.envs)
        first = audit.step(np.array([[.5, 0.], [0., .5], [-.5, 0.]]))[-1]['pixels']
        saved = first.copy()
        for _ in range(9):
            last = audit.step(np.array([[.5, 0.], [0., .5], [-.5, 0.]]))[-1]['pixels']
        np.testing.assert_array_equal(first, saved)
        assert not np.array_equal(first, last)
        summary = audit.summary()
        assert summary['performance_interpretation_valid']
        assert all(case['agent_path_length'] > 1. for case in summary['cases'])
        assert all(case['step_calls'] == 10 for case in summary['cases'])
        audit.step(np.ones((3, 2)), mask=np.array([True, False, False]))
        assert [r['step_calls'] for r in audit.summary()['cases']] == [11, 10, 10]
    finally:
        world.envs.close()


def test_summary_counts_invalid_and_near_zero_values():
    summary = action_summary([0., 1., np.nan])
    assert summary['nonfinite_count'] == 1
    assert summary['near_zero_fraction'] == .5


def test_invalid_or_ineffective_actions_are_not_valid_performance():
    from types import SimpleNamespace
    audit = EnvironmentAudit(SimpleNamespace(num_envs=1, step=lambda: None))
    for actions, expected in [([float('nan'), float('nan')], 'nonfinite'), ([.5, .5], 'no agent movement')]:
        audit.records = [{'actions': [actions], 'mask': [True],
                          'state_before': [[0.]*7], 'state_after': [[0.]*7]}]
        result = audit.summary()
        assert not result['performance_interpretation_valid']
        assert expected in result['cases'][0]['warning']


def test_random_control_advances_rng_and_is_reproducible():
    from mylewm.evaluation.evaluate_pusht_random import RandomControl
    world = swm.World(env_name='swm/PushT-v1', num_envs=3, max_episode_steps=50, image_shape=(224, 224))
    try:
        a, b = RandomControl(42), RandomControl(42)
        world.set_policy(a)
        world.set_policy(b)
        first, second = a.get_action({}), a.get_action({})
        assert not np.array_equal(first, second)
        np.testing.assert_array_equal(first, b.get_action({}))
        np.testing.assert_array_equal(second, b.get_action({}))
    finally:
        world.envs.close()
