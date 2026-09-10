"""Read-only action/state diagnostics and owned observation snapshots."""
from copy import deepcopy

import numpy as np


def action_summary(values, threshold=1e-3):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return {'count': int(values.size), 'nonfinite_count': int(values.size-finite.size),
            'near_zero_threshold': threshold,
            **({key: float(value) for key, value in {
                'min': finite.min(), 'max': finite.max(), 'mean': finite.mean(),
                'std': finite.std(), 'mean_abs': np.abs(finite).mean(),
                'near_zero_fraction': (np.abs(finite) < threshold).mean(),
            }.items()} if finite.size else {})}


class EnvironmentAudit:
    def __init__(self, pool):
        self.pool = pool
        self.original_step = pool.step
        self.records = []

    def step(self, actions, mask=None):
        active = np.ones(self.pool.num_envs, dtype=bool) if mask is None else np.asarray(mask)
        before = np.stack([env.unwrapped._get_obs().copy() for env in self.pool.envs])
        result = self.original_step(actions, mask=mask)
        after = np.stack([env.unwrapped._get_obs().copy() for env in self.pool.envs])
        self.records.append({'actions': np.asarray(actions).tolist(), 'mask': active.tolist(),
                             'state_before': before.tolist(), 'state_after': after.tolist()})
        # EnvPool reuses its arrays in-place; SWM's video collector retains views.
        # Give each step its own storage so earlier video frames remain unchanged.
        return (*result[:-1], deepcopy(result[-1]))

    def summary(self):
        cases = []
        for i in range(self.pool.num_envs):
            rows = [r for r in self.records if r['mask'][i]]
            if not rows:
                cases.append({'case': i, 'step_calls': 0})
                continue
            initial = np.asarray(rows[0]['state_before'][i])
            final = np.asarray(rows[-1]['state_after'][i])
            deltas = np.asarray([np.asarray(r['state_after'][i])-r['state_before'][i] for r in rows])
            actions = action_summary([r['actions'][i] for r in rows])
            path = float(np.linalg.norm(deltas[:, :2], axis=1).sum())
            cases.append({'case': i, 'step_calls': len(rows), 'physical_actions': actions,
                          'initial_state': initial.tolist(), 'final_state': final.tolist(),
                          'agent_displacement': float(np.linalg.norm(final[:2]-initial[:2])),
                          'agent_path_length': path,
                          'object_displacement': float(np.linalg.norm(final[2:4]-initial[2:4])),
                          'warning': ('nonfinite active actions' if actions['nonfinite_count'] else
                                      'nontrivial actions but no agent movement' if actions.get('mean_abs', 0) > 1e-3 and path < 1e-6 else None)})
        return {'step_calls': len(self.records), 'cases': cases,
                'performance_interpretation_valid': all(r.get('step_calls', 0) and not r.get('warning') and not r.get('physical_actions', {}).get('nonfinite_count') for r in cases)}
