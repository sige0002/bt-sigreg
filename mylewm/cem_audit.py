"""Passive CEM instrumentation used only by explicit PushT audits."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from stable_worldmodel.solver import CEMSolver


def tensor_hash(value):
    value = value.detach().cpu().contiguous().numpy() if torch.is_tensor(value) else np.asarray(value)
    return hashlib.sha256(value.tobytes()).hexdigest()


class AuditedCEMSolver(CEMSolver):
    """Run the upstream solver unchanged while recording its cost evaluations."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        path = os.environ.get('BT_CEM_AUDIT_PATH')
        if not path:
            raise RuntimeError('BT_CEM_AUDIT_PATH is required')
        self.audit_path = Path(path)
        self.goal_cases = {}
        self.replans = {}

    @torch.inference_mode()
    def solve(self, info_dict, init_action=None):
        calls = []
        original = self.model.get_cost

        def observed(info, candidates):
            costs = original(info, candidates)
            goal = info['goal'][:, 0]
            def first_sample(value, row):
                if torch.is_tensor(value) or isinstance(value, np.ndarray):
                    return value[row:row + 1, :1]
                if isinstance(value, list):
                    return value[row:row + 1]
                return value
            for row in range(costs.shape[0]):
                fingerprint = tensor_hash(goal[row])
                if fingerprint not in self.goal_cases:
                    self.goal_cases[fingerprint] = len(self.goal_cases)
                values = costs[row].detach().float().cpu()
                best = int(values.argmin())
                calls.append({'case': self.goal_cases[fingerprint], 'goal_sha256': fingerprint,
                    'cost_min': float(values.min()), 'cost_mean': float(values.mean()),
                    'cost_std': float(values.std(unbiased=False)),
                    'cost_median': float(values.median()), 'best_index': best,
                    'best_action': candidates[row, best].detach().float().cpu().tolist(),
                    '_info': {k: first_sample(v, row) for k, v in info.items()}})
            return costs

        self.model.get_cost = observed
        try:
            outputs = super().solve(info_dict, init_action=init_action)
        finally:
            self.model.get_cost = original
        expected = len(outputs['actions']) * self.n_steps
        if len(calls) != expected:
            raise RuntimeError(f'unexpected CEM call count: {len(calls)} != {expected}')
        records = []
        for row in range(len(outputs['actions'])):
            history = calls[row * self.n_steps:(row + 1) * self.n_steps]
            case = history[0]['case']
            if any(item['case'] != case for item in history):
                raise RuntimeError('CEM batch/case ordering changed')
            chosen = outputs['actions'][row:row + 1].to(self.device).unsqueeze(1)
            chosen_cost = original(history[-1].pop('_info'), chosen).detach().float().cpu()
            for item in history[:-1]:
                item.pop('_info')
            replan = self.replans.get(case, 0)
            self.replans[case] = replan + 1
            records.append({'case': case, 'replan': replan,
                'iterations': [{k: v for k, v in item.items() if k not in ('best_action', '_info')}
                               for item in history],
                'final_best_action': history[-1]['best_action'],
                'selected_mean_action': outputs['actions'][row].float().tolist(),
                'selected_predicted_cost': float(chosen_cost[0, 0]),
                'final_best_sample_cost': history[-1]['cost_min']})
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open('a') as stream:
            for record in records:
                stream.write(json.dumps(record) + '\n')
        return outputs
