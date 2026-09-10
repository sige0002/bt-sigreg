"""CPU random-policy control on the same fixed cases as a completed evaluation."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
import stable_worldmodel as swm

from lewm.eval import get_dataset
from mylewm.evaluation_contract import array_hash
from mylewm.pusht_action_audit import EnvironmentAudit


class RandomControl(swm.policy.BasePolicy):
    """Keep one seeded space: EnvPool.action_space constructs a fresh copy."""
    def __init__(self, seed):
        super().__init__()
        self.seed = seed

    def set_env(self, env):
        super().set_env(env)
        self.space = env.action_space
        self.space.seed(self.seed)

    def get_action(self, info, **kwargs):
        return self.space.sample()

    def set_seed(self, seed):
        self.seed = seed
        self.space.seed(seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if not output.is_relative_to(root/'output') or output == root/'output':
        parser.error('output must be a new directory under output/')
    reference = json.loads(args.reference_result.read_text())
    cfg = OmegaConf.create(reference['config'])
    output.mkdir(parents=True, exist_ok=False)
    status = output/'status.json'
    status.write_text(json.dumps({'state': 'running'}))
    world = None
    try:
        torch.set_num_threads(4)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        world = swm.World(**cfg.world, image_shape=(224, 224))
        world.set_policy(RandomControl(seed=cfg.seed))
        dataset = get_dataset(cfg, cfg.eval.dataset_name)
        audit = EnvironmentAudit(world.envs)
        world.envs.step = audit.step
        original = world._get_actions
        initial_hashes = []
        def action():
            if not initial_hashes:
                for i, expected in enumerate(reference['initial_runtime_hashes']):
                    initial_hashes.append({k: array_hash(world.infos[k][i]) for k in expected})
                if initial_hashes != reference['initial_runtime_hashes']:
                    raise ValueError('Random control initial states/goals differ from reference')
            return original()
        world._get_actions = action
        world.reset(seed=cfg.seed)
        metrics = world.evaluate(dataset=dataset, episodes_idx=reference['episodes'],
                                 start_steps=reference['starts'], goal_offset=cfg.eval.goal_offset_steps,
                                 eval_budget=cfg.eval.eval_budget,
                                 callables=OmegaConf.to_container(cfg.eval.callables), video=output)
        result = {'policy': 'random', 'reference_result': str(args.reference_result),
                  'episodes': reference['episodes'], 'starts': reference['starts'],
                  'initial_runtime_hashes': initial_hashes,
                  'successes': np.asarray(metrics['episode_successes']).tolist(),
                  'success_rate': float(metrics['success_rate']),
                  'physical_actions': audit.records, 'action_path_diagnostics': audit.summary()}
        (output/'results.json').write_text(json.dumps(result, indent=2))
        status.write_text(json.dumps({'state': 'succeeded', 'exit_code': 0}))
        print(json.dumps({'success_rate': result['success_rate'], 'cases': len(result['episodes']),
                          'action_path_diagnostics': result['action_path_diagnostics']}), flush=True)
    except BaseException:
        status.write_text(json.dumps({'state': 'failed'}))
        raise
    finally:
        if world is not None:
            world.envs.close()


if __name__ == '__main__':
    main()
