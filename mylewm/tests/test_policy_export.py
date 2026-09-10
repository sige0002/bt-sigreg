"""Shared CEM dynamics, actual JEPA export, official LeRobot factory/processors."""
import copy
import json
from mylewm.paths import CONFIGS
from pathlib import Path

import pytest
import torch

from mylewm.data.contract import PREPROCESSING
from mylewm.policy.runtime import WorldModelController, load_world_model, validate_planner


class LinearWorld(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.action_encoder = torch.nn.Identity()
        self.register_buffer('training_action_mean', torch.zeros(2, dtype=torch.float64))
        self.register_buffer('training_action_std', torch.ones(2, dtype=torch.float64))
        self.input_contract = dict(domain='linear', camera='rgb', action_names=['x', 'y'],
            action_units=['m', 'm'], action_convention='delta', fps=10, frameskip=1,
            history=3, preprocessing=PREPROCESSING)

    def encode(self, batch):
        return dict(batch, emb=batch['pixels'].mean((-1, -2))[..., :2])

    def predict(self, z, a):
        return z + a


def planner(k=1):
    return dict(horizon=k, samples=32, elites=8, iterations=3, seed=5,
                action_low=[-1., -1.], action_high=[1., 1.])


def test_cem_finds_action_for_known_dynamics_and_isolates_rng():
    runtime = WorldModelController(LinearWorld(), planner())
    runtime.history = torch.zeros(1, 3, 2)
    runtime.past_actions = torch.zeros(1, 2, 1, 2)
    runtime.goal = torch.tensor([[.7, -.4]])
    before = torch.get_rng_state()
    action = runtime.plan()
    torch.testing.assert_close(torch.get_rng_state(), before)
    assert ((action - runtime.goal).square().mean() < .01)
    assert (action.abs() <= 1).all()


def test_history_execution_timing_and_reset():
    runtime = WorldModelController(LinearWorld(), planner())
    images = torch.zeros(3, 3, 8, 8, dtype=torch.uint8)
    with pytest.raises(ValueError, match='prime'):
        runtime.select_action(images[:1], .2)
    runtime.prime(images, torch.zeros(2, 1, 2), images[:1], [0., .1, .2])
    first = runtime.select_action(images[:1], .2)
    with pytest.raises(ValueError, match='actually executed'):
        runtime.select_action(images[:1], .3)
    actual = torch.tensor([.3, .4])
    runtime.select_action(images[:1], .3, actual)
    torch.testing.assert_close(runtime.past_actions[0, -1, 0], actual)
    with pytest.raises(ValueError, match='timing'):
        runtime.select_action(images[:1], .7, actual)
    runtime.prime(images, torch.zeros(2, 1, 2), images[:1], [0., .1, .2])
    torch.testing.assert_close(runtime.select_action(images[:1], .2), first, rtol=0, atol=0)
    runtime.reset()
    assert runtime.history is None and not runtime.queue


@pytest.mark.parametrize('bad', [dict(elites=1), dict(action_low=[0.]), dict(samples=0),
                                dict(action_high=[float('inf'), 1.]), dict(iterations=1.5)])
def test_invalid_planner_refused(bad):
    with pytest.raises(ValueError):
        validate_planner(dict(planner(), **bad), 2)


@pytest.fixture(scope='module')
def exported(tmp_path_factory):
    pytest.importorskip('lerobot')
    from mylewm.training.train import build_model
    from mylewm.policy.export_lerobot_policy import export
    root = tmp_path_factory.mktemp('policy_export')
    c = json.loads((CONFIGS / 'lerobot_pusht_input.json').read_text())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = build_model().eval()
    model.input_contract = c
    model.register_buffer('training_action_mean', torch.tensor([100., 200.], dtype=torch.float64))
    model.register_buffer('training_action_std', torch.tensor([20., 30.], dtype=torch.float64))
    source = root / 'object.ckpt'
    torch.save(model, source)
    output = root / 'policy'
    export(source, output, c, dict(planner(), horizon=1, samples=4, elites=2, iterations=1,
                                  action_low=[0., 0.], action_high=[512., 512.]))
    return source, output


def test_actual_model_weights_latents_actions_and_processors(exported):
    from mylewm.policy.lerobot import BTSIGRegPolicy
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from mylewm.policy.runtime import load_checkpoint_controller
    source, output = exported
    policy = BTSIGRegPolicy.from_pretrained(output)
    assert get_policy_class('bt_sigreg') is BTSIGRegPolicy
    native = load_checkpoint_controller(source, policy.config.planner)
    rgb = (torch.arange(3 * 3 * 28 * 28).reshape(3, 3, 28, 28) % 256).to(torch.uint8)
    actions = torch.arange(20).reshape(2, 5, 2).float()
    native.prime(rgb, actions, rgb[-1:], [0., .5, 1.])
    policy.prime(rgb, actions, rgb[-1:], [0., .5, 1.])
    torch.testing.assert_close(native.history, policy.controller.history, rtol=0, atol=0)
    pre, post = make_pre_post_processors(policy.config, pretrained_path=output)
    action = post(policy.select_action(pre({'observation.image': rgb[-1].float() / 255,
                                           'observation.timestamp': torch.tensor(1.)})))
    torch.testing.assert_close(action[0], native.select_action(rgb[-1:], 1.), rtol=0, atol=0)
    assert action.shape == (1, 2)
    for tick in range(1, 6):
        executed = torch.tensor([100. + tick, 200. - tick])
        batch = pre({'observation.image': rgb[-1].float() / 255,
                     'observation.timestamp': torch.tensor(1. + tick / 10),
                     'observation.executed_action': executed})
        actual = post(policy.select_action(batch))[0]
        expected = native.select_action(rgb[-1:], 1. + tick / 10, executed)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Factory construction (without pretrained processor files) is also supported.
    make_pre_post_processors(policy.config, dataset_stats={'action': {'mean': torch.ones(2) * 999}})


def test_export_refuses_overwrite_and_incompatible_contract(exported):
    from mylewm.policy.export_lerobot_policy import export
    from mylewm.policy.lerobot import BTSIGRegPolicy
    source, output = exported
    policy = BTSIGRegPolicy.from_pretrained(output)
    with pytest.raises(FileExistsError):
        export(source, output, policy.config.input_contract, policy.config.planner)
    c = copy.deepcopy(policy.config)
    c.input_contract['action_units'] = ['m', 'm']
    with pytest.raises(ValueError, match='override'):
        BTSIGRegPolicy.from_pretrained(output, config=c)
    with pytest.raises(ValueError, match='strict'):
        BTSIGRegPolicy.from_pretrained(output, strict=False)


def test_training_checkpoint_extracts_only_model(exported, tmp_path):
    source, _ = exported
    model = load_world_model(source)
    state = {'model.' + k: v for k, v in model.state_dict().items()}
    state['sigreg.transport.fake'] = torch.ones(3)
    path = tmp_path / 'training.ckpt'
    torch.save({'recipe': {'input_contract': model.input_contract}, 'state_dict': state}, path)
    extracted = load_world_model(path)
    assert extracted.input_contract == model.input_contract
    for key, value in model.state_dict().items():
        torch.testing.assert_close(extracted.state_dict()[key], value, rtol=0, atol=0)


def test_missing_weights_fail_strict_reload(exported, tmp_path):
    from mylewm.policy.lerobot import BTSIGRegPolicy
    from safetensors.torch import load_file, save_file
    source, output = exported
    damaged = tmp_path / 'damaged'
    damaged.mkdir()
    (damaged / 'config.json').write_bytes((output / 'config.json').read_bytes())
    state = load_file(output / 'model.safetensors')
    state.pop('model.encoder.embeddings.patch_embeddings.projection.weight')
    save_file(state, damaged / 'model.safetensors')
    with pytest.raises((RuntimeError, ValueError)):
        BTSIGRegPolicy.from_pretrained(damaged)
