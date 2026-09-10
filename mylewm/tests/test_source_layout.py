"""Regression boundaries for source relocation, legacy objects, and host paths."""
import os
import subprocess
import sys

import pytest
import torch

from mylewm.paths import ROOT, libero_paths


@pytest.mark.parametrize('module', [
    'mylewm.training.train', 'mylewm.training.train_libero',
    'mylewm.data.prepare_dataset', 'mylewm.evaluation.compare_paired',
    'mylewm.evaluation.evaluate_upstream_pusht',
    'mylewm.policy.infer_trajectories', 'mylewm.policy.export_lerobot_policy',
    'mylewm.policy.check_policy_export',
])
def test_installed_cli_outside_checkout_without_pythonpath(tmp_path, module):
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, '-m', module, '--help'], cwd=tmp_path,
                            env=env, text=True, capture_output=True, timeout=45)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout


def test_historical_libero_object_class_resolves(tmp_path):
    # Serialize with the genuine old module identity in a separate interpreter.
    # No class mutation leaks into current model exports or other tests.
    checkpoint = tmp_path / 'legacy.ckpt'
    script = '''
import torch
from mylewm.libero_model import TwoViewJEPA
TwoViewJEPA.__module__ = 'mylewm.libero_model'
model = TwoViewJEPA.__new__(TwoViewJEPA)
torch.nn.Module.__init__(model)
model.register_buffer('saved_value', torch.tensor([3.]))
torch.save(model, __import__('sys').argv[1])
'''
    subprocess.run([sys.executable, '-c', script, str(checkpoint)], check=True)
    from mylewm.algorithms.libero_model import TwoViewJEPA
    model = torch.load(checkpoint, weights_only=False)
    assert type(model) is TwoViewJEPA
    assert model.saved_value.item() == 3.


def test_external_runtime_paths_are_configurable(tmp_path, monkeypatch):
    root, runtime = tmp_path/'libero source', tmp_path/'runtime'
    monkeypatch.setenv('LIBERO_ROOT', str(root))
    monkeypatch.setenv('LIBERO_MUJOCO_PATH', str(runtime))
    assert libero_paths() == (root, runtime)


def test_libero_shell_preserves_explicit_host_paths(tmp_path):
    mesa = tmp_path/'custom mesa'; mesa.mkdir()
    (mesa/'libOSMesa.so.8').touch()
    bindir = tmp_path/'bin'; bindir.mkdir()
    launcher = bindir/'uv'
    launcher.write_text('#!/bin/sh\nexec /usr/bin/env\n')
    launcher.chmod(0o755)
    env = dict(os.environ, PATH=str(bindir)+os.pathsep+os.environ['PATH'],
               LIBERO_OSMESA_DIR=str(mesa), LIBERO_CONFIG_PATH=str(tmp_path/'config'),
               LIBERO_ROOT=str(tmp_path/'source'), LIBERO_MUJOCO_PATH=str(tmp_path/'runtime'))
    result = subprocess.run(['bash', str(ROOT/'scripts/run_libero.sh'), '--help'],
                            env=env, text=True, capture_output=True, check=True)
    actual = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    for name in ('LIBERO_CONFIG_PATH', 'LIBERO_ROOT', 'LIBERO_MUJOCO_PATH'):
        assert actual[name] == env[name]
    assert actual['LD_LIBRARY_PATH'].split(os.pathsep)[0] == str(mesa)
    assert actual['MUJOCO_GL'] == 'osmesa'
