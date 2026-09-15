import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import types

import pytest

from mylewm.data import regenerate_libero_openvla as regen


@pytest.fixture
def args(tmp_path, monkeypatch):
    monkeypatch.setattr(regen, 'ROOT', tmp_path)
    raw = tmp_path / 'raw'
    raw.mkdir()
    for i in range(10):
        (raw / f'task{i}_demo.hdf5').write_bytes(b'original')
    return argparse.Namespace(raw_data=raw, output=tmp_path / 'output/new', execute=False)


def test_dry_run_and_existing_output_are_read_only(args, monkeypatch):
    monkeypatch.setattr(regen.urllib.request, 'urlopen', lambda *a, **kw: pytest.fail('download'))
    result = regen.run(args)
    assert result['resolution'] == 256 and result['split'] is None
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    with pytest.raises(ValueError, match='fresh'):
        regen.run(args)


def test_download_failure_is_recorded_without_touching_input(args, monkeypatch):
    args.execute = True
    monkeypatch.setattr(regen.urllib.request, 'urlopen', lambda *a, **kw: io.BytesIO(b'wrong source'))
    with pytest.raises(ValueError, match='SHA-256'):
        regen.run(args)
    assert json.loads((args.output / 'status.json').read_text())['state'] == 'failed'
    assert all(p.read_bytes() == b'original' for p in args.raw_data.iterdir())


def test_execution_isolates_upstream_metadata_and_restores_modules(args, monkeypatch):
    args.execute = True
    payload = b'test source'
    monkeypatch.setattr(regen, 'SOURCE_HASHES', {regen.REPLAY: hashlib.sha256(payload).hexdigest()})
    monkeypatch.setattr(regen.urllib.request, 'urlopen', lambda *a, **kw: io.BytesIO(payload))
    monkeypatch.setitem(sys.modules, 'mujoco', types.SimpleNamespace(__version__='test'))
    monkeypatch.setattr(regen.importlib.metadata, 'version', lambda name: 'test')
    key = 'experiments.robot.libero.libero_utils'
    prior = types.ModuleType(key)
    monkeypatch.setitem(sys.modules, key, prior)
    cwd = Path.cwd()

    def upstream_main(options):
        assert Path.cwd() == args.output
        assert sys.modules[key].get_libero_dummy_action('llava') == [0, 0, 0, 0, 0, 0, -1]
        assert Path(options.libero_raw_data_dir) == args.raw_data
        target = Path(options.libero_target_dir)
        assert not target.exists()
        target.mkdir()
        for i in range(10):
            (target / f'task{i}_demo.hdf5').write_bytes(b'regenerated')
        Path('experiments/robot/libero/libero_10_metainfo.json').write_text('{}')

    monkeypatch.setattr(regen.runpy, 'run_path', lambda path: {'main': upstream_main})
    regen.run(args)
    assert Path.cwd() == cwd and sys.modules[key] is prior
    assert json.loads((args.output / 'status.json').read_text())['state'] == 'succeeded'
    assert len(json.loads((args.output / 'dataset_sha256.json').read_text())) == 10
    assert all(p.read_bytes() == b'original' for p in args.raw_data.iterdir())


def test_environment_helpers_preserve_upstream_seed_and_resolution(monkeypatch):
    calls = []
    class Env:
        def __init__(self, **kwargs):
            calls.append(kwargs)
        def seed(self, value):
            calls.append(value)
    monkeypatch.setitem(sys.modules, 'libero.libero', types.SimpleNamespace(get_libero_path=lambda key: '/bddl'))
    monkeypatch.setitem(sys.modules, 'libero.libero.envs', types.SimpleNamespace(OffScreenRenderEnv=Env))
    environments = []
    helpers = regen.environment_utils(environments)
    env, instruction = helpers.get_libero_env(
        types.SimpleNamespace(problem_folder='suite', bddl_file='task.bddl', language='task'), 'llava')
    assert environments == [env] and instruction == 'task'
    assert calls == [{'bddl_file_name': '/bddl/suite/task.bddl', 'camera_heights': 256, 'camera_widths': 256}, 0]
    assert helpers.get_libero_dummy_action('llava') == [0, 0, 0, 0, 0, 0, -1]
