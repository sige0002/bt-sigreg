"""Run pinned OpenVLA HDF5 regeneration in a fresh directory (dry-run by default).

The upstream replay/filter/save implementation runs unchanged. Only its two
environment utilities are supplied locally to avoid importing TensorFlow/VLA.
This is OpenVLA data preparation, not an exact TC-LeWM reproduction.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import sys
import types
import urllib.request

from mylewm.data.verification import file_sha256
from mylewm.paths import ROOT

REVISION = 'c8f03f48af692657d3060c19588038c7220e9af9'
REPLAY = 'experiments/robot/libero/regenerate_libero_dataset.py'
UTILS = 'experiments/robot/libero/libero_utils.py'
SOURCE_HASHES = {
    REPLAY: 'f17dd49e61659b515ccd691c0ee57d20fb57909af3b8b53ef7e7e6bee55ceb49',
    UTILS: '6c162ce0954c8659018ee8ff1604a33904b50ac0631a8ca0f8bf1f7a4d25181d',
    'LICENSE': 'c4ee2ba5958af03d74b1d3dfa174e3749171d689fb4c23cc170749c05ae3eeb5',
}


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def environment_utils(environments):
    """Exactly the two relevant helpers in pinned libero_utils.py, lines 16–27."""
    module = types.ModuleType('experiments.robot.libero.libero_utils')

    def get_libero_env(task, model_family, resolution=256):
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        env = OffScreenRenderEnv(
            bddl_file_name=str(Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file),
            camera_heights=resolution, camera_widths=resolution)
        environments.append(env)
        env.seed(0)
        return env, task.language

    module.get_libero_env = get_libero_env
    module.get_libero_dummy_action = lambda model_family: [0, 0, 0, 0, 0, 0, -1]
    return module


def run(args):
    raw, output = args.raw_data.resolve(), args.output.resolve()
    if not raw.is_dir() or len(list(raw.glob('*_demo.hdf5'))) != 10:
        raise ValueError('Require the ten original LIBERO-10 HDF5 files')
    if output.exists() or not output.is_relative_to(ROOT / 'output'):
        raise ValueError('Use a fresh directory under output/')
    if raw.is_relative_to(output) or output.is_relative_to(raw):
        raise ValueError('Keep original and regenerated data separate')
    config = {
        'schema': 'openvla_libero_regeneration_v1', 'raw_data': str(raw),
        'output': str(output), 'upstream_revision': REVISION, 'upstream_sha256': SOURCE_HASHES,
        'resolution': 256, 'suite': 'libero_10', 'environment_seed': 0,
        'settling_steps': 10, 'settling_action': [0, 0, 0, 0, 0, 0, -1],
        'images': 'native HDF5; no RLDS rotation/JPEG conversion',
        'split': None, 'tclewm_exact_match': False,
        'implementation': 'unchanged pinned replay main; two environment helpers supplied locally',
        'launcher_sha256': file_sha256(Path(__file__)),
    }
    if not args.execute:
        print(json.dumps(config, indent=2))
        print('Dry-run: no download, output, simulation, training or evaluation.')
        return config
    output.mkdir(parents=True)
    write_json(output / 'config.json', config)
    write_json(output / 'status.json', {'state': 'running', 'phase': 'sources'})
    previous_cwd = Path.cwd()
    key = 'experiments.robot.libero.libero_utils'
    previous_module = sys.modules.get(key)
    environments = []
    try:
        source = output / 'upstream'
        for name, expected in SOURCE_HASHES.items():
            url = f'https://raw.githubusercontent.com/openvla/openvla/{REVISION}/{name}'
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f'Upstream SHA-256 mismatch: {name}')
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        config['input_sha256'] = {p.name: file_sha256(p) for p in sorted(raw.glob('*_demo.hdf5'))}
        config['dependencies'] = {n: importlib.metadata.version(n) for n in ('numpy', 'h5py', 'robosuite')}
        import mujoco
        config['mujoco'] = mujoco.__version__
        config['render_backend'] = os.environ.get('MUJOCO_GL')
        write_json(output / 'config.json', config)
        # Upstream writes metainfo relative to cwd. Isolate it from external/ and old runs.
        (output / 'experiments/robot/libero').mkdir(parents=True)
        os.chdir(output)
        sys.modules[key] = environment_utils(environments)
        namespace = runpy.run_path(str(source / REPLAY))
        write_json(output / 'status.json', {'state': 'running', 'phase': 'replay'})
        namespace['main'](argparse.Namespace(libero_task_suite='libero_10',
                         libero_raw_data_dir=str(raw), libero_target_dir=str(output / 'data')))
        hashes = {p.name: file_sha256(p) for p in sorted((output / 'data').glob('*.hdf5'))}
        write_json(output / 'dataset_sha256.json', hashes)
        write_json(output / 'status.json', {'state': 'succeeded', 'files': len(hashes),
                                          'data': 'data', 'training_started': False})
    except BaseException as exc:
        write_json(output / 'status.json', {'state': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
        raise
    finally:
        os.chdir(previous_cwd)
        if previous_module is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = previous_module
        for env in environments:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
