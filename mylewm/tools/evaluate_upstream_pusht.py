"""Evaluate one trusted PushT checkpoint with the pinned upstream LeWM eval."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import types


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / 'external/le-wm'
COMMIT = '8edfeb336732b5f3ce7b8b210d0ba370a09e2cac'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New directory under output/.')
    p.add_argument('--seed', type=int, default=42, help='Unmodified upstream cfg.seed.')
    args = p.parse_args()
    checkpoint, output = args.checkpoint.resolve(), args.output.resolve()
    if not checkpoint.is_file() or not checkpoint.name.endswith('_object.ckpt'):
        p.error('--checkpoint must be a trusted *_object.ckpt')
    if output.exists() or not output.is_relative_to((ROOT / 'output').resolve()):
        p.error('--output must be a new directory under output/')
    if subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip() != COMMIT:
        raise RuntimeError('Pinned upstream checkout is unavailable')

    source = output / 'source'
    for relative in ['eval.py', 'config/eval/pusht.yaml', 'config/eval/solver/cem.yaml',
                     'config/eval/launcher/local.yaml']:
        data = subprocess.check_output(['git', '-C', str(UPSTREAM), 'show', f'{COMMIT}:{relative}'])
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    dataset = ROOT / '.cache/stable-wm/datasets/pusht_expert_train.h5'
    if not dataset.is_file():
        raise RuntimeError('Local PushT HDF5 is missing')
    launch = {'upstream_commit': COMMIT, 'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
              'upstream_eval_sha256': sha(source / 'eval.py'), 'seed': args.seed,
              'protocol': 'byte-identical upstream eval.py; random starts and all-data StandardScaler; no action adapter'}
    (output / 'launch.json').write_text(json.dumps(launch, indent=2))

    # The pinned source is executed unchanged. These two compatibility shims only
    # bridge the installed HDF5 loader and trusted local checkpoint location.
    os.environ['STABLEWM_HOME'] = str(ROOT / '.cache/stable-wm')
    import torch
    import stable_worldmodel as swm
    with dataset.open('rb') as stream:
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    torch.cuda.init()
    probe = torch.empty(1, device='cuda'); torch.cuda.synchronize(); del probe
    torch.cuda.reset_peak_memory_stats()
    sys.path[:0] = [str(ROOT), str(ROOT / 'lewm')]
    if not hasattr(swm, 'wm'):
        swm.wm = types.SimpleNamespace()
    if not hasattr(swm.wm, 'utils'):
        swm.wm.utils = types.SimpleNamespace()
    swm.wm.utils.load_pretrained = lambda policy: torch.load(checkpoint, map_location='cpu', weights_only=False)
    original = swm.World.evaluate

    def audited(self, *a, **kw):
        episodes, starts = kw['episodes_idx'], kw['start_steps']
        (output / 'cases.json').write_text(json.dumps({'episodes': episodes, 'starts': starts}, indent=2))
        result = original(self, *a, **kw)
        plain = lambda x: x.tolist() if hasattr(x, 'tolist') else x
        result = {k: plain(v) for k, v in result.items()}
        result.update(checkpoint_sha256=sha(checkpoint), episodes=episodes, starts=starts)
        (output / 'results.json').write_text(json.dumps(result, indent=2))
        return result
    swm.World.evaluate = audited
    sys.argv = [str(source / 'eval.py'), '--config-name=pusht.yaml', f'policy={output}/checkpoint',
                f'seed={args.seed}', 'output.filename=results.txt', f'hydra.run.dir={output}/hydra']
    with (output / 'console.log').open('x') as stream:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = stream
        try:
            runpy.run_path(str(source / 'eval.py'), run_name='__main__')
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
    result = json.loads((output / 'results.json').read_text())
    flags = result['episode_successes']
    if len(flags) != 50 or result['checkpoint_sha256'] != launch['checkpoint_sha256']:
        raise RuntimeError('Upstream evaluation contract failed')
    summary = {'state': 'succeeded', 'successes': sum(flags), 'cases': len(flags),
               'success_rate': result['success_rate']}
    (output / 'status.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
