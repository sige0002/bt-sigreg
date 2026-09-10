#!/usr/bin/env bash
# Audited single-checkpoint evaluation. Dry-run unless --execute is supplied.
set -euo pipefail
EVAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$EVAL_ROOT"
exec uv run python - "$EVAL_ROOT" "$@" <<'PY'
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import atexit
from datetime import datetime, timezone

root = Path(sys.argv.pop(1))
parser = argparse.ArgumentParser(description='Evaluate a trusted PushT inference checkpoint; default is dry-run.')
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True, help='New directory under repository output/')
parser.add_argument('--manifest', type=Path, default=root/'.cache/stable-wm/pusht/rbg_v0/manifest.json')
parser.add_argument('--num-eval', type=int, default=50)
parser.add_argument('--dataset', type=Path, help='Relocated HDF5; otherwise use manifest dataset')
parser.add_argument('--offset', type=int, default=0, help='Index into the fixed confirm cases')
parser.add_argument('--seed', type=int, default=42, help='CEM/environment seed; confirm cases stay fixed')
parser.add_argument('--gb10-cache-workaround', action='store_true', help='Release only the dataset clean file cache before CUDA initialization')
parser.add_argument('--cem-audit', action='store_true', help='Record CEM convergence and selected-plan diagnostics')
parser.add_argument('--execute', action='store_true', help='Actually run evaluation on GPU')
parser.add_argument('--verify-data', action='store_true', help='Explicit full dataset SHA-256 scan')
args = parser.parse_args()
checkpoint = args.checkpoint.resolve()
manifest = args.manifest.resolve()
output = args.output.resolve()
if not checkpoint.is_file() or not checkpoint.name.endswith('_object.ckpt'):
    parser.error('--checkpoint must be an existing trusted *_object.ckpt, not resume.pt')
if not manifest.is_file():
    parser.error(f'Manifest is missing: {manifest}')
if output == root/'output' or not output.is_relative_to((root/'output').resolve()):
    parser.error('--output must be a new subdirectory of repository output/')
if output.exists() or args.output.is_symlink():
    parser.error('Output already exists; use a new name (no overwrite or implicit resume)')
data = json.loads(manifest.read_text())
dataset = (args.dataset or Path(data['dataset'])).expanduser().resolve()
if not dataset.is_file():
    parser.error(f'PushT HDF5 is missing: {dataset}')
if args.dataset and 'dataset_size' not in data:
    parser.error('--dataset relocation requires dataset_size in manifest')
if 'dataset_size' in data and dataset.stat().st_size != data['dataset_size']:
    parser.error(f'Dataset size differs from manifest: {dataset}')
cases = data['confirm'][args.offset:args.offset + args.num_eval]
if args.offset < 0 or args.num_eval < 1 or len(cases) != args.num_eval:
    parser.error('Requested cases are outside the confirm partition')
if len({case['episode'] for case in cases}) != len(cases):
    parser.error('Evaluation requires distinct episodes')

# CUDA initialization precedes large dataset reads. The evaluator then resets
# all its seeds, so initialization does not change the CEM random sequence.
bootstrap = '''
import os, sys, runpy
from pathlib import Path
root, dataset, release = sys.argv[1:4]
sys.path.insert(0, root)
from mylewm.pusht_eval_data import validate_schema
print('Checking PushT HDF5 schema', flush=True)
validate_schema(dataset)
import torch
if release == '1':
    with open(dataset, 'rb') as stream:
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    print('Released dataset clean cache only', flush=True)
torch.cuda.init()
probe = torch.empty(1, device='cuda')
torch.cuda.synchronize()
del probe
torch.cuda.reset_peak_memory_stats()
print('CUDA initialized before evaluation imports/data reads', flush=True)
sys.path.insert(0, str(Path(root)/'lewm'))
sys.argv = [str(Path(root)/'lewm/eval.py'), *sys.argv[4:]]
runpy.run_path(sys.argv[0], run_name='__main__')
print('evaluation_cuda_peak_allocated_bytes', torch.cuda.max_memory_allocated())
print('evaluation_cuda_peak_reserved_bytes', torch.cuda.max_memory_reserved())
'''
command = [sys.executable, '-c', bootstrap, str(root), str(dataset),
           str(int(args.gb10_cache_workaround)), f'policy={output}/checkpoint',
           f'eval.num_eval={args.num_eval}', f'+eval.manifest={manifest}',
           '+eval.partition=confirm', f'+eval.offset={args.offset}',
           f'seed={args.seed}',
           f'+eval.verify_data={str(args.verify_data).lower()}',
           '+eval.dataset_path=' + json.dumps(str(dataset)),
           '+eval.audit_provenance=true', '+eval.shared_physical_search=true',
           'output.filename=results.txt', f'hydra.run.dir={output}/hydra']
if args.cem_audit:
    command.append('solver._target_=mylewm.cem_audit.AuditedCEMSolver')
plan = {'checkpoint': str(checkpoint), 'dataset': str(dataset), 'output': str(output), 'cases': len(cases),
        'offset': args.offset, 'seed': args.seed, 'partition': 'confirm', 'execute': args.execute,
        'cem_audit': args.cem_audit,
        'verify_data': args.verify_data,
        'gb10_cache_workaround': args.gb10_cache_workaround,
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'launcher_sha256': hashlib.sha256((root/'mylewm/tools/evaluate_pusht.sh').read_bytes()).hexdigest(),
        'command': command}
print(json.dumps({key: value for key, value in plan.items() if key != 'command'}, indent=2), flush=True)
if not args.execute:
    print('Dry-run only: no GPU initialization, cache release, or output creation.')
    sys.exit(0)
output.mkdir(parents=True, exist_ok=False)
terminal = False
def record_status(state, **details):
    record = dict(state=state, updated_at=datetime.now(timezone.utc).isoformat(),
                  pid=os.getpid(), checkpoint=str(checkpoint), **details)
    temporary = output/'status.json.tmp'
    temporary.write_text(json.dumps(record, indent=2))
    temporary.replace(output/'status.json')
def unfinished_exit():
    if not terminal:
        record_status('failed', reason='Launcher exited before validated completion; inspect console.log and service status')
atexit.register(unfinished_exit)
record_status('running')
(output/'checkpoint_object.ckpt').symlink_to(checkpoint)
(output/'launch.json').write_text(json.dumps(plan, indent=2))
env = dict(os.environ, STABLEWM_HOME=str(root/'.cache/stable-wm'),
           PYTHONPATH=os.pathsep.join([str(root), str(root/'lewm')]),
           PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
if args.cem_audit:
    env['BT_CEM_AUDIT_PATH'] = str(output/'cem_audit.jsonl')
print(f"Running; follow progress with: tail -f {output/'console.log'}", flush=True)
with (output/'console.log').open('x') as stream:
    import threading
    stopped = threading.Event()
    def follow():
        with (output/'console.log').open() as reader:
            while not stopped.is_set():
                text = reader.read()
                if text:
                    print(text, end='', flush=True)
                stopped.wait(.2)
            print(reader.read(), end='', flush=True)
    follower = threading.Thread(target=follow, daemon=True)
    follower.start()
    try:
        result = subprocess.run(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
    finally:
        stream.flush()
        stopped.set()
        follower.join()
if result.returncode:
    print(f"Evaluation failed; keep {output/'console.log'} for diagnosis. No success rate asserted.", file=sys.stderr)
    sys.exit(1)
sys.path.insert(0, str(root))
from mylewm.evaluation_contract import validate_result, protocol
metrics = json.loads((output/'results.txt.json').read_text())
validate_result(metrics, cases, metrics['provenance'], protocol(metrics['config']))
if args.cem_audit and (not (output/'cem_audit.jsonl').is_file() or not (output/'cem_audit.jsonl').stat().st_size):
    raise ValueError('CEM audit was requested but no records were written')
if metrics['checkpoint_sha256'] != plan['checkpoint_sha256'] or metrics['provenance']['manifest_sha256'] != plan['manifest_sha256']:
    raise ValueError('Checkpoint or manifest changed during evaluation')
successes = sum(metrics['successes'])
if abs(metrics['success_rate'] - 100*successes/len(cases)) > 1e-6:
    raise ValueError('Reported success rate differs from case flags')
print(f'Completed: {successes}/{len(cases)} successes ({metrics["success_rate"]:.1f}%).')
record_status('succeeded', successes=successes, cases=len(cases), success_rate=metrics['success_rate'])
terminal = True
print('This is a fixed-case checkpoint evaluation, not a guarantee or a matched-training-budget comparison.')
PY
