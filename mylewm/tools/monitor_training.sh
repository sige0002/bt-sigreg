#!/usr/bin/env bash
# Read-only monitor: no torch imports, checkpoint loads, or training controls.
set -euo pipefail
MONITOR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python3 - "$MONITOR_ROOT" "$@" <<'PY'
import argparse
from collections import deque
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

root = Path(sys.argv.pop(1))
parser = argparse.ArgumentParser(description='Read-only training monitor. Ctrl-C exits without stopping training.')
parser.add_argument('--run', type=Path, help='Run directory containing config.json and metrics.jsonl')
parser.add_argument('--unit', help='Optional systemd user service (automatically set for the default run)')
parser.add_argument('--interval', type=float, default=5, help='Refresh seconds (default: 5)')
parser.add_argument('--once', action='store_true', help='Print once and exit')
args = parser.parse_args()
if not math.isfinite(args.interval) or args.interval <= 0:
    parser.error('--interval must be finite and positive')
default_run = args.run is None
run = args.run or root / 'output/pusht/bt_spectral_v2_100k_s3072'
unit = args.unit or ('bt-pusht-spectral-v2-100k-s3072.service' if default_run else None)
metrics = run / 'metrics.jsonl'
rows = deque(maxlen=200)
position = 0
identity = None
validation = diagnostic = None
bad_lines = 0

def number(value):
    return f'{value:.6g}' if isinstance(value, (int, float)) else '-'

def refresh():
    global position, identity, validation, diagnostic, bad_lines
    if not metrics.exists():
        return
    with metrics.open() as stream:
        stat = metrics.stat()
        current = (stat.st_dev, stat.st_ino)
        if identity != current or stat.st_size < position:
            position = 0
            rows.clear()
            validation = diagnostic = None
            bad_lines = 0
        identity = current
        stream.seek(position)
        while True:
            line = stream.readline()
            if not line or not line.endswith('\n'):
                break  # Writer may still be appending this line; retry next refresh.
            position = stream.tell()
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get('step'), int):
                    raise ValueError('Invalid metric row')
                rows.append(row)
                if 'validation' in row:
                    validation = row
                if 'transport_diagnostics_pre_update' in row:
                    diagnostic = row
            except (ValueError, TypeError):
                bad_lines += 1

def display():
    refresh()
    if sys.stdout.isatty() and not args.once:
        print('\033[2J\033[H', end='')
    print(f'Training monitor | {datetime.now().astimezone().isoformat(timespec="seconds")}')
    print(f'Run: {str(run)!r}')
    if unit:
        try:
            status = subprocess.run(['systemctl', '--user', 'show', '--property=ActiveState',
                                     '--property=SubState', '--property=MainPID', '--', unit],
                                    capture_output=True, text=True, timeout=3)
            print('Service: ' + (' '.join(status.stdout.splitlines()) or 'unknown'))
        except (OSError, subprocess.TimeoutExpired):
            print('Service: unavailable')
    try:
        config = json.loads((run / 'config.json').read_text())
    except (OSError, ValueError):
        config = {}
    if not rows:
        print('Waiting for complete metrics (hashing/initialization may still be running).')
    else:
        row = rows[-1]
        total = config.get('steps')
        progress = f' ({100*row["step"]/total:.2f}%)' if isinstance(total, int) and total > 0 else ''
        print(f'Step: {row["step"]:,} / {total or "?"}{progress}')
        print(f'Loss: {number(row.get("loss"))} | prediction: {number(row.get("prediction"))} | SIGReg: {number(row.get("gaussian"))}')
        recent = [r['loss'] for r in list(rows)[-100:] if isinstance(r.get('loss'), (float, int))]
        print(f'Loss mean (last {len(recent)}): {number(statistics.mean(recent)) if recent else "-"}')
        print(f'Grad: {number(row.get("grad_norm"))} | T grad: {number(row.get("transport_grad_norm"))} | latent std: {number(row.get("latent_std"))}')
        print('LR: ' + ', '.join(number(x) for x in row.get('learning_rates', [])))
        speed = []
        for a, b in zip(rows, list(rows)[1:]):
            dt = b.get('elapsed_session', 0)-a.get('elapsed_session', 0)
            ds = b['step']-a['step']
            if dt > 0 and ds > 0:
                speed.append(dt/ds)
        if speed:
            seconds = statistics.median(speed)
            eta = f'{max(0,total-row["step"])*seconds/3600:.1f} h' if isinstance(total, int) else '-'
            print(f'Speed (recent median): {seconds:.3f} s/step | ETA estimate: {eta}')
        age = max(0, time.time()-metrics.stat().st_mtime)
        print(f'Log age: {age:.0f} s' + (' | WARNING: no log write for 10 minutes' if age >= 600 else ''))
        if any(isinstance(row.get(k), (float,int)) and not math.isfinite(row[k]) for k in ('loss','prediction','gaussian','grad_norm','transport_grad_norm')):
            print('WARNING: non-finite loss/gradient')
        if validation:
            v = validation['validation']
            print(f'Validation @ {validation["step"]}: prediction={number(v.get("prediction"))}, SIGReg={number(v.get("gaussian"))}')
        if diagnostic:
            d = diagnostic['transport_diagnostics_pre_update']
            print(f'T diagnostics @ {diagnostic["step"]}: relative displacement={number(d.get("relative_displacement_rms"))}, variance z/u={number(d.get("z_variance_trace"))}/{number(d.get("u_variance_trace"))}')
    if bad_lines:
        print(f'WARNING: skipped {bad_lines} malformed complete metric lines')
    print('Read-only. Ctrl-C stops this monitor, not training.', flush=True)

try:
    while True:
        display()
        if args.once:
            break
        time.sleep(args.interval)
except KeyboardInterrupt:
    pass
PY
