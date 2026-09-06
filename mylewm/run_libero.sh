#!/usr/bin/env bash
# Local LIBERO rendering runtime; does not alter the PushT environment.
set -euo pipefail
research_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mesa_path="$research_root/.cache/libero-osmesa/usr/lib/aarch64-linux-gnu"
if [[ ! -f "$mesa_path/libOSMesa.so.8" ]]; then
  echo 'Missing local OSMesa runtime; see mylewm/README.md.' >&2
  exit 1
fi
export LD_LIBRARY_PATH="$mesa_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LIBERO_CONFIG_PATH="$research_root/.cache/libero-config"
export PYTHONPATH="$research_root/.cache/libero-runtime:$research_root/external/libero:$research_root${PYTHONPATH:+:$PYTHONPATH}"
exec "$research_root/.venv/bin/python" "$@"
