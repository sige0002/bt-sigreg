#!/usr/bin/env bash
# Local LIBERO rendering runtime; does not alter the PushT environment.
set -euo pipefail
research_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mesa_path="${LIBERO_OSMESA_DIR:-$research_root/.cache/libero-osmesa/usr/lib/$(uname -m)-linux-gnu}"
if [[ ! -f "$mesa_path/libOSMesa.so.8" ]]; then
  echo "Missing $mesa_path/libOSMesa.so.8; set LIBERO_OSMESA_DIR to the directory containing it." >&2
  exit 1
fi
export LD_LIBRARY_PATH="$mesa_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$research_root/.cache/libero-config}"
export LIBERO_ROOT="${LIBERO_ROOT:-$research_root/external/libero}"
export LIBERO_MUJOCO_PATH="${LIBERO_MUJOCO_PATH:-$research_root/.cache/libero-runtime}"
export PYTHONPATH="$LIBERO_MUJOCO_PATH:$LIBERO_ROOT:$research_root/src:$research_root${PYTHONPATH:+:$PYTHONPATH}"
cd "$research_root"
exec uv run --no-sync python "$@"
