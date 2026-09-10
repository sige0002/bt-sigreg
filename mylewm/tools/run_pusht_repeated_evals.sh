#!/usr/bin/env bash
# Five paired seeds for fixed-confirm and pinned-upstream PushT evaluation.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT"
BT='output/pusht/bt_spectral_v2_100k_s3072/step_100000_object.ckpt'
OFFICIAL='.cache/stable-wm/pusht/lewm_object.ckpt'
MANIFEST='.cache/stable-wm/pusht/rbg_v0/manifest.json'

for seed in 42 43 44 45 46; do
  for entry in "bt:$BT" "official:$OFFICIAL"; do
    name="${entry%%:*}"
    checkpoint="${entry#*:}"
    fixed="output/pusht/repeated_eval/fixed_${name}_seed${seed}"
    upstream="output/pusht/repeated_eval/upstream_${name}_seed${seed}"
    if [[ ! -f "$fixed/status.json" ]] || ! grep -q '"state": "succeeded"' "$fixed/status.json"; then
      bash mylewm/tools/evaluate_pusht.sh --checkpoint "$checkpoint" --manifest "$MANIFEST" \
        --seed "$seed" --gb10-cache-workaround --execute --output "$fixed"
    fi
    if [[ ! -f "$upstream/status.json" ]] || ! grep -q '"state": "succeeded"' "$upstream/status.json"; then
      uv run python mylewm/tools/evaluate_upstream_pusht.py --checkpoint "$checkpoint" --seed "$seed" \
        --output "$upstream"
    fi
  done
done
