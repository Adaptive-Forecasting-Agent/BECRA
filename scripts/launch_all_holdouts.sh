#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/activate_becra.sh"
BATCH_ID="${BECRA_BATCH_ID:-$(date +%Y%m%d_%H%M%S)}"
BATCH_LOG_DIR="$ROOT/logs/all_holdouts_${BATCH_ID}"
mkdir -p "$BATCH_LOG_DIR"
nohup env BECRA_BATCH_ID="$BATCH_ID" bash "$ROOT/scripts/run_all_holdouts_sequential.sh" "$@" \
  >"$BATCH_LOG_DIR/nohup.out" 2>&1 &
echo $! >"$BATCH_LOG_DIR/batch.pid"
echo "PID=$(cat "$BATCH_LOG_DIR/batch.pid")"
echo "BATCH_LOG_DIR=$BATCH_LOG_DIR"
echo "BATCH_ID=$BATCH_ID"
echo "tail -f $BATCH_LOG_DIR/nohup.out"
