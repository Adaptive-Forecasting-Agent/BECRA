#!/usr/bin/env bash
# Sequentially run all six BECRA hold-outs (4 stages each; wait until one finishes).
# Order: Weather -> ETTm2 -> Electricity -> ETTh1 -> ETTh2 -> ETTm1
#
# Usage:
#   ./scripts/run_all_holdouts_sequential.sh
#   ./scripts/run_all_holdouts_sequential.sh --nohup
#   ./scripts/run_all_holdouts_sequential.sh --from electricity
#   ./scripts/run_all_holdouts_sequential.sh --only ettm2
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/activate_becra.sh"

BATCH_ID="${BECRA_BATCH_ID:-$(date +%Y%m%d_%H%M%S)}"
BATCH_LOG_DIR="$ROOT/logs/all_holdouts_${BATCH_ID}"
mkdir -p "$BATCH_LOG_DIR"

ORDER=(weather ettm2 electricity etth1 etth2 ettm1)

script_for() {
  case "$1" in
    weather)     echo "$ROOT/scripts/run_weather_holdout_pipeline.sh" ;;
    ettm2)       echo "$ROOT/scripts/run_ettm2_holdout_pipeline.sh" ;;
    electricity) echo "$ROOT/scripts/run_electricity_holdout_pipeline.sh" ;;
    etth1)       echo "$ROOT/scripts/run_etth1_holdout_pipeline.sh" ;;
    etth2)       echo "$ROOT/scripts/run_etth2_holdout_pipeline.sh" ;;
    ettm1)       echo "$ROOT/scripts/run_ettm1_holdout_pipeline.sh" ;;
    *) echo "unknown holdout key: $1" >&2; return 1 ;;
  esac
}

FROM=""
ONLY=""
DETACH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nohup) DETACH=1; shift ;;
    --from)
      FROM="${2:?--from needs a key}"
      shift 2
      ;;
    --only)
      ONLY="${2:?--only needs a key}"
      shift 2
      ;;
    -h|--help)
      sed -n '1,20p' "$0"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ "$DETACH" -eq 1 ]]; then
  echo "Starting detached all-holdouts batch -> $BATCH_LOG_DIR/nohup.out"
  nohup env BECRA_BATCH_ID="$BATCH_ID" bash "$ROOT/scripts/run_all_holdouts_sequential.sh" \
    ${FROM:+--from "$FROM"} ${ONLY:+--only "$ONLY"} \
    >"$BATCH_LOG_DIR/nohup.out" 2>&1 &
  echo $! >"$BATCH_LOG_DIR/batch.pid"
  echo "PID=$(cat "$BATCH_LOG_DIR/batch.pid")"
  echo "BATCH_LOG_DIR=$BATCH_LOG_DIR"
  echo "tail -f $BATCH_LOG_DIR/nohup.out"
  exit 0
fi

exec > >(tee -a "$BATCH_LOG_DIR/batch.log") 2>&1
echo "=== All six hold-outs sequential ==="
echo "started: $(date -Is)"
echo "batch_id: $BATCH_ID"
echo "batch_log: $BATCH_LOG_DIR"
echo "order: ${ORDER[*]}"
echo "toolchain_pool: $(python -c 'from becra.config import TOOLCHAINS; print(len(TOOLCHAINS))')"
python - <<'PY'
from becra.config import TOOLCHAINS
from becra.ucb import UNPULLED_PRIOR_STD
must = [
    "none_none_none_fft_standard_timexer",
    "none_none_none_fft_standard_multipatchformer",
    "none_none_none_none_standard_timexer",
    "none_none_none_none_standard_multipatchformer",
]
for name in must:
    print(f"  grid has {name}: {name in TOOLCHAINS}")
print(f"  TiDE in pool: {sum(1 for n in TOOLCHAINS if 'tide' in n)}")
print(f"  UNPULLED_PRIOR_STD: {UNPULLED_PRIOR_STD}")
PY

KEYS=("${ORDER[@]}")
if [[ -n "$ONLY" ]]; then
  KEYS=("$ONLY")
elif [[ -n "$FROM" ]]; then
  start=-1
  for i in "${!ORDER[@]}"; do
    if [[ "${ORDER[$i]}" == "$FROM" ]]; then start=$i; break; fi
  done
  if [[ $start -lt 0 ]]; then
    echo "unknown --from key: $FROM" >&2
    exit 1
  fi
  KEYS=("${ORDER[@]:$start}")
fi

STATUS_JSONL="$BATCH_LOG_DIR/status.jsonl"
: >"$STATUS_JSONL"

for key in "${KEYS[@]}"; do
  script="$(script_for "$key")"
  echo ""
  echo ">>>>>>>>>> START $key $(date -Is) <<<<<<<<<<"
  echo "{\"event\":\"start\",\"holdout\":\"$key\",\"time\":\"$(date -Is)\"}" >>"$STATUS_JSONL"
  BECRA_RUN_ID="${BATCH_ID}_${key}" bash "$script"
  echo ">>>>>>>>>> FINISHED $key $(date -Is) <<<<<<<<<<"
  echo "{\"event\":\"finish\",\"holdout\":\"$key\",\"time\":\"$(date -Is)\"}" >>"$STATUS_JSONL"
done

echo ""
echo "=== BATCH DONE $(date -Is) ==="
for key in weather ettm2 electricity etth1 etth2 ettm1; do
  echo "  ${key}: $ROOT/outputs/${key}_zeroshot.csv"
done
