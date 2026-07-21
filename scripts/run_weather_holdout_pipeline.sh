#!/usr/bin/env bash
# Weather hold-out: BECRA pipeline + paper appendix prompts.
# Stage 4: --planning-inject-scope matched (all strict-matched injected).
#
# Usage:
#   ./scripts/run_weather_holdout_pipeline.sh
#   ./scripts/run_weather_holdout_pipeline.sh --nohup
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/activate_becra.sh"

RUN_ID="${BECRA_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$ROOT/logs/weather_holdout_${RUN_ID}"
OUT="$ROOT/outputs"
ARCHIVE="$OUT/archive_weather_${RUN_ID}"

mkdir -p "$LOG_DIR" "$ARCHIVE"

archive_if_exists() {
  local f=$1
  if [[ -f "$f" ]]; then
    mv "$f" "$ARCHIVE/"
    echo "archived $(basename "$f") -> $ARCHIVE/"
  fi
}

echo "Archiving previous Weather hold-out outputs (if any)..."
for f in \
  weather_explore_results.csv weather_explore_results.ucb.json weather_explore_log.jsonl \
  weather_candidate_lessons.json weather_verified_lessons.json weather_paired_rollout_cache.json \
  weather_zeroshot.csv weather_zeroshot_mechanism_test.csv weather_verified_mechanism_test.json; do
  archive_if_exists "$OUT/$f"
done
if [[ -d "$OUT/weather_lesson_induction_prompts" ]]; then
  mv "$OUT/weather_lesson_induction_prompts" "$ARCHIVE/" 2>/dev/null || true
fi
if [[ -d "$OUT/weather_lesson_induction_llm_raw" ]]; then
  mv "$OUT/weather_lesson_induction_llm_raw" "$ARCHIVE/" 2>/dev/null || true
fi
if compgen -G "$ROOT/logs/weather_holdout_*" >/dev/null 2>&1; then
  mkdir -p "$ARCHIVE/logs"
  for d in "$ROOT"/logs/weather_holdout_*; do
    [[ -d "$d" ]] || continue
    [[ "$d" == "$LOG_DIR" ]] && continue
    mv "$d" "$ARCHIVE/logs/"
    echo "archived log $(basename "$d") -> $ARCHIVE/logs/"
  done
fi

if [[ "${1:-}" == "--nohup" ]]; then
  shift
  echo "Starting detached Weather pipeline -> $LOG_DIR/nohup.out"
  nohup env BECRA_RUN_ID="$RUN_ID" bash "$ROOT/scripts/run_weather_holdout_pipeline.sh" "$@" \
    >"$LOG_DIR/nohup.out" 2>&1 &
  echo $! >"$LOG_DIR/pipeline.pid"
  echo "PID=$(cat "$LOG_DIR/pipeline.pid")"
  echo "tail -f $LOG_DIR/nohup.out"
  exit 0
fi

exec > >(tee -a "$LOG_DIR/pipeline.log") 2>&1
set -o pipefail
echo "=== BECRA Weather hold-out ==="
echo "started: $(date -Is)"
echo "log_dir: $LOG_DIR"
echo "source datasets (explore/induce/verify): ETTh1 ETTh2 ETTm1 ETTm2 Electricity"
echo "hold-out target: Weather"
echo "verify_planner: llm"
echo "planning_inject_scope: matched (inject up to 8 diverse strict-matched)"
echo "prompts: paper appendix (becra/prompt_templates.py)"
echo "toolchain_pool: $(python -c 'from becra.config import TOOLCHAINS; print(len(TOOLCHAINS))')"

SRC_DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Electricity)
GPU_ARGS=(--gpus ${BECRA_GPUS:-0 1} --max-parallel 2)
PY=(python scripts/run_becra_long_term.py)

stage() {
  local name=$1
  shift
  echo ""
  echo "========== STAGE: $name =========="
  echo "time: $(date -Is)"
  "$@" 2>&1 | tee -a "$LOG_DIR/${name}.log"
}

stage explore "${PY[@]}" explore \
  --datasets "${SRC_DATASETS[@]}" \
  --pred-lens 96 192 336 720 \
  --rounds 12 --k-pos 4 --k-neg 2 --T-noise 3 --ucb-lambda 0.6 \
  --ucb-unpulled-bonus 1.0 --ucb-exploration-c 0.3 \
  --explore-stratify-forecasting \
  "${GPU_ARGS[@]}" \
  --results-csv "$OUT/weather_explore_results.csv" \
  --log-path "$OUT/weather_explore_log.jsonl"

stage induce "${PY[@]}" induce \
  --use-llm --no-heuristic --overwrite \
  --datasets "${SRC_DATASETS[@]}" \
  --results-csv "$OUT/weather_explore_results.csv" \
  --prompt-dir "$OUT/weather_lesson_induction_prompts" \
  --llm-raw-dir "$OUT/weather_lesson_induction_llm_raw" \
  --lesson-json "$OUT/weather_candidate_lessons.json" \
  --threshold-quantile 0.5 \
  --induction-mode paper \
  --max-induction-groups 25 --max-lessons-per-group 7

stage verify "${PY[@]}" verify \
  --paired-rollout --rollout-mode execute \
  --verify-planner llm \
  --no-seed \
  --datasets "${SRC_DATASETS[@]}" \
  --pred-lens 96 192 336 720 \
  "${GPU_ARGS[@]}" \
  --results-csv "$OUT/weather_explore_results.csv" \
  --lesson-json "$OUT/weather_candidate_lessons.json" \
  --verified-json "$OUT/weather_verified_lessons.json" \
  --rollout-cache "$OUT/weather_paired_rollout_cache.json" \
  --overwrite \
  --top-alpha 0.35 --min-effect 0.0 --min-verify-support 2 --target-verify-datasets 3

stage run "${PY[@]}" run \
  --planner llm \
  --datasets Weather \
  --pred-lens 96 192 336 720 \
  "${GPU_ARGS[@]}" \
  --no-seed \
  --planning-mode paper \
  --planning-inject-scope matched \
  --planning-top-k 8 \
  --planning-llm-retries 3 \
  --lessons-json "$OUT/weather_verified_lessons.json" \
  --results-csv "$OUT/weather_zeroshot.csv"

echo ""
echo "=== FINISHED $(date -Is) ==="
echo "explore: $OUT/weather_explore_results.csv"
echo "verified: $OUT/weather_verified_lessons.json"
echo "weather: $OUT/weather_zeroshot.csv"
