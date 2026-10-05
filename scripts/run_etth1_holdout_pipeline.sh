#!/usr/bin/env bash
# ETTh1 hold-out: BECRA pipeline + paper appendix prompts.
# Stage 4: --planning-inject-scope matched (up to 8 diverse strict-matched lessons injected).
#
# Usage:
#   ./scripts/run_etth1_holdout_pipeline.sh
#   ./scripts/run_etth1_holdout_pipeline.sh --nohup
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/activate_becra.sh"

RUN_ID="${BECRA_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$ROOT/logs/etth1_holdout_${RUN_ID}"
OUT="$ROOT/outputs"
ARCHIVE="$OUT/archive_etth1_${RUN_ID}"

mkdir -p "$ARCHIVE"

archive_if_exists() {
  local f=$1
  if [[ -f "$f" ]]; then
    mv "$f" "$ARCHIVE/"
    echo "archived $(basename "$f") -> $ARCHIVE/"
  fi
}

echo "Archiving previous ETTh1 hold-out outputs (if any)..."
for f in \
  etth1_explore_results.csv etth1_explore_results.ucb.json etth1_explore_log.jsonl \
  etth1_candidate_lessons.json etth1_verified_lessons.json etth1_paired_rollout_cache.json \
  etth1_zeroshot.csv; do
  archive_if_exists "$OUT/$f"
done
if [[ -d "$OUT/etth1_lesson_induction_prompts" ]]; then
  mv "$OUT/etth1_lesson_induction_prompts" "$ARCHIVE/" 2>/dev/null || true
fi
if [[ -d "$OUT/etth1_lesson_induction_llm_raw" ]]; then
  mv "$OUT/etth1_lesson_induction_llm_raw" "$ARCHIVE/" 2>/dev/null || true
fi
LOG_ARCHIVE="$ARCHIVE/logs"
mkdir -p "$LOG_ARCHIVE"
shopt -s nullglob
for d in "$ROOT/logs/etth1_"*; do
  if [[ -d "$d" && "$d" != "$LOG_DIR" ]]; then
    mv "$d" "$LOG_ARCHIVE/"
    echo "archived log dir $(basename "$d") -> $LOG_ARCHIVE/"
  fi
done
shopt -u nullglob

mkdir -p "$LOG_DIR"

if [[ "${1:-}" == "--nohup" ]]; then
  shift
  echo "Starting detached ETTh1 pipeline -> $LOG_DIR/nohup.out"
  nohup env BECRA_RUN_ID="$RUN_ID" bash "$ROOT/scripts/run_etth1_holdout_pipeline.sh" "$@" \
    >"$LOG_DIR/nohup.out" 2>&1 &
  echo $! >"$LOG_DIR/pipeline.pid"
  echo "PID=$(cat "$LOG_DIR/pipeline.pid")"
  echo "LOG_DIR=$LOG_DIR"
  echo "tail -f $LOG_DIR/nohup.out"
  exit 0
fi

exec > >(tee -a "$LOG_DIR/pipeline.log") 2>&1
set -o pipefail
echo "=== BECRA ETTh1 hold-out ==="
echo "started: $(date -Is)"
echo "log_dir: $LOG_DIR"
echo "source datasets (explore/induce/verify): Weather ETTh2 ETTm1 ETTm2 Electricity"
echo "hold-out target: ETTh1"
echo "verify_planner: llm"
echo "planning_inject_scope: matched (inject up to 8 diverse strict-matched)"
echo "prompts: paper appendix (becra/prompt_templates.py)"
echo "toolchain_pool: $(python -c 'from becra.config import TOOLCHAINS; print(len(TOOLCHAINS))')"

SRC_DATASETS=(Weather ETTh2 ETTm1 ETTm2 Electricity)
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
  --results-csv "$OUT/etth1_explore_results.csv" \
  --log-path "$OUT/etth1_explore_log.jsonl"

stage induce "${PY[@]}" induce \
  --overwrite \
  --datasets "${SRC_DATASETS[@]}" \
  --results-csv "$OUT/etth1_explore_results.csv" \
  --prompt-dir "$OUT/etth1_lesson_induction_prompts" \
  --llm-raw-dir "$OUT/etth1_lesson_induction_llm_raw" \
  --lesson-json "$OUT/etth1_candidate_lessons.json" \
  --threshold-quantile 0.5 \
  --max-induction-groups 25 --max-lessons-per-group 7

stage verify "${PY[@]}" verify \
  --datasets "${SRC_DATASETS[@]}" \
  --pred-lens 96 192 336 720 \
  "${GPU_ARGS[@]}" \
  --results-csv "$OUT/etth1_explore_results.csv" \
  --lesson-json "$OUT/etth1_candidate_lessons.json" \
  --verified-json "$OUT/etth1_verified_lessons.json" \
  --rollout-cache "$OUT/etth1_paired_rollout_cache.json" \
  --overwrite \
  --top-alpha 0.35 --min-effect 0.0 --min-verify-support 2 --target-verify-datasets 3

stage run "${PY[@]}" run \
  --datasets ETTh1 \
  --pred-lens 96 192 336 720 \
  "${GPU_ARGS[@]}" \
  --planning-inject-scope matched \
  --planning-top-k 8 \
  --planning-llm-retries 3 \
  --lessons-json "$OUT/etth1_verified_lessons.json" \
  --results-csv "$OUT/etth1_zeroshot.csv" \
  --explore-rewiden-csv "$OUT/etth1_explore_results.csv" \
  --rewiden-source-datasets "${SRC_DATASETS[@]}"

echo ""
echo "=== FINISHED $(date -Is) ==="
echo "explore: $OUT/etth1_explore_results.csv"
echo "verified: $OUT/etth1_verified_lessons.json"
echo "etth1: $OUT/etth1_zeroshot.csv"
