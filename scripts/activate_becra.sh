#!/usr/bin/env bash
# Source from pipeline scripts:  source "$(dirname "$0")/activate_becra.sh"
set -euo pipefail

export BECRA_ROOT="${BECRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export TMPDIR="${TMPDIR:-${BECRA_ROOT}/.tmp}"
mkdir -p "$TMPDIR" "$BECRA_ROOT/logs" "$BECRA_ROOT/outputs"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate becra
fi

cd "$BECRA_ROOT"
export PYTHONUNBUFFERED=1
