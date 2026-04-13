#!/usr/bin/env bash
# Run KOA batch pipelines in one go:
#   1) JSN  — joint space → CSV only (no figure mode in scripts/jsn.py).
#   2) OST  — osteophyte → CSV + per-pair figures (requires *_0000 images where applicable).
#   3) SCL  — osteosclerosis → CSV + per-case figures (requires *_0000 images where applicable).
#
# Paths and outputs come from each config unless overridden via env:
#   JSN_OUTPUT, OST_OUTPUT_CSV, OST_FIGURE_DIR, SCL_OUTPUT_CSV, SCL_FIGURE_DIR
#
# Usage:
#   chmod +x scripts/run_jsn_osteophyte_sclerosis.sh
#   ./scripts/run_jsn_osteophyte_sclerosis.sh
#   QUIET=1 MPLBACKEND=Agg ./scripts/run_jsn_osteophyte_sclerosis.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KOA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${KOA_ROOT}"

PYTHON="${PYTHON:-python3}"
FIG_EXT="${FIG_EXT:-png}"

# Headless-safe matplotlib unless user already set MPLBACKEND
export MPLBACKEND="${MPLBACKEND:-Agg}"

echo "==> KOA_ROOT=${KOA_ROOT}"
echo "==> Using PYTHON=${PYTHON} FIG_EXT=${FIG_EXT}"

JSN_ARGS=()
if [[ -n "${JSN_OUTPUT:-}" ]]; then
  JSN_ARGS+=(--output "${JSN_OUTPUT}")
fi

OST_ARGS=(--batch-csv-and-figures --figure-ext "${FIG_EXT}" --no-show)
if [[ -n "${OST_OUTPUT_CSV:-}" ]]; then
  OST_ARGS+=(--output-csv "${OST_OUTPUT_CSV}")
fi
if [[ -n "${OST_FIGURE_DIR:-}" ]]; then
  OST_ARGS+=(--output-figure-dir "${OST_FIGURE_DIR}")
fi
if [[ -n "${QUIET:-}" ]]; then
  OST_ARGS+=(--quiet)
fi

SCL_ARGS=(--batch-csv-and-figures --figure-ext "${FIG_EXT}" --no-show)
if [[ -n "${SCL_OUTPUT_CSV:-}" ]]; then
  SCL_ARGS+=(--output-csv "${SCL_OUTPUT_CSV}")
fi
if [[ -n "${SCL_FIGURE_DIR:-}" ]]; then
  SCL_ARGS+=(--output-figure-dir "${SCL_FIGURE_DIR}")
fi
if [[ -n "${QUIET:-}" ]]; then
  SCL_ARGS+=(--quiet)
fi

echo ""
echo "========== (1/3) JSN =========="
"${PYTHON}" "${KOA_ROOT}/scripts/jsn.py" "${JSN_ARGS[@]}"

echo ""
echo "========== (2/3) OSTEOPHYTE =========="
"${PYTHON}" "${KOA_ROOT}/scripts/osteophyte.py" "${OST_ARGS[@]}"

echo ""
echo "========== (3/3) OSTEOSCLEROSIS =========="
"${PYTHON}" "${KOA_ROOT}/scripts/osteoscierosis.py" "${SCL_ARGS[@]}"

echo ""
echo "All three steps finished."
