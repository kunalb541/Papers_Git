#!/usr/bin/env bash
# build_full.sh
# Usage: bash build_full.sh [--workers N] [--n-pred N] [--n-caus N]
# Runs: mkdir -> evol_battery.py -> paper_analysis.py -> latexmk
set -e

# ── Defaults ─────────────────────────────────────────────────────────────────
WORKERS=6
N_PRED=100000
N_CAUS=100000
OUTDIR="."

while [[ $# -gt 0 ]]; do
    case $1 in
        --workers)  WORKERS="$2";  shift 2 ;;
        --n-pred)   N_PRED="$2";   shift 2 ;;
        --n-caus)   N_CAUS="$2";   shift 2 ;;
        --outdir)   OUTDIR="$2";   shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Step 1: directories ───────────────────────────────────────────────────────
echo "=== Step 1: create output directories ==="
mkdir -p outputs/figures
mkdir -p outputs/tables
mkdir -p outputs/data
mkdir -p outputs/logs

# ── Step 2: run battery ───────────────────────────────────────────────────────
echo ""
echo "=== Step 2: run evol_battery.py ==="
echo "    workers=$WORKERS  n-pred=$N_PRED  n-caus=$N_CAUS  outdir=$OUTDIR"
python evol_battery.py \
    --workers "$WORKERS" \
    --n-pred  "$N_PRED"  \
    --n-caus  "$N_CAUS"  \
    --outdir  "$OUTDIR"


# ── Step 3: generate figures / tables / macros ────────────────────────────────
echo ""
echo "=== Step 3: paper_analysis.py ==="
python paper_analysis.py --results "$OUTDIR/results.json"

# ── Step 4: compile LaTeX ─────────────────────────────────────────────────────
echo ""
echo "=== Step 4: latexmk (three passes) ==="
latexmk -pdf \
    -auxdir=outputs/logs \
    -outdir=outputs/logs \
    paper.tex

cp outputs/logs/paper.pdf ./paper.pdf
echo ""
echo "Build complete -> paper.pdf"

