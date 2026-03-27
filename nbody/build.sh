#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

mkdir -p outputs/data outputs/figures outputs/tables outputs/logs

# ---- Move stale aux files into logs ----------------------------------------
shopt -s nullglob
for f in *.aux *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz *.bbl *.blg; do
  mv "$f" outputs/logs/
done
shopt -u nullglob

# ---- Step 1: Generate all numerical outputs --------------------------------
echo "=== Step 1: nbody_paper.py ==="
python nbody_paper.py "$@"

# ---- Step 2: Compile paper -------------------------------------------------
echo ""
echo "=== Step 2: latexmk ==="
latexmk -pdf \
        -interaction=nonstopmode \
        -halt-on-error \
        -outdir=outputs/logs \
        paper.tex

# ---- Copy final PDF to project root ----------------------------------------
cp outputs/logs/paper.pdf ./paper.pdf

echo ""
echo "Done."
echo ""
echo "Top level:"
ls -1
echo ""
echo "Figures:"
ls -1 outputs/figures 2>/dev/null || true
