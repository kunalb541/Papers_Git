#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

mkdir -p outputs/data outputs/figures outputs/tables outputs/logs

shopt -s nullglob
for f in *.aux *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz *.bbl *.blg *.nav *.snm; do
  mv "$f" outputs/logs/ 2>/dev/null || true
done
shopt -u nullglob

# Run BH package
python bh.py

# Compile manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=outputs/logs paper.tex

cp outputs/logs/paper.pdf ./paper.pdf

echo
echo "Done."
ls -1
echo "Figures:"; ls -1 outputs/figures 2>/dev/null || true
echo "Tables:"; ls -1 outputs/tables 2>/dev/null || true