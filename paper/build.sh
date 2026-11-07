#!/usr/bin/env bash
# Build the manuscript. Source: manuscript.md (pandoc Markdown), references.bib.
# Output: build/manuscript.docx (Word, for submission as PDF after export) and build/manuscript.tex.
# JDMDH asks for a PDF at submission and reserves its template for accepted papers:
# export the .docx to PDF in Word, or compile the .tex on a machine with TeX.
set -euo pipefail
cd "$(dirname "$0")"
# Needs pandoc >= 2.11 for --citeproc. Set PANDOC=/path/to/pandoc if the system one is older.
PANDOC="${PANDOC:-pandoc}"
mkdir -p build figures
# SVG figures are converted to PNG when a converter exists; otherwise the figures stay as links.
for f in ../protocol/docs/figures/*.svg; do
  b=$(basename "$f" .svg); case "$b" in *-dark) continue;; esac
  if command -v rsvg-convert >/dev/null; then rsvg-convert -w 1600 "$f" -o "figures/$b.png"
  elif command -v inkscape >/dev/null; then inkscape "$f" -o "figures/$b.png" -w 1600 >/dev/null 2>&1
  else echo "note: no SVG converter found; figure $b not converted" >&2; fi
done
SRC=manuscript.md
if ls figures/*.png >/dev/null 2>&1; then sed 's#\.\./protocol/docs/figures/\([^)]*\)\.svg#figures/\1.png#g' manuscript.md > build/manuscript.png.md; SRC=build/manuscript.png.md; fi
"$PANDOC" "$SRC" --citeproc --bibliography=references.bib -o build/manuscript.docx
"$PANDOC" "$SRC" --citeproc --bibliography=references.bib -s -o build/manuscript.tex
echo "built: build/manuscript.docx build/manuscript.tex"
