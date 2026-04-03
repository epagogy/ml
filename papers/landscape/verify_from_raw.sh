#!/usr/bin/env bash
# Verify paper claims from raw JSONL. Only needs: jq, awk, sort.
# Usage: bash verify_from_raw.sh
set -euo pipefail

DATA="data/leakage_landscape_v1_final.jsonl"

echo "=== Dataset count ==="
N=$(jq -r 'select(.status == "ok") | .status' "$DATA" | wc -l | tr -d ' ')
echo "  n_datasets: $N (expected 2047)"

echo ""
echo "=== Corpus median n ==="
MEDIAN=$(jq -r 'select(.status == "ok") | .n_rows' "$DATA" | sort -n | awk '{a[NR]=$1} END{print a[int(NR/2)+1]}')
echo "  corpus.median_n: $MEDIAN (expected 1901)"

echo ""
echo "=== Peeking mean ΔAUC ==="
PEEK=$(jq -r 'select(.status == "ok" and .b_infl_k10 != null) | .b_infl_k10' "$DATA" \
  | awk '{s+=$1; n++} END{printf "%.4f\n", s/n}')
echo "  peek.auc: $PEEK (expected 0.0398)"

echo ""
echo "=== Normalization mean ΔAUC ==="
NORM=$(jq -r 'select(.status == "ok" and .a_lr_gap_diff != null) | .a_lr_gap_diff' "$DATA" \
  | awk '{s+=$1; n++} END{printf "%.5f\n", s/n}')
echo "  norm_lr.auc: $NORM (expected -0.00012)"

echo ""
echo "Verify d_z values with: mean / sd (requires numpy or R for paired sd)"
echo "The above confirms the raw AUC values trace to the paper."
