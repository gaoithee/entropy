#!/bin/bash
set -e

REPO_ID="saracandu/entropy-traces"
DATASETS=("aime_2024" "aime2025" "aime_2026" "math-500" "zebralogic" "gpqa")

for ds in "${DATASETS[@]}"; do
    echo "=== Uploading eval results for $ds ==="
    hf upload "$REPO_ID" \
        "data/${ds}" \
        "${ds}/results" \
        --repo-type dataset \
        --include "*_eval_*.json" \
        --exclude "*teacher_traces.json"
done

echo "Done."
