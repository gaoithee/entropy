#!/usr/bin/env bash
# =============================================================================
# sync_to_hf.sh
#
# Sincronizza data/{dataset}/ locale verso saracandu/entropy-traces su HF,
# instradando i file per PATTERN DEL NOME (non per posizione locale, che puo'
# variare tra dataset: alcuni hanno gia' results/results_sentence/, altri
# hanno tutto piatto). Stessa logica di classify() nel vecchio push_to_hf.py:
#
#   *_teacher_traces.json              -> {dataset}/{fname}            (root)
#   *_eval_sentence_*_nopatch.json     -> {dataset}/results_sentence/{fname}
#   *_eval_*.json (senza "_sentence_") -> {dataset}/results/{fname}
#
# hf upload fa gia' il diff via hash: i file identici a quelli gia' presenti
# sul repo vengono saltati automaticamente, quindi puoi rilanciare questo
# script tutte le volte che vuoi senza ricaricare tutto.
#
# Uso:
#   bash sync_to_hf.sh                  # tutti i dataset trovati in data/
#   bash sync_to_hf.sh aime2025 gpqa    # solo alcuni dataset
# =============================================================================

set -euo pipefail

REPO_ID="saracandu/entropy-traces"
DATA_ROOT="data"

if [ "$#" -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=()
    for d in "$DATA_ROOT"/*/; do
        DATASETS+=("$(basename "$d")")
    done
fi

stage_and_upload () {
    # $1 = descrizione, $2 = remote subpath (puo' essere vuoto per la root),
    # resto = lista di file locali da linkare
    local desc="$1"; shift
    local remote_sub="$1"; shift
    local files=("$@")

    if [ "${#files[@]}" -eq 0 ]; then
        return
    fi

    local staging
    staging="$(mktemp -d)"
    for f in "${files[@]}"; do
        ln -s "$(readlink -f "$f")" "$staging/$(basename "$f")"
    done

    echo "-- upload $desc (${#files[@]} file)..."
    if [ -z "$remote_sub" ]; then
        hf upload "$REPO_ID" "$staging/" "$dataset/" --repo-type dataset
    else
        hf upload "$REPO_ID" "$staging/" "$dataset/$remote_sub/" --repo-type dataset
    fi
    rm -rf "$staging"
}

for dataset in "${DATASETS[@]}"; do
    local_dir="$DATA_ROOT/$dataset"
    if [ ! -d "$local_dir" ]; then
        echo "!! $local_dir non esiste, salto."
        continue
    fi

    echo "=== $dataset ==="

    root_files=()
    results_files=()
    results_sentence_files=()

    while IFS= read -r -d '' f; do
        fname="$(basename "$f")"
        case "$fname" in
            *_teacher_traces.json)
                root_files+=("$f")
                ;;
            *_eval_sentence_*_nopatch.json)
                results_sentence_files+=("$f")
                ;;
            *_eval_*.json)
                results_files+=("$f")
                ;;
            *)
                :
                ;;
        esac
    done < <(find "$local_dir" -maxdepth 2 -type f -name '*.json' -print0)

    if [ "${#root_files[@]}" -gt 0 ]; then
        stage_and_upload "file root" "" "${root_files[@]}"
    fi
    if [ "${#results_files[@]}" -gt 0 ]; then
        stage_and_upload "results/" "results" "${results_files[@]}"
    fi
    if [ "${#results_sentence_files[@]}" -gt 0 ]; then
        stage_and_upload "results_sentence/" "results_sentence" "${results_sentence_files[@]}"
    fi

    echo ""
done

echo "Sync completato."
