#!/bin/bash
# ============================================================================
# Trova i task troncati guardando i log SLURM (.out) - criterio: l'ultima
# riga "processed N/M questions" ha N<M - e cancella SOLO i json prodotti da
# quello specifico (dataset, selector) per quel modello, per tutti i
# retention_rate (visto che ogni task scrive uno o piu' file per rate).
#
# Uso:
#   ./find_and_clean_truncated.sh            # dry-run, mostra cosa cancellerebbe
#   ./find_and_clean_truncated.sh --delete   # cancella davvero
#
# Esegui da ~/entropy (deve vedere sia slurm_outputs/ che data/)
# ============================================================================

set -uo pipefail

DELETE=false
[ "${1:-}" = "--delete" ] && DELETE=true

LOG_DIR="slurm_outputs"
DATA_DIR="data"

DATASETS=("aime_2024" "aime2025" "aime_2026" "math-500" "zebralogic" "gpqa")

# selettori per gli sbatch "sentence" (5 selettori)
SENTENCE_SELECTORS=("low_entropy" "high_entropy" "numbers" "low_entropy_no_numbers" "random")
# selettori per gli sbatch "splice"/token-level originali (6 selettori)
SPLICE_SELECTORS=("low_entropy" "high_entropy" "numbers" "newlines" "end_of_sentence" "random")

# job_base -> (model_tag, tipo: sentence|splice, n_selectors)
declare -A JOB_MODEL=(
    [48925325]="gpt-oss-20b:sentence"
    [48925546]="gpt-oss-120b:sentence"
    [48925607]="gemma-4-E4B-it:sentence"
    [48935275]="gemma-4-26B-A4B-it:sentence"
    [48926657]="Qwen3-4B:sentence"
    [48926658]="Qwen3-14B:sentence"
    [48867209]="gpt-oss-120b:splice"
    [48867510]="gpt-oss-120b:splice"
)

total_truncated=0
total_json_deleted=0

for jobid in "${!JOB_MODEL[@]}"; do
    entry="${JOB_MODEL[$jobid]}"
    model_tag="${entry%%:*}"
    kind="${entry##*:}"

    if [ "$kind" = "sentence" ]; then
        selectors=("${SENTENCE_SELECTORS[@]}")
    else
        selectors=("${SPLICE_SELECTORS[@]}")
    fi
    n_sel=${#selectors[@]}

    # trova tutti i log .out per questo jobid (qualunque prefisso di nome job)
    for logf in "$LOG_DIR"/*"${jobid}"_*.out; do
        [ -f "$logf" ] || continue
        taskid=$(echo "$logf" | grep -oE "${jobid}_[0-9]+" | grep -oE "[0-9]+$")
        [ -z "$taskid" ] && continue

        last=$(grep -oE "processed [0-9]+/[0-9]+ questions" "$logf" | tail -1)
        [ -z "$last" ] && continue  # nessuna riga "processed", non e' un task valido/avviato

        n=$(echo "$last" | grep -oE "[0-9]+" | sed -n '1p')
        m=$(echo "$last" | grep -oE "[0-9]+" | sed -n '2p')
        if [ -z "$n" ] || [ -z "$m" ]; then
            echo "AVVISO: impossibile leggere N/M da '$last' in $logf, salto"
            continue
        fi

        if [ "$n" -lt "$m" ]; then
            d_idx=$(( taskid / n_sel ))
            s_idx=$(( taskid % n_sel ))
            dataset="${DATASETS[$d_idx]}"
            selector="${selectors[$s_idx]}"

            total_truncated=$((total_truncated + 1))
            echo "TRONCATO: job=${jobid} task=${taskid} model=${model_tag} dataset=${dataset} selector=${selector} (${n}/${m})"

            # pattern dei json prodotti da questa combinazione (tutti i retention_rate)
            if [ "$kind" = "sentence" ]; then
                json_glob="${DATA_DIR}/${dataset}/${model_tag}_teacher_traces_eval_sentence_${selector}_r*_nopatch.json"
            else
                json_glob="${DATA_DIR}/${dataset}/${model_tag}_teacher_traces_eval_${selector}_r*.json"
            fi
            matched=$(ls $json_glob 2>/dev/null)
            if [ -n "$matched" ]; then
                echo "$matched" | while read -r jf; do
                    if $DELETE; then
                        rm -v "$jf"
                    else
                        echo "  [dry-run] rimuoverei: $jf"
                    fi
                done
                n_matched=$(echo "$matched" | wc -l)
                total_json_deleted=$((total_json_deleted + n_matched))
            else
                echo "  (nessun json trovato con pattern: $json_glob)"
            fi
        fi
    done
done

echo ""
echo "--- Riepilogo ---"
echo "Task troncati trovati: ${total_truncated}"
if $DELETE; then
    echo "JSON cancellati: ${total_json_deleted}"
else
    echo "JSON che VERREBBERO cancellati (dry-run): ${total_json_deleted}"
    echo "Rilancia con --delete per cancellare davvero."
fi
