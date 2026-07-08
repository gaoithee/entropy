#!/bin/bash
# Smoke test: 1 trace, 1 domanda, 1 selector, 1 retention rate
# Verifica solo che la pipeline giri (full/reached) su Gemma-4-E4B-it,
# Gemma-4-A26B-it e Qwen3-4B, prima dello sweep completo su Leonardo.

set -uo pipefail  # niente -e: vogliamo che continui anche se un modello fallisce

DATA_DIR="/share/ai-lab/scandussio/entropy/data/aime_2024"
RATE="0.30"
SELECTOR="low_entropy"
SUFFIX="_smoketest"

declare -A MODELS=(
    ["google/gemma-4-E4B-it"]="gemma-4-E4B-it"
    ["google/gemma-4-26B-A4B-it"]="gemma-4-26B-A4B-it"
    ["Qwen/Qwen3-4B"]="Qwen3-4B"
)

for MODEL_ID in "${!MODELS[@]}"; do
    MODEL_TAG="${MODELS[$MODEL_ID]}"
    TRACES_FILE="${DATA_DIR}/${MODEL_TAG}_teacher_traces.json"

    echo "============================================================"
    echo ">> Modello: ${MODEL_ID}"
    echo ">> Traces file atteso: ${TRACES_FILE}"

    if [ ! -f "${TRACES_FILE}" ]; then
        echo "!! ATTENZIONE: traces file non trovato, salto ${MODEL_ID}."
        echo "   (va generato prima con lo step di teacher trace generation)"
        continue
    fi

    LOG_FILE="smoketest_${MODEL_TAG}.log"

    python -u evaluate_entropy_sentence.py \
        --model "${MODEL_ID}" \
        --traces_file "${TRACES_FILE}" \
        --retention_rate "${RATE}" \
        --selector "${SELECTOR}" \
        --suffix_variant therefore_boxed \
        --max_traces 1 \
        --max_questions 1 \
        --skip_patched True \
        2>&1 | tee "${LOG_FILE}"

    STATUS=${PIPESTATUS[0]}
    if [ "${STATUS}" -ne 0 ]; then
        echo "!! ${MODEL_ID}: FALLITO (exit code ${STATUS}). Controlla ${LOG_FILE}."
    else
        echo ">> ${MODEL_ID}: eseguito senza errori. Riepilogo full/reached:"
        grep -E "full=|reached=" "${LOG_FILE}" || echo "   (nessuna riga full=/reached= trovata nel log, controlla manualmente)"
    fi
    echo "============================================================"
    echo
done

echo "Smoke test completato per tutti i modelli disponibili."
