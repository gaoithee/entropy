#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="collect-traces"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_outputs/collect-traces-%A_%a.out
#SBATCH --export=ALL

# ===========================================================================
# Job array — solo i job mancanti
#
# lovelace (A100 80GB) — gemma-4-26B su aime_2024, aime_2026, non-math-mmlu-pro
#   sbatch --partition=lovelace --gres=gpu:a100:1 --array=0-2 collect_traces_array.sh
#
# main (A100 40GB) — gemma-4-E4B, Qwen3-4B, Qwen3-14B (parziale), Phi-4 (parziale)
#   sbatch --partition=main --gres=gpu:a100:1 --array=3-23 collect_traces_array.sh
# ===========================================================================

PROJECT_DIR="/u/scandussio/entropy"
cd "$PROJECT_DIR"
source .venv-traces/bin/activate

export CUDA_HOME=/usr/local/cuda-12.4
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# ---------------------------------------------------------------------------
# Jobs espliciti: (model, dataset)
# ---------------------------------------------------------------------------
JOBS=(
    # lovelace — gemma-4-26B mancanti (0-2)
    "google/gemma-4-26B-A4B-it aime_2024"
    "google/gemma-4-26B-A4B-it aime_2026"
    "google/gemma-4-26B-A4B-it non-math-mmlu-pro"
    # main — gemma-4-E4B tutti i dataset (3-9)
    "google/gemma-4-E4B-it aime2025"
    "google/gemma-4-E4B-it aime_2024"
    "google/gemma-4-E4B-it aime_2026"
    "google/gemma-4-E4B-it zebralogic"
    "google/gemma-4-E4B-it math-500"
    "google/gemma-4-E4B-it non-math-mmlu-pro"
    "google/gemma-4-E4B-it gpqa"
    # main — Qwen3-4B mancanti (10-15)
    "Qwen/Qwen3-4B aime_2024"
    "Qwen/Qwen3-4B aime_2026"
    "Qwen/Qwen3-4B zebralogic"
    "Qwen/Qwen3-4B math-500"
    "Qwen/Qwen3-4B non-math-mmlu-pro"
    "Qwen/Qwen3-4B gpqa"
    # main — Qwen3-14B mancanti (16-18)
    "Qwen/Qwen3-14B aime_2024"
    "Qwen/Qwen3-14B aime_2026"
    "Qwen/Qwen3-14B non-math-mmlu-pro"
    # per ora lasciati fuori — Phi-4 mancanti (19-21)
    "microsoft/Phi-4-reasoning-plus aime_2024"
    "microsoft/Phi-4-reasoning-plus aime_2026"
    "microsoft/Phi-4-reasoning-plus non-math-mmlu-pro"
    # main — Qwen3-4B aime2025 che mi sono dimenticata prima (22)
    "Qwen/Qwen3-4B aime2025"
    # lovelace — gpt-oss-120b tutti i dataset (23-29)
    "openai/gpt-oss-120b aime2025"
    "openai/gpt-oss-120b aime_2024"
    "openai/gpt-oss-120b aime_2026"
    "openai/gpt-oss-120b zebralogic"
    "openai/gpt-oss-120b math-500"
    "openai/gpt-oss-120b non-math-mmlu-pro"
    "openai/gpt-oss-120b gpqa"
)

JOB="${JOBS[$SLURM_ARRAY_TASK_ID]}"
MODEL=$(echo "$JOB" | awk '{print $1}')
DATASET=$(echo "$JOB" | awk '{print $2}')

echo "Job $SLURM_ARRAY_TASK_ID: model=$MODEL dataset=$DATASET"

# ---------------------------------------------------------------------------
# Parametri per-dataset
# ---------------------------------------------------------------------------
case "$DATASET" in
    "zebralogic")
        NUM_QUESTIONS=50
        ;;
    "math-500")
        NUM_QUESTIONS=100
        ;;
    *)
        NUM_QUESTIONS=""
        ;;
esac

# ---------------------------------------------------------------------------
# Lancia collect_traces.py
# ---------------------------------------------------------------------------
CMD=(
    python scripts/collect_traces.py
    --model_name "$MODEL"
    --data_name "$DATASET"
    --num_out 16
    --batch_size 64
    --resume True
    --max_tokens 16384
    --gpu_memory_utilization 0.9
    --max_model_len 22000
)

if [[ -n "$NUM_QUESTIONS" ]]; then
    CMD+=(--num_questions "$NUM_QUESTIONS")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
