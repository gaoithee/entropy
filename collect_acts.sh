#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="collect-acts"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_outputs/collect-acts-%A_%a.out
#SBATCH --export=ALL

# ===========================================================================
# Job array per raccogliere activations + entropies HF
#
# lovelace (A100 80GB) — modelli grandi (0-15)
#   sbatch --partition=lovelace --gres=gpu:a100:1 --array=0-15 collect_acts.sh
#
# babbage (A100 40GB) — gemma-4-E4B (16)
#   sbatch --partition=main --gres=gpu:a100:1 --array=16 collect_acts.sh
# ===========================================================================

PROJECT_DIR="/u/scandussio/entropy"
cd "$PROJECT_DIR"

source .venv/bin/activate

# ---------------------------------------------------------------------------
# Jobs espliciti: (model, dataset)
# ---------------------------------------------------------------------------
JOBS=(
    # lovelace — gpt-oss-20b mancanti (0-2)
    "openai/gpt-oss-20b aime_2026"
    "openai/gpt-oss-20b math-500"
    "openai/gpt-oss-20b gpqa"
    # lovelace — gemma-4-26B (3-8)
    "google/gemma-4-26B-A4B-it aime2025"
    "google/gemma-4-26B-A4B-it aime_2024"
    "google/gemma-4-26B-A4B-it aime_2026"
    "google/gemma-4-26B-A4B-it zebralogic"
    "google/gemma-4-26B-A4B-it math-500"
    "google/gemma-4-26B-A4B-it gpqa"
    # lovelace — Qwen3-14B (9-12)
    "Qwen/Qwen3-14B aime2025"
    "Qwen/Qwen3-14B zebralogic"
    "Qwen/Qwen3-14B math-500"
    "Qwen/Qwen3-14B gpqa"
    # lovelace — Phi-4 (13-16)
    "microsoft/Phi-4-reasoning-plus aime2025"
    "microsoft/Phi-4-reasoning-plus zebralogic"
    "microsoft/Phi-4-reasoning-plus math-500"
    "microsoft/Phi-4-reasoning-plus gpqa"
    # babbage A100 40GB — gemma-4-E4B (17-19)
    "google/gemma-4-E4B-it aime2025"
    "google/gemma-4-E4B-it aime_2024"
    "google/gemma-4-E4B-it aime_2026"
)

JOB="${JOBS[$SLURM_ARRAY_TASK_ID]}"
MODEL=$(echo "$JOB" | awk '{print $1}')
DATASET=$(echo "$JOB" | awk '{print $2}')

echo "Job $SLURM_ARRAY_TASK_ID: model=$MODEL dataset=$DATASET"

OUTPUT_DIR="outputs/${MODEL//\//_}/${DATASET}"

python scripts/collect_activations.py \
    --model_name "$MODEL" \
    --data_name "$DATASET" \
    --output_dir "$OUTPUT_DIR"

echo "Done. Output: $OUTPUT_DIR"