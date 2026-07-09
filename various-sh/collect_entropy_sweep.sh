#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="entropy-sweep-aime"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_outputs/entropy-sweep-aime-%A_%a.out
#SBATCH --export=ALL
# ===========================================================================
# Sweep of evaluate_entropy_splice.py, ONE model load per job covering ALL
# retention rates at once (--retention_rate accepts a comma-separated list
# and loops in-process, see evaluate_entropy_splice.py docstring).
#
#   datasets:         aime2025, aime_2024, aime_2026
#   retention_rates:   1 2 5 10 15 20 25 30 40 50 60 70 80 90 100  (%) — ALL
#                      swept in one process per job
#   selector:          one per job (default: low_entropy). To sweep several
#                      selectors too, either pass a comma-separated
#                      --selector list (also handled in one process — see
#                      SELECTOR_MODE below) or add more array entries.
#   model:             openai/gpt-oss-20b  (edit MODEL below for other models)
#
# Default mode: one job per dataset (3 jobs total, --array=0-2), each
# sweeping all 15 retention rates for a single selector (low_entropy).
#
# Usage:
#   sbatch --partition=lovelace --gres=gpu:a100:1 --array=0-2 collect_entropy_sweep.sh
#
# To also sweep every selector within the SAME process (still one model
# load per job — cheaper than separate jobs), set SELECTOR_MODE=all below
# and resubmit; this makes each job cover 15 rates x 6 selectors = 90
# combos in one model load.
# ===========================================================================

PROJECT_DIR="/u/scandussio/entropy"
cd "$PROJECT_DIR"
source .venv/bin/activate

# --- Bypass CUDA 13.0 / flashinfer JIT mismatch: forza CUDA 12.4 ---
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
# ---------------------------------------------------------------------------

MODEL="openai/gpt-oss-20b"

DATASETS=(aime2025 aime_2024 aime_2026 zebralogic gpqa math-500 non-math-mmlu-pro)
RATES="0.01,0.02,0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00"

# Set to "all" to sweep every selector in the same process (90 combos/job
# instead of 15), or leave as a single selector name for a lighter run.
SELECTOR_MODE="low_entropy"
if [[ "$SELECTOR_MODE" == "all" ]]; then
    SELECTOR="low_entropy,high_entropy,numbers,newlines,end_of_sentence,random"
else
    SELECTOR="$SELECTOR_MODE"
fi

N_DATASETS=${#DATASETS[@]}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID not set. Submit with --array=0-$((N_DATASETS - 1))."
    exit 1
fi

if [[ "$SLURM_ARRAY_TASK_ID" -ge "$N_DATASETS" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range (max $((N_DATASETS - 1)))."
    exit 1
fi

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
TRACES_FILE="data/${DATASET}/gpt-oss-20b_teacher_traces.json"

# Per-dataset overrides
MAX_QUESTIONS_ARGS=()
if [[ "$DATASET" == "math-500" ]]; then
    MAX_QUESTIONS_ARGS=(--max_questions 100)
fi

echo "Job $SLURM_ARRAY_TASK_ID: dataset=$DATASET selector(s)=$SELECTOR"
echo "traces_file=$TRACES_FILE"
echo "retention_rates=$RATES"
if [[ ${#MAX_QUESTIONS_ARGS[@]} -gt 0 ]]; then
    echo "override: ${MAX_QUESTIONS_ARGS[*]}"
fi

if [[ ! -f "$TRACES_FILE" ]]; then
    echo "ERROR: $TRACES_FILE not found, skipping."
    exit 1
fi

python evaluate_entropy_splice.py \
    --model "$MODEL" \
    --traces_file "$TRACES_FILE" \
    --retention_rate "$RATES" \
    --selector "$SELECTOR" \
    --max_traces 8 \
    --max_new_tokens 50 \
    --skip_patched True \
    "${MAX_QUESTIONS_ARGS[@]}"