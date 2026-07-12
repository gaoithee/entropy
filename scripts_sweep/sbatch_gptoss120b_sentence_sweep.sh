#!/bin/bash
#SBATCH --account=uTS26_Bortolus
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name="gptoss120b-sentence-sweep"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=60:00:00
#SBATCH --qos=boost_qos_lprod
#SBATCH --array=14-29
#SBATCH --output=slurm_outputs/gptoss120b-sentence-sweep-%A_%a.out
#SBATCH --export=ALL

# ===========================================================================
# Full retention/selector sweep for openai/gpt-oss-120b, SENTENCE-LEVEL
# (evaluate_entropy_sentence.py, --skip_patched True), one (dataset,
# selector) combo per array task. --array=14-29 (tasks 0-13 already done,
# resuming from the 14th onward; 6 datasets x 5 selectors total = 0-29).
#
# gres/cpus lowered from the original 4 GPU / 32 CPU: gpt-oss-120b only
# needs 2 GPU per current sizing, and cpus-per-task=4 to cut CPU-hour
# burn rate (project monthly budget was getting throttled at 32 CPU/task).
# ===========================================================================

PROJECT_DIR="$HOME/entropy"
cd "$PROJECT_DIR"
source .venv/bin/activate
export HF_HOME=$WORK/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR="$WORK/triton_cache/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TRITON_CACHE_DIR"

RATES="0.01,0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.0"
MAX_TRACES=8

DATASETS=(
    "data/aime_2024/gpt-oss-120b_teacher_traces.json"
    "data/aime2025/gpt-oss-120b_teacher_traces.json"
    "data/aime_2026/gpt-oss-120b_teacher_traces.json"
    "data/math-500/gpt-oss-120b_teacher_traces.json"
    "data/zebralogic/gpt-oss-120b_teacher_traces.json"
    "data/gpqa/gpt-oss-120b_teacher_traces.json"
)
# max_questions per dataset, aligned by index with DATASETS above (0 = all)
MAX_Q=(0 0 0 100 50 50)

ALL_SELECTORS=("low_entropy" "high_entropy" "numbers" "low_entropy_no_numbers" "random")

N_DATASETS=${#DATASETS[@]}
N_SELECTORS=${#ALL_SELECTORS[@]}

DATASET_IDX=$(( SLURM_ARRAY_TASK_ID / N_SELECTORS ))
SELECTOR_IDX=$(( SLURM_ARRAY_TASK_ID % N_SELECTORS ))

TRACES_FILE="${DATASETS[$DATASET_IDX]}"
Q_COUNT="${MAX_Q[$DATASET_IDX]}"
SEL="${ALL_SELECTORS[$SELECTOR_IDX]}"

echo "Task $SLURM_ARRAY_TASK_ID: traces_file=$TRACES_FILE selector=$SEL max_questions=$Q_COUNT"

if [ "$Q_COUNT" = "0" ]; then
    MAX_Q_ARG=""
else
    MAX_Q_ARG="--max_questions $Q_COUNT"
fi

python -u evaluate_entropy_sentence.py \
    --model openai/gpt-oss-120b \
    --traces_file "$TRACES_FILE" \
    --retention_rate "$RATES" \
    --selector "$SEL" \
    --suffix_variant therefore_boxed \
    --max_traces $MAX_TRACES \
    --skip_patched True \
    $MAX_Q_ARG

echo "Task $SLURM_ARRAY_TASK_ID done."
