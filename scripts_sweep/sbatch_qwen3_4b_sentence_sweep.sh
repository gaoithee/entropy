#!/bin/bash
#SBATCH --account=uTS26_Bortolus
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name="qwen3-4b-sentence-sweep"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_outputs/qwen3-4b-sentence-sweep-%A_%a.out
#SBATCH --export=ALL

# ===========================================================================
# Full retention/selector sweep for Qwen/Qwen3-4B, SENTENCE-LEVEL
# (evaluate_entropy_sentence.py), one (dataset, selector) combo per array
# task. --array=0-29 (6 datasets x 5 selectors).
#
# NOTE: no empirical per-task timing yet for the sentence-level script
# (smoke test was 1 trace / 1 question only). --time is a conservative
# placeholder copied from the gpt-oss-20b splice sweep; adjust after the
# first task(s) complete and you have a real minutes/question figure.
# ===========================================================================

PROJECT_DIR="$HOME/entropy"
cd "$PROJECT_DIR"
source .venv/bin/activate
export HF_HOME=$WORK/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RATES="0.01,0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.0"
MAX_TRACES=8

DATASETS=(
    "data/aime_2024/Qwen3-4B_teacher_traces.json"
    "data/aime2025/Qwen3-4B_teacher_traces.json"
    "data/aime_2026/Qwen3-4B_teacher_traces.json"
    "data/math-500/Qwen3-4B_teacher_traces.json"
    "data/zebralogic/Qwen3-4B_teacher_traces.json"
    "data/gpqa/Qwen3-4B_teacher_traces.json"
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
    --model Qwen/Qwen3-4B \
    --traces_file "$TRACES_FILE" \
    --retention_rate "$RATES" \
    --selector "$SEL" \
    --suffix_variant therefore_boxed \
    --max_traces $MAX_TRACES \
    --skip_patched True \
    $MAX_Q_ARG

echo "Task $SLURM_ARRAY_TASK_ID done."
