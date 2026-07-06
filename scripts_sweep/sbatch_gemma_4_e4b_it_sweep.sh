#!/bin/bash
#SBATCH --account=uTS26_Bortolus
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name="gemma_4_e4b_it-sweep"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=slurm_outputs/gemma_4_e4b_it-sweep-%A_%a.out
#SBATCH --export=ALL

# ===========================================================================
# Full retention/selector sweep for gpt-oss-20b, one whole dataset per array
# task (no question chunking). --array=0-5.
# Per-task estimated duration (~18.4 min/question, measured empirically):
#   aime_2024   (30 q) ~9.2h    aime2025    (30 q) ~9.2h
#   aime_2026   (30 q) ~9.2h    math-500   (100 q) ~30.7h  <- longest
#   zebralogic  (50 q) ~15.3h   gpqa        (50 q) ~15.3h
# --time is set to the worst case (math-500) plus margin.
# ===========================================================================

PROJECT_DIR="$HOME/entropy"
cd "$PROJECT_DIR"
source .venv/bin/activate
export HF_HOME=$WORK/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RATES="0.01,0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.0"
SELECTORS="low_entropy,high_entropy,numbers,newlines,end_of_sentence,random"
MAX_TRACES=8

# ---------------------------------------------------------------------------
# JOBS: (traces_file, max_questions)  -- max_questions=0 means "all"
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# JOBS: (traces_file, selector) -- one selector per task, full retention_rate
# sweep still handled internally (model loaded once, all 12 rates run in the
# same process). Splitting per (dataset, selector) instead of per question
# chunk means: (a) every task stays well under the 24h cap without needing to
# split math-500, and (b) a failed/timed-out task only loses ONE selector's
# results for ONE dataset, not an arbitrary chunk of questions -- there is no
# resume/checkpoint in evaluate_entropy_splice.py, so granularity here is the
# only blast-radius control we have.
# ---------------------------------------------------------------------------
DATASETS=(
    "data/aime_2024/gemma-4-E4B-it_teacher_traces.json"
    "data/aime2025/gemma-4-E4B-it_teacher_traces.json"
    "data/aime_2026/gemma-4-E4B-it_teacher_traces.json"
    "data/math-500/gemma-4-E4B-it_teacher_traces.json"
    "data/zebralogic/gemma-4-E4B-it_teacher_traces.json"
    "data/gpqa/gemma-4-E4B-it_teacher_traces.json"
)
# max_questions per dataset, aligned by index with DATASETS above (0 = all)
MAX_Q=(0 0 0 100 50 50)

ALL_SELECTORS=("low_entropy" "high_entropy" "numbers" "newlines" "end_of_sentence" "random")

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

python -u evaluate_entropy_splice.py \
    --model google/gemma-4-E4B-it \
    --traces_file "$TRACES_FILE" \
    --retention_rate "$RATES" \
    --selector "$SEL" \
    --suffix_variant therefore_boxed \
    --max_traces $MAX_TRACES \
    $MAX_Q_ARG

echo "Task $SLURM_ARRAY_TASK_ID done."
