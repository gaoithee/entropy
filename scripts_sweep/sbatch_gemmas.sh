#!/bin/bash
#SBATCH --account=uTS26_Bortolus
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name="lowent-nonum-gemma26b-2trace"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --output=slurm_outputs/lowent-nonum-gemma26b-2trace-%A_%a.out
#SBATCH --export=ALL

# ===========================================================================
# Ablation low_entropy_no_numbers -- gemma-4-26B-A4B-it on math-500 and gpqa,
# previously cancelled for being too slow (traces up to 16384 tokens,
# frequent truncation). Re-launched with MAX_TRACES=2 instead of 8 to get a
# usable (if smaller) sample within a reasonable time budget, instead of
# waiting indefinitely or skipping these two cells entirely.
# ===========================================================================

PROJECT_DIR="$HOME/entropy"
cd "$PROJECT_DIR"
source .venv/bin/activate

export HF_HOME=$WORK/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RATES="0.01,0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.0"
SELECTOR="low_entropy_no_numbers"
MODEL_ID="google/gemma-4-26B-A4B-it"
MODEL_PREFIX="gemma-4-26B-A4B-it"

DATASETS=("math-500" "gpqa")
MAX_Q=(100 50)

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
Q_COUNT="${MAX_Q[$SLURM_ARRAY_TASK_ID]}"

TRACES_FILE="data/${DATASET}/${MODEL_PREFIX}_teacher_traces.json"

echo "Task $SLURM_ARRAY_TASK_ID: model=$MODEL_ID traces_file=$TRACES_FILE max_questions=$Q_COUNT max_traces=2"

python -u evaluate_entropy_splice.py \
    --model "$MODEL_ID" \
    --traces_file "$TRACES_FILE" \
    --retention_rate "$RATES" \
    --selector "$SELECTOR" \
    --suffix_variant therefore_boxed \
    --max_traces 2 \
    --max_questions $Q_COUNT

if [ $? -ne 0 ]; then
    echo "Task $SLURM_ARRAY_TASK_ID FAILED"
    exit 1
fi
echo "Task $SLURM_ARRAY_TASK_ID done."
