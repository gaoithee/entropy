#!/bin/bash
#SBATCH --account=uTS26_Bortolus
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name="lowent-nonum-zebra-1gpu"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=60:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm_outputs/lowent-nonum-zebra-1gpu-%A_%a.out
#SBATCH --export=ALL

# ===========================================================================
# Ablation low_entropy_no_numbers -- zebralogic, 1-GPU models.
# gpt-oss-20b, gemma-4-E4B-it, Qwen3-14B, Qwen3-4B.
# 4 tasks. 60h on boost_qos_lprod.
# ===========================================================================

PROJECT_DIR="$HOME/entropy"
cd "$PROJECT_DIR"
source .venv/bin/activate

export HF_HOME=$WORK/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RATES="0.01,0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.0"
SELECTOR="low_entropy_no_numbers"
MAX_TRACES=8

MODEL_IDS=(
    "openai/gpt-oss-20b"
    "google/gemma-4-E4B-it"
    "Qwen/Qwen3-14B"
    "Qwen/Qwen3-4B"
)
MODEL_PREFIXES=(
    "gpt-oss-20b"
    "gemma-4-E4B-it"
    "Qwen3-14B"
    "Qwen3-4B"
)

DATASET="zebralogic"
Q_COUNT=50

MODEL_ID="${MODEL_IDS[$SLURM_ARRAY_TASK_ID]}"
MODEL_PREFIX="${MODEL_PREFIXES[$SLURM_ARRAY_TASK_ID]}"

TRACES_FILE="data/${DATASET}/${MODEL_PREFIX}_teacher_traces.json"

echo "Task $SLURM_ARRAY_TASK_ID: model=$MODEL_ID traces_file=$TRACES_FILE max_questions=$Q_COUNT"

python -u evaluate_entropy_splice.py \
    --model "$MODEL_ID" \
    --traces_file "$TRACES_FILE" \
    --retention_rate "$RATES" \
    --selector "$SELECTOR" \
    --suffix_variant therefore_boxed \
    --max_traces $MAX_TRACES \
    --max_questions $Q_COUNT

if [ $? -ne 0 ]; then
    echo "Task $SLURM_ARRAY_TASK_ID FAILED"
    exit 1
fi
echo "Task $SLURM_ARRAY_TASK_ID done."
