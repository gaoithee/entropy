#!/bin/bash
#SBATCH --account=uTS26_Bortolus
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name="lowent-nonum-zebra-2gpu"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=60:00:00
#SBATCH --array=0-1
#SBATCH --output=slurm_outputs/lowent-nonum-zebra-2gpu-%A_%a.out
#SBATCH --export=ALL

# ===========================================================================
# Ablation low_entropy_no_numbers -- zebralogic, 2-GPU models.
# gemma-4-26B-A4B-it, gpt-oss-120b.
# 2 tasks. 60h on boost_qos_lprod.
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
    "google/gemma-4-26B-A4B-it"
    "openai/gpt-oss-120b"
)
MODEL_PREFIXES=(
    "gemma-4-26B-A4B-it"
    "gpt-oss-120b"
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
