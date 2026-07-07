#!/bin/bash
#SBATCH --account=uTS26_Bortolus
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name="lowent-nonum"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=60:00:00
#SBATCH --array=36-71
#SBATCH --output=slurm_outputs/lowent-nonum-%A_%a.out
#SBATCH --export=ALL

# ===========================================================================
# Ablation: low_entropy selector with numeric-token candidates excluded from
# the pool ("low_entropy_no_numbers"). One (model, dataset) pair per array
# task, full retention_rate sweep handled internally. Indices start at 36 to
# avoid clashing with the existing gptoss120b-sweep numbering (0-35).
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

# --------------------------------------------------------------------------
# MODELS: (hf_model_id, traces_file_prefix)
# traces_file_prefix must match the *_teacher_traces.json filename already
# present under data/<dataset>/ -- verify these against your actual model
# identifiers before submitting, especially the gemma-4 ones.
# --------------------------------------------------------------------------
MODEL_IDS=(
    "openai/gpt-oss-120b"
    "openai/gpt-oss-20b"
    "google/gemma-4-26B-A4B-it"
    "google/gemma-4-E4B-it"
    "Qwen/Qwen3-14B"
    "Qwen/Qwen3-4B"
)
MODEL_PREFIXES=(
    "gpt-oss-120b"
    "gpt-oss-20b"
    "gemma-4-26B-A4B-it"
    "gemma-4-E4B-it"
    "Qwen3-14B"
    "Qwen3-4B"
)

DATASETS=("aime_2024" "aime2025" "aime_2026" "math-500" "zebralogic" "gpqa")
MAX_Q=(0 0 0 100 50 50)

N_MODELS=${#MODEL_IDS[@]}
N_DATASETS=${#DATASETS[@]}

# Offset so this array's task IDs (36-71) map back to a 0-based index
BASE_IDX=$(( SLURM_ARRAY_TASK_ID - 36 ))

MODEL_IDX=$(( BASE_IDX / N_DATASETS ))
DATASET_IDX=$(( BASE_IDX % N_DATASETS ))

MODEL_ID="${MODEL_IDS[$MODEL_IDX]}"
MODEL_PREFIX="${MODEL_PREFIXES[$MODEL_IDX]}"
DATASET="${DATASETS[$DATASET_IDX]}"
Q_COUNT="${MAX_Q[$DATASET_IDX]}"

TRACES_FILE="data/${DATASET}/${MODEL_PREFIX}_teacher_traces.json"

echo "Task $SLURM_ARRAY_TASK_ID: model=$MODEL_ID traces_file=$TRACES_FILE max_questions=$Q_COUNT"

if [ "$Q_COUNT" = "0" ]; then
    MAX_Q_ARG=""
else
    MAX_Q_ARG="--max_questions $Q_COUNT"
fi

python -u evaluate_entropy_splice.py \
    --model "$MODEL_ID" \
    --traces_file "$TRACES_FILE" \
    --retention_rate "$RATES" \
    --selector "$SELECTOR" \
    --suffix_variant therefore_boxed \
    --max_traces $MAX_TRACES \
    $MAX_Q_ARG

if [ $? -ne 0 ]; then
    echo "Task $SLURM_ARRAY_TASK_ID FAILED"
    exit 1
fi
echo "Task $SLURM_ARRAY_TASK_ID done."
