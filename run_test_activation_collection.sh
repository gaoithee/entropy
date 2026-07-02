#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="test-act-collection"
#SBATCH --partition=lovelace
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_outputs/test-act-collection-%j.out
#SBATCH --export=ALL

PROJECT_DIR="/u/scandussio/entropy"
cd "$PROJECT_DIR"

source .venv/bin/activate

.venv/bin/python -m pytest tests/test_activation_collection.py -v -s -k "gpt-oss-20b"
