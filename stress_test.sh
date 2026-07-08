#!/bin/bash
python -u evaluate_entropy_sentence.py \
    --model openai/gpt-oss-20b \
    --traces_file /share/ai-lab/scandussio/entropy/data/aime_2024/gpt-oss-20b_teacher_traces.json \
    --retention_rate "0.5,1.0" \
    --selector "low_entropy,high_entropy,numbers,low_entropy_no_numbers,random" \
    --suffix_variant therefore_boxed \
    --max_traces 1 \
    --max_questions 2 \
    --debug True \
    --skip_patched True
