#!/bin/bash
python -u evaluate_entropy_sentence.py \
    --model openai/gpt-oss-20b \
    --traces_file /share/ai-lab/scandussio/entropy/data/aime_2024/gpt-oss-20b_teacher_traces.json \
    --retention_rate "0.05,0.10,0.20,0.30,0.50,0.70,1.0" \
    --selector "low_entropy,high_entropy,numbers,low_entropy_no_numbers,random" \
    --suffix_variant therefore_boxed \
    --max_traces 3 \
    --max_questions 5 \
    --skip_patched True
