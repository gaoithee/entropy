#!/usr/bin/env python
"""Run CoT compression experiment (token selection + activation pooling).

Usage
-----
python scripts/run_compression.py \\
    --model_name     openai/gpt-oss-20b \\
    --input_dir      outputs/gpt-oss-20b/AIME2025/trace_act_ent \\
    --output_dir     outputs/gpt-oss-20b/AIME2025/compression \\
    --retention_rate 0.1 \\
    [--selection_methods "low_entropy random high_entropy before_entropy after_entropy newline end_of_sentence numbers"] \\
    [--force_boxed_answer True] \\
    [--attn_implementation flash_attention_2]

Available selection methods:
    low_entropy, high_entropy, random,
    before_entropy, after_entropy,
    newline, end_of_sentence, numbers
"""
import fire
from entropy.experiments.compression.pooling import CompressionPooling, CompressionPoolingCfg


def main(
    model_name: str,
    input_dir: str,
    output_dir: str,
    retention_rate: float = 0.1,
    selection_methods: str | None = None,
    pooling: str = "mean",
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    force_boxed_answer: bool = True,
    filter_gt_answer: bool = True,
    quantization: str | None = None,
    attn_implementation: str | None = None,
):
    from entropy.experiments.compression.pooling import ALL_METHODS
    methods = selection_methods.split() if isinstance(selection_methods, str) else list(ALL_METHODS)

    cfg = CompressionPoolingCfg(
        model_name=model_name,
        input_dir=input_dir,
        output_dir=output_dir,
        retention_rate=retention_rate,
        selection_methods=methods,
        pooling=pooling,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        force_boxed_answer=force_boxed_answer,
        filter_gt_answer=filter_gt_answer,
        quantization=quantization,
        attn_implementation=attn_implementation,
    )
    CompressionPooling(cfg).run()


if __name__ == "__main__":
    fire.Fire(main)
