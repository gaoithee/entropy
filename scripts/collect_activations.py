#!/usr/bin/env python
"""Collect activations and per-token entropy from pre-generated traces.
Equivalent to:
    python -m neurohike.run --exp trace_act_ent ...
Usage
-----
python scripts/collect_activations.py \\
    --model_name  openai/gpt-oss-20b \\
    --data_name   opencompass/AIME2025 \\
    --output_dir  outputs/gpt-oss-20b/AIME2025/trace_act_ent \\
    [--top_k_for_entropy 20] \\
    [--max_seq_length 7000] \\
    [--max_traces 16] \\
    [--max_questions 1] \\
    [--quantization 4bit] \\
    [--attn_implementation flash_attention_2]
"""
import fire
from entropy.experiments.trace_collection import TraceActivationEntropy, TraceCollectionCfg


def main(
    model_name: str,
    data_name: str,
    output_dir: str,
    top_k_for_entropy: int = 20,
    max_traces: int | None = None,
    max_seq_length: int | None = 7000,
    max_questions: int | None = None,
    quantization: str | None = None,
    attn_implementation: str | None = None,
    layers: str | None = None,  # e.g. "0.25 0.5 0.75 1.0" for percentiles
):
    parsed_layers = None
    if layers:
        vals = [float(x) if "." in x else int(x) for x in layers.split()]
        parsed_layers = vals
    cfg = TraceCollectionCfg(
        model_name=model_name,
        data_name=data_name,
        output_dir=output_dir,
        top_k_for_entropy=top_k_for_entropy,
        max_traces=max_traces,
        max_seq_length=max_seq_length,
        max_questions=max_questions,
        quantization=quantization,
        attn_implementation=attn_implementation,
        layers=parsed_layers,
    )
    TraceActivationEntropy(cfg).run()


if __name__ == "__main__":
    fire.Fire(main)