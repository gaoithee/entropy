# The Fool's Gold

## Datasets and models

| Model | Dataset | Traces | Activations | Entropies |
|---|---|---|---|---|
| gpt-oss-20b | AIME 2024 | ✓ | ✓ | ✓ |
| gpt-oss-20b | AIME 2025 | ✓ | ✓ | ✓ |
| gpt-oss-20b | ZebraLogic | ✓ | ✓ | ✓ |
| gpt-oss-20b | MMLU-Pro | ✓ | ✓ | ✓ |
| gpt-oss-20b | AIME 2026 | Demetra | — | — |
| gemma-4-26b | AIME 2025 | ✓ | — | — |
| Qwen3-14B | AIME 2025 | ✓ | — | — |
| Phi-4-r-plus | AIME 2025 | ✓ | — | — |

## Structure

```
entropy/
├── entropy/
│   ├── core/
│   │   ├── model_loader.py      # HF model + tokenizer loading
│   │   ├── trace_loader.py      # Load pre-generated vLLM traces from HF
│   │   └── utils.py             # extract_boxed_answer, etc.
│   ├── models/
│   │   └── registry.py          # Thinking token delimiters per model family
│   ├── datasets/
│   │   └── registry.py          # Dataset HF identifiers and field mappings
│   └── experiments/
│       ├── trace_collection.py  # Step 1: collect activations + entropies
│       └── compression/
│           ├── common.py        # CompressionResult, KL div, entropy utils
│           └── pooling.py       # Step 2: token selection + pooling + eval
├── scripts/
│   ├── collect_activations.py   # CLI for trace_collection
│   └── run_compression.py       # CLI for compression/pooling
└── configs/                     # Per-run YAML configs
```

## Quick start

```bash
pip install -e .

# Step 1 — collect activations and entropies from pre-generated traces
python scripts/collect_activations.py \
    --model_name openai/gpt-oss-20b \
    --data_name  opencompass/AIME2025 \
    --output_dir outputs/gpt-oss-20b/AIME2025/trace_act_ent \
    --max_seq_length 7000

# Step 2 — run compression experiment
python scripts/run_compression.py \
    --model_name     openai/gpt-oss-20b \
    --input_dir      outputs/gpt-oss-20b/AIME2025/trace_act_ent \
    --output_dir     outputs/gpt-oss-20b/AIME2025/compression \
    --retention_rate 0.1 \
    --selection_methods "low_entropy random high_entropy before_entropy after_entropy newline end_of_sentence"
```

## Selection methods

| Method | Description |
|---|---|
| `low_entropy` | Top-k lowest entropy tokens — **main finding** |
| `high_entropy` | Top-k highest entropy tokens (ablation) |
| `random` | Random sample (baseline) |
| `before_entropy` | Token immediately before each high-entropy peak |
| `after_entropy` | Token immediately after each high-entropy peak |
| `newline` | Tokens containing `\n` |
| `end_of_sentence` | Tokens at sentence boundaries (`. \n` or `. `) |
| `numbers` | Tokens whose decoded text is purely numeric |

## Adding a new model

1. Add thinking token delimiters to `entropy/models/registry.py`
2. Add a stub file `entropy/models/<name>.py` for any tokenizer quirks
3. Add a config in `configs/`

## Adding a new dataset

1. Add HF identifier + field mapping to `entropy/datasets/registry.py`
2. Register the trace HF repo in `entropy/core/trace_loader.py`

