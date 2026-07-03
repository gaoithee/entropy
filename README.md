# All that is gold does not glitter

Demistifying entropy-related token selection for CoT compression in reasoning LLMs.

## Setup

```bash
cd entropy
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```

Dependencies are pinned in `pyproject.toml` (exact versions for `transformers`,
`kernels`, `torchao`, `triton`).

Optional extras:
```bash
uv pip install -e ".[dev]"    # pytest
uv pip install -e ".[quant]"  # bitsandbytes
uv pip install -e ".[flash]"  # flash-attn
```

## Data: reasoning traces

The traces live on HuggingFace: https://huggingface.co/datasets/saracandu/entropy-traces

```bash
hf download saracandu/entropy-traces --repo-type dataset --local-dir data
```

Expected layout under `data/`:
```
data/{dataset_name}/{model_name}_teacher_traces.json
```
where `dataset_name` is one of `aime_2024`, `aime2025`, `aime_2026`, `gpqa`, `math-500`,
`non-math-mmlu-pro`, `zebralogic`, and `model_name` is one of `gpt-oss-20b`,
`gpt-oss-120b`, `gemma-4-26B-A4B-it`, `gemma-4-E4B-it`, `Qwen3-4B`, `Qwen3-14B`,
`Phi-4-reasoning-plus` (the last one is not available for every dataset).

`entropy/core/trace_loader.py` (`get_reasoning_traces`) loads from here, falling back
to a local path if an HF repo isn't registered in `_TRACE_REPOS`.


## Pipeline

### 0. Trace collection (vLLM)

Generate reasoning traces with a model, via vLLM:

```bash
python scripts/collect_traces.py \
    --model_name openai/gpt-oss-20b \
    --data_name aime2025 \
    --num_out 16 \
    --batch_size 500 \
    --num_questions 30
```

Main parameters:
- `num_out`: number of sampled traces per question
- `resume` (default `True`): resumes from already-written batches
- `quantization`, `gpu_memory_utilization`, `max_model_len`: memory/vLLM tuning
- `**gen_kwargs`: passthrough to vLLM (defaults: `temperature=0.7, top_p=0.9, max_tokens=16384`)

### 1. Activation collection

Collects activations + per-token entropies from HF checkpoints (doesn't go through
vLLM — loads the model and runs direct forward passes). **Note:** at the moment only
the activations are needed, not the HF entropies already available in the HF dataset.

```bash
python scripts/collect_activations.py \
    --model_name openai/gpt-oss-20b \
    --data_name aime2025 \
    --output_dir outputs/openai_gpt-oss-20b/aime2025
```

Useful parameters:
- `layers`: percentile string like `"0.25 0.5 0.75 1.0"` to sample only some layers
  instead of all
- `max_seq_length` (default 7000), `max_traces`, `max_questions`: to cap cost
- `top_k_for_entropy` (default 20)

On a slurm cluster, a job array is already set up in `collect_acts.sh`:
```bash
# lovelace (A100 80GB) — larger models
sbatch --partition=lovelace --gres=gpu:a100:1 --array=0-16 collect_acts.sh

# babbage (A100 40GB) — gemma-4-E4B
sbatch --partition=main --gres=gpu:a100:1 --array=17-19 collect_acts.sh
```
The `(model, dataset) -> array index` mapping is hardcoded in the `JOBS` array inside
the script — update it there if you add combinations.

### 2. Activation patching / splicing (entropy-based selection)

`evaluate_entropy_splice.py` is the main script for patching experiments — selects a
subset of tokens (via `selector`, e.g. `low_entropy`) and tests the effect of patching
on a subset of layers.

```bash
python evaluate_entropy_splice.py \
    --model openai/gpt-oss-20b \
    --traces_file data/aime2025/gpt-oss-20b_teacher_traces.json \
    --retention_rate 0.1 \
    --selector low_entropy \
    --patch_layers "18:24" \
    --suffix_variant therefore_boxed
```

**`--patch_layers`** — syntax:
- omitted / `None`: patches all layers (legacy behavior, unchanged)
- `"a:b"` (slice, end-exclusive): contiguous window, e.g. `"18:24"` = last 6 layers out
  of 24. **Prefer this form** for results you intend to report — sparse patching over
  isolated layers is typically weaker/noisier than a contiguous window (causal tracing
  literature, ROME-style: adjacent untouched layers can "route around" a single
  patched point)
- `"a,b,c"` (explicit indices): supported, but must be passed **as a quoted string**,
  because Fire coerces `--patch_layers 0,4,8,12,16,20` (even quoted on the shell) into
  a Python tuple *before* it reaches the script — if you see it fail with
  `ValueError: invalid literal for int() with base 10: '(0'`, this is why. Fixed in
  `parse_patch_layers`, but keep it in mind if you write external wrappers/configs.

Other relevant parameters:
- `retention_rate`: fraction of tokens kept by the selector
- `suffix_variant`: forced text after end-of-thinking, before the final answer —
  defaults to `therefore_boxed`; this is a deliberate methodological choice (forces
  the model to answer using only what survived inside the thinking region, not
  free-form reasoning done afterward)
- `skip_patched`: skips the patched baseline (natural generation only)
- `activations_pth_dir`: if activations are already saved to disk, avoids recomputing
  them
- `max_questions`, `max_traces`, `question_offset`, `trace_offset`: to run subsets
  (useful for parallel sweeps via job arrays)

### 3. Static trace correctness / quality evaluation

`evaluate_trace_correctness.py` batch-checks traces in `data/*/*_teacher_traces.json`
for correctness, truncation, missing `\boxed{}`, etc.

```bash
python evaluate_trace_correctness.py \
    --data_glob "data/*/*teacher_traces.json" \
    --answer_domain math \
    --max_tokens 16384 \
    --write False
```

Always run dry-run first (`--write False`, default) to see the report without
modifying anything; pass `--write True` only after checking the output.

## Repo structure

```
entropy/
├── entropy/                  # installable package
│   ├── core/                 # model/trace loading, utils
│   ├── datasets/             # dataset registry (AIME, GPQA, MATH-500, ...)
│   ├── experiments/          # trace_collection, compression/
│   └── models/                # model registry
├── scripts/                  # CLI entrypoints (collect_traces, collect_activations, run_compression, plot_results)
├── configs/                  # YAML configs for experiments
├── tests/                    # pytest
│   └── debug_archive/        # old one-off debug scripts, not maintained
├── data -> /share/.../data       # symlink, traces (see Data section)
├── outputs -> /share/.../outputs # symlink, experiment outputs (not versioned)
├── check_empty_extractions.py    # utility: finds empty/malformed extractions in traces
├── dedup_traces.py               # utility: removes duplicate traces
├── evaluate_entropy_splice.py    # patching/splicing experiments (main script)
└── evaluate_trace_correctness.py # trace correctness report
```

## Notes

- `data/` and `outputs/` are symlinks to shared storage
  (`/share/ai-lab/scandussio/entropy/`) — not versioned, multi-GB in size.
- Slurm logs (`slurm_outputs/`) and sweep logs (`sweep_logs/`) stay on local disk,
  excluded from git.