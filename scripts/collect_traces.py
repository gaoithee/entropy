"""Generate reasoning traces using vLLM and save them for later activation collection.

Ported from neurohike/scripts/vllm_traces_generator.py.
Changes vs original:
  - Imports from entropy.core instead of shared
  - enable_thinking passed to apply_chat_template for gemma/qwen models
  - Output format compatible with TraceActivationEntropy input schema
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from typing import Optional

import fire
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from entropy.core.data_utils import get_data, extract_boxed_answer
from entropy.models.registry import get_thinking_tokens


def main(
    model_name: str = "openai/gpt-oss-20b",
    data_name: str = "aime2025",
    num_out: int = 16,
    batch_size: int = 500,
    resume: bool = True,
    quantization: Optional[str] = None,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 22000,
    num_questions: Optional[int] = None,
    **gen_kwargs,
):
    print(f"Loading dataset: {data_name}")
    reasoning_dataset = get_data(data_name)
    print(f"Loaded {len(reasoning_dataset)} examples")
    if num_questions is not None:
        reasoning_dataset = reasoning_dataset[:num_questions]
        print(f"Limiting to {num_questions} question(s)")

    print(f"Loading model: {model_name}")
    llm = LLM(model=model_name, tensor_parallel_size=1, trust_remote_code=True, quantization=quantization, gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # enable_thinking for models that require it
    cfg = get_thinking_tokens(model_name)
    template_kwargs = {}
    if cfg.get("enable_thinking"):
        template_kwargs["enable_thinking"] = True

    default_gen_kwargs = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 16_384}
    gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
    top_k_entropy = 20
    sampling_params = SamplingParams(n=num_out, logprobs=top_k_entropy, skip_special_tokens=False, **gen_kwargs)

    dataset_short_name = data_name.split("/")[-1]
    model_short_name = model_name.split("/")[-1]
    output_dir = Path("./data") / dataset_short_name
    output_dir.mkdir(parents=True, exist_ok=True)
    final_file = output_dir / f"{model_short_name}_teacher_traces.json"

    prompts = []
    for question, _ in reasoning_dataset:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            add_generation_prompt=True,
            tokenize=False,
            **template_kwargs,
        )
        prompts.append(prompt)

    # Resume support
    if not resume and final_file.exists():
        print(f"resume=False: overwriting existing {final_file}")
        final_file.unlink()

    processed_indices = set()
    if resume and final_file.exists():
        try:
            with open(final_file) as f:
                existing_data = json.load(f)
                processed_indices = {item["input_text"] for item in existing_data}
                print(f"Resuming: {len(processed_indices)} already processed.")
        except Exception as e:
            print(f"Could not read existing file: {e}. Starting fresh.")

    indices_to_process = [
        i for i, (question, _) in enumerate(reasoning_dataset)
        if question not in processed_indices
    ]

    if not indices_to_process:
        print("All examples already processed!")
        return

    if not final_file.exists():
        with open(final_file, "w") as f:
            json.dump([], f)

    batch_dir = output_dir / "batches"
    batch_dir.mkdir(exist_ok=True)

    for batch_start in tqdm(range(0, len(indices_to_process), batch_size), desc="Batches"):
        batch_indices = indices_to_process[batch_start:batch_start + batch_size]
        batch_prompts  = [prompts[i] for i in batch_indices]
        batch_questions = [reasoning_dataset[i][0] for i in batch_indices]
        batch_gt_answers = [reasoning_dataset[i][1] for i in batch_indices]

        outputs = llm.generate(batch_prompts, sampling_params)

        batch_results = []
        for idx_in_batch, output in enumerate(outputs):
            traces         = [o.text for o in output.outputs]
            traces_tokens  = [list(o.token_ids) for o in output.outputs]
            extracted      = [extract_boxed_answer(t) for t in traces]

            traces_entropy = []
            for completion in output.outputs:
                token_entropies = []
                for step in completion.logprobs:
                    log_probs = np.array([v.logprob for v in step.values()])
                    probs = np.exp(log_probs)
                    probs = probs / probs.sum()
                    token_entropies.append(float(-np.sum(probs * np.log(probs + 1e-12))))
                traces_entropy.append(token_entropies)

            batch_results.append({
                "input_text":    batch_questions[idx_in_batch],
                "prompt_tokens": list(output.prompt_token_ids),
                "traces":        traces,
                "traces_tokens": traces_tokens,
                "traces_entropy": traces_entropy,
                "extracted_answers": extracted,
                "GT_answer":     batch_gt_answers[idx_in_batch],
            })

        batch_file = batch_dir / f"batch_{batch_start:06d}.json"
        with open(batch_file, "w") as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False)

        with open(final_file) as f:
            all_results = json.load(f)
        all_results.extend(batch_results)
        with open(final_file, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"Batch done. Total: {len(all_results)}")

    print(f"\nDone! Saved to {final_file}")


if __name__ == "__main__":
    fire.Fire(main)