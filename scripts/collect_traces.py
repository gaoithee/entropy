import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fire
from typing import Optional

import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json

from tqdm import tqdm

from shared.data_utils import get_data, extract_boxed_answer


def main(
    model_name: str = "Qwen/Qwen3-4B",
    data_name: str = "openai/gsm8k",
    num_out: int = 16,
    batch_size: int = 500,
    resume: bool = True,          # automatically resume if partial results exist
    quantization: Optional[str] = None,  # quantization mode: "awq", "gptq", "squeezellm", "fp8", etc.
    **gen_kwargs,
):
    print(f"Loading dataset: {data_name}")
    reasoning_dataset = get_data(data_name)
    print(f"Loaded {len(reasoning_dataset)} examples")

    print(f"Loading model: {model_name}")
    if quantization:
        print(f"Using quantization: {quantization}")
    llm = LLM(model=model_name, tensor_parallel_size=1, quantization=quantization)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    default_gen_kwargs = {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 16_384,
    }
    gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
    top_k_entropy = 20
    sampling_params = SamplingParams(n=num_out, logprobs=top_k_entropy, **gen_kwargs)

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
        )
        prompts.append(prompt)

    # Determine which indices to process (support resume)
    processed_indices = set()
    if resume and final_file.exists():
        print(f"Checking existing progress in {final_file}...")
        try:
            with open(final_file, "r") as f:
                existing_data = json.load(f)
                processed_indices = {item["input_text"] for item in existing_data}
                print(f"Found {len(processed_indices)} already processed examples. Resuming...")
        except Exception as e:
            print(f"Could not read existing file (maybe corrupted): {e}. Starting from scratch.")

    indices_to_process = [
        i for i, (question, _) in enumerate(reasoning_dataset)
        if question not in processed_indices
    ]

    if not indices_to_process:
        print("All examples already processed!")
        return

    print(f"Will process {len(indices_to_process)} remaining examples in batches of {batch_size}")

    if not final_file.exists():
        # Create empty list
        with open(final_file, "w") as f:
            json.dump([], f)

    # safety
    batch_dir = output_dir / "batches"
    batch_dir.mkdir(exist_ok=True)

    # Process in batches
    for batch_start in tqdm(range(0, len(indices_to_process), batch_size), desc="Batches"):

        batch_indices = indices_to_process[batch_start:batch_start + batch_size]
        batch_prompts = [prompts[i] for i in batch_indices]
        batch_questions = [reasoning_dataset[i][0] for i in batch_indices]
        batch_gt_answers = [reasoning_dataset[i][1] for i in batch_indices]

        print(f"\nGenerating batch {batch_start // batch_size + 1} "
              f"({len(batch_prompts)} prompts)...")

        outputs = llm.generate(batch_prompts, sampling_params)

        batch_results = []
        for idx_in_batch, output in enumerate(outputs):
            global_idx = batch_indices[idx_in_batch]
            question = batch_questions[idx_in_batch]
            gt_answer = batch_gt_answers[idx_in_batch]

            traces = [output.outputs[x].text for x in range(len(output.outputs))]
            traces_tokens = [list(output.outputs[x].token_ids) for x in range(len(output.outputs))]
            extracted = [extract_boxed_answer(t) for t in traces]

            # Compute top-k entropy for each token in each trace
            traces_entropy = []
            for completion in output.outputs:
                token_entropies = []
                for step in completion.logprobs:
                    log_probs = np.array([v.logprob for v in step.values()])
                    probs = np.exp(log_probs)
                    probs = probs / probs.sum()  # renormalize over top-k
                    entropy = -np.sum(probs * np.log(probs + 1e-12))
                    token_entropies.append(float(entropy))
                traces_entropy.append(token_entropies)

            result = {
                "input_text": question,
                "prompt_tokens": list(output.prompt_token_ids),
                "traces": traces,
                "traces_tokens": traces_tokens,
                "traces_entropy": traces_entropy,
                "extracted_answers": extracted,
                "GT_answer": gt_answer,
            }
            batch_results.append(result)

        # save this batch separately (safety)
        batch_file = batch_dir / f"batch_{batch_start:06d}_{batch_start + len(batch_results):06d}.json"
        with open(batch_file, "w") as f:
            json.dump(batch_results, f, indent=4, ensure_ascii=False)

        # append to the main file
        # load
        with open(final_file, "r") as f:
            all_results = json.load(f)
        # extedn
        all_results.extend(batch_results)
        # save
        with open(final_file, "w") as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)

        print(f"Batch saved. Total processed: {len(all_results)}")

    # Save a human-readable JSONL (entropy lengths instead of raw lists)
    readable_file = output_dir / f"{model_short_name}_teacher_traces_readable.jsonl"
    with open(readable_file, "w") as f:
        for item in all_results:
            readable = {
                "input_text": item["input_text"],
                "prompt_tokens_len": len(item["prompt_tokens"]),
                "num_traces": len(item["traces"]),
                "traces_tokens_len": [len(t) for t in item["traces_tokens"]],
                "traces_entropy_len": [len(e) for e in item["traces_entropy"]],
                "extracted_answers": item["extracted_answers"],
                "GT_answer": item["GT_answer"],
            }
            f.write(json.dumps(readable, ensure_ascii=False) + "\n")

    print(f"\nAll done! Final results saved to:\n   {final_file}")
    print(f"Readable summary: {readable_file}")
    print(f"Individual batches are in: {batch_dir}")


if __name__ == "__main__":
    fire.Fire(main)