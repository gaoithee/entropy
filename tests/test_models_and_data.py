"""Tests for model loading, dataset loading, answer extraction, and trace collection.

Requires GPU and network access (downloads models and datasets from HuggingFace).

Run with:
    python -m pytest tests/test_models_and_data.py -v -s
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch


# ===========================================================================
# Models and datasets
# ===========================================================================

MODELS = [
    "openai/gpt-oss-20b",
    # "openai/gpt-oss-120b",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-4B",
    "google/gemma-4-26B-A4B-it",
    "google/gemma-4-E4B-it",
    "microsoft/Phi-4-reasoning-plus",
]

DATASET_NAMES = [
    "aime2025",
    "aime_2024",
    "aime_2026",
    "zebralogic",
    "math-500",
    "non-math-mmlu-pro",
    "gpqa",
]


# ===========================================================================
# Session-scoped fixture — loads model ONCE for the entire test session
# ===========================================================================

@pytest.fixture(scope="session", params=MODELS)
def model_and_tokenizer(request):
    """Load each model once per session. Yields (model_name, model, tokenizer, config)."""
    import gc
    from entropy.core.model_loader import load_model_and_tokenizer
    model_name = request.param
    model, tokenizer, config = load_model_and_tokenizer(model_name, model_type="hf")
    yield model_name, model, tokenizer, config
    del model
    gc.collect()
    torch.cuda.empty_cache()



# ===========================================================================
# 1. DATASET LOADING — one question per dataset
# ===========================================================================

class TestDatasetLoading:
    """Load one question from each dataset and check format."""

    @pytest.mark.parametrize("data_name", DATASET_NAMES)
    def test_loads_at_least_one_question(self, data_name):
        from entropy.core.data_utils import get_data
        data = get_data(data_name)
        assert len(data) > 0, f"{data_name}: empty dataset"

    @pytest.mark.parametrize("data_name", DATASET_NAMES)
    def test_first_question_is_nonempty_string(self, data_name):
        from entropy.core.data_utils import get_data
        question, answer = get_data(data_name)[0]
        assert isinstance(question, str) and len(question) > 10, \
            f"{data_name}: question too short or not a string"

    @pytest.mark.parametrize("data_name", DATASET_NAMES)
    def test_first_answer_is_nonempty(self, data_name):
        from entropy.core.data_utils import get_data
        question, answer = get_data(data_name)[0]
        assert answer is not None and str(answer).strip() != "", \
            f"{data_name}: answer is empty"

    @pytest.mark.parametrize("data_name", DATASET_NAMES)
    def test_answer_domain_is_valid(self, data_name):
        from entropy.core.data_utils import get_answer_domain
        domain = get_answer_domain(data_name)
        assert domain in ("math", "mcq"), f"{data_name}: unknown domain {domain!r}"

    def test_math_datasets_have_numeric_or_string_answers(self):
        from entropy.core.data_utils import get_data
        for name in ("aime2025", "aime_2024", "gsm8k", "math-500"):
            _, answer = get_data(name)[0]
            assert isinstance(answer, (int, float, str)), \
                f"{name}: answer is {type(answer)}"

    def test_mcq_datasets_have_letter_answers(self):
        from entropy.core.data_utils import get_data
        for name in ("non-math-mmlu-pro", "gpqa"):
            _, answer = get_data(name)[0]
            assert isinstance(answer, str) and len(answer) == 1 and answer.isalpha(), \
                f"{name}: expected single letter answer, got {answer!r}"

    def test_dataset_sizes(self):
        """Verify expected sizes for each dataset."""
        from entropy.core.data_utils import get_data
        expected = {
            "aime2025":          30,
            "aime_2024":         30,
            "aime_2026":         30,
            "zebralogic":        50,
            "math-500":         100,
            "non-math-mmlu-pro": 65,
            "gpqa":             197,
        }
        for name, expected_len in expected.items():
            data = get_data(name)
            assert len(data) == expected_len, \
                f"{name}: expected {expected_len} items, got {len(data)}"

    def test_unknown_dataset_raises(self):
        from entropy.core.data_utils import get_data
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_data("nonexistent/dataset")


# ===========================================================================
# 2. ANSWER EXTRACTION
# ===========================================================================

class TestAnswerExtraction:
    """Test extract_boxed_answer on realistic outputs from each domain."""

    def test_math_simple(self):
        from entropy.core.utils import extract_boxed_answer
        text = r"After computing, the answer is \boxed{42}."
        assert extract_boxed_answer(text) == "42"

    def test_math_nested_fraction(self):
        from entropy.core.utils import extract_boxed_answer
        text = r"Therefore \boxed{\frac{3}{4}}"
        assert extract_boxed_answer(text) == r"\frac{3}{4}"

    def test_math_last_boxed_wins(self):
        from entropy.core.utils import extract_boxed_answer
        text = r"First I get \boxed{3}, then \boxed{7}."
        assert extract_boxed_answer(text) == "7"

    def test_mcq_letter_in_boxed(self):
        from entropy.core.utils import extract_boxed_answer
        text = r"The correct option is \boxed{A}."
        assert extract_boxed_answer(text) == "A"

    def test_zebralogic_option_in_boxed(self):
        from entropy.core.utils import extract_boxed_answer
        text = r"Based on the clues, \boxed{B. The painter lives in the red house}."
        result = extract_boxed_answer(text)
        assert result.startswith("B")

    def test_no_boxed_returns_empty(self):
        from entropy.core.utils import extract_boxed_answer
        assert extract_boxed_answer("No answer here.") == ""

    def test_answer_suffix_plus_generation(self):
        """Simulate what happens at inference: suffix prepended to generation."""
        from entropy.core.utils import extract_boxed_answer
        from entropy.core.data_utils import get_answer_suffix
        suffix = get_answer_suffix("aime2025")   # r"Therefore, the final answer is \boxed{"
        generation = "204}"                       # model continues after the opening brace
        full = suffix + generation
        assert extract_boxed_answer(full) == "204"


# ===========================================================================
# 3. MODEL LOADING + THINKING TOKEN CONFIG
# ===========================================================================

class TestModelLoading:
    """Load each model once via session fixture and verify config."""

    def test_model_loads(self, model_and_tokenizer):
        model_name, model, tokenizer, config = model_and_tokenizer
        assert model is not None
        assert tokenizer is not None
        assert config["num_hidden_layers"] > 0
        assert config["hidden_size"] > 0

    def test_thinking_token_ids_in_vocab(self, model_and_tokenizer):
        from entropy.models.registry import get_thinking_tokens
        model_name, _, tokenizer, _ = model_and_tokenizer
        cfg = get_thinking_tokens(model_name)
        vocab_size = tokenizer.vocab_size
        if cfg["start_token_ids"] is not None:
            for tid in cfg["start_token_ids"]:
                assert 0 <= tid < vocab_size + 1000
            for tid in cfg["end_token_ids"]:
                assert 0 <= tid < vocab_size + 1000
        else:
            start_ids = tokenizer.encode(cfg["start_token"], add_special_tokens=False)
            end_ids   = tokenizer.encode(cfg["end_token"],   add_special_tokens=False)
            assert len(start_ids) > 0
            assert len(end_ids)   > 0


# ===========================================================================
# 4. CHAT TEMPLATE
# ===========================================================================

class TestChatTemplate:
    """Verify chat template and generation — one load per model via session fixture."""

    def test_chat_template_produces_nonempty_tokens(self, model_and_tokenizer):
        from entropy.core.data_utils import get_data
        model_name, _, tokenizer, _ = model_and_tokenizer
        question, _ = get_data("aime2025")[0]

        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        assert isinstance(prompt, str)
        assert len(prompt) > len(question)
        assert len(prompt_tokens) > 0

    def test_chat_template_contains_question_text(self, model_and_tokenizer):
        from entropy.core.data_utils import get_data
        model_name, _, tokenizer, _ = model_and_tokenizer
        question, _ = get_data("aime2025")[0]

        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert question[:50] in prompt, "question text not found in templated prompt"

    def test_chat_template_ends_with_assistant_turn(self, model_and_tokenizer):
        from entropy.core.data_utils import get_data
        model_name, _, tokenizer, _ = model_and_tokenizer
        question, _ = get_data("aime2025")[0]

        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print(f"\n[{model_name}] PROMPT TAIL: ...{prompt[-200:]!r}")
        # Different models use different role names: "assistant", "model", etc.
        assert any(role in prompt.lower() for role in ("assistant", "model")), \
            f"No assistant/model turn marker found in prompt: {prompt[-100:]!r}"

    def test_model_generates_thinking_start(self, model_and_tokenizer):
        """Generate 20 tokens and verify the model opens a thinking region."""
        from entropy.core.data_utils import get_data
        from entropy.models.registry import get_thinking_tokens

        model_name, model, tokenizer, _ = model_and_tokenizer
        question, _ = get_data("aime2025")[0]

        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=20, do_sample=False)

        generated_ids = out[0][input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        print(f"\n[{model_name}] GENERATED: {generated_text!r}")

        cfg = get_thinking_tokens(model_name)
        print(f"[{model_name}] START TOKEN IN REGISTRY: {cfg['start_token']!r}")

        if cfg["start_token_ids"] is not None:
            print(f"[{model_name}] FIRST GENERATED IDS: {generated_ids[:len(cfg['start_token_ids'])].tolist()}")
            assert generated_ids[:len(cfg["start_token_ids"])].tolist() == cfg["start_token_ids"], \
                f"{model_name}: start_token_ids {cfg['start_token_ids']} don't match generated {generated_ids[:5].tolist()}"
        else:
            # For models where start_token is in the prompt template (e.g. gemma4),
            # the model generates reasoning content directly — verify the prompt contains it
            full_context = prompt + generated_text
            assert cfg["start_token"] in full_context, \
                f"{model_name}: start_token {cfg['start_token']!r} not found in prompt+generation"
            print(f"[{model_name}] OK — start_token found in prompt template (generated content follows)")


# ===========================================================================
# 5. TRACE COLLECTION (synthetic — no real forward pass)
#    Verifies that TraceCollectionCfg + TraceActivationEntropy wire up
#    correctly given a pre-saved synthetic .pth as if it were a real trace.
# ===========================================================================

class TestTraceCollectionSynthetic:
    """Simulate trace collection on one synthetic question without GPU."""

    def _make_fake_traces(self, model_name, data_name, tmp_path):
        """Write a fake reasoning_traces JSON that mimics the vLLM output."""
        import json
        from entropy.models.registry import get_thinking_tokens

        cfg = get_thinking_tokens(model_name)
        start_ids = cfg["start_token_ids"] or [1]
        end_ids   = cfg["end_token_ids"]   or [2]

        prompt_tokens = [10, 20, 30]
        cot_tokens    = list(range(100, 150))
        trace_tokens  = start_ids + cot_tokens + end_ids

        record = {
            "input_text": "What is 6 × 7?",
            "prompt_tokens": prompt_tokens,
            "GT_answer": "42",
            "traces": [r"After thinking, \boxed{42}."],
            "traces_tokens": [trace_tokens],
            "traces_entropy": [[0.1] * len(trace_tokens)],
        }

        out = tmp_path / "traces.json"
        out.write_text(json.dumps([record]))
        return out, record

    def test_cfg_accepts_all_fields(self, tmp_path):
        from entropy.experiments.trace_collection import TraceCollectionCfg
        cfg = TraceCollectionCfg(
            model_name="openai/gpt-oss-20b",
            data_name="aime2025",
            output_dir=str(tmp_path),
            top_k_for_entropy=20,
            max_traces=1,
            max_seq_length=512,
        )
        assert cfg.top_k_for_entropy == 20
        assert cfg.max_traces == 1

    def test_synthetic_pth_roundtrip(self, tmp_path):
        """Save a synthetic question_0000.pth and reload it — checks format contract."""
        n_tok, n_layers, hidden = 56, 4, 64
        from entropy.models.registry import get_thinking_tokens
        cfg = get_thinking_tokens("openai/gpt-oss-20b")
        start = cfg["start_token_ids"]
        end   = cfg["end_token_ids"]
        tokens = [10, 20, 30] + start + list(range(100, 100 + 47)) + end

        data = {
            "input_text": "What is 6 × 7?",
            "prompt_tokens": [10, 20, 30],
            "GT_answer": "42",
            "traces": [{
                "text": r"After thinking, \boxed{42}.",
                "tokens": tokens,
                "activations": torch.randn(len(tokens), n_layers, hidden),
                "entropies_hf": [0.1] * len(tokens),
                "entropies_vllm": [0.1] * len(tokens),
                "answer": "42",
            }],
            "num_layers": n_layers,
            "model_num_layers": n_layers,
            "layers_collected": list(range(n_layers)),
        }

        path = tmp_path / "question_0000.pth"
        torch.save(data, path)

        loaded = torch.load(path, weights_only=False)
        assert loaded["GT_answer"] == "42"
        trace = loaded["traces"][0]
        assert trace["activations"].shape == (len(tokens), n_layers, hidden)
        assert len(trace["entropies_hf"]) == len(tokens)

    def test_thinking_boundaries_in_synthetic_tokens(self, tmp_path):
        """Check that _find_thinking_boundaries correctly locates CoT in synthetic tokens."""
        from unittest.mock import MagicMock, patch
        from entropy.experiments.compression.pooling import CompressionPooling, CompressionPoolingCfg
        from entropy.models.registry import get_thinking_tokens

        model_name = "openai/gpt-oss-20b"
        cfg_tokens = get_thinking_tokens(model_name)
        start = cfg_tokens["start_token_ids"]
        end   = cfg_tokens["end_token_ids"]

        prompt = [10, 20, 30]
        cot    = list(range(100, 120))
        tokens = prompt + start + cot + end

        # Need at least one .pth to pass base.__init__ check
        pth_data = {
            "input_text": "q", "prompt_tokens": prompt, "GT_answer": "42",
            "traces": [{"tokens": tokens, "activations": torch.randn(len(tokens), 4, 64),
                        "entropies_hf": [0.1]*len(tokens), "answer": "42"}],
            "num_layers": 4, "model_num_layers": 4, "layers_collected": list(range(4)),
        }
        torch.save(pth_data, tmp_path / "question_0000.pth")

        cfg = CompressionPoolingCfg(
            model_name=model_name,
            input_dir=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            retention_rate=0.1,
            selection_methods=["low_entropy"],
        )
        with patch("entropy.experiments.compression.base.load_model_and_tokenizer") as mock_load:
            mock_tok = MagicMock()
            mock_tok.pad_token = "<pad>"; mock_tok.pad_token_id = 0
            mock_tok.eos_token_id = 1; mock_tok.all_special_tokens = []
            mock_tok.encode = lambda s, **kw: [99]
            mock_tok.decode = lambda ids, **kw: "x"
            mock_load.return_value = (MagicMock(), mock_tok, {"num_hidden_layers": 4, "hidden_size": 64})
            obj = CompressionPooling(cfg)

        bounds = obj._find_thinking_boundaries(tokens)
        assert bounds is not None
        s, e = bounds
        assert s == len(prompt) + len(start)
        assert e == len(prompt) + len(start) + len(cot)
        assert tokens[s:e] == cot


# ===========================================================================
# 6. MODEL × DATASET CROSS TEST
#    For every (model, dataset) pair: apply chat template and show output
# ===========================================================================

class TestModelDatasetCross:
    """Test chat template + first generation tokens for every model × dataset pair.
    
    Parametrized on models via session fixture. Iterates over all datasets
    within each test so we don't create N*M fixtures (which would reload
    the model N*M times).
    """

    def test_chat_template_all_datasets(self, model_and_tokenizer):
        """Apply chat template for every dataset and verify basic format."""
        from entropy.core.data_utils import get_data
        from entropy.models.registry import get_thinking_tokens
        model_name, _, tokenizer, _ = model_and_tokenizer
        cfg = get_thinking_tokens(model_name)
        template_kwargs = {}
        if cfg.get("enable_thinking"):
            template_kwargs["enable_thinking"] = True

        for data_name in DATASET_NAMES:
            question, answer = get_data(data_name)[0]
            messages = [{"role": "user", "content": question}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **template_kwargs
            )
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

            print(f"\n[{model_name}] [{data_name}]")
            print(f"  Q: {question[:80]!r}")
            print(f"  A: {str(answer)!r}")
            print(f"  PROMPT TAIL: ...{prompt[-150:]!r}")
            print(f"  N tokens: {len(prompt_tokens)}")

            assert len(prompt_tokens) > 0, f"{model_name} × {data_name}: empty prompt"
            assert question[:30] in prompt, f"{model_name} × {data_name}: question not in prompt"

    def test_generation_all_datasets(self, model_and_tokenizer):
        """Generate 10 tokens for one question per dataset and print output."""
        from entropy.core.data_utils import get_data
        from entropy.models.registry import get_thinking_tokens

        model_name, model, tokenizer, _ = model_and_tokenizer
        cfg = get_thinking_tokens(model_name)
        template_kwargs = {}
        if cfg.get("enable_thinking"):
            template_kwargs["enable_thinking"] = True

        for data_name in DATASET_NAMES:
            question, answer = get_data(data_name)[0]
            messages = [{"role": "user", "content": question}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **template_kwargs
            )
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                out = model.generate(input_ids, max_new_tokens=10, do_sample=False)

            generated_ids = out[0][input_ids.shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

            print(f"\n[{model_name}] [{data_name}]")
            print(f"  GENERATED: {generated_text!r}")

            # Verify thinking starts either in prompt or in generation
            full = prompt + generated_text
            thinking_present = (
                cfg["start_token"] in full
                if cfg["start_token_ids"] is None
                else generated_ids[:len(cfg["start_token_ids"])].tolist() == cfg["start_token_ids"]
            )
            assert thinking_present, \
                f"{model_name} × {data_name}: thinking start not found in prompt+generation"


# ===========================================================================
# 7. ACTIVATION AND ENTROPY COLLECTION
#    Step 1: collect_traces.py (vLLM) generates traces for one question
#    Step 2: TraceActivationEntropy._collect_from_trace (HF) extracts acts + entropies
# ===========================================================================

class TestActivationEntropyCollection:
    """Test the full activation + entropy collection pipeline.

    Step 1: Run scripts/collect_traces.py with num_out=1, max 1 question.
            This uses vLLM and saves a JSON with traces_tokens + traces_entropy.
    Step 2: Feed those tokens to TraceActivationEntropy._collect_from_trace
            (HF forward pass) and verify activations and entropies_hf.
    """

    def _run_collect_traces(self, model_name, data_name, output_dir):
        """Run scripts/collect_traces.py for 1 question, 1 trace."""
        import subprocess, sys, json
        result = subprocess.run(
            [sys.executable, "scripts/collect_traces.py",
             "--model_name", model_name,
             "--data_name", data_name,
             "--num_out", "1",
             "--batch_size", "1",
             "--resume", "False",
             "--max_tokens", "128",
             "--gpu_memory_utilization", "0.35",
            ],
            capture_output=True, text=True, cwd=str(Path(__file__).parent.parent),
            env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": "0"},
        )
        print(result.stdout[-2000:] if result.stdout else "")
        print(result.stderr[-2000:] if result.stderr else "")
        assert result.returncode == 0, f"collect_traces.py failed:\n{result.stderr[-1000:]}"

        # Find the output JSON
        import glob
        dataset_short = data_name.split("/")[-1]
        model_short = model_name.split("/")[-1]
        json_path = Path(__file__).parent.parent / "data" / dataset_short / f"{model_short}_teacher_traces.json"
        assert json_path.exists(), f"Output not found: {json_path}"
        with open(json_path) as f:
            traces = json.load(f)
        assert len(traces) >= 1
        return traces[0]

    def _make_collector(self, model, tokenizer, config, tmp_dir, model_name):
        """Instantiate TraceActivationEntropy bypassing model loading."""
        from entropy.experiments.trace_collection import TraceActivationEntropy, TraceCollectionCfg
        n_layers = config["num_hidden_layers"]
        cfg = TraceCollectionCfg(
            model_name=model_name,
            data_name="opencompass/AIME2025",
            output_dir=tmp_dir,
            top_k_for_entropy=20,
        )
        collector = object.__new__(TraceActivationEntropy)
        collector.cfg = cfg
        collector.model = model
        collector.tokenizer = tokenizer
        collector.num_layers = n_layers
        collector.layers_to_collect = list(range(n_layers))
        collector.output_path = Path(tmp_dir)
        collector._partial = False
        collector._suffix = ""
        return collector, n_layers

    def test_collect_from_trace(self, model_and_tokenizer):
        """Full pipeline: collect_traces.py → _collect_from_trace → check acts + entropies."""
        import tempfile
        model_name, model, tokenizer, config = model_and_tokenizer

        # Move HF model to CPU so vLLM subprocess has full GPU budget
        model.cpu()
        torch.cuda.empty_cache()

        # Step 1: generate trace with vLLM via collect_traces.py
        trace_record = self._run_collect_traces(model_name, "aime2025", "data")
        assert len(trace_record["traces_tokens"]) >= 1

        prompt_tokens = trace_record["prompt_tokens"]
        trace_tokens  = trace_record["traces_tokens"][0]
        vllm_entropy  = trace_record["traces_entropy"][0]
        trace_text    = trace_record["traces"][0]

        print(f"\n[{model_name}] prompt_len={len(prompt_tokens)} trace_len={len(trace_tokens)}")
        print(f"  trace[:80]: {trace_text[:80]!r}")

        # Step 2: HF forward pass — move model back to GPU
        model.cuda()
        with tempfile.TemporaryDirectory() as tmp_dir:
            collector, n_layers = self._make_collector(
                model, tokenizer, config, tmp_dir, model_name
            )
            result = collector._collect_from_trace(
                trace_text, prompt_tokens, trace_tokens, vllm_entropy
            )

        acts     = result["activations"]
        ents_hf  = result["entropies_hf"]
        ents_vllm = result["entropies_vllm"]

        print(f"  activations shape: {list(acts.shape)}")
        print(f"  entropies_hf[:5]:   {[f'{e:.4f}' for e in ents_hf[:5]]}")
        print(f"  entropies_vllm[:5]: {[f'{e:.4f}' for e in ents_vllm[:5]]}")

        assert acts.ndim == 3
        assert acts.shape[0] == len(trace_tokens)
        assert acts.shape[1] == n_layers
        assert len(ents_hf)   == len(trace_tokens)
        assert len(ents_vllm) == len(trace_tokens)

        max_ent = __import__("torch").log(__import__("torch").tensor(20.0)).item()
        for i, e in enumerate(ents_hf):
            assert 0.0 <= e <= max_ent + 0.01, f"entropies_hf[{i}]={e:.4f} out of range"

        assert __import__("torch").isfinite(acts).all()
        assert acts.abs().sum() > 0
        assert result["entropies_vllm"] == vllm_entropy
        print(f"  ✓ all checks passed")

    def test_pth_roundtrip(self, model_and_tokenizer):
        """Verify .pth save/load preserves all fields correctly."""
        import tempfile
        model_name, model, tokenizer, config = model_and_tokenizer

        # Move HF model to CPU so vLLM subprocess has full GPU budget
        model.cpu()
        torch.cuda.empty_cache()

        trace_record = self._run_collect_traces(model_name, "aime2025", "data")
        prompt_tokens = trace_record["prompt_tokens"]
        trace_tokens  = trace_record["traces_tokens"][0]
        vllm_entropy  = trace_record["traces_entropy"][0]
        trace_text    = trace_record["traces"][0]
        gt_answer     = str(trace_record["GT_answer"])

        # Move model back to GPU for HF forward pass
        model.cuda()
        with tempfile.TemporaryDirectory() as tmp_dir:
            collector, n_layers = self._make_collector(
                model, tokenizer, config, tmp_dir, model_name
            )
            trace_result = collector._collect_from_trace(
                trace_text, prompt_tokens, trace_tokens, vllm_entropy
            )
            question_result = {
                "input_text":    trace_record["input_text"],
                "prompt_tokens": prompt_tokens,
                "GT_answer":     gt_answer,
                "traces":        [trace_result],
                "num_layers":    n_layers,
                "model_num_layers": n_layers,
                "layers_collected": list(range(n_layers)),
            }
            pth_path = Path(tmp_dir) / "question_0000.pth"
            __import__("torch").save(question_result, pth_path)
            loaded = __import__("torch").load(pth_path, weights_only=False)

        assert loaded["GT_answer"] == gt_answer
        t = loaded["traces"][0]
        assert t["activations"].shape == (len(trace_tokens), n_layers, config["hidden_size"])
        assert len(t["entropies_hf"])   == len(trace_tokens)
        assert len(t["entropies_vllm"]) == len(trace_tokens)
        print(f"\n[{model_name}] .pth roundtrip ✓ shape={list(t['activations'].shape)}")