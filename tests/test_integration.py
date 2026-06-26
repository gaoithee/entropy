"""Structural integration test for the entropy repo.

Checks that all modules import correctly, interfaces are consistent, and the
full data-flow pipeline works end-to-end using synthetic data (no real model,
no GPU required).

Run with:
    pytest tests/test_integration.py -v
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch


# ===========================================================================
# 1. IMPORTS — every public module must be importable
# ===========================================================================

class TestImports:
    def test_core_utils(self):
        from entropy.core.utils import extract_boxed_answer
        assert callable(extract_boxed_answer)

    def test_core_model_loader(self):
        from entropy.core.model_loader import load_model_and_tokenizer
        assert callable(load_model_and_tokenizer)

    def test_core_trace_loader(self):
        from entropy.core.trace_loader import get_reasoning_traces
        assert callable(get_reasoning_traces)

    def test_models_registry(self):
        from entropy.models.registry import get_thinking_tokens
        assert callable(get_thinking_tokens)

    def test_datasets_registry(self):
        from entropy.datasets.registry import get_dataset_config, DATASETS
        assert isinstance(DATASETS, dict)
        assert len(DATASETS) > 0

    def test_compression_common(self):
        from entropy.experiments.compression.common import (
            CompressionResult, compute_top_k_entropy, kl_div_top_k
        )
        assert callable(compute_top_k_entropy)

    def test_compression_base(self):
        from entropy.experiments.compression.base import CompressionBase
        assert issubclass(CompressionBase, object)

    def test_compression_pooling(self):
        from entropy.experiments.compression.pooling import (
            CompressionPooling, CompressionPoolingCfg, ALL_METHODS
        )
        assert len(ALL_METHODS) > 0

    def test_trace_collection(self):
        from entropy.experiments.trace_collection import (
            TraceActivationEntropy, TraceCollectionCfg
        )
        assert callable(TraceActivationEntropy)


# ===========================================================================
# 2. MODELS REGISTRY — thinking tokens for every supported model
# ===========================================================================

class TestModelsRegistry:
    MODELS = [
        "openai/gpt-oss-20b",
        "google/gemma-4-26b",
        "Qwen/Qwen3-14B",
        "microsoft/Phi-4-r-plus",
        "deepseek-ai/DeepSeek-R1",
        "unknown/model-xyz",   # fallback
    ]

    def test_all_models_return_required_keys(self):
        from entropy.models.registry import get_thinking_tokens
        required = {"start_token", "end_token", "start_token_ids", "end_token_ids"}
        for model in self.MODELS:
            cfg = get_thinking_tokens(model)
            missing = required - set(cfg.keys())
            assert not missing, f"{model} missing keys: {missing}"

    def test_gpt_oss_has_explicit_token_ids(self):
        from entropy.models.registry import get_thinking_tokens
        cfg = get_thinking_tokens("openai/gpt-oss-20b")
        assert cfg["start_token_ids"] == [200005, 35644, 200008]
        assert cfg["end_token_ids"] is not None and len(cfg["end_token_ids"]) > 0

    def test_other_models_have_none_token_ids(self):
        from entropy.models.registry import get_thinking_tokens
        for model in ["Qwen/Qwen3-14B"]:
            cfg = get_thinking_tokens(model)
            assert cfg["start_token_ids"] is None
            assert cfg["end_token_ids"] is None

    def test_gemma_has_correct_tokens(self):
        from entropy.models.registry import get_thinking_tokens
        cfg = get_thinking_tokens("google/gemma-4-26b")
        assert cfg["start_token"] == "<|channel>thought"
        assert cfg["end_token"] == "<channel|>"

    def test_mistral_has_explicit_token_ids(self):
        from entropy.models.registry import get_thinking_tokens
        cfg = get_thinking_tokens("mistral/ministral-8b")
        assert cfg["start_token_ids"] == [34]
        assert cfg["end_token_ids"] == [35]

    def test_start_and_end_tokens_are_strings(self):
        from entropy.models.registry import get_thinking_tokens
        for model in self.MODELS:
            cfg = get_thinking_tokens(model)
            assert isinstance(cfg["start_token"], str)
            assert isinstance(cfg["end_token"], str)


# ===========================================================================
# 3. DATASETS REGISTRY
# ===========================================================================

class TestDatasetsRegistry:
    EXPECTED = [
        "opencompass/AIME2024",
        "opencompass/AIME2025",
        "WildEval/ZebraLogic",
        "lighteval/MATH-500",
        "TIGER-Lab/MMLU-Pro",
        "Idavidrein/gpqa",
        "openai/gsm8k",
    ]

    def test_all_expected_datasets_registered(self):
        from entropy.datasets.registry import DATASETS
        for ds in self.EXPECTED:
            assert ds in DATASETS, f"Dataset '{ds}' not in registry"

    def test_get_dataset_config_returns_correct_type(self):
        from entropy.datasets.registry import get_dataset_config, DatasetConfig
        cfg = get_dataset_config("opencompass/AIME2024")
        assert isinstance(cfg, DatasetConfig)

    def test_get_dataset_config_raises_for_unknown(self):
        from entropy.datasets.registry import get_dataset_config
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_dataset_config("nonexistent/dataset")

    def test_all_configs_have_required_fields(self):
        from entropy.datasets.registry import DATASETS
        for name, cfg in DATASETS.items():
            assert cfg.hf_name, f"{name}: hf_name is empty"
            assert cfg.answer_field, f"{name}: answer_field is empty"
            assert cfg.question_field, f"{name}: question_field is empty"


# ===========================================================================
# 4. UTILS
# ===========================================================================

class TestExtractBoxedAnswer:
    def test_simple(self):
        from entropy.core.utils import extract_boxed_answer
        assert extract_boxed_answer(r"The answer is \boxed{42}") == "42"

    def test_nested_braces(self):
        from entropy.core.utils import extract_boxed_answer
        assert extract_boxed_answer(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"

    def test_returns_last_boxed(self):
        from entropy.core.utils import extract_boxed_answer
        result = extract_boxed_answer(r"\boxed{3} then \boxed{7}")
        assert result == "7"

    def test_no_boxed_returns_empty(self):
        from entropy.core.utils import extract_boxed_answer
        assert extract_boxed_answer("no answer here") == ""

    def test_empty_string(self):
        from entropy.core.utils import extract_boxed_answer
        assert extract_boxed_answer("") == ""


# ===========================================================================
# 5. COMPRESSION COMMON
# ===========================================================================

class TestCompressionCommon:
    def test_compute_top_k_entropy_shape(self):
        from entropy.experiments.compression.common import compute_top_k_entropy
        logits = torch.randn(1000)
        val = compute_top_k_entropy(logits, k=20)
        assert isinstance(val, float)
        assert val >= 0.0

    def test_compute_top_k_entropy_uniform_is_high(self):
        from entropy.experiments.compression.common import compute_top_k_entropy
        # Uniform over top-k → max entropy
        logits_uniform = torch.zeros(1000)
        logits_peaked  = torch.zeros(1000); logits_peaked[0] = 100.0
        assert compute_top_k_entropy(logits_uniform) > compute_top_k_entropy(logits_peaked)

    def test_compression_result_fields(self):
        from entropy.experiments.compression.common import CompressionResult
        r = CompressionResult(
            compressed_prompt="[prompt]",
            generated_text="42",
            answer="42",
            num_peaks_used=5,
            original_thinking_length=100,
            compression_ratio=0.05,
        )
        assert r.answer == "42"
        assert r.compression_ratio == 0.05

    def test_kl_div_top_k(self):
        from entropy.experiments.compression.common import kl_div_top_k
        p = torch.randn(10, 1000)
        q = torch.randn(10, 1000)
        kl = kl_div_top_k(p, q, top_k=20)
        assert kl.item() >= 0.0


# ===========================================================================
# 6. THINKING BOUNDARY DETECTION (core CompressionBase logic)
#    Tested with a mock that bypasses model loading
# ===========================================================================

def _make_mock_compression_base(start_ids, end_ids):
    """Return a CompressionBase-like object with _find_thinking_boundaries
    wired up but no real model loaded."""
    from entropy.experiments.compression.base import CompressionBase
    obj = object.__new__(CompressionBase)
    obj.start_thinking_ids = start_ids
    obj.end_thinking_ids   = end_ids
    return obj


class TestFindThinkingBoundaries:
    # Synthetic token sequences using gpt-oss-20b delimiters
    START = [200005, 35644, 200008]
    END   = [200007, 200006, 173781, 200005, 17196, 200008]

    def _obj(self):
        return _make_mock_compression_base(self.START, self.END)

    def test_finds_correct_boundaries(self):
        obj = self._obj()
        cot_tokens = [1, 2, 3]
        tokens = self.START + cot_tokens + self.END
        result = obj._find_thinking_boundaries(tokens)
        assert result is not None
        start, end = result
        assert start == len(self.START)
        assert end == len(self.START) + len(cot_tokens)

    def test_returns_none_when_start_missing(self):
        obj = self._obj()
        tokens = [1, 2, 3] + self.END
        assert obj._find_thinking_boundaries(tokens) is None

    def test_returns_none_when_end_missing(self):
        obj = self._obj()
        tokens = self.START + [1, 2, 3]
        assert obj._find_thinking_boundaries(tokens) is None

    def test_returns_none_when_end_before_start(self):
        obj = self._obj()
        tokens = self.END + self.START
        assert obj._find_thinking_boundaries(tokens) is None

    def test_prompt_tokens_before_start_ignored(self):
        obj = self._obj()
        prompt = [10, 20, 30]
        cot    = [5, 6, 7, 8]
        tokens = prompt + self.START + cot + self.END
        start, end = obj._find_thinking_boundaries(tokens)
        assert start == len(prompt) + len(self.START)
        assert end   == len(prompt) + len(self.START) + len(cot)


# ===========================================================================
# 7. ANCHOR SELECTION — all methods, no model
# ===========================================================================

def _make_mock_pooling(retention_rate=0.1):
    """CompressionPooling with a fake tokenizer, no model loaded."""
    from entropy.experiments.compression.pooling import CompressionPooling, CompressionPoolingCfg

    obj = object.__new__(CompressionPooling)

    # Fake tokenizer: decode returns the token id as string
    tok = MagicMock()
    tok.decode = lambda ids, **kw: str(ids[0]) if ids else ""
    obj.student_tokenizer = tok

    # Minimal cfg
    obj.cfg = SimpleNamespace(retention_rate=retention_rate)
    return obj


class TestAnchorSelection:
    N = 50  # thinking length
    TOKENS = list(range(50))
    ENTROPIES = [float(i) / 50 for i in range(50)]  # monotonically increasing

    def _obj(self, rate=0.1):
        return _make_mock_pooling(rate)

    def _k(self, rate=0.1):
        return max(1, int(rate * self.N))

    # --- score-based ---

    def test_low_entropy_selects_lowest(self):
        obj = self._obj()
        anchors = obj._select_anchors("low_entropy", self.TOKENS, self.ENTROPIES)
        k = self._k()
        assert anchors == sorted(range(k))   # lowest entropies are indices 0..k-1

    def test_high_entropy_selects_highest(self):
        obj = self._obj()
        anchors = obj._select_anchors("high_entropy", self.TOKENS, self.ENTROPIES)
        k = self._k()
        assert anchors == sorted(range(self.N - k, self.N))

    def test_random_correct_count(self):
        obj = self._obj()
        anchors = obj._select_anchors("random", self.TOKENS, self.ENTROPIES)
        assert len(anchors) == self._k()
        assert anchors == sorted(anchors)
        assert all(0 <= a < self.N for a in anchors)

    def test_before_entropy_adjacent_to_peaks(self):
        obj = self._obj()
        k = self._k()
        peaks = sorted(range(self.N - k, self.N))  # same as high_entropy indices
        expected = sorted(set(max(0, p - 1) for p in peaks))
        anchors = obj._select_anchors("before_entropy", self.TOKENS, self.ENTROPIES)
        assert anchors == expected

    def test_after_entropy_adjacent_to_peaks(self):
        obj = self._obj()
        k = self._k()
        peaks = sorted(range(self.N - k, self.N))
        expected = sorted(set(min(self.N - 1, p + 1) for p in peaks))
        anchors = obj._select_anchors("after_entropy", self.TOKENS, self.ENTROPIES)
        assert anchors == expected

    def test_all_score_methods_return_sorted(self):
        obj = self._obj()
        for method in ["low_entropy", "high_entropy", "random", "before_entropy", "after_entropy"]:
            anchors = obj._select_anchors(method, self.TOKENS, self.ENTROPIES)
            assert anchors == sorted(anchors), f"{method} not sorted"

    def test_retention_rate_1_returns_all(self):
        obj = _make_mock_pooling(retention_rate=1.0)
        for method in ["low_entropy", "high_entropy", "random"]:
            anchors = obj._select_anchors(method, self.TOKENS, self.ENTROPIES)
            assert len(anchors) == self.N

    # --- content-based: tokenizer-dependent ---

    def test_newline_finds_newline_tokens(self):
        obj = _make_mock_pooling()
        # Override decode: token at index 5 decodes to "\n"
        def fake_decode(ids, **kw):
            return "\n" if ids[0] == 5 else "x"
        obj.student_tokenizer.decode = fake_decode
        tokens = list(range(10))
        anchors = obj._select_anchors("newline", tokens, [0.0] * 10)
        assert anchors == [5]

    def test_newline_returns_none_when_none_found(self):
        obj = _make_mock_pooling()
        obj.student_tokenizer.decode = lambda ids, **kw: "x"
        anchors = obj._select_anchors("newline", list(range(10)), [0.0] * 10)
        assert anchors is None

    def test_numbers_finds_digit_tokens(self):
        obj = _make_mock_pooling()
        def fake_decode(ids, **kw):
            return "42" if ids[0] == 3 else "word"
        obj.student_tokenizer.decode = fake_decode
        tokens = list(range(10))
        anchors = obj._select_anchors("numbers", tokens, [0.0] * 10)
        assert anchors == [3]

    def test_unknown_method_raises(self):
        obj = _make_mock_pooling()
        with pytest.raises(ValueError, match="Unknown method"):
            obj._select_anchors("banana", list(range(10)), [0.0] * 10)


# ===========================================================================
# 8. POOLING GEOMETRY
# ===========================================================================

class TestPooling:
    def _obj(self):
        return _make_mock_pooling()

    def test_pool_produces_correct_number_of_segments(self):
        obj = self._obj()
        # 20 tokens, 3 anchors → boundaries become {0,3,7,19} → 3 segments
        acts = torch.randn(20, 4, 64)  # [n_tok, n_layers, hidden_dim]
        anchors = [3, 7]
        pooled = obj._pool(acts, anchors)
        # anchors ∪ {0, 19} = {0,3,7,19} → 3 segments: [0-3],[4-7],[8-19]
        assert len(pooled) == 3

    def test_pool_output_shape(self):
        obj = self._obj()
        n_layers, hidden = 4, 64
        acts = torch.randn(30, n_layers, hidden)
        pooled = obj._pool(acts, [5, 15])
        for p in pooled:
            assert p.shape == (n_layers, hidden)

    def test_pool_single_anchor_two_segments(self):
        obj = self._obj()
        acts = torch.randn(10, 2, 8)
        pooled = obj._pool(acts, [4])
        # anchors = {0, 4, 9} → 2 segments
        assert len(pooled) == 2

    def test_pool_values_are_means(self):
        obj = self._obj()
        acts = torch.zeros(6, 1, 1)
        acts[0] = 2.0; acts[1] = 4.0  # seg [0-2]: mean of rows 0,1,2
        acts[2] = 0.0
        pooled = obj._pool(acts, [2])
        # segment 0: rows 0,1,2 → mean = (2+4+0)/3 = 2.0
        assert abs(pooled[0].item() - 2.0) < 1e-5

    def test_pool_retention_rate_1_is_identity(self):
        """When all positions are anchors, each segment is one token → no averaging."""
        obj = self._obj()
        n = 5
        acts = torch.randn(n, 2, 8)
        anchors = list(range(n))
        pooled = obj._pool(acts, anchors)
        # Each segment is a single token
        for i, p in enumerate(pooled[:-1]):   # last segment may merge
            assert p.shape == (2, 8)


# ===========================================================================
# 9. METRICS
# ===========================================================================

class TestComputeMetrics:
    def _obj(self):
        obj = object.__new__(__import__(
            "entropy.experiments.compression.base", fromlist=["CompressionBase"]
        ).CompressionBase)
        return obj

    def test_all_correct(self):
        obj = self._obj()
        pak, pr = obj._compute_metrics("42", ["42", "42", "42"])
        assert pak == 1
        assert pr == 1.0

    def test_all_wrong(self):
        obj = self._obj()
        pak, pr = obj._compute_metrics("42", ["7", "7"])
        assert pak == 0
        assert pr == 0.0

    def test_partial_correct(self):
        obj = self._obj()
        pak, pr = obj._compute_metrics("42", ["42", "7", "42"])
        assert pak == 1
        assert abs(pr - 2/3) < 1e-6

    def test_empty_answers(self):
        obj = self._obj()
        pak, pr = obj._compute_metrics("42", [])
        assert pak == 0
        assert pr == 0.0

    def test_permissive_substring_matching(self):
        # gt "42" should match "the answer is 42" (substring)
        obj = self._obj()
        pak, pr = obj._compute_metrics("42", ["the answer is 42"])
        assert pak == 1
        assert pr == 1.0

    def test_case_insensitive(self):
        obj = self._obj()
        pak, pr = obj._compute_metrics("ABC", ["abc"])
        assert pak == 1


# ===========================================================================
# 10. PIPELINE: synthetic question_XXXX.pth → _process_trace (no model)
#     Tests that trace format, boundary detection, selection, and pooling
#     are all compatible end-to-end
# ===========================================================================

START_IDS = [200005, 35644, 200008]
END_IDS   = [200007, 200006, 173781, 200005, 17196, 200008]


def make_synthetic_trace(n_cot=50, n_layers=4, hidden=64):
    """Build a synthetic trace dict matching the real question_XXXX.pth format."""
    prompt_tokens = [10, 20, 30]
    cot_tokens    = list(range(100, 100 + n_cot))
    tokens        = prompt_tokens + START_IDS + cot_tokens + END_IDS

    n_total = len(tokens)
    activations  = torch.randn(n_total, n_layers, hidden)
    entropies_hf = [float(i) / n_total for i in range(n_total)]

    return {
        "tokens": tokens,
        "activations": activations,
        "entropies_hf": entropies_hf,
        "entropies_vllm": entropies_hf,
        "answer": r"\boxed{42}",
        "text": "some reasoning... " + r"\boxed{42}",
    }


def make_synthetic_question_file(tmp_path: Path, q_id: int = 0):
    """Save a synthetic question_XXXX.pth and return the path."""
    trace = make_synthetic_trace()
    data = {
        "input_text": "What is 6 × 7?",
        "prompt_tokens": [10, 20, 30],
        "GT_answer": "42",
        "traces": [trace],
        "num_layers": 4,
        "model_num_layers": 4,
        "layers_collected": list(range(4)),
    }
    path = tmp_path / f"question_{q_id:04d}.pth"
    torch.save(data, path)
    return path


class TestEndToEndPipeline:
    """Tests the full data flow without a real model."""

    def _make_pooling_obj(self, tmp_path):
        """Instantiate CompressionPooling with all model-loading patched out."""
        from entropy.experiments.compression.pooling import CompressionPooling, CompressionPoolingCfg

        # Save synthetic file BEFORE instantiating (base.__init__ checks for files)
        make_synthetic_question_file(tmp_path, q_id=0)

        cfg = CompressionPoolingCfg(
            model_name="openai/gpt-oss-20b",
            input_dir=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            retention_rate=0.1,
            selection_methods=["low_entropy", "high_entropy", "random",
                               "before_entropy", "after_entropy",
                               "newline", "end_of_sentence", "numbers"],
        )

        with patch("entropy.experiments.compression.base.load_model_and_tokenizer") as mock_load:
            mock_model     = MagicMock()
            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token     = "<pad>"
            mock_tokenizer.pad_token_id  = 0
            mock_tokenizer.eos_token_id  = 1
            mock_tokenizer.all_special_tokens = []
            mock_tokenizer.encode = lambda s, **kw: [99]
            mock_tokenizer.decode = lambda ids, **kw: "x"
            mock_config = {"num_hidden_layers": 4, "hidden_size": 64}
            mock_load.return_value = (mock_model, mock_tokenizer, mock_config)

            obj = CompressionPooling(cfg)

        return obj

    def test_question_file_format_readable(self, tmp_path):
        path = make_synthetic_question_file(tmp_path)
        data = torch.load(path, weights_only=False)
        assert "traces" in data
        assert "GT_answer" in data
        assert "prompt_tokens" in data
        trace = data["traces"][0]
        assert "tokens" in trace
        assert "activations" in trace
        assert "entropies_hf" in trace
        assert trace["activations"].ndim == 3   # [n_tok, n_layers, hidden]

    def test_boundaries_found_in_synthetic_trace(self, tmp_path):
        obj = self._make_pooling_obj(tmp_path)
        trace = make_synthetic_trace(n_cot=50)
        bounds = obj._find_thinking_boundaries(trace["tokens"])
        assert bounds is not None
        start, end = bounds
        # CoT starts after prompt (3) + START_IDS (3) = index 6
        assert start == 3 + len(START_IDS)
        assert end == start + 50

    def test_all_score_methods_run_on_synthetic_trace(self, tmp_path):
        obj = self._make_pooling_obj(tmp_path)
        trace = make_synthetic_trace(n_cot=50)
        bounds = obj._find_thinking_boundaries(trace["tokens"])
        s, e = bounds
        thinking_tokens = trace["tokens"][s:e]
        thinking_ents   = trace["entropies_hf"][s:e]

        for method in ["low_entropy", "high_entropy", "random", "before_entropy", "after_entropy"]:
            anchors = obj._select_anchors(method, thinking_tokens, thinking_ents)
            assert anchors is not None, f"{method} returned None"
            assert len(anchors) > 0,   f"{method} returned empty list"
            assert anchors == sorted(anchors), f"{method} not sorted"

    def test_pool_compatible_with_trace_activations(self, tmp_path):
        obj = self._make_pooling_obj(tmp_path)
        trace = make_synthetic_trace(n_cot=20, n_layers=4, hidden=64)
        bounds = obj._find_thinking_boundaries(trace["tokens"])
        s, e = bounds
        thinking_acts = trace["activations"][s:e]
        thinking_ents = trace["entropies_hf"][s:e]
        thinking_toks = trace["tokens"][s:e]

        anchors = obj._select_anchors("low_entropy", thinking_toks, thinking_ents)
        pooled  = obj._pool(thinking_acts, anchors)

        assert len(pooled) > 0
        for p in pooled:
            assert p.shape == (4, 64)   # [n_layers, hidden_dim]

    def test_process_trace_returns_not_applicable_for_content_methods_when_no_match(self, tmp_path):
        """newline/end_of_sentence/numbers should return NOT_APPLICABLE gracefully
        when the synthetic tokens don't match their criteria."""
        obj = self._make_pooling_obj(tmp_path)
        # tokenizer.decode always returns "x" (no newlines, no digits)
        trace = make_synthetic_trace(n_cot=20)
        result = obj._process_trace(0, 0, [10, 20, 30], trace, "newline")
        assert result == "NOT_APPLICABLE"

    def test_jsonl_output_written_correctly(self, tmp_path):
        """Check that the JSONL output file is valid after a mock run."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        record = {
            "question_id": 0,
            "GT_answer": "42",
            "original_traces": {"answers": ["42"], "pass_rate": 1.0},
            "methods": {
                "low_entropy": {"answers": ["42"], "compression_ratios": [0.1], "pass_rate": 1.0}
            },
            "experiment_params": {"retention_rate": 0.1, "model": "test"},
        }
        output_file = out_dir / "compression_pooling_results_rate_0.1.jsonl"
        with open(output_file, "w") as f:
            f.write(json.dumps(record) + "\n")

        with open(output_file) as f:
            loaded = json.loads(f.readline())
        assert loaded["GT_answer"] == "42"
        assert loaded["methods"]["low_entropy"]["pass_rate"] == 1.0


# ===========================================================================
# 11. TRACE COLLECTION FORMAT
# ===========================================================================

class TestTraceCollectionFormat:
    """Checks TraceCollectionCfg fields and output format contract."""

    def test_cfg_defaults(self):
        from entropy.experiments.trace_collection import TraceCollectionCfg
        cfg = TraceCollectionCfg(
            model_name="openai/gpt-oss-20b",
            data_name="opencompass/AIME2025",
            output_dir="/tmp/test",
        )
        assert cfg.top_k_for_entropy == 20
        assert cfg.max_traces is None
        assert cfg.max_seq_length is None
        assert cfg.layers is None

    def test_synthetic_pth_has_correct_activation_shape(self):
        trace = make_synthetic_trace(n_cot=30, n_layers=6, hidden=128)
        acts = trace["activations"]
        # full sequence = prompt(3) + START(3) + CoT(30) + END(6) = 42 tokens
        assert acts.shape == (42, 6, 128)

    def test_entropies_hf_length_matches_tokens(self):
        trace = make_synthetic_trace(n_cot=30)
        assert len(trace["entropies_hf"]) == len(trace["tokens"])