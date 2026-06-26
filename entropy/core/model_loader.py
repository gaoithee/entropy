"""Model and tokenizer loading utilities.

Ported from neurohike.core.models.
Supports both plain HuggingFace (for trace collection) and nnsight
(for compression experiments with activation patching).
"""
from __future__ import annotations
from typing import Any, Literal

import torch


def load_model_and_tokenizer(
    model_name: str,
    model_type: Literal["hf", "nnsight"] = "hf",
    quantization: str | None = None,
    attn_implementation: str | None = None,
) -> tuple[Any, Any, dict]:
    """Load a causal LM and its tokenizer.

    Parameters
    ----------
    model_name          : HuggingFace model identifier
    model_type          : "hf" for plain transformers (trace collection)
                          "nnsight" for LanguageModel wrapper (compression)
    quantization        : None | "4bit" | "8bit"
    attn_implementation : None | "flash_attention_2" | "sdpa" | "eager"

    Returns
    -------
    (model, tokenizer, config_dict)
    config_dict always contains {"num_hidden_layers": int, "hidden_size": int}
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if quantization == "4bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    elif quantization == "8bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_type == "nnsight":
        from nnsight import LanguageModel
        model = LanguageModel(model_name, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        model.eval()

    raw_config = (
        model._model.config.to_dict()
        if model_type == "nnsight" and hasattr(model, "_model")
        else model.config.to_dict()
    )
    # Gemma4 and other multimodal models nest text config
    text_config = raw_config.get("text_config", raw_config)
    config = {
        "num_hidden_layers": text_config.get("num_hidden_layers", text_config.get("n_layer", 0)),
        "hidden_size":       text_config.get("hidden_size",       text_config.get("n_embd", 0)),
    }
    return model, tokenizer, config