"""Shared utilities for compression experiments.

Ported directly from neurohike/experiments/compression/common.py.
Model-specific thinking token logic moved to entropy.models.registry.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class CompressionResult:
    """Result of a single compressed generation."""
    compressed_prompt: str
    generated_text: str
    answer: str
    num_peaks_used: int
    original_thinking_length: int
    compression_ratio: float


def kl_div_top_k(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    top_k: int = 20,
) -> torch.Tensor:
    """Compute KL(p || q) using only the top-k tokens from p per position.

    Ported from neurohike unchanged.
    """
    _, top_indices = torch.topk(logits_p, top_k, dim=-1)
    p_topk = torch.gather(logits_p, dim=-1, index=top_indices)
    q_topk = torch.gather(logits_q, dim=-1, index=top_indices)
    log_p = F.log_softmax(p_topk.float(), dim=-1)
    log_q = F.log_softmax(q_topk.float(), dim=-1)
    return F.kl_div(log_q, log_p, log_target=True, reduction="batchmean")


def compute_top_k_entropy(logits: torch.Tensor, k: int = 20) -> float:
    """Compute entropy over top-k tokens from logits."""
    top_k_logits, _ = torch.topk(logits, k, dim=-1)
    probs = F.softmax(top_k_logits, dim=-1)
    log_probs = torch.log(probs + 1e-10)
    return (-torch.sum(probs * log_probs, dim=-1)).item()
