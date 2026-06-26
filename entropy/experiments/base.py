"""Base classes for experiments."""

from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Optional

from ..core import get_data


@dataclass
class BaseCfg:
    """Base configuration for all experiments."""

    def __init__(
        self,
        student_model_name: str,
        teacher_model_name: str,
        data_name: str,
        output_path: str,
        quantization: Optional[Literal["4bit", "8bit"]] = None,
        attn_implementation: Optional[Literal["flash_attention_2", "sdpa", "eager"]] = None,
    ):
        self.student_model_name = student_model_name
        self.teacher_model_name = teacher_model_name
        self.data_name = data_name
        self.quantization = quantization
        self.attn_implementation = attn_implementation

        output_path_obj: Path = Path(output_path)
        output_path_obj.mkdir(parents=True, exist_ok=True)
        self.output_path: Path = output_path_obj


class Experiment:
    """Base class for all experiments."""

    def __init__(self, base_cfg: BaseCfg):
        self.base_cfg = base_cfg
        self.reasoning_dataset = get_data(base_cfg.data_name)

    def run(self):
        raise NotImplementedError