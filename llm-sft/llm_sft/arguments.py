from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-0.5B")
    use_lora: Optional[bool] = field(default=False)


@dataclass
class DataArguments:
    cache_dir_data: str
    data_path: str = field(default=None, metadata={
        "help": "Path to the training data."})
    source_length: int = field(default=512)
    target_length: int = field(default=512)


@dataclass
class TrainArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_deepspeed: bool = field(default=False)
