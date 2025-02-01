from dataclasses import dataclass, field
from typing import Optional, Literal

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/")
    use_lora: bool = field(default=False)
    lora_rank: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0)

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    cutoff_len: int = field(
        default=1024, metadata={"help": "The maximum length of the input."}
    )
    cache_dir: str = field(
        default=None, metadata={"help": "use data cache"}
    )

@dataclass
class FinetuningArguments:
    dpo_type: Literal["dpo", "simpo"] = field(
        default="dpo",
        metadata={"help": "Which dpo method to use."},
    )
    pref_beta: float = field(
        default=0.1,
        metadata={"help": "The beta parameter in the preference loss."},
    )
    pref_fix: float = field(
        default=0.0,
        metadata={"help": "The supervised fine-tuning loss coefficient in DPO training."},
    )
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "The robust DPO label smoothing parameter in cDPO that should be between 0 and 0.5."},
    )
    disable_shuffling: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable the shuffling of the training set."},
    )
    simpo_gamma: float = field(
        default=0.5,
        metadata={"help": "The target reward margin term in SimPO loss."},
    )