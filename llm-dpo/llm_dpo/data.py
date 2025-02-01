from datasets import load_dataset
from .arguments import DataArguments
from transformers import TrainingArguments, PreTrainedTokenizer, DataCollatorForSeq2Seq
from dataclasses import dataclass
from functools import partial
import logging
from typing import Dict
from collections import defaultdict
from typing import Optional, Sequence, Tuple, Any
import torch

logger = logging.getLogger(__name__)
IGNORE_INDEX = -100

def infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> Tuple[int, int]:
    r"""
    Computes the real sequence length after truncation by the cutoff_len.
    """
    if target_len * 2 < cutoff_len:  # truncate source
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:  # truncate target
        max_target_len = cutoff_len - source_len
    else:  # truncate both
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    return new_source_len, new_target_len

def preprocess(examples: Dict, tokenizer: PreTrainedTokenizer, cutoff_len: int):
    model_inputs = defaultdict(list)
    prompts = examples['prompt']
    chosens = examples['chosen']
    rejecteds = examples['rejected']
    for prompt, chosen, rejected in zip(prompts, chosens, rejecteds):
        prompt = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        prompt_text = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True
        )
        prompt_ids = tokenizer(prompt_text)['input_ids']
        chosen_ids = tokenizer(chosen + tokenizer.eos_token)['input_ids']
        rejected_ids = tokenizer(rejected + tokenizer.eos_token)['input_ids']

        source_len, target_len = infer_seqlen(len(prompt_ids), max(len(chosen_ids), len(rejected_ids)), cutoff_len)
        prompt_ids = prompt_ids[:source_len]
        chosen_ids = chosen_ids[:target_len]
        rejected_ids = rejected_ids[:target_len]

        chosen_input_ids = prompt_ids + chosen_ids
        chosen_labels = [IGNORE_INDEX] * len(prompt_ids) + chosen_ids
        rejected_input_ids = prompt_ids + rejected_ids
        rejected_labels = [IGNORE_INDEX] * len(prompt_ids) + rejected_ids
        model_inputs["chosen_input_ids"].append(chosen_input_ids)
        model_inputs["chosen_attention_mask"].append([1] * len(chosen_input_ids))
        model_inputs["chosen_labels"].append(chosen_labels)
        model_inputs["rejected_input_ids"].append(rejected_input_ids)
        model_inputs["rejected_attention_mask"].append([1] * len(rejected_input_ids))
        model_inputs["rejected_labels"].append(rejected_labels)
    return model_inputs

def make_train_dataset(data_args: DataArguments, tokenizer: PreTrainedTokenizer):
    dataset = load_dataset(
        "json",
        data_files=data_args.data_path,
        cache_dir=data_args.cache_dir,
    )['train']
    preprocess_func = partial(preprocess, tokenizer=tokenizer, cutoff_len=data_args.cutoff_len)
    dataset = dataset.map(
        preprocess_func,
        batched=True,
        desc="Tokenizing training dataset",
        num_proc=16,
    )
    return dataset

def get_dataset(data_args: DataArguments, train_args: TrainingArguments, tokenizer: PreTrainedTokenizer):
    with train_args.main_process_first(desc="make_train_dataset"):
        dataset = make_train_dataset(
            tokenizer=tokenizer, data_args=data_args
        )
    return dataset

@dataclass
class PairwiseDataCollatorWithPadding(DataCollatorForSeq2Seq):
    """
    Data collator for pairwise data.
    """

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        r"""
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        concatenated_features = []
        for key in ("chosen", "rejected"):
            for feature in features:
                target_feature = {
                    "input_ids": feature[f"{key}_input_ids"],
                    "attention_mask": feature[f"{key}_attention_mask"],
                    "labels": feature[f"{key}_labels"],
                }
                concatenated_features.append(target_feature)
        return super().__call__(concatenated_features)