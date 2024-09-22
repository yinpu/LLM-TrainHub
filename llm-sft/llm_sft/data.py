from datasets import load_dataset
from pathlib import Path
from typing import Dict, Optional, Sequence, List
import logging
from functools import partial
from torch.utils.data import Dataset
from .arguments import DataArguments, TrainArguments
import transformers
from transformers import DataCollatorForSeq2Seq
import os

logger = logging.getLogger(__name__)
IGNORE_INDEX = -100


def build_source_text(prompt: str, tokenizer: transformers.PreTrainedTokenizer):

    if not hasattr(tokenizer, 'apply_chat_template'):
        raise AttributeError("The tokenizer does not have the method 'apply_chat_template'")
    
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text

def get_all_datapath(dir_name: str) -> List[str]:
    all_file_list = [str(path) for path in Path(dir_name).rglob('*') if path.is_file()]
    return all_file_list

def load_dataset_from_path(data_path: Optional[str] = None,
                           cache_dir: Optional[str] = "cache_data") -> Dataset:
    all_file_list = get_all_datapath(data_path)
    data_files = {'train': all_file_list}
    extension = all_file_list[0].split(".")[-1]

    logger.info("load files %d number", len(all_file_list))

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=cache_dir,
    )['train']
    return raw_datasets


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized = tokenizer(
        strings,
        return_tensors="pt",
        padding=True,
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    input_ids = labels = tokenized['input_ids']
    ne_pad_token_id = IGNORE_INDEX if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    input_ids_lens = labels_lens = (input_ids != ne_pad_token_id).sum(dim=1).tolist()
    return {
        "input_ids": input_ids,
        "labels": labels,
        "input_ids_lens": input_ids_lens,
        "labels_lens": labels_lens,
    }


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict[str, List]:
    """Preprocess the data by tokenizing."""
    combined_texts = [s + t for s, t in zip(sources, targets)]
    examples_tokenized = _tokenize_fn(combined_texts, tokenizer)
    sources_tokenized = _tokenize_fn(sources, tokenizer)

    input_ids = examples_tokenized["input_ids"]
    labels = input_ids.clone()

    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return  {
        "input_ids": input_ids,
        "labels": labels
    }


def make_train_dataset(
    tokenizer: transformers.PreTrainedTokenizer,
    data_path: str,
    data_args: DataArguments,
) -> Dataset:
    logger.warning("Loading data...")
    dataset = load_dataset_from_path(
        data_path=data_path,
    )
    logger.warning("Formatting inputs...")

    def generate_sources_targets(examples: Dict):
        instructions = examples.get("instruction", [])
        inputs = examples.get("input", [])
        outputs = examples.get("output", [])

        sources, targets = [], []
        for instruction, input, output in zip(instructions, inputs, outputs):
            prompt = (instruction or "") + (input or "")
            prompt = prompt[: data_args.source_length]
            source = build_source_text(prompt, tokenizer)
            target = (output or "")[:data_args.target_length - 1] + tokenizer.eos_token
            sources.append(source)
            targets.append(target)

        input_output = preprocess(sources=sources, targets=targets, tokenizer=tokenizer)
        examples["input_ids"] = input_output["input_ids"]
        examples["labels"] = input_output["labels"]
        return examples

    dataset = dataset.map(
        function=generate_sources_targets,
        batched=True,
        desc="Tokenizing training dataset",
        num_proc=min(40, os.cpu_count() or 1),
    )
    return dataset

def get_dataset(data_args: DataArguments, train_args: TrainArguments, tokenizer: transformers.PreTrainedTokenizer):
    with train_args.main_process_first(desc="make_train_dataset"):
        dataset = make_train_dataset(
            tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args
        )
    return dataset

def get_data_collator(tokenizer: transformers.PreTrainedTokenizer, model: transformers.PreTrainedModel):
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, label_pad_token_id=IGNORE_INDEX
    )
    return data_collator
