from typing import List

import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer


class EmbeddingModel4Qwen2(nn.Module):
    def __init__(
        self, model_name_or_path: str, device: str = "cuda", max_length: int = 4096
    ) -> None:
        super(EmbeddingModel4Qwen2, self).__init__()
        self.max_length = max_length
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        # if self.device == "cuda":
        #     self.model.to("cuda")

    def forward(
        self,
        text: List[str],
        max_len: int = None,
    ):
        max_length = max_len if max_len is not None else self.max_length
        # Tokenize the input texts
        batch_dict = self.tokenizer(
            text,
            max_length=max_length - 1,
            return_attention_mask=False,
            padding=False,
            truncation=True,
        )
        # append eos_token_id to every input_ids
        batch_dict["input_ids"] = [
            input_ids + [self.tokenizer.eos_token_id]
            for input_ids in batch_dict["input_ids"]
        ]
        batch_dict = self.tokenizer.pad(
            batch_dict, padding=True, return_attention_mask=True, return_tensors="pt"
        )

        for tempkey in batch_dict.keys():
            batch_dict[tempkey] = batch_dict[tempkey].to(self.model.device)

        modeloutput = self.model(**batch_dict)

        embeddings = self.last_token_pool(
            modeloutput.last_hidden_state, batch_dict["attention_mask"]
        )
        return embeddings

    def last_token_pool(
        self, last_hidden_states: Tensor, attention_mask: Tensor
    ) -> Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]