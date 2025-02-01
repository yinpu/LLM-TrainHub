import warnings
from collections import defaultdict
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import Trainer
from trl import DPOTrainer
from trl.trainer import disable_dropout_in_model
from typing_extensions import override
from packaging import version
from typing import TYPE_CHECKING
import importlib.metadata
import importlib.util
from .arguments import FinetuningArguments

IGNORE_INDEX = -100

if TYPE_CHECKING:
    from packaging.version import Version

def _get_package_version(name: str) -> "Version":
    try:
        return version.parse(importlib.metadata.version(name))
    except Exception:
        return version.parse("0.0.0")

def is_transformers_version_equal_to_4_46():
    return version.parse("4.46.0") <= _get_package_version("transformers") <= version.parse("4.46.1")

if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

def get_batch_logps(
    logits: "torch.Tensor", labels: "torch.Tensor", label_pad_token_id: int = IGNORE_INDEX
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """
    Computes the log probabilities of the given labels under the given logits.
    Returns:
        logps: A tensor of shape (batch_size,) containing the sum of log probabilities.
        valid_length: A tensor of shape (batch_size,) containing the number of non-masked tokens.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batchsize x seq_len) and labels must have the same shape.")
    
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id
    labels[labels == label_pad_token_id] = 0  # dummy token
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)

class CustomDPOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        disable_dropout: bool = True,
        **kwargs,
    ):
        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

        self.label_pad_token_id = IGNORE_INDEX
        self.ref_model = ref_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self.precompute_ref_log_probs = False
        self._peft_has_been_casted_to_bf16 = False
        self.use_dpo_data_collator = True
        self.reference_free = False
        self.f_divergence_type = "reverse_kl"
        self.loss_type = 'sigmoid'
        # dpo hyperparams
        self.beta = finetuning_args.pref_beta
        self.ftx_gamma = finetuning_args.pref_ftx
        self.label_smoothing = finetuning_args.label_smoothing

        self.dpo_type = finetuning_args.dpo_type
        self.simplo_gamma = finetuning_args.simplo_gamma
        print(f"DPO_TYPE: {self.dpo_type}")

        Trainer.__init__(self, model=model, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
                else:
                    self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
            self.ref_model.eval()

        if getattr(self.args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
                print("enable_input_require_grads")
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
                print("enable_input_embeddings_require_grads")

    def simplo_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        """
        Computes SimPLO loss for batched log probabilities of the policy model.
        """
        pi_logratios = chosen_logps - rejected_logps
        gamma_logratios = self.simplo_gamma / self.beta
        logits = pi_logratios - gamma_logratios
        simplo_loss = -F.logsigmoid(self.beta * logits)
        return simplo_loss

    def compute_preference_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: Optional["torch.Tensor"],
        reference_rejected_logps: Optional["torch.Tensor"],
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """
        Computes loss for preference learning.
        """