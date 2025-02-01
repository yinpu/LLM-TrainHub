from .arguments import ModelArguments
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

def get_model(model_args: ModelArguments, is_ref_model=False):
    if is_ref_model == False and model_args.use_lora == True:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype="auto",
        )
        from peft import LoraConfig, get_peft_model, TaskType
        config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            inference_mode=False,
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
        return model

    elif is_ref_model == False and model_args.use_lora == False:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype="auto",
        )
        return model

    elif is_ref_model == True and model_args.use_lora == True:
        return None

    elif is_ref_model == True and model_args.use_lora == False:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype="auto",
        )
        for param in model.parameters():
            param.requires_grad = False
        model = model.eval()
        return model