import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from .arguments import ModelArguments, TrainArguments

logger = logging.getLogger(__name__)

def get_model_and_tokenizer(
    model_args: ModelArguments,
    training_args: TrainArguments,
) -> tuple:
    if training_args.use_deepspeed:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )

    if model_args.use_lora:
        logger.warning("Loading model to Lora")

        from peft import LoraConfig, get_peft_model

        LORA_R = 32
        #LORA_DROPOUT = 0.05
        #TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

        config = LoraConfig(
            r=LORA_R,
            # lora_alpha=LORA_ALPHA,
            #target_modules=TARGET_MODULES,
            #lora_dropout=LORA_DROPOUT,
            #bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )

    return model, tokenizer