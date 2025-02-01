import logging
import sys
import transformers
from transformers import TrainingArguments, AutoTokenizer
from transformers import HfArgumentParser, set_seed
from transformers.utils.logging import (enable_default_handler, enable_explicit_format, set_verbosity)
from llm_dpo.model import get_model
from llm_dpo.data import get_dataset, PairwiseDataCollatorWithPadding
from llm_dpo.arguments import ModelArguments, DataArguments, FinetuningArguments
from llm_dpo.trainer import CustomDPOTrainer

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, FinetuningArguments))
    model_args, data_args, training_args, finetuning_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
        + f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    logger.info(f'Training/evaluation parameters {training_args}')
    logger.info(f'Finetuning arguments: {finetuning_args}')
    logger.info(f'Model args: {model_args}')
    logger.info(f'Data args: {data_args}')

    set_seed(training_args.seed)

    # 加载模型
    policy_model = get_model(model_args, is_ref_model=False)
    ref_model = get_model(model_args, is_ref_model=True)
    logger.info("POLICY MODEL PARAM:")
    for name, param in policy_model.named_parameters():
        logger.info(f"{name}\t{param.requires_grad}")

    if ref_model:
        logger.info("REF MODEL PARAM:")
        for name, param in policy_model.named_parameters():
            logger.info(f"{name}\t{param.requires_grad}")

    # 加载数据集
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    dataset = get_dataset(data_args, training_args, tokenizer)
    data_collator = PairwiseDataCollatorWithPadding(tokenizer=tokenizer)

    # 训练
    training_args.remove_unused_columns = False
    trainer = CustomDPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        finetuning_args=finetuning_args,
    )

    train_result = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

if __name__ == "__main__":
    main()