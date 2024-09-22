import logging
from transformers import HfArgumentParser, Trainer

from llm_sft.arguments import TrainArguments, DataArguments, ModelArguments
from llm_sft.data import get_dataset, get_data_collator
from llm_sft.model import get_model_and_tokenizer
from llm_sft.utils import count_parameters

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((TrainArguments, DataArguments, ModelArguments))
    train_args, data_args, model_args = parser.parse_args_into_dataclasses()
    train_args: TrainArguments
    data_args: DataArguments
    model_args: ModelArguments

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if train_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        train_args.local_rank,
        train_args.device,
        train_args.n_gpu,
        bool(train_args.local_rank != -1),
        train_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", train_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    model, tokenizer = get_model_and_tokenizer(model_args, train_args)
    dataset = get_dataset(data_args, train_args, tokenizer)
    data_collator = get_data_collator(tokenizer, model)

    trainable_params, all_param = count_parameters(model)
    param_stats = "trainable params: {:,} || all params: {:,} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
    )
    logger.info(param_stats)
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=train_args.output_dir)

if __name__ == "__main__":
    main()