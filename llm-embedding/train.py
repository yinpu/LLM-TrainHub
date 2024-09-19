import logging
from transformers import HfArgumentParser

from llm_embedding.arguments import TrainArguments, ModelDataarguments
from llm_embedding.data import EmbeddingCollator, TrainDatasetForEmbedding
from llm_embedding.model import EmbeddingModel4Qwen2, EmbeddingModel4Qwen2Adapter
from llm_embedding.trainer import EmbeddingTrainer
from llm_embedding.utils import count_parameters

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelDataarguments, TrainArguments))
    modeldata_args, training_args = parser.parse_args_into_dataclasses()
    modeldata_args: ModelDataarguments
    training_args: TrainArguments

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model Data parameters %s", modeldata_args)

    if modeldata_args.use_adapter:
        model = EmbeddingModel4Qwen2Adapter(
                model_name_or_path=modeldata_args.model_name_or_path
        )
    else:
        model = EmbeddingModel4Qwen2(
                model_name_or_path=modeldata_args.model_name_or_path
        )

    trainable_params, all_param = count_parameters(model)
    param_stats = "trainable params: {:,} || all params: {:,} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
    )
    logger.info(param_stats)
    
    dataset = TrainDatasetForEmbedding(args=modeldata_args)

    trainer = EmbeddingTrainer(
        model=model,
        args=training_args,
        data_collator=EmbeddingCollator(),
        train_dataset=dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()