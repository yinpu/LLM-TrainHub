import argparse
import torch
from typing import List
from llm_embedding.model import EmbeddingModel4Qwen2, EmbeddingModel4Qwen2Adapter

def parse_args():
    parser = argparse.ArgumentParser(description="Text Embedding Generation Script")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path or name of the pretrained model, e.g., 'Qwen-2'.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use, 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--texts",
        type=str,
        nargs='+',
        required=True,
        help="Texts to generate embeddings for. You can input multiple texts.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="EmbeddingModel4Qwen2",
        choices=["EmbeddingModel4Qwen2", "EmbeddingModel4Qwen2Adapter"],
        help="Type of model to use:  EmbeddingModel4Qwen2 or  EmbeddingModel4Qwen2Adapter.",
    )
    parser.add_argument(
        "--adapter_weights_path",
        type=str,
        default=None,
        help="Path to the adapter weights. Required if --model_type is 'EmbeddingModel4Qwen2Adapter'.",
    )
    parser.add_argument(
        "--adapter_output_size",
        type=int,
        default=256,
        help="Output size for the adapter model. Only applicable if --model_type is 'EmbeddingModel4Qwen2Adapter'.",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Switching to CPU.")
        device = "cpu"
    else:
        device = args.device

    if args.model_type == "EmbeddingModel4Qwen2":
        model = EmbeddingModel4Qwen2(
            model_name_or_path=args.model_path,
            device=device
        )
    elif args.model_type == "adapter":
        if args.adapter_weights_path is None:
            raise ValueError("Adapter weights path must be provided when using the adapter model.")
        model = EmbeddingModel4Qwen2Adapter(
            model_name_or_path=args.model_path,
            device=device,
            load_weights_path=args.adapter_weights_path
        )

    model.eval()
    embeddings = model(args.texts)
    print(embeddings)

if __name__ == "__main__":
    main()