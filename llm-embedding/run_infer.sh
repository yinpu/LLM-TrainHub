
# # 
python infer.py \
    --model_path  Qwen/Qwen2.5-0.5B \
    --texts hello \
    --model_type EmbeddingModel4Qwen2Adapter \
    --adapter_weights_path saved/Qwen-Emb

# python infer.py \
#     --model_path  saved/Qwen-Emb \
#     --texts hello \
#     --model_type EmbeddingModel4Qwen2