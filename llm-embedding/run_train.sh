torchrun --nnodes 1 \
    --nproc-per-node 1 \
    train.py \
    --output_dir saved/Qwen2-1.5B-Emb \
    --model_name_or_path /home/yinpu/Projects/llm-tutorial/llm-lora-simple/Qwen/Qwen2-1.5B-Instruct \
    --data_dir data \
    --cache_dir_data cache_data \
    --learning_rate 2e-5 \
    --fp16 true \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --query_max_len 64 \
    --passage_max_len 512 \
    --save_strategy epoch \
    --save_total_limit 3 \
    --temperature 0.05 \
    --logging_steps 5 \
    --remove_unused_columns False \
    --deepspeed ds_z2_config.json