torchrun --nnodes 1 \
    --nproc-per-node 1 \
    train.py \
    --output_dir saved/Qwen-DPO \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --data_path data/alpaca_emoji.json \
    --cache_dir cache_data \
    --learning_rate 2e-5 \
    --bf16 True \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_strategy epoch \
    --save_total_limit 3 \
    --logging_steps 5 \
    --remove_unused_columns False \
    --use_lora True \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --pref_beta 0.1 \
    --pref_ftx 0.1 \
    --label_smoothing 0.0 \
    --dpo_type dpo \
    --simpo_gamma 0.0 \
    --overwrite_output_dir True \
    --evaluation_strategy no \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --optim adamw_torch \
    --gradient_checkpointing False \
    --report_to tensorboard \
    --deepspeed ds_z2_config.json
    

# 使用simpo算法训练
torchrun --nnodes 1 \
    --nproc-per-node 1 \
    train.py \
    --output_dir saved/Qwen-Simpo \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --data_path data/alpaca_emoji.json \
    --cache_dir cache_data \
    --learning_rate 2e-5 \
    --bf16 True \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_strategy epoch \
    --save_total_limit 3 \
    --logging_steps 5 \
    --remove_unused_columns False \
    --use_lora True \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --pref_beta 0.1 \
    --pref_ftx 0.1 \
    --label_smoothing 0.0 \
    --dpo_type simpo \
    --simpo_gamma 0.5 \
    --overwrite_output_dir True \
    --evaluation_strategy no \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --optim adamw_torch \
    --gradient_checkpointing False \
    --report_to tensorboard \
    --deepspeed ds_z2_config.json
