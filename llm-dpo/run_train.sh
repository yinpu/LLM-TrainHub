torchrun --nnodes 1 \
    --nproc-per-node 1 \
    train.py \
    --output_dir saved/Qwen-DPO \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --data_path data/preference_data.json \
    --cache_dir_data cache_data \
    --learning_rate 2e-5 \
    --fp16 true \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --save_strategy epoch \
    --save_total_limit 3 \
    --logging_steps 5 \
    --remove_unused_columns False \
    --use_lora True \
    --pref_beta 0.1 \
    --pref_ftx 0.1 \
    --label_smoothing 0.0 \
    --dpo_type dpo \
    --simplo_gamma 0.0 \
    --deepspeed ds_z3_config.json

# 使用simplo算法训练
torchrun --nnodes 1 \
    --nproc-per-node 1 \
    train.py \
    --output_dir saved/Qwen-Simplo \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --data_path data/preference_data.json \
    --cache_dir_data cache_data \
    --learning_rate 2e-5 \
    --fp16 true \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --save_strategy epoch \
    --save_total_limit 3 \
    --logging_steps 5 \
    --remove_unused_columns False \
    --use_lora True \
    --pref_beta 0.1 \
    --pref_ftx 0.1 \
    --label_smoothing 0.0 \
    --dpo_type simplo \
    --simplo_gamma 0.5 \
    --deepspeed ds_z3_config.json
