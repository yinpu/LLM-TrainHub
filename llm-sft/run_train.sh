torchrun --nnodes 1 \
    --nproc-per-node 1 \
    train.py \
    --output_dir saved/Qwen-SFT \
    --model_name_or_path /home/yinpu/Projects/llm-tutorial/llm-embedding/Qwen/Qwen2.5-0.5B \
    --data_path data \
    --cache_dir_data cache_data \
    --learning_rate 2e-5 \
    --fp16 true \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --save_strategy epoch \
    --save_total_limit 3 \
    --logging_steps 5 \
    --use_deepspeed True \
    --use_lora True \
    --deepspeed ds_z3_config.json
