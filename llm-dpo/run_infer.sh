#!/bin/bash


BASE_MODEL_PATH="/models/Qwen/Qwen2.5-7B-Instruct"
PEFT_MODEL_PATH=""


python3 infer.py --base_model_path "$BASE_MODEL_PATH" --peft_model_path "$PEFT_MODEL_PATH"
