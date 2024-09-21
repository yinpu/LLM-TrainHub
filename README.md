# LLM Training and Inference Framework

This repository provides a customizable framework for training and inference of large language models (LLM). It is designed for flexibility, ease of modification, and maintaining a consistent style across projects. The core objective is to enable users to easily adapt the code for their own needs, particularly around custom data loading, model architecture adjustments, and custom loss functions, while supporting common LLM training setups.

## Features

- **Modular Projects**: Each project is independent and follows a unified structure for easy navigation and customization.
- **Pre-built Scripts**: Example data, training, and inference scripts are provided for quick setup and usage.
- **Native Transformers Integration**: Wherever possible, the training framework utilizes Hugging Face's `transformers` library, specifically the `Trainer` class, for a seamless experience.
- **Multi-GPU Support**: Distributed Data Parallel (DDP) and DeepSpeed are supported for multi-GPU training.
- **Compatibility**: Verified to work with the Qwen2 series of models.

## Goals

- **Customizable Workflows**: Allow for easy modifications in data loading, model architecture, and loss functions.
- **Efficient Training**: Supports state-of-the-art techniques for LLM training, including LoRA fine-tuning and embedding generation.
- **Flexible Model Handling**: Provides utilities for custom model saving and loading workflows.

## Current Projects

- [x] **llm-lora-simple**: A simplified single-machine LoRA fine-tuning setup for large language models.
- [x] **llm-embedding**: A project to generate embeddings using LLMs, with support for training on custom datasets.


目标：
- 每个项目相对独立，方便修改，提供示例数据、训练、预测脚本
- 尽可能使用原生transformers的trainer
- 支持ddp、deepspeed
- 在qwen2-1.5b和7b跑通