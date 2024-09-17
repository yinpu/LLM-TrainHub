import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

def replace_name(row, eos_token):
    text = row['instruction'] + " " + row['output'] + eos_token
    row['text'] = text.replace("{{", "{").replace("}}", "}").format(name="yLLM", author="yinpu")
    return row


# 加载model、tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B-Instruct", device_map="auto", trust_remote_code=True)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

model = get_peft_model(model, peft_config)

# 加载数据集
dataset = load_dataset("json", data_files="data/identity.json")
dataset = dataset.map(lambda x: replace_name(x, eos_token=tokenizer.eos_token))
dataset = dataset.map(lambda samples: tokenizer(samples['text']), batched=True)

# 训练
trainer = Trainer(
    model=model, 
    train_dataset=dataset['train'],
    args=TrainingArguments(
        per_device_train_batch_size=2, 
        gradient_accumulation_steps=2,
        warmup_steps=20, 
        max_steps=100, 
        learning_rate=2e-4, 
        fp16=True,
        logging_steps=1, 
        output_dir='outputs'
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
trainer.train()

# 提问：你是什么模型？ 你是谁开发的？
batch = tokenizer("你好，给我介绍一下你？", return_tensors='pt').to("cuda")

with torch.cuda.amp.autocast():
    output_tokens = model.generate(**batch, max_new_tokens=1000)

print("\n\n")
print(tokenizer.decode(output_tokens[0], skip_special_tokens=True))

# 保存
model.save_pretrained("lora_model")


