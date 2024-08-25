import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


model = AutoPeftModelForCausalLM.from_pretrained("lora_model")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")

model = model.to("cuda")
model.eval()
inputs = tokenizer("请介绍一下你是什么模型?", return_tensors="pt")

outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=500)
print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])