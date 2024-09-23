import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

from torch.utils.data import Dataset

tokenizer = AutoTokenizer.from_pretrained("/home/yinpu/Projects/llm-tutorial/llm-embedding/Qwen/Qwen2.5-0.5B", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained("/home/yinpu/Projects/llm-tutorial/llm-embedding/Qwen/Qwen2.5-0.5B", device_map="auto", trust_remote_code=True)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

model = get_peft_model(model, peft_config)

model_ref = AutoModelForCausalLM.from_pretrained("/home/yinpu/Projects/llm-tutorial/llm-embedding/Qwen/Qwen2.5-0.5B", device_map="auto", trust_remote_code=True)



class dpo_dataset(Dataset):
    def __init__(self,file,tokenizer,max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        # 打开json文件 用transformers
        self.data_list = load_dataset("json",data_files=file)['train']
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self,index):
        # 取出data_list的一条数据  --> {"chosen":xxx,"rejected":xxx,"prompt":xxx} 一条数据是这样的格式
        data = self.data_list[index]

        # 对prompt reject和chosen进行tokenize  判断是否需要截断 保证所有的input_ids都一样 不够长度的直接padding  
        # 适配qwen 的 template  添加eos token
        prompt_input_ids = self.tokenizer.encode('<|im_start|>' + data['prompt'] + '<|im_end|>',add_special_tokens=False)
        chosen_input_ids = self.tokenizer.encode(data['chosen'],add_special_tokens=False)
        rejected_input_ids = self.tokenizer.encode(data['rejected'],add_special_tokens=False)

        prompt_input_ids = prompt_input_ids + [self.tokenizer.pad_token_id]
        # 设置labels
        chosen_labels = [-100] * len(prompt_input_ids) + chosen_input_ids + [self.tokenizer.pad_token_id]
        rejected_labels = [-100] * len(prompt_input_ids) + rejected_input_ids + [self.tokenizer.pad_token_id]
        chosen_input_ids = prompt_input_ids + chosen_input_ids + [self.tokenizer.pad_token_id]
        rejected_input_ids = prompt_input_ids + rejected_input_ids + [self.tokenizer.pad_token_id]

        assert len(chosen_labels) == len(chosen_input_ids)
        assert len(rejected_labels) == len(rejected_input_ids)

        inputs = dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=[1]*len(prompt_input_ids),
            chosen_input_ids=chosen_input_ids,
            chosen_attention_mask=[1]*len(chosen_input_ids),
            chosen_labels=chosen_labels,
            rejected_input_ids=rejected_input_ids,
            rejected_attention_mask=[1]*len(rejected_input_ids),
            rejected_labels=rejected_labels,
        )
        return inputs
    def map(self, func, **kwargs):
        return self
    


train_dataset = dpo_dataset(file = 'data.json', tokenizer = tokenizer, max_seq_length = 50)
from trl import DPOTrainer, DPOConfig

training_args = DPOConfig(
        num_train_epochs = 1,
        per_device_train_batch_size=2,
        learning_rate=3e-4,
        output_dir="./",
        save_total_limit = 1,
        logging_strategy = "steps",
        logging_steps = 50,
        seed = 103,
        fp16 = True,
        warmup_steps = 100,
)

dpo_trainer = DPOTrainer(
        model,
        model_ref,
        beta=0.1, # DPO 的温度超参
        train_dataset=train_dataset, # 上文准备好的数据集
        tokenizer=tokenizer, # 分词器
        args=training_args)

dpo_trainer.train()