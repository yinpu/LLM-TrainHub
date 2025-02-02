from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from argparse import ArgumentParser
from peft import PeftModel

def infer(model, tokenizer, prompt):
    set_seed(42)
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=500
    )
    generated_ids = [
        output_ids[len(model_inputs.input_ids[0]):] for output_ids in generated_ids
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return response


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="/models/Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--peft_model_path", type=str, default=None)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
                args.base_model_path,
                torch_dtype="auto",
    ).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)

    if args.peft_model_path:
        model = PeftModel.from_pretrained(model, args.peft_model_path).eval()
        print("PEFT model loaded.")
    else:
        print("Base model loaded.")

    # List of example prompts
    prompts = [
        "请告诉我定期锻炼的好处。",
        "日本的首都是什么？",
        "如何提高在家工作时的生产力？",
        "用简单的术语解释量子计算。",
        "什么是可持续生活方式的最佳实践？",
        "能否总结一下乔治·奥威尔的小说《1984》的情节？",
        "告诉我一些关于太空探索的有趣事实。",
        "如何有效地为工作面试做准备？",
        "给我一些快速学习新语言的技巧。",
        "今年人工智能领域的主要趋势是什么？"
    ]

    # Display the responses for all prompts
    for idx, prompt in enumerate(prompts):
        print(f"Prompt {idx+1}: {prompt}")
        response = infer(model, tokenizer, prompt)
        print(f"Response {idx+1}: {response[0]}\n" + "-"*80)
