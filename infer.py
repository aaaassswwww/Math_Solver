import json
import torch
from tqdm import tqdm
from modelscope import AutoTokenizer
from transformers import AutoModelForCausalLM
from peft import PeftModel
import re

def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
     
    return response

def extract_answer(text):
    # 匹配 "answer:" 后的所有字符（非贪婪模式，避免跨行匹配）
    match = re.search(r'answer:\s*(.*?)(\n|$)', text, re.IGNORECASE)
    # 如果没有换行符或结尾符，直接匹配到末尾
    if not match:
        match = re.search(r'answer:\s*(.*)', text, re.IGNORECASE)
    return match.group(1).strip() if match else None

test_json_new_path = "test.json"

with open(test_json_new_path, 'r', encoding='utf-8') as file:
    test_data = json.load(file)

tokenizer = AutoTokenizer.from_pretrained("./Qwen/Qwen3-0.6B/", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./Qwen/Qwen3-0.6B/", device_map="auto", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, model_id="./output/Qwen/checkpoint-6925/")

with open("submit.csv", 'w', encoding='utf-8') as file:
    for idx, row in tqdm(enumerate(test_data)):
        instruction = row['instruction']
        input_value = row['question']
        id = row['id']
        
        messages = [
            {"role": "system", "content": f"{instruction}"},
            {"role": "user", "content": f"{input_value}"}
        ]
        response = predict(messages, model, tokenizer)
        response = response.replace('\n', ' ')
        response = extract_answer(response)
        file.write(f"{id},{response}\n")


