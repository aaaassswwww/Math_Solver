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
        max_new_tokens=700
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
     
    return response

def extract_answer(text):
    pattern = r'-?(?:\d+\.\d+|\d+|\.\d+)'
    numbers = re.findall(pattern,text)

    if not numbers:
        return None

    last_number_str = numbers[-1]

    #转换为数值类型
    if '.' in last_number_str:
        return float(last_number_str)
    else:
        return int(last_number_str)

def extract_answer_e(text):
    """
    从文本中提取答案，优先匹配最后一个「answer:」后的内容，若无则匹配最后一个「答案：」后的内容。
    若均未找到，返回 None。
    
    参数:
        text (str): 包含答案的原始文本
        
    返回:
        str: 提取的答案（去除前后空格），若无匹配返回 None
    """
    # # 匹配英文格式（支持全角/半角冒号及前后空格）
    # en_matches = list(re.finditer(r'answer\s*[：:]', text, re.IGNORECASE))
    # if en_matches:
    #     last_en = en_matches[-1]
    #     answer = text[last_en.end():].strip()
    #     return answer if answer else None
    
    # 匹配中文格式（支持全角/半角冒号及前后空格）
    zh_matches = list(re.finditer(r'最终答案\s*[：:]', text))
    if zh_matches:
        last_zh = zh_matches[-1]
        answer = text[last_zh.end():].strip()
        return answer if answer else None
    
    return None

def extract_last_answer_number(text: str) -> str | None:
    """
    从文本中提取最后一个「答案：」后的数字（支持整数、小数、分数）
    
    参数:
        text (str): 输入文本
        
    返回:
        str | None: 提取到的数字字符串，未找到则返回 None
        
    示例:
        >>> extract_last_answer_number("答案：28")
        '28'
        >>> extract_last_answer_number("问题：... 答案：3.14 答案：1/4")
        '1/4'
        >>> extract_last_answer_number("答案：无效答案文本")
        None
    """
    # 定位最后一个「答案：」的位置
    last_answer_pos = text.rfind('答案：')
    if last_answer_pos == -1:
        return None
    
    # 提取关键区域文本并去除左侧空白
    target_text = text[last_answer_pos+3:].lstrip()
    
    # 正则表达式模式（优先级：分数 > 小数 > 整数）
    pattern = r"""
        -?                  # 可选负号
        (?:
            \d+/\d+        # 分数格式如 1/4、-3/5
            |              # 或
            \d+\.\d*       # 小数格式如 3.14、5.
            |              # 或
            \.\d+          # 小数格式如 .5
            |              # 或
            \d+            # 整数
        )
    """
    
    # 执行正则匹配
    match = re.search(pattern, target_text, re.VERBOSE)
    return match.group() if match else None

test_json_new_path = "test.json"

with open(test_json_new_path, 'r', encoding='utf-8') as file:
    test_data = json.load(file)

tokenizer = AutoTokenizer.from_pretrained("./Qwen/Qwen2.5-0.5B-Instruct/", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./Qwen/Qwen2.5-0.5B-Instruct/", device_map="auto", torch_dtype=torch.bfloat16)
# model = PeftModel.from_pretrained(model, model_id="./output/Qwen3/checkpoint-935/")

exa = "示例如下\nquestion: 食堂运来105千克的萝卜，运来的青菜是萝卜的3倍，运来青菜多少千克？\nthink: 青菜的重量是萝卜的3倍，所以105千克 × 3 = 315千克。\nanswer:315"
ins = f"这是小学数学1-6年级的校内题目，请给出推理过程，最后给出答案,不需要单位,示例如下\n{exa}"
with open("submit.csv", 'w', encoding='utf-8') as file:
    for idx, row in tqdm(enumerate(test_data)):
        # instruction = row['instruction']
        instruction = ins
        input_value = row['question']
        id = row['id']
        
        messages = [
            {"role": "system", "content": f"{instruction}"},
            {"role": "user", "content": f"{input_value}"}
        ]
        response = predict(messages, model, tokenizer)
        response = response.replace('\n', ' ')
        print(response)
        response = extract_last_answer_number(response)
        print(response)
        file.write(f"{id},{response}\n")


