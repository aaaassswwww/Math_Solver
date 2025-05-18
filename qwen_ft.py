import json
import torch
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.huggingface import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, LogitsProcessor, LogitsProcessorList
import swanlab
import re
from collections import Counter

exa = "示例如下\nquestion: 食堂运来105千克的萝卜，运来的青菜是萝卜的3倍，运来青菜多少千克？\nthink: 青菜的重量是萝卜的3倍，所以105千克 × 3 = 315千克。\nanswer:315"
            
def process_func(example):
    """
    将数据集进行预处理
    """
    MAX_LENGTH = 384 
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
            f"<|im_start|>system\n请逐步分析这个数学问题，最后给出答案,{exa}<|im_end|>\n<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant{example['output']}<|im_end|>\n",
        add_special_tokens=False,
    )
    # response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}   

class CoTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        #原始向前传播
        outputs = model(
                    input_ids = inputs['input_ids'],
                    attention_mask = inputs['attention_mask'],
                    labels = inputs['labels']
                )
        # 提取模型生成的logits和labels
        logits = outputs.logits
        labels = inputs["labels"]
        
        # 仅计算答案部分的损失（labels中非-100的部分）
        loss_mask = (labels != -100).float()  # 获取有效loss区域
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_labels = labels[..., 1:].contiguous()
        shifted_loss_mask = loss_mask[..., 1:].contiguous()
        
        # 计算交叉熵损失（仅关注答案部分）
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1))
        loss = (loss * shifted_loss_mask.view(-1)).sum() / shifted_loss_mask.sum()
         # ------------------------- Self-Consistency 逻辑 -------------------------
        # 生成多个候选答案（示例生成3次）
        num_samples = 3  # 可调整生成数量
        consistency_loss = 0.0
        
        # 禁用梯度以加速生成（不影响参数更新）
        with torch.no_grad():
            consistency_score = 0.0
            inputs_id = inputs['input_ids']
            for i in range(inputs_id.size(0)):
                # 切片提取第i行，并保持维度为 [1, 369]
                input_part = inputs_id[i:i+1, :]  # 关键操作
                # 生成多样化的候选答案
                generated_responses = []
                for _ in range(num_samples):
                    # 调整生成参数（temperature、top_p等）
                    response = model.generate(
                        input_ids=input_part,
                        max_length=512,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id
                    )
                    generated_responses.append(response)
                
                # 提取答案并投票
                answers = []
                for resp in generated_responses:
                    text = tokenizer.decode(resp[0], skip_special_tokens=True)
                    ans = extract_answer(text)  # 使用现有的提取函数
                    if ans is not None:
                        answers.append(ans)
                
                # 计算一致性得分（示例：多数答案占比）
                counter = Counter(answers)
                if len(counter) > 0:
                    majority_vote = counter.most_common(1)[0][0]
                    consistency_score += counter[majority_vote] / num_samples
                else:
                    consistency_score += 0.0
        
        # ------------------------- 混合损失计算 -------------------------
        # 定义一致性损失权重（可调整）
        alpha = 0.2  # 控制一致性损失的影响
        consistency_loss = 1.0 - consistency_score / 8 # 一致性越高，损失越小
        
        total_loss = loss + alpha * consistency_loss       
        return (total_loss, outputs) if return_outputs else total_loss

    
def extract_answer(text):
    """
    从文本中提取答案，优先匹配最后一个「answer:」后的内容，若无则匹配最后一个「答案：」后的内容。
    若均未找到，返回 None。
    
    参数:
        text (str): 包含答案的原始文本
        
    返回:
        str: 提取的答案（去除前后空格），若无匹配返回 None
    """
    # 匹配英文格式（支持全角/半角冒号及前后空格）
    en_matches = list(re.finditer(r'answer\s*[：:]', text, re.IGNORECASE))
    if en_matches:
        last_en = en_matches[-1]
        answer = text[last_en.end():].strip()
        return answer if answer else None
    
    # 匹配中文格式（支持全角/半角冒号及前后空格）
    zh_matches = list(re.finditer(r'答案\s*[：:]', text))
    if zh_matches:
        last_zh = zh_matches[-1]
        answer = text[last_zh.end():].strip()
        return answer if answer else None
    
    return None

model_dir = snapshot_download("Qwen/Qwen3-0.6B", cache_dir="./", revision="master")

# Transformers加载模型权重

tokenizer = AutoTokenizer.from_pretrained("./Qwen/Qwen3-0.6B/", use_fast=False, trust_remote_code=True, padding_side="left")
model = AutoModelForCausalLM.from_pretrained("./Qwen/Qwen3-0.6B/", device_map="auto", torch_dtype=torch.bfloat16)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

train_json_new_path = "school_math.json"

with open(train_json_new_path, 'r', encoding='utf-8') as file:
    train_data = json.load(file)
train_dataset = []
for d in train_data:
    train_dataset.append(process_func(d))

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
)

model = get_peft_model(model, config)

args = TrainingArguments(
    output_dir="./output/Qwen",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=16,
    logging_steps=10,
    num_train_epochs=5,
    save_steps=1000,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

swanlab_callback = SwanLabCallback(
    project="Qwen2.5-0.5B-fintune",
    experiment_name="Qwen/Qwen3-0.6B",
    config={
        "model": "Qwen/Qwen3-0.6B",
        "dataset": "news",
    }
)

trainer = CoTTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)

trainer.train()

swanlab.finish()
