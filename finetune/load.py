import os
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from modelscope import snapshot_download

# 设置常量和路径
OUTPUT_DIR = "./models/qwen-coder-finetuned"
DATASET_PATH = "./question_store/finetune_django_claude.jsonl"
MAX_LENGTH = 2048

# 设置随机种子以确保可重现性
torch.manual_seed(42)
np.random.seed(42)

# 1. 加载模型和tokenizer
print("正在加载模型...")
model_dir = snapshot_download("Qwen/Qwen2.5-Coder-7B-Instruct", cache_dir="./cache/autodl-tmp", revision="master")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

# 使用Unsloth加载和优化模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name_or_path=model_dir,
    max_seq_length=MAX_LENGTH,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)

# 2. 配置LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# 3. 加载数据集
print("正在加载数据集...")
dataset = load_dataset("json", data_files=DATASET_PATH)["train"]

# 处理为Unsloth期望的格式
def format_instruction(example):
    messages = example["messages"]
    system_prompt = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
    user_msg = next((msg["content"] for msg in messages if msg["role"] == "human"), "")
    assistant_msg = next((msg["content"] for msg in messages if msg["role"] == "assistant"), "")
    
    # 构建完整指令
    if system_prompt:
        instruction = f"{system_prompt}\n\n用户: {user_msg}"
    else:
        instruction = f"用户: {user_msg}"
    
    return {
        "instruction": instruction,
        "output": assistant_msg
    }

formatted_dataset = dataset.map(format_instruction)
formatted_dataset = formatted_dataset.select_columns(["instruction", "output"])

# 4. 设置训练参数并开始训练
trainer = FastLanguageModel.get_trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset,
    dataset_text_field=None,  # 使用指令格式而不是text_field
    max_seq_length=MAX_LENGTH,
    batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    optimizer="adamw_torch",
    lr_scheduler_type="cosine",
    output_dir=OUTPUT_DIR,
    mixed_precision="bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "no",
    save_strategy="epoch",
    logging_steps=10,
    save_total_limit=2,
)

# 5. 开始训练
print("开始训练...")
trainer.train()

# 6. 保存模型
print("保存模型...")
FastLanguageModel.save_pretrained(model, tokenizer, OUTPUT_DIR)

print(f"微调完成! 模型已保存到: {OUTPUT_DIR}")






