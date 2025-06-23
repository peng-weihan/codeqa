import json
import os
import csv
import pandas as pd

class DataCreator:
    def __init__(self, mode: str, question_store_path: str, repo_name: str):
        """
        初始化数据创建器
        
        参数:
            mode: 输出格式模式，如'claude'、'openai'、'llama'等
            question_store_path: 问答对存储路径
            repo_name: 代码仓库名称
        """
        self.mode = mode
        self.question_store_path = question_store_path
        self.repo_name = repo_name
        
        # 读取问答对数据
        self.qa_pairs_path = os.path.join(question_store_path, "updated_questions.json")
        if not os.path.exists(self.qa_pairs_path):
            self.qa_pairs_path = os.path.join(question_store_path, "questions.json")
        
        with open(self.qa_pairs_path, "r") as f:
            self.qa_pairs = json.load(f)
            
        # 配置不同仓库的system instruction
        self.repo_configs = {
            "django": {
                "system": "你是Django框架的专家，精通Web开发、ORM和MVC架构。请用中文回答关于Django源码的问题。",
                "repo_description": "Django是一个高级Python Web框架，鼓励快速开发和简洁实用的设计。"
            },
            "pytorch": {
                "system": "你是PyTorch深度学习框架的专家，精通神经网络和机器学习算法。请用中文回答关于PyTorch源码的问题。",
                "repo_description": "PyTorch是一个开源的机器学习库，提供灵活的神经网络构建工具。"
            },
            # 可添加更多仓库配置
        }
        
        # 获取当前仓库配置
        self.repo_config = self.repo_configs.get(self.repo_name, {
            "system": f"你是{self.repo_name}代码库的专家，精通相关技术和架构。请用中文回答问题。",
            "repo_description": f"{self.repo_name}是一个代码仓库。"
        })
        
        # 输出格式配置
        self.mode_configs = {
            "claude": {
                "format": "jsonl",
                "message_format": True
            },
            "openai": {
                "format": "jsonl", 
                "chat_format": True
            },
            "llama": {
                "format": "jsonl",
                "alpaca_format": True
            },
            "general": {
                "format": "jsonl",
                "include_repo_description": True
            },
            "csv": {
                "format": "csv"
            }
        }
        
        # 获取当前模式配置
        self.mode_config = self.mode_configs.get(self.mode, self.mode_configs["general"])
        
    def create_training_data(self):
        """生成训练数据文件"""
        output_format = self.mode_config.get("format", "jsonl")
        output_file = os.path.join(self.question_store_path, f"finetune_{self.repo_name}_{self.mode}.{output_format}")
        
        if output_format == "csv":
            self._create_csv_data(output_file)
        else:
            self._create_jsonl_data(output_file)
            
        print(f"已将训练数据保存为{self.mode}格式: {output_file}")
        return output_file
        
    def _create_jsonl_data(self, output_file):
        """创建JSONL格式的训练数据"""
        with open(output_file, "w", encoding="utf-8") as f:
            for qa in self.qa_pairs:
                # 解析qa对象，根据格式可能需要调整
                question = qa.get("question", "")
                answer = qa.get("answer", "")
                
                if not question or not answer:
                    continue
                    
                if self.mode_config.get("message_format"):  # Claude 格式
                    data = {
                        "messages": [
                            {"role": "system", "content": self.repo_config["system"]},
                            {"role": "human", "content": question},
                            {"role": "assistant", "content": answer}
                        ]
                    }
                elif self.mode_config.get("chat_format"):  # OpenAI 格式
                    data = {
                        "messages": [
                            {"role": "system", "content": self.repo_config["system"]},
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer}
                        ]
                    }
                elif self.mode_config.get("alpaca_format"):  # Llama/Alpaca 格式
                    data = {
                        "system": self.repo_config["system"],
                        "instruction": question,
                        "output": answer
                    }
                    # 如果需要包含仓库描述
                    if self.mode_config.get("include_repo_description"):
                        data["context"] = self.repo_config["repo_description"]
                else:  # 通用格式
                    data = {
                        "system_instruction": self.repo_config["system"],
                        "question": question,
                        "answer": answer
                    }
                    # 如果需要包含仓库描述
                    if self.mode_config.get("include_repo_description"):
                        data["context"] = self.repo_config["repo_description"]
                        
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    def _create_csv_data(self, output_file):
        """创建CSV格式的训练数据"""
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # 添加列标题
            writer.writerow(["system", "question", "answer", "context"])
            
            for qa in self.qa_pairs:
                question = qa.get("question", "")
                answer = qa.get("answer", "")
                
                if not question or not answer:
                    continue
                    
                context = self.repo_config.get("repo_description", "") if self.mode_config.get("include_repo_description") else ""
                writer.writerow([self.repo_config["system"], question, answer, context])