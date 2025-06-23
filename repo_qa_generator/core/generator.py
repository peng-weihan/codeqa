
import ast
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
import os
from openai import OpenAI
import dotenv
import logging

dotenv.load_dotenv()

class BaseGenerator:
    """在prompt中使用RAG技术，增加相应的代码内容，回答用户的问题"""
    
    def __init__(self, baseurl: str = None, apikey: str = None):
        self.llm_client = OpenAI(
            base_url=baseurl or os.environ.get("OPENAI_URL"),
            api_key=apikey or os.environ.get("OPENAI_API_KEY")
        )

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        调用LLM获取回答
        
        Args:
            system_prompt: 提交给LLM的系统提示文本
            user_prompt: 提交给LLM的用户提示文本
            
        Returns:
            LLM的回答
        """
        if not self.llm_client:
            return "无法调用LLM，请确保LLM客户端已正确配置。"
        
        try:
            # 调用LLM
            response = self.llm_client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                stream=False,
                temperature=0.2
            )
            # 提取回答
            return response.choices[0].message.content.strip("```json").strip("```")
        except Exception as e:
            return f"调用LLM时发生错误: {str(e)}"
    
