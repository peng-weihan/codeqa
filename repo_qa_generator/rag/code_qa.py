import ast
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from repo_qa_generator.models.data_models import RepositoryStructure, QAPair
import os
from openai import OpenAI
from repo_qa_generator.core.generator import BaseGenerator
from repo_qa_generator.models.data_models import CodeNode
from format.code_formatting import format_code_from_list
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = "You are a professional code analysis assistant, you are good at explaining code and answering programming questions."
class RecordedRAGCodeQA(BaseGenerator):
    """在prompt中使用RAG技术，增加相应的代码内容，回答用户的问题"""
    
    def __init__(self, repo_structure: RepositoryStructure = None, mode = "internel"):
        super().__init__()
        
        self.repo_structure = repo_structure
        if mode == "external":
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            self._build_embeddings()
        else:
            self.embed_model = None
        
        
    def _build_embeddings(self):
        """构建所有代码元素的embeddings"""
        # 准备所有需要编码的文本
        self.elements = []
        self.element_types = []  # 记录每个元素的类型
        
        # 处理类
        for cls in self.repo_structure.classes:
            text = f"class {cls.name}: {cls.docstring}"
            self.elements.append(text)
            self.element_types.append(('class', cls))
            
        # 处理函数
        for func in self.repo_structure.functions:
            prefix = "method" if func.is_method else "函数"
            class_prefix = f"{func.class_name}." if func.is_method else ""
            text = f"{prefix} {class_prefix}{func.name}: {func.docstring}"
            self.elements.append(text)
            self.element_types.append(('function', func))
            
        # 处理属性
        for attr in self.repo_structure.attributes:
            text = f"attribute {attr.class_name}.{attr.name}"
            self.elements.append(text)
            self.element_types.append(('attribute', attr))
            
        # 计算embeddings
        if self.elements:
            self.embeddings = self.embed_model.encode(self.elements)
        else:
            self.embeddings = np.array([])
            
    def find_relevant_code(self, query: str, top_k: int = 5) -> List[CodeNode]:
        """
        查找与查询最相关的代码元素
        
        Args:
            query: 查询文本
            top_k: 返回的最相关元素数量
            
        Returns:
            List of CodeNode
        """
        # 计算查询的embedding
        query_embedding = self.embed_model.encode(query)
        
        # 计算相似度
        if len(self.embeddings) > 0:
            similarities = np.dot(self.embeddings, query_embedding) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # 获取最相关的元素
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                element_type, element = self.element_types[idx]
                
                # 读取相关代码
                try:
                    file_path = os.path.join(element.relative_code.belongs_to.upper_path, element.relative_code.belongs_to.file_name)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # 提取相关代码段
                    code_lines = content.split('\n')
                    start_line = 0
                    end_line = 0
                    code_content = ""
                    
                    if element_type == 'class':
                        start_line = element.relative_code.start_line
                        end_line = element.relative_code.end_line
                        code_content = '\n'.join(code_lines[start_line-1:end_line])
                    elif element_type == 'function':
                        start_line = element.relative_code.start_line
                        end_line = element.relative_code.end_line
                        code_content = '\n'.join(code_lines[start_line-1:end_line])
                    else:  # attribute
                        start_line = element.relative_code.defined_line
                        end_line = element.relative_code.defined_line
                        code_content = code_lines[start_line-1]
                    
                    # 创建文件节点
                    file_node = {
                        "file_name": element.relative_code.belongs_to.file_name,
                        "upper_path": element.relative_code.belongs_to.upper_path,
                        "module": element_type,
                        "define_class": [element.class_name] if hasattr(element, 'class_name') else [],
                        "imports": []
                    }
                    
                    # 创建代码节点
                    code_node = CodeNode(
                        start_line=start_line,
                        end_line=end_line,
                        belongs_to=file_node,
                        relative_function=[],
                        code=code_content
                    )
                    
                    results.append(code_node)
                except Exception as e:
                    print(f"警告：读取文件 {file_path} 时发生错误：{str(e)}")
                    continue
                    
            return results
        return []
    
    def make_question_prompt(self, question: str) -> str:
        """
        组装传给LLM代码的问题
        
        Args:
            question: 用户的问题
            
        Returns:
            回答内容
        """
        # 找到相关的代码元素
        relevant_code = self.find_relevant_code(question)
        
        if not relevant_code:
            return "抱歉，没有找到相关的代码信息。"
            
        # 构建回答
        answer = "根据代码分析，以下是相关信息：\n\n"
        
        for code_content, similarity, element_type in relevant_code:
            answer += f"相关度: {similarity:.2f}\n"
            answer += f"类型: {element_type}\n"
            answer += "相关代码:\n```python\n"
            answer += code_content
            answer += "\n```\n\n"
            
        return answer
    
    def process_qa_pair(self, qa_pair: QAPair) -> QAPair:
        """
        处理QA Pair，找到相关代码并使用LLM回答问题
        
        Args:
            qa_pair: 包含问题和回答的QA Pair对象
        """
        if not qa_pair.relative_code_list:
            relevant_code_list = self.find_relevant_code(qa_pair.question)
        else:
            relevant_code_list = qa_pair.relative_code_list
        answer = self.process_answer(qa_pair.question, relevant_code_list)
        qa_pair.answer = answer
        return qa_pair
    
    def process_answer(self, question: str, relevant_code_list: List[CodeNode]) -> str:
        """
        处理QA Pair，找到相关代码并使用LLM回答问题
        
        Args:
            qa_pair: 包含问题和回答的QA Pair对象
            
        Returns:
            更新后的QA Pair对象，其中包含新的答案
        """

        if not relevant_code_list:
            # 如果没有找到相关代码，则原样返回
            # relevant_code_list = self.find_relevant_code(qa_pair.question)
            return "No relevant code found. No sufficient information to answer the question."
        # 2. 构建提交给LLM的提示
        prompt = self._build_llm_prompt(question, relevant_code_list)
        
        # 3. 调用LLM获取回答
        answer = self._call_llm(system_prompt=SYSTEM_PROMPT, user_prompt=prompt)
        return answer

    def _build_llm_prompt(self, question: str, relevant_code_list: List[CodeNode]) -> str:
        """
        构建提交给LLM的提示
        
        Args:
            question: 用户的问题
            relevant_code_list: 相关代码列表，每个元素是(代码内容, 相似度, 元素类型)的元组
            
        Returns:
            完整的提示文本
        """
        prompt = "You are a professional code analysis assistant. Please answer the question based on the following code snippets.\n\n"
        prompt += f"Question: {question}\n\n"
        prompt += format_code_from_list(relevant_code_list)
        
        prompt += "Please answer the question based on the above code snippets. Explain key concepts and code logic, ensuring the answer is accurate, comprehensive, and easy to understand."
        return prompt
    
    def process_qa_pairs(self, qa_pairs: List[QAPair]) -> List[QAPair]:
        """
        批量处理多个QA Pair
        
        Args:
            qa_pairs: QA Pair列表
            
        Returns:
            更新后的QA Pair列表
        """
        updated_pairs = []
        for qa_pair in qa_pairs:
            updated_pair = self.process_qa_pair(qa_pair)
            updated_pairs.append(updated_pair)
        return updated_pairs 
