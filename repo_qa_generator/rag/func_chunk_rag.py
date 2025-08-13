import sys
sys.path.append("/data3/pwh/codeqa")

import ast
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
import requests
import torch
from repo_qa_generator.models.data_models import RepositoryStructure, QAPair
import os
from openai import OpenAI
from repo_qa_generator.core.generator import BaseGenerator
from repo_qa_generator.models.data_models import CodeNode
from format.code_formatting import format_code_from_list
from dotenv import load_dotenv
import faiss
import pickle

load_dotenv()

SYSTEM_PROMPT = "You are a professional code analysis assistant, you are good at explaining code and answering programming questions."

class VoyageEmbeddingModel:
    """Voyage AI embedding model wrapper"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError("Voyage API key is required. Set VOYAGE_API_KEY environment variable or pass api_key parameter.")
        
        self.base_url = "https://api.voyageai.com/v1/embeddings"
        self.model = "voyage-code-2"
    
    def encode(self, texts, batch_size=16, show_progress_bar=True):
        """
        Encode texts using Voyage AI API
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for API calls
            show_progress_bar: Whether to show progress bar
            
        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "input": batch_texts,
            }
            
            try:
                response = requests.post(self.base_url, headers=headers, json=data)
                response.raise_for_status()
                
                result = response.json()
                batch_embeddings = [item["embedding"] for item in result["data"]]
                all_embeddings.extend(batch_embeddings)
                
                if show_progress_bar:
                    print(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                    
            except Exception as e:
                print(f"Error encoding batch {i//batch_size + 1}: {e}")
                raise
        
        return np.array(all_embeddings)
    
    def get_dimension(self) -> int:
        """
        获取embedding维度
        
        Returns:
            embedding维度
        """
        return self.dimension

class RAGFullContextCodeQA(BaseGenerator):
    """在prompt中使用RAG技术，增加相应的代码内容，回答用户的问题"""
    
    def __init__(self, filepath : str, save_path: str, mode = "external"):
        self.save_path = save_path
        self.faiss_index_path = save_path.replace('.json', '_faiss.index')
        self.metadata_path = save_path.replace('.json', '_metadata.pkl')
        super().__init__()
        
        def read_code_nodes_from_json(filepath: str):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        
        self.code_nodes = read_code_nodes_from_json(filepath)
        print(f"len(self.code_nodes): {len(self.code_nodes)}")
        
        # 初始化FAISS索引和元数据
        self.faiss_index = None
        self.code_metadata = []
        
        if mode == "external":
            self.embed_model = VoyageEmbeddingModel()
            self._build_embeddings()
        else:
            self.embed_model = None
        
    def _build_embeddings(self, load_if_exists=True):
        """构建所有代码元素的embeddings，使用FAISS进行存储和检索"""

        # 如果已有FAISS索引文件且允许加载
        if load_if_exists and os.path.exists(self.faiss_index_path) and os.path.exists(self.metadata_path):
            print(f"加载已有 FAISS 索引: {self.faiss_index_path}")
            self.faiss_index = faiss.read_index(self.faiss_index_path)
            with open(self.metadata_path, 'rb') as f:
                self.code_metadata = pickle.load(f)
            print(f"成功加载 FAISS 索引，包含 {self.faiss_index.ntotal} 个向量")
            return
    
        # 准备所有需要编码的文本
        self.elements = []
        self.code_metadata = []  # 存储代码元数据
        
        for code_node in self.code_nodes:
            self.elements.append(code_node["code"][:8192])
            self.code_metadata.append(code_node)

        print(f"len(self.elements): {len(self.elements)}")
        
        # 计算embeddings
        def encode_with_fallback(embed_model, elements, batch_sizes=[16, 12, 8, 4]):
            if not elements:
                return np.array([])

            for batch_size in batch_sizes:
                try:
                    embeddings = embed_model.encode(elements, batch_size=batch_size, show_progress_bar=True)
                    print(f"成功使用 batch_size={batch_size}")
                    return embeddings
                except Exception as e:
                    print(f"batch_size={batch_size} 出错，准备降批次重试，错误信息：{e}")

            raise RuntimeError("所有 batch_size 尝试均失败，无法编码。")
        
        if self.elements:
            embeddings = encode_with_fallback(self.embed_model, self.elements)
        else:
            embeddings = np.array([])

        # 创建FAISS索引
        if len(embeddings) > 0:
            dimension = embeddings.shape[1]
            print(f"Embedding 维度: {dimension}")
            
            # 使用IVFFlat索引，适合中等规模数据集
            if len(embeddings) > 1000:
                # 对于大数据集，使用IVFFlat
                nlist = min(100, len(embeddings) // 10)  # 聚类中心数量
                quantizer = faiss.IndexFlatIP(dimension)
                self.faiss_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
                
                # 训练索引
                print("训练 FAISS 索引...")
                self.faiss_index.train(embeddings.astype('float32'))
            else:
                # 对于小数据集，使用Flat索引
                self.faiss_index = faiss.IndexFlatIP(dimension)
            
            # 添加向量到索引
            print("添加向量到 FAISS 索引...")
            self.faiss_index.add(embeddings.astype('float32'))
            
            # 保存FAISS索引和元数据
            faiss.write_index(self.faiss_index, self.faiss_index_path)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.code_metadata, f)
            
            print(f"已保存 FAISS 索引到: {self.faiss_index_path}")
            print(f"已保存元数据到: {self.metadata_path}")
        else:
            print("没有embedding数据需要保存")
            
    def find_relevant_code(self, query: str, top_k: int = 5) -> List[CodeNode]:
        """
        使用FAISS查找与查询最相关的代码元素
        
        Args:
            query: 查询文本
            top_k: 返回的最相关元素数量
            
        Returns:
            List of CodeNode
        """
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            print("FAISS索引未初始化或为空")
            return []
        
        # 计算查询的embedding
        query_embedding = self.embed_model.encode(query)[0]  # Voyage AI returns list, take first element
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # 使用FAISS进行相似性搜索
        try:
            # 对于IVF索引，需要设置nprobe参数
            if hasattr(self.faiss_index, 'nprobe'):
                self.faiss_index.nprobe = min(10, self.faiss_index.nlist)
            
            # 执行搜索
            similarities, indices = self.faiss_index.search(query_embedding, top_k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx == -1:  # FAISS返回-1表示没有找到结果
                    continue
                    
                element = self.code_metadata[idx]
                similarity = similarities[0][i]
                
                # 读取相关代码
                try:
                    file_path = os.path.join(element["path"], element["file"])
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # 提取相关代码段
                    code_lines = content.split('\n')
                    start_line = element["start_line"]
                    end_line = element["end_line"]
                    code_content = '\n'.join(code_lines[start_line-1:end_line])
                                        
                    # 创建文件节点
                    file_node = {
                        "file_name": element["file"],
                        "upper_path": element["path"],
                        "module": element["type"],
                        "define_class": [],
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
                    print(f"找到相关代码，相似度: {similarity:.4f}, 文件: {element['file']}")
                    
                except Exception as e:
                    print(f"警告：读取文件 {file_path} 时发生错误：{str(e)}")
                    continue
                    
            return results
            
        except Exception as e:
            print(f"FAISS搜索出错: {str(e)}")
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
    
    def add_code_to_index(self, code_nodes: List[Dict]) -> None:
        """
        向FAISS索引中添加新的代码片段
        
        Args:
            code_nodes: 新的代码节点列表
        """
        if self.faiss_index is None:
            print("FAISS索引未初始化")
            return
        
        # 准备新的代码文本
        new_elements = []
        new_metadata = []
        
        for code_node in code_nodes:
            new_elements.append(code_node["code"][:8192])
            new_metadata.append(code_node)
        
        if not new_elements:
            return
        
        # 计算新代码的embeddings
        new_embeddings = self.embed_model.encode(new_elements)
        
        # 添加到FAISS索引
        self.faiss_index.add(new_embeddings.astype('float32'))
        
        # 更新元数据
        self.code_metadata.extend(new_metadata)
        
        # 保存更新后的索引和元数据
        faiss.write_index(self.faiss_index, self.faiss_index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.code_metadata, f)
        
        print(f"成功添加 {len(new_elements)} 个代码片段到FAISS索引")
    
    def get_index_stats(self) -> Dict:
        """
        获取FAISS索引统计信息
        
        Returns:
            包含索引统计信息的字典
        """
        if self.faiss_index is None:
            return {"error": "FAISS索引未初始化"}
        
        stats = {
            "total_vectors": self.faiss_index.ntotal,
            "dimension": self.faiss_index.d,
            "is_trained": self.faiss_index.is_trained if hasattr(self.faiss_index, 'is_trained') else True
        }
        
        if hasattr(self.faiss_index, 'nlist'):
            stats["nlist"] = self.faiss_index.nlist
        
        return stats
    
    def clear_index(self) -> None:
        """
        清空FAISS索引
        """
        if self.faiss_index is not None:
            self.faiss_index.reset()
            self.code_metadata = []
            
            # 删除索引文件
            if os.path.exists(self.faiss_index_path):
                os.remove(self.faiss_index_path)
            if os.path.exists(self.metadata_path):
                os.remove(self.metadata_path)
            
            print("FAISS索引已清空") 

    