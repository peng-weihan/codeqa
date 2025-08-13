import sys
sys.path.append("/data3/pwh/codeqa")

import ast
import json
import os
import pickle
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import faiss
from voyageai import Client
from repo_qa_generator.models.data_models import RepositoryStructure, QAPair, CodeNode
from repo_qa_generator.core.generator import BaseGenerator
from format.code_formatting import format_code_from_list
from dotenv import load_dotenv
import tiktoken
import concurrent.futures
import threading
from tqdm import tqdm

load_dotenv()

SYSTEM_PROMPT = "You are a professional code analysis assistant, you are good at explaining code and answering programming questions."

class SlidingWindowsRAG(BaseGenerator):
    """使用sliding windows方法对Python文件进行chunking，使用Voyage API进行embedding，存储到FAISS中的RAG系统"""
    
    def __init__(self, repo_path: str, voyage_api_key: str = None, 
                 chunk_size: int = 1000, overlap: int = 100, 
                 embedding_model: str = "voyage-code-3", 
                 faiss_index_path: str = None):
        """
        初始化SlidingWindowsRAG
        
        Args:
            repo_path: 仓库路径
            voyage_api_key: Voyage API密钥，如果为None则从环境变量获取
            chunk_size: 每个chunk的token数量
            overlap: 相邻chunk之间的重叠token数量
            embedding_model: Voyage embedding模型名称
            faiss_index_path: FAISS索引保存路径
        """
        super().__init__()
        
        self.repo_path = Path(repo_path)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.embedding_model = embedding_model
        
        load_dotenv()
        voyage_api_key = os.getenv("VOYAGE_API_KEY")
        if not voyage_api_key:
            raise ValueError("Voyage API key is required. Set VOYAGE_API_KEY environment variable or pass it as parameter.")
        
        self.voyage_client = Client(voyage_api_key)
        
        # 初始化tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # FAISS索引相关
        self.faiss_index_path = faiss_index_path or f"{self.repo_path.name}_faiss_index"
        self.metadata_path = f"{self.faiss_index_path}_metadata.pkl"
        
        # 存储chunks和embeddings
        self.chunks = []
        self.chunk_metadata = []
        self.faiss_index = None
        self.dimension = None
        
        # 如果索引已存在，则加载
        if self._index_exists():
            self._load_index()
        else:
            self._build_index()
    
    def _index_exists(self) -> bool:
        """检查FAISS索引是否存在"""
        return (os.path.exists(self.faiss_index_path) and 
                os.path.exists(self.metadata_path))
    
    def _load_index(self):
        """加载已存在的FAISS索引"""
        print(f"Loading existing FAISS index from {self.faiss_index_path}")
        
        # 加载FAISS索引
        self.faiss_index = faiss.read_index(self.faiss_index_path)
        self.dimension = self.faiss_index.d
        
        # 加载元数据
        with open(self.metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.chunk_metadata = data['metadata']
        
        print(f"Loaded {len(self.chunks)} chunks with dimension {self.dimension}")
    
    def _save_index(self):
        """保存FAISS索引和元数据"""
        print(f"Saving FAISS index to {self.faiss_index_path}")
        
        # 保存FAISS索引
        faiss.write_index(self.faiss_index, self.faiss_index_path)
        
        # 保存元数据
        data = {
            'chunks': self.chunks,
            'metadata': self.chunk_metadata
        }
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved {len(self.chunks)} chunks")
    
    def _find_python_files(self) -> List[Path]:
        """查找仓库中所有的Python文件"""
        python_files = []
        for root, dirs, files in os.walk(self.repo_path):
            # 跳过一些常见的非代码目录
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.pytest_cache', 'node_modules', 'venv', 'env', '.venv'}]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        
        return python_files
    
    def _tokenize_text(self, text: str) -> List[str]:
        """将文本tokenize"""
        return self.tokenizer.encode(text)
    
    def _detokenize_text(self, tokens: List[int]) -> str:
        """将tokens转换回文本"""
        return self.tokenizer.decode(tokens)
    
    def _create_sliding_windows(self, text: str, file_path: Path, show_progress: bool = False) -> List[Dict[str, Any]]:
        """使用sliding windows方法创建chunks"""
        tokens = self._tokenize_text(text)
        chunks = []
        
        if len(tokens) <= self.chunk_size:
            # 如果文本长度小于chunk_size，直接作为一个chunk
            chunk_text = self._detokenize_text(tokens)
            chunks.append({
                'text': chunk_text,
                'start_token': 0,
                'end_token': len(tokens),
                'file_path': str(file_path),
                'start_line': 1,
                'end_line': len(text.split('\n'))
            })
        else:
            # 使用sliding windows创建chunks
            start = 0
            chunk_count = 0
            
            # 计算总chunk数量用于进度条
            total_chunks = (len(tokens) - self.chunk_size) // (self.chunk_size - self.overlap) + 1
            
            # 创建进度条（如果需要）
            if show_progress and total_chunks > 10:  # 只有chunk数量较多时才显示进度条
                pbar = tqdm(total=total_chunks, desc=f"Creating chunks for {file_path.name}", leave=False)
            else:
                pbar = None
            
            while start < len(tokens):
                end = min(start + self.chunk_size, len(tokens))
                
                # 提取当前chunk的tokens
                chunk_tokens = tokens[start:end]
                chunk_text = self._detokenize_text(chunk_tokens)
                
                # 计算对应的行号范围（近似）
                lines = text.split('\n')
                start_line = 1
                end_line = len(lines)
                
                # 尝试更精确地计算行号
                if start > 0:
                    # 计算start位置对应的行号
                    start_text = self._detokenize_text(tokens[:start])
                    start_line = len(start_text.split('\n'))
                
                if end < len(tokens):
                    # 计算end位置对应的行号
                    end_text = self._detokenize_text(tokens[:end])
                    end_line = len(end_text.split('\n'))
                
                chunks.append({
                    'text': chunk_text,
                    'start_token': start,
                    'end_token': end,
                    'file_path': str(file_path),
                    'start_line': start_line,
                    'end_line': end_line
                })
                
                chunk_count += 1
                if pbar:
                    pbar.update(1)
                
                # 移动到下一个chunk，考虑重叠
                start = end - self.overlap
                if start >= len(tokens):
                    break
            
            if pbar:
                pbar.close()
        
        return chunks
    
    def _get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """批量获取embeddings"""
        try:
            embeddings = self.voyage_client.embed(texts, model=self.embedding_model)
            return np.array(embeddings.embeddings, dtype=np.float32)
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            raise
    
    def _get_embeddings_parallel(self, texts: List[str], max_workers: int = 4, batch_size: int = 50) -> np.ndarray:
        """并行批量获取embeddings"""
        if not texts:
            return np.array([])
        
        # 创建线程锁，确保API调用的线程安全
        lock = threading.Lock()
        
        def process_batch(batch_texts):
            with lock:
                try:
                    embeddings = self.voyage_client.embed(batch_texts, model=self.embedding_model)
                    return np.array(embeddings.embeddings, dtype=np.float32)
                except Exception as e:
                    print(f"Error in batch processing: {e}")
                    # 返回零向量作为fallback
                    return np.zeros((len(batch_texts), 1024), dtype=np.float32)
        
        # 分批处理
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        all_embeddings = []
        
        # 使用线程池并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有批次的任务
            future_to_batch = {executor.submit(process_batch, batch): i for i, batch in enumerate(batches)}
            
            # 使用tqdm显示进度
            with tqdm(total=len(batches), desc="Getting embeddings") as pbar:
                for future in concurrent.futures.as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    try:
                        batch_embeddings = future.result()
                        all_embeddings.append(batch_embeddings)
                    except Exception as e:
                        print(f"Batch {batch_idx} failed: {e}")
                        # 添加零向量作为fallback
                        batch_size_actual = len(batches[batch_idx])
                        all_embeddings.append(np.zeros((batch_size_actual, 1024), dtype=np.float32))
                    finally:
                        pbar.update(1)
        
        if all_embeddings:
            return np.vstack(all_embeddings)
        else:
            return np.array([])
    
    def _build_index(self):
        """构建FAISS索引"""
        print("Building FAISS index...")
        
        python_files = self._find_python_files()
        print(f"Found {len(python_files)} Python files")
        
        all_chunks = []
        all_metadata = []
        
        # 添加文件处理进度条
        with tqdm(python_files, desc="Processing files", unit="file") as pbar:
            for file_path in pbar:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 创建sliding windows chunks
                    # 对于大型文件显示chunk创建进度
                    show_chunk_progress = len(content) > 10000  # 文件大于10KB时显示进度
                    chunks = self._create_sliding_windows(content, file_path, show_progress=show_chunk_progress)
                    all_chunks.extend(chunks)
                    
                    # 添加元数据
                    for chunk in chunks:
                        metadata = {
                            'file_path': chunk['file_path'],
                            'start_line': chunk['start_line'],
                            'end_line': chunk['end_line'],
                            'start_token': chunk['start_token'],
                            'end_token': chunk['end_token']
                        }
                        all_metadata.append(metadata)
                    
                    # 更新进度条描述
                    pbar.set_postfix({
                        'chunks': len(all_chunks),
                        'current_file': file_path.name
                    })
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue
        
        if not all_chunks:
            print("No chunks created!")
            return
        
        # 获取embeddings - 使用并行处理
        print("Getting embeddings with parallel processing...")
        texts = [chunk['text'] for chunk in all_chunks]
        
        # 使用并行处理获取embeddings
        embeddings = self._get_embeddings_parallel(texts, max_workers=4, batch_size=50)
        
        # 创建FAISS索引
        print("Creating FAISS index...")
        self.dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(self.dimension)  # 使用内积相似度
        self.faiss_index.add(embeddings)
        
        # 保存chunks和元数据
        self.chunks = all_chunks
        self.chunk_metadata = all_metadata
        
        # 保存索引
        self._save_index()
        
        # 显示最终统计信息
        print(f"\n✅ Index building completed!")
        print(f"📊 Statistics:")
        print(f"   - Total files processed: {len(python_files)}")
        print(f"   - Total chunks created: {len(all_chunks)}")
        print(f"   - Embedding dimension: {self.dimension}")
        print(f"   - Average chunks per file: {len(all_chunks) / len(python_files):.1f}")
        print(f"   - Index saved to: {self.faiss_index_path}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        搜索最相关的代码chunks
        
        Args:
            query: 查询文本
            top_k: 返回的最相关chunks数量
            
        Returns:
            List of dictionaries containing chunk information and similarity scores
        """
        # 获取查询的embedding
        query_embedding = self._get_embeddings_batch([query])
        
        # 搜索最相似的chunks
        similarities, indices = self.faiss_index.search(query_embedding, top_k)
        
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                metadata = self.chunk_metadata[idx]
                
                result = {
                    'text': chunk['text'],
                    'similarity': float(similarity),
                    'file_path': metadata['file_path'],
                    'start_line': metadata['start_line'],
                    'end_line': metadata['end_line'],
                    'start_token': metadata['start_token'],
                    'end_token': metadata['end_token']
                }
                results.append(result)
        
        return results
    
    def search_batch(self, queries: List[str], top_k: int = 5, max_workers: int = 4) -> List[List[Dict[str, Any]]]:
        """
        批量搜索多个查询，使用并行处理
        
        Args:
            queries: 查询文本列表
            top_k: 每个查询返回的最相关chunks数量
            max_workers: 并行工作线程数
            
        Returns:
            List of search results for each query
        """
        def search_single(query):
            return self.search(query, top_k)
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_query = {executor.submit(search_single, query): i for i, query in enumerate(queries)}
            
            # 初始化结果列表
            results = [None] * len(queries)
            
            for future in concurrent.futures.as_completed(future_to_query):
                query_idx = future_to_query[future]
                try:
                    result = future.result()
                    results[query_idx] = result
                except Exception as e:
                    print(f"Search for query {query_idx} failed: {e}")
                    results[query_idx] = []
        
        return results
    
    def find_relevant_code(self, query: str, top_k: int = 5) -> List[CodeNode]:
        """
        查找与查询最相关的代码元素，返回CodeNode格式
        
        Args:
            query: 查询文本
            top_k: 返回的最相关元素数量
            
        Returns:
            List of CodeNode
        """
        search_results = self.search(query, top_k)
        
        results = []
        for result in search_results:
            try:
                # 读取文件内容
                file_path = result['file_path']
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 提取相关代码段
                lines = content.split('\n')
                start_line = result['start_line']
                end_line = result['end_line']
                
                # 确保行号在有效范围内
                start_line = max(1, min(start_line, len(lines)))
                end_line = max(start_line, min(end_line, len(lines)))
                
                code_content = '\n'.join(lines[start_line-1:end_line])
                
                # 创建文件节点
                file_node = {
                    "file_name": os.path.basename(file_path),
                    "upper_path": os.path.dirname(file_path),
                    "module": "code_chunk",
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
                
            except Exception as e:
                print(f"Warning: Error reading file {file_path}: {str(e)}")
                continue
        
        return results
    
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
        处理问题，使用相关代码生成回答
        
        Args:
            question: 用户的问题
            relevant_code_list: 相关代码列表
            
        Returns:
            生成的回答
        """
        if not relevant_code_list:
            return "No relevant code found. No sufficient information to answer the question."
        
        # 构建提交给LLM的提示
        prompt = self._build_llm_prompt(question, relevant_code_list)
        
        # 调用LLM获取回答
        answer = self._call_llm(system_prompt=SYSTEM_PROMPT, user_prompt=prompt)
        print(f"LLM response answer: {answer}")
        return answer
    
    def _build_llm_prompt(self, question: str, relevant_code_list: List[CodeNode]) -> str:
        """
        构建提交给LLM的提示
        
        Args:
            question: 用户的问题
            relevant_code_list: 相关代码列表
            
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
    
    def get_index_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        return {
            'total_chunks': len(self.chunks),
            'dimension': self.dimension,
            'index_path': self.faiss_index_path,
            'files_processed': len(set(chunk['file_path'] for chunk in self.chunks))
        } 

    