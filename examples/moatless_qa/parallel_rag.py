
import json
import time
from dotenv import load_dotenv
import sys
import os
project_root = os.path.abspath(os.path.join(os.getcwd(), ''))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"已添加 {project_root} 到Python路径")

from codeqa.examples.repo_parser.repo_analyzer_example import analyze_repository
from codeqa.repo_qa_generator.models.data_models import QAPair, ResultPair
from codeqa.repo_qa_generator.rag.code_qa import RecordedRAGCodeQA
from moatless_qa.benchmark.utils import get_moatless_instance
from moatless_qa.completion.completion import CompletionModel
from moatless_qa.completion.completion import LLMResponseFormat
from moatless_qa.benchmark.swebench import create_repository
from moatless_qa.index import CodeIndex
from moatless_qa.file_context import FileContext
from moatless_qa.selector import BestFirstSelector
from moatless_qa.feedback import GroundTruthFeedbackGenerator
from moatless_qa.value_function.base import ValueFunction
from moatless_qa.actions import FindClass, FindFunction, FindCodeSnippet, SemanticSearch, ViewCode, Finish, FindCalledObject
from moatless_qa.agent.code_qa_agent import CodeQAAgent
from moatless_qa.agent.code_qa_prompts import *
from moatless_qa.code_qa.search_tree import CodeQASearchTree
from moatless_qa.completion.completion import (
    LLMResponseFormat,
    CompletionModel,
)
import threading
lock = threading.Lock()  # 文件写入锁，防止并发写乱序
import litellm
load_dotenv()
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
base_dir = os.path.dirname(base_dir)
index_store_dir = os.path.join(base_dir, "dataset/index_store")
repo_base_dir = os.path.join(base_dir, "dataset/repos")
persist_path = os.path.join(base_dir, "dataset/trajectory.json")

# def load_data_from_jsonl(path):
#     data_list = []
#     with open(path, 'r', encoding='utf-8') as f:
#         for line in f:
#             try:
#                 data = json.loads(line)
#                 data_list.append(data)
#             except Exception as e:
#                 print(f"[跳过] 无效 JSON 行: {e}")
#     return data_list

def load_data_from_jsonl(path, max_lines=64):
    data_list = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            try:
                data = json.loads(line)
                qa_pair = QAPair.model_validate(data)
                data_list.append(qa_pair)
            except Exception as e:
                print(f"[跳过] 无效 JSON 行: {e}")
    return data_list

def append_data_to_jsonl(path, data):
    with lock:
        with open(path, 'a', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')

def process_single_question(message: QAPair, rag: RecordedRAGCodeQA):
    """
    处理单个问题，使用RAG模型获取答案
    """
    try:
        rag_qa_pair = rag.process_qa_pair(message)
        result_json = json.loads(rag_qa_pair.answer)
        result_pair = ResultPair.model_validate(result_json)
        result = {
            "question": message.question,
            "rag_answer": result_pair.answer,
            "rag_ground_truth": result_pair.ground_truth,
            "rag_thought": result_pair.thought,
            "rag_score": 0
        }
        return result
    except Exception as e:
        print(f"处理问题失败: {str(e)}")
        return {"error": str(e)}

import concurrent.futures

def run_questions_concurrently(input_path, output_path, max_workers=16):
    repo_path = "/data3/pwh/sympy"
    repository = analyze_repository(repo_path=repo_path, repo_root=repo_path)
    
    rag = RecordedRAGCodeQA(repo_structure=repository.structure,mode="external")
    data_list = load_data_from_jsonl(input_path)
    results = []

    def task(data):
        try:
            # 只用 question 调用接口，获得新的 answer
            res = process_single_question(data, rag)
            data_json = data.model_dump()  # 转换为字典格式
            data_json["rag_answer"] = res.get("rag_answer", "无答案")  # 这里写回新的答案
            append_data_to_jsonl(output_path, data_json)  # 立即写入
            print(f"[完成] 问题: {data}\n新答案: {data_json['rag_answer']}\n")
        except Exception as e:
            print(f"[错误] 处理问题失败: {data}, 错误: {e}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(task, data_list)  # 并发执行，结果写入由task处理

if __name__ == "__main__":
    input_jsonl = "/data3/pwh/codeqa/dataset/generated_answers/sympy_answers.jsonl"
    output_jsonl = "/data3/pwh/codeqa/dataset/generated_answers/sympy_answers_rag.jsonl"
    start_time = time.time()
    results = run_questions_concurrently(input_jsonl, output_jsonl, max_workers=16
)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n✨ 所有问题处理完成，总耗时：{total_time:.2f} 秒")