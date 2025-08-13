
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


from codeqa.repo_qa_generator.rag.func_chunk_rag import RAGFullContextCodeQA
from examples.repo_parser.repo_analyzer_example import analyze_repository
from repo_qa_generator.models.data_models import QAPair, Repository, ResultPair, load_repository_from_json
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

def load_data_from_jsonl(path, max_lines=50):
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

def process_single_question(message: QAPair, rag: RAGFullContextCodeQA):
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

def run_questions_concurrently(input_path: str, output_path: str, code_nodes_json_path: str, embeddings_save_path: str, max_workers=32):
    # repo_path = "/data3/pwh/sympy"
    # repository = analyze_repository(repo_path=repo_path, repo_root=repo_path)
 
    rag = RAGFullContextCodeQA(filepath=code_nodes_json_path, save_path=embeddings_save_path)
    data_list = load_data_from_jsonl(input_path)
    print(f"len(data_list): {len(data_list)}")

    def task(data):
        try:
            data_json = data.model_dump()  # 转换为字典格式
            # 只用 question 调用接口，获得新的 answer
            res = process_single_question(data, rag)
            if res.get("rag_answer") is not None and res.get("rag_answer") != "null":
                data_json["rag_answer"] = res.get("rag_answer")  # 这里写回新的答案
                append_data_to_jsonl(output_path, data_json)  # 立即写入
                print(f"[完成] 问题: {data}\n新答案: {data_json['rag_answer']}\n")
        except Exception as e:
            print(f"[错误] 处理问题失败: {data}, 错误: {e}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(task, data_list)  # 并发执行，结果写入由task处理

if __name__ == "__main__":
    queue = [
    #   {
    #       "input_jsonl": "/data3/pwh/questions/flask.jsonl",
    #       "output_jsonl": "/data3/pwh/answers/rag_func/flask_rag.jsonl",
    #       "full_json_path": "/data3/pwh/repo_analysis/full_code_for_embedding/flask/flask_code_nodes.json",
    #       "save_path": "/data3/pwh/voyage_faiss/flask_embeddings.json"
    #   },
      {
          "input_jsonl": "/data3/pwh/questions/astropy.jsonl",
          "output_jsonl": "/data3/pwh/answers/rag_func/astropy_rag.jsonl",
          "full_json_path": "/data3/pwh/repo_analysis/full_code_for_embedding/astropy/astropy_code_nodes.json",
          "save_path": "/data3/pwh/voyage_faiss/astropy_embeddings.json"
      },
      {
          "input_jsonl": "/data3/pwh/questions/matplotlib.jsonl",
          "output_jsonl": "/data3/pwh/answers/rag_func/matplotlib_rag.jsonl",
          "full_json_path": "/data3/pwh/repo_analysis/full_code_for_embedding/matplotlib/matplotlib_code_nodes.json",
          "save_path": "/data3/pwh/voyage_faiss/matplotlib_embeddings.json"
      },
      {
          "input_jsonl": "/data3/pwh/questions/pylint.jsonl",
          "output_jsonl": "/data3/pwh/answers/rag_func/pylint_rag.jsonl",    
          "full_json_path": "/data3/pwh/repo_analysis/full_code_for_embedding/pylint/pylint_code_nodes.json",
          "save_path": "/data3/pwh/voyage_faiss/pylint_embeddings.json"
      },
      {
          "input_jsonl": "/data3/pwh/questions/pytest.jsonl",
          "output_jsonl": "/data3/pwh/answers/rag_func/pytest_rag.jsonl",
          "full_json_path": "/data3/pwh/repo_analysis/full_code_for_embedding/pytest/pytest_code_nodes.json",
          "save_path": "/data3/pwh/voyage_faiss/pytest_embeddings.json"
      },
      {
          "input_jsonl": "/data3/pwh/questions/requests.jsonl",
          "output_jsonl": "/data3/pwh/answers/rag_func/requests_rag.jsonl",
          "full_json_path": "/data3/pwh/repo_analysis/full_code_for_embedding/requests/requests_code_nodes.json",
          "save_path": "/data3/pwh/voyage_faiss/requests_embeddings.json"
      },
      {
          "input_jsonl": "/data3/pwh/questions/scikit-learn.jsonl",
          "output_jsonl": "/data3/pwh/answers/rag_func/scikit-learn_rag.jsonl",
          "full_json_path": "/data3/pwh/repo_analysis/full_code_for_embedding/scikit-learn/scikit-learn_code_nodes.json",
          "save_path": "/data3/pwh/voyage_faiss/scikit-learn_embeddings.json"
      },
      {
          "input_jsonl": "/data3/pwh/questions/sphinx.jsonl",
          "output_jsonl": "/data3/pwh/answers/rag_func/sphinx_rag.jsonl",
          "full_json_path": "/data3/pwh/repo_analysis/full_code_for_embedding/sphinx/sphinx_code_nodes.json",
          "save_path": "/data3/pwh/voyage_faiss/sphinx_embeddings.json"
      },
      {
          "input_jsonl": "/data3/pwh/questions/sqlfluff.jsonl",
          "output_jsonl": "/data3/pwh/answers/rag_func/sqlfluff_rag.jsonl",
          "full_json_path": "/data3/pwh/repo_analysis/full_code_for_embedding/sqlfluff/sqlfluff_code_nodes.json",
          "save_path": "/data3/pwh/voyage_faiss/sqlfluff_embeddings.json"
      },
      {
          "input_jsonl": "/data3/pwh/questions/sympy.jsonl",
          "output_jsonl": "/data3/pwh/answers/rag_func/sympy_rag.jsonl",
          "full_json_path": "/data3/pwh/repo_analysis/full_code_for_embedding/sympy/sympy_code_nodes.json",
          "save_path": "/data3/pwh/voyage_faiss/sympy_embeddings.json"
      },
      {
          "input_jsonl": "/data3/pwh/questions/xarray.jsonl",
          "output_jsonl": "/data3/pwh/answers/rag_func/xarray_rag.jsonl",
          "full_json_path": "/data3/pwh/repo_analysis/full_code_for_embedding/xarray/xarray_code_nodes.json",
          "save_path": "/data3/pwh/voyage_faiss/xarray_embeddings.json"
      },
      {
          "input_jsonl": "/data3/pwh/questions/django.jsonl",
          "output_jsonl": "/data3/pwh/answers/rag_func/django_rag.jsonl",
          "full_json_path": "/data3/pwh/repo_analysis/full_code_for_embedding/django/django_code_nodes.json",
          "save_path": "/data3/pwh/voyage_faiss/django_embeddings.json"
      }
    ]
    for item in queue:
        try:
            start_time = time.time()
            results = run_questions_concurrently(
                input_path=item["input_jsonl"], 
                output_path=item["output_jsonl"], 
                code_nodes_json_path=item["full_json_path"],
                embeddings_save_path=item["save_path"],
                max_workers=32
            )
            end_time = time.time()
            total_time = end_time - start_time
            print(f"\n✨ 仓库 {item['input_jsonl']} 处理完成，总耗时：{total_time:.2f} 秒")
        except Exception as e:
            print(f"⚠️ 处理仓库 {item['input_jsonl']} 时发生错误，跳过该仓库，错误信息：{e}")

    # input_jsonl = "/data3/pwh/questions/flask.jsonl"
    # output_jsonl = "/data3/pwh/answers/rag_doc/flask_rag.jsonl"
    # start_time = time.time()
    # results = run_questions_concurrently(input_jsonl, output_jsonl, max_workers=32)
    # end_time = time.time()
    # total_time = end_time - start_time
    # print(f"\n✨ 所有问题处理完成，总耗时：{total_time:.2f} 秒")