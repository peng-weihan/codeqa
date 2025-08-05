
import json
import time
from dotenv import load_dotenv
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.getcwd(), ''))
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"已添加 {project_root} 到Python路径")
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
from moatless_qa.agent.agent import Setup_logging
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

def load_data_from_jsonl(path, max_lines=150):
    data_list = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            try:
                data = json.loads(line)
                data_list.append(data)
            except Exception as e:
                print(f"[跳过] 无效 JSON 行: {e}")
    return data_list


def append_data_to_jsonl(path, data):
    with lock:
        with open(path, 'a', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')

def process_single_question(message: str):
    Setup_logging(f"\nProcessing question: {message}")
    # 以下内容与你现有代码完全一致，只是将 message 替换为函数参数
    completion_model = CompletionModel(
        model="deepseek/deepseek-chat",
        temperature=0.7,
    )
    completion_model.response_format = LLMResponseFormat.TOOLS

    repository = create_repository(repo_path="/data3/pwh/fineract", repo_base_dir=repo_base_dir)

    code_index = CodeIndex.from_persist_dir(persist_dir="/data3/pwh/codeqa/dataset/index_store/fineract", file_repo=repository)

    file_context = FileContext(repo=repository)

    selector = BestFirstSelector()
    value_function = ValueFunction(completion_model=completion_model)

    actions = [
        FindClass(completion_model=completion_model, code_index=code_index, repository=repository),
        FindFunction(completion_model=completion_model, code_index=code_index, repository=repository),
        FindCodeSnippet(completion_model=completion_model, code_index=code_index, repository=repository),
        FindCalledObject(completion_model=completion_model, code_index=code_index, repository=repository),
        SemanticSearch(completion_model=completion_model, code_index=code_index, repository=repository),
        ViewCode(completion_model=completion_model, repository=repository),
        Finish(),
    ]

    agent = CodeQAAgent.create(repository=repository, completion_model=completion_model,
                               code_index=code_index, preset_actions=actions)
    feedback_generator = GroundTruthFeedbackGenerator(
        completion_model=agent.completion
    )

    search_tree = CodeQASearchTree.create(
        message=message,
        agent=agent,
        file_context=file_context,
        selector=selector,
        value_function=value_function,
        feedback_generator=feedback_generator,
        max_iterations=100,
        max_expansions=3,
        max_depth=25,
        persist_path=persist_path,
    )

    node = search_tree.run_search()
    Setup_logging(f"\nQuestion Answer: {node.observation.message if node else '搜索失败'}")
    return {
        "question": message,
        "answer": node.observation.message if node else "搜索失败"
    }

import concurrent.futures

def run_questions_concurrently(input_path, output_path, max_workers=1):
    data_list = load_data_from_jsonl(input_path)
    durations = []  # 存储每个任务的耗时
    results = []

    def task(data):
        question = data.get("question", "")
        if not question:
            print("[警告] 数据中无 question 字段，跳过。")
            return 
        
        start_time = time.perf_counter()  # 精确计时开始
        try:
            # 只用 question 调用接口，获得新的 answer
            res = process_single_question(question)
            data["answer"] = res.get("answer", "无答案")  # 这里写回新的答案
            append_data_to_jsonl(output_path, data)  # 立即写入
            end_time = time.perf_counter()
            duration = end_time - start_time
            durations.append(duration)

            print(f"✔ [完成] 问题: {question}\n MCTS 答案: {data['answer']}\n")

        except Exception as e:
            end_time = time.perf_counter()
            durations.append(end_time - start_time)  # 即使出错也计入耗时
            print(f"❌ [错误] 处理问题失败: {question}, 错误: {e}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(task, data_list)  # 并发执行，结果写入由task处理

    if durations:
        avg_duration = sum(durations) / len(durations)
        print(f"\n[统计] 任务总数: {len(durations)}, 平均耗时: {avg_duration:.2f} 秒")
    else:
        print("[警告] 没有记录到任何任务耗时。")

if __name__ == "__main__":
    input_jsonl = "/data3/pwh/codeqa/dataset/generated_questions/fineract_questions.jsonl"
    output_jsonl = "/data3/pwh/codeqa/dataset/generated_answers/fineract_answers_2.jsonl"
    start_time = time.time()
    results = run_questions_concurrently(input_jsonl, output_jsonl, max_workers=1)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n✨ 所有问题处理完成，全程总耗时：{total_time:.2f} 秒")