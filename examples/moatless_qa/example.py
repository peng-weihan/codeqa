
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
import logging

# logging.basicConfig(
#     filename='/data3/pwh/codeqa/dataset/log/exg.log',  # 日志会写入此文件
#     level=logging.INFO,  # 会记录 INFO 及以上级别
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )

import litellm
# 加载环境变量
load_dotenv()
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
base_dir = os.path.dirname(base_dir)
index_store_dir = os.path.join(base_dir, "dataset/index_store")
repo_base_dir = os.path.join(base_dir, "dataset/repos")
persist_path = os.path.join(base_dir, "dataset/trajectory.json")
instance_id = "sphinx-doc__sphinx-8551"
instance_path = os.path.join(base_dir, f"dataset/trajectory/{instance_id}/")
instance = get_moatless_instance(instance_id)

completion_model = CompletionModel(
    model="deepseek/deepseek-chat",
    temperature=0.7,
)
completion_model.response_format = LLMResponseFormat.TOOLS

# repository = create_repository(instance, repo_base_dir=repo_base_dir)
repository = create_repository(repo_path="/data3/pwh/fineract", repo_base_dir=repo_base_dir)

# code_index = CodeIndex.from_index_name(
#     instance["instance_id"], index_store_dir=index_store_dir, file_repo=repository
# )
code_index = CodeIndex.from_repository(repo_path="/data3/pwh/fineract", index_store_dir=index_store_dir, file_repo=repository)

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

agent = CodeQAAgent.create(repository=repository, completion_model=completion_model,code_index=code_index,preset_actions=actions)
agent.actions = actions
feedback_generator = GroundTruthFeedbackGenerator(completion_model=agent.completion, instance_dir=instance_path)
search_tree = CodeQASearchTree.create(
    # message="Where can i find the function `verify_needs_extensions` in the code base?",
    # message="Where can I find the `create_repository` function in the codebase?",
    # message="Where can I find the implementation of 'DataDeclarationDocumenter' in the codebase?",
    # message=" Sphinx 项目中哪些模块负责解析 reStructuredText 文档？它们之间是如何协作的？",

    message="Fineract 如何实现 Command-Query 分离（CQRS）的？",
    # message="Fineract中的 Maker‑Checker 工作流是怎样实现的？",
    # message="当发起 GET /clients 查询时，处理链里哪些组件参与权限校验、数据读取与 JSON 序列化？",
    # message="Fineract 如何支持多租户（multi-tenancy）？",
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
print(node.observation.message)