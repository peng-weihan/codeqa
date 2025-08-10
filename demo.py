import os
import sys

def main():
    # 模拟参数
    repo_path = "/data3/pwh/jsoup"  # 替换为你的目标代码仓库路径
    question = "parseFragment函数在哪里？总结一下其做了什么事情"  # 替换为你要提问的问题
    print(sys.path)
    # 设置项目根目录到PYTHONPATH（更优雅的方式）
    from pathlib import Path
    # 获取项目根目录
    current_dir = Path(__file__).parent
    sys.path.append(current_dir)
    parent_dir = current_dir.parent
    sys.path.append(parent_dir)
    print(sys.path)

    # 导入项目模块
    from moatless_qa.benchmark.utils import get_moatless_instance
    from moatless_qa.completion.completion import CompletionModel, LLMResponseFormat
    from moatless_qa.benchmark.swebench import create_repository
    from moatless_qa.index import CodeIndex
    from moatless_qa.file_context import FileContext
    from moatless_qa.selector import BestFirstSelector
    from moatless_qa.feedback import GroundTruthFeedbackGenerator
    from moatless_qa.value_function.base import ValueFunction
    from moatless_qa.actions import FindClass, FindFunction, FindCodeSnippet, SemanticSearch, ViewCode, Finish, FindCalledObject
    from moatless_qa.agent.code_qa_agent import CodeQAAgent
    from moatless_qa.code_qa.search_tree import CodeQASearchTree

    # 构造必要路径
    index_store_dir = "tmp/index_store"
    repo_base_dir = "tmp/repos"
    persist_path = "tmp/trajectory.json"

    # 创建模型
    completion_model = CompletionModel(
        model="deepseek/deepseek-chat",
        temperature=0.7,
    )

    completion_model.response_format = LLMResponseFormat.TOOLS

    # 创建代码仓库对象
    repository = create_repository(repo_path=repo_path, repo_base_dir=repo_base_dir)

    # 创建索引
    code_index = CodeIndex.from_repository(
        repo_path=repo_path,
        index_store_dir=index_store_dir,
        file_repo=repository,
    )

    # 文件上下文
    file_context = FileContext(repo=repository)

    # Selector 和 Value Function
    selector = BestFirstSelector()
    value_function = ValueFunction(completion_model=completion_model)

    # 动作集合
    actions = [
        FindClass(completion_model=completion_model, code_index=code_index, repository=repository),
        FindFunction(completion_model=completion_model, code_index=code_index, repository=repository),
        FindCodeSnippet(completion_model=completion_model, code_index=code_index, repository=repository),
        FindCalledObject(completion_model=completion_model, code_index=code_index, repository=repository),
        SemanticSearch(completion_model=completion_model, code_index=code_index, repository=repository),
        ViewCode(completion_model=completion_model, repository=repository),
        Finish(),
    ]

    # 创建 Agent
    agent = CodeQAAgent.create(
        repository=repository,
        completion_model=completion_model,
        code_index=code_index,
        preset_actions=actions,
    )
    agent.actions = actions

    # 反馈生成器
    feedback_generator = GroundTruthFeedbackGenerator(
        completion_model=agent.completion,
    )

    # 创建搜索树
    search_tree = CodeQASearchTree.create(
        message=question,
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

    # 执行搜索
    node = search_tree.run_search()

    # 输出答案
    print("\n🔍 最终回答：")
    print(node.observation.message)

if __name__ == "__main__":
    main()
