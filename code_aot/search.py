# code_aot/external_search.py
from moatless_qa.actions import *
from moatless_qa.agent.agent import ActionAgent
from moatless_qa.node import Node
from moatless_qa.index import CodeIndex
from moatless_qa.repository.repository import Repository
from code_aot.llm import gen
import traceback
import logging
import os
logger = logging.getLogger(__name__)

# 支持的外部搜索动作
EXTERNAL_SEARCH_ACTIONS = [
    FindClass,
    FindCodeSnippet,
    FindFunction,
    SemanticSearch,
    ViewCode,
    FindCalledObject
]

class ExternalSearchManager:
    def __init__(self, repository, code_index, completion_model=None):
        """
        初始化外部搜索管理器
        
        参数:
        - repository: 代码仓库对象
        - code_index: 代码索引对象
        - completion_model: 完成模型（可选）
        """
        self.repository = repository
        self.code_index = code_index
        self.completion_model = completion_model
        self.actions = self._init_actions()
        
    def _init_actions(self):
        """初始化所有搜索动作"""
        actions = []
        for action_class in EXTERNAL_SEARCH_ACTIONS:
            action = action_class(
                code_index=self.code_index,
                repository=self.repository,
                completion_model=self.completion_model
            )
            actions.append(action)
        return actions
    
    def search(self, node, query):
        """
        执行外部搜索
        
        参数:
        - node: 当前节点
        - query: 搜索查询
        
        返回:
        - 搜索结果，或None表示无结果
        """
        if node.action:
            logger.info(f"Node{node.node_id}: 重置节点")
            node.reset()
        
        node.possible_actions = [action.__class__.__name__ for action in self.actions]
        
        # 为搜索任务构建提示
        search_prompt = f"""
        Analyze the following query and determine if it can be solved in one action call.
        If it can be solved, select the most appropriate action and return it.
        If it cannot be solved, clearly indicate that it cannot be solved in one call.
        Only return one action, and only execute it when you are very sure that it can be solved in one call.
        Query: {query}
        Actions: {[action.__class__.__name__ for action in self.actions]}
        """
        
        try:
            # 调用LLM获取搜索动作建议
            msg = [
                {"role": "system", "content": "You are a code search assistant, helping to determine the most appropriate code search method."},
                {"role": "user", "content": search_prompt}
            ]
            completion_response = gen(msg)
            
            # 解析返回的动作
            selected_action = None
            for action in self.actions:
                if action.__class__.__name__ in completion_response:
                    selected_action = action
                    break
            
            if not selected_action:
                logger.info(f"未找到适合查询的动作: {query}")
                return None
            
            # 构建动作参数
            action_args = {'query': query}
            
            # 执行搜索动作
            result = selected_action.run(**action_args)
            return result
            
        except Exception as e:
            logger.warning(f"外部搜索失败: {e}")
            return None

def create_external_search_manager(repo_path, index_path=None):
    """
    创建外部搜索管理器的工厂函数
    
    参数:
    - repo_path: 代码仓库路径
    - index_path: 索引存储路径（可选）
    
    返回:
    - 配置好的ExternalSearchManager实例
    """
    from moatless_qa.benchmark.swebench import create_repository
    from moatless_qa.completion.completion import CompletionModel
    
    # 创建仓库对象
    repository = Repository(repo_dir=repo_path)
    
    # 创建或加载代码索引
    if index_path:
        code_index = CodeIndex.from_index_name(
            os.path.basename(repo_path),
            index_store_dir=index_path,
            file_repo=repository
        )
    else:
        # 如果没有提供索引路径，创建新索引
        code_index = CodeIndex.from_repository(repository)
    
    # 创建完成模型
    completion_model = CompletionModel(
        model="deepseek/deepseek-chat", 
        temperature=0.7
    )
    
    return ExternalSearchManager(
        repository=repository,
        code_index=code_index,
        completion_model=completion_model
    )