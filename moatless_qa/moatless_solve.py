from repo_qa_generator.models.data_models import QAPair
from moatless_qa.agent.code_qa_agent import CodeQAAgent
from moatless_qa.completion import CompletionModel
from moatless_qa.actions import FindClass, FindFunction, FindCodeSnippet, SemanticSearch, ViewCode, Finish, FindCalledObject
from moatless_qa.benchmark.swebench import create_repository
from moatless_qa.index import CodeIndex
from moatless_qa.file_context import FileContext
from moatless_qa.selector import BestFirstSelector
from moatless_qa.feedback import GroundTruthFeedbackGenerator
from moatless_qa.value_function.base import ValueFunction
from moatless_qa.completion.completion import LLMResponseFormat
from moatless_qa.code_qa.search_tree import CodeQASearchTree
from moatless_qa.benchmark.utils import get_moatless_instance

index_store_dir = "./data/index_store"
repo_base_dir = "./data/repos"
persist_path = "./data/trajectory.json"
instance_id = "sphinx-doc__sphinx-8551"
instance_path = f'./data/trajectory/{instance_id}/'
instance = get_moatless_instance(instance_id)

class MoatlessSolve:
    def __init__(self, repo_name:str, repo_path: str):
        # Global variables are used here as defined above the class:
        # instance, repo_base_dir, index_store_dir, instance_path, persist_path
    
        completion_model = CompletionModel(model="deepseek/deepseek-chat", temperature=0.7)
        completion_model.response_format = LLMResponseFormat.TOOLS
        repository = create_repository(instance, repo_base_dir=repo_base_dir)

        code_index = CodeIndex.from_index_name(
            instance["instance_id"], index_store_dir=index_store_dir, file_repo=repository
        )
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
        feedback_generator = GroundTruthFeedbackGenerator(completion_model=agent.completion, instance_dir=instance_path)
        
        self.search_args = {
            "agent": agent,
            "file_context": file_context,
            "selector": selector,
            "value_function": value_function,
            "feedback_generator": feedback_generator,
        }
        self.persist_path = persist_path

    def moatless_solve(self, question: str):
        search_tree = CodeQASearchTree.create(
            message=question, 
            **self.search_args,
            max_iterations=100,
            max_expansions=3,
            max_depth=25,
            persist_path=self.persist_path
        )
        res_node = search_tree.run_search()
        return res_node.observation.message if res_node else None
