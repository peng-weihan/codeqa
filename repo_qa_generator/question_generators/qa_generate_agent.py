import os
from typing import List
import openai
from dotenv import load_dotenv
from typing import List
from repo_qa_generator.models.data_models import QAGeneratorResponseList, QAPair, QAPairListResponse, RepositoryStructure
from repo_qa_generator.core.generator import BaseGenerator
import json
from repo_qa_generator.question_generators.utils import load_template_questions, format_code_relationship_list
from tqdm import tqdm
load_dotenv()
SYSTEM_PROMPT = """You are a professional code analysis assistant, you are good at generating high quality questions about code repository.
Generate as many questions as possible."""
class AgentQAGenerator(BaseGenerator):
    def __init__(self, questions_dir: str = None):
        super().__init__()
        if questions_dir is None:
            # 使用默认模板问题目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.questions_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'src', 'assets', 'questions')
        else:
            self.questions_dir = questions_dir
        self.template_questions = load_template_questions(self.questions_dir, ["Api_Undirect","Relationship_Undirect","High-Level"]) 

    def generate_summary_of_repo(self, repository_structure: RepositoryStructure) -> str:
         # 提取依赖图的简化表示
        dep_graph_summary = {}
        for module, dependencies in repository_structure.dependency_graph.items():
            module_name = os.path.basename(module)
            dep_graph_summary[module_name] = [os.path.basename(dep) for dep in dependencies]
        
        
        # 创建类和函数的简化表示
        classes_summary = []
        for cls in repository_structure.classes:
            classes_summary.append({
                "name": cls.name,
                "methods": [method.name for method in cls.methods],
                "attributes": [attr.name for attr in cls.attributes]
            })
        
        functions_summary = []
        for func in repository_structure.functions:
            if not func.is_method:  # 只包括顶级函数
                functions_summary.append({
                    "name": func.name,
                    "parameters": func.parameters,
                    "calls": func.calls[:5] if len(func.calls) > 5 else func.calls  # 限制调用列表大小
                })
        
        # 构建模块树的简化表示
        def module_to_dict(module):
            return {
                "name": module.name,
                "is_package": module.is_package,
                "files": [file.file_name for file in module.files],
                "sub_modules": [module_to_dict(submodule) for submodule in module.sub_modules]
            }
        
        module_tree_summary = [module_to_dict(module) for module in repository_structure.root_modules]
        
        # 构建输入提示
        summary = {
            "dependency_graph": dep_graph_summary,
            "classes": classes_summary,  # 限制类数量
            "functions": functions_summary,  # 限制函数数量
            "module_tree": module_tree_summary
        }
        
        summary_str = json.dumps(summary, ensure_ascii=False, indent=2)
        return summary_str
    
    def _generate_qa_pairs_with_llm(self, prompt_content: str) -> List[QAPair]:
        """
        Helper method to generate QA pairs using LLM
        
        Args:
            prompt_content: The prompt content to send to the LLM
            
        Returns:
            List of QAPair objects
        """
        schema = QAGeneratorResponseList.model_json_schema()
        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
        
        prompt = f"{prompt_content}\n\nPlease return the response in the following JSON format:\n{schema_str}"
        
        result_text = self._call_llm(system_prompt=SYSTEM_PROMPT, user_prompt=prompt)
        try:
            result_json = json.loads(result_text)
            result_model = QAGeneratorResponseList.model_validate(result_json)
            return [
                QAPair(
                    question=item.question,
                    ground_truth=item.ground_truth,
                ) for item in result_model.qa_pairs
            ]
        except Exception as e:
            try:
                result_json = json.loads(result_text)
                if "qa_pairs" in result_json:
                    return [
                        QAPair(
                            question=item.get("question", ""),
                            answer=item.get("answer", ""),
                            related_code=item.get("related_code")
                        ) for item in result_json["qa_pairs"]
                    ]
            except:
                pass
            
            return []

    def generate_questions(self, repo_structure: RepositoryStructure) -> List[QAPair]:
        """Generate questions based on seed questions"""
        print("Starting question generation...")
        questions = []
        
        # Set up generation tasks
        print(self.template_questions)
        generation_tasks = [
            ("Relationship Questions", lambda: self.generate_relationship_questions(repo_structure, self.template_questions["Relationship_Undirect"])),
            ("API Questions", lambda: self.generate_api_questions(repo_structure, self.template_questions["Api_Undirect"])),
            ("High-Level Questions", lambda: self.generate_high_level_questions(repo_structure, self.template_questions["High-Level"]))
        ]
        
        # Use tqdm to display generation progress
        for task_name, task_func in tqdm(generation_tasks, desc="Generating question categories"):
            print(f"Generating {task_name}...")
            task_questions = task_func()
            questions.extend(task_questions)
            print(f"{task_name}: Generated {len(task_questions)} questions")
        
        print(f"Total questions generated: {len(questions)}")
        return questions
    
    def generate_relationship_questions(self, repo_structure: RepositoryStructure, seed_questions:list[str]) -> List[QAPair]:
        """Generate questions based on file dependencies"""
        
        prompt = """
        Based on the following relationship list, generate high quality questions about API and code relationships using the most important information of the repostory:
        
        {relationships}
        
        # Instructions:
        Please generate questions covering the following aspects:
        1. Module dependencies and their impact
        2. Class inheritance relationships
        3. Function call relationships

        # Examples:
        Question List: {seed_questions}
        

        Based on your understanding of the code, store important information that will help solve the problem in the ground_truth field.
        """
        relationships_str = format_code_relationship_list(repo_structure.relationships)
        questions = []
        for rel in relationships_str:
            prompt = prompt.format(relationships=rel, seed_questions=seed_questions)
            result = self._generate_qa_pairs_with_llm(prompt)
            questions.extend(result)
        return questions
        
    def generate_api_questions(self, repository_structure: RepositoryStructure,question_list:list[str]) -> List[QAPair]:
        """
        Generate high quality Q&A pairs about API based on the repository structure.
        
        Args:
            repository_structure: RepositoryStructure object containing dependency graph, module tree, and code relationships
            model: OpenAI model to use
            
        Returns:
            List of QAPair objects
        """
       

        summary_str = self.generate_summary_of_repo(repository_structure)
        prompt = f"""
        Based on the following repository structure, generate high quality Q&A pairs about API and code relationships using the most important information of the repostory:
        
        {summary_str}
        
        # Instructions:
        Please generate questions covering the following aspects:
        1. API usage patterns and best practices
        2. Module dependencies and their impact

        # Examples:
        Question List: {question_list}
        

        Based on your understanding of the code, store important information that will help solve the problem in the ground_truth field.
        """
        

        return self._generate_qa_pairs_with_llm(prompt)
        
    def generate_high_level_questions(self, repository_structure: RepositoryStructure,question_list:list[str]) -> List[QAPair]:
        """
        Generate high quality Q&A pairs about high level questions based on the repository structure.
        """
        summary_str = self.generate_summary_of_repo(repository_structure)
        
        print(summary_str)
        prompt = f"""
        Based on the following repository structure, generate as many high quality Q&A pairs as possible about high level questions using the most important information of the repostory:
        {summary_str}

        # Examples:
        Question List: {question_list}
        """

        return self._generate_qa_pairs_with_llm(prompt)