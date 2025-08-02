import os
import random
from typing import List
import concurrent
import openai
from dotenv import load_dotenv
from typing import List
from repo_qa_generator.models.data_models import QAGeneratorResponseList, QAPair, QAPairListResponse, RepositoryStructure
from repo_qa_generator.core.generator import BaseGenerator
import json
from repo_qa_generator.question_generators.utils import load_template_questions_v2, format_code_relationship_list
from tqdm import tqdm
load_dotenv()
SYSTEM_PROMPT = """You are a professional code analysis assistant, you are good at generating high quality questions about code repository.
Generate as many questions as possible."""

class AgentQAGeneratorV2(BaseGenerator):
    def __init__(self, questions_dir: str = None):
        super().__init__()
        if questions_dir is None:
            # 使用默认模板问题目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.questions_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'dataset', 'seed_questions_v2', 'llm')
        else:
            self.questions_dir = questions_dir

        self.template_questions = load_template_questions_v2(self.questions_dir, ["how","what","why","where"]) 
        for key, question_list in self.template_questions.items():
            print(f"Loaded {len(question_list)} questions for category '{key}' from {self.questions_dir}")

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
        
        print(f"Prompt for LLM:\n{prompt}\n")
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

    def generate_questions(self, repo_structure: RepositoryStructure, file: str) -> List[QAPair]:
        """Generate questions based on seed questions"""
        print("Starting question generation...")
        questions = []
        questions.extend(self.generate_questions_by_class_parallel(repo_structure, file))
        questions.extend(self.generate_questions_by_function_parallel(repo_structure, file))
        return questions
    
    def generate_for_class(self, cls, prompt_template, file) -> List[QAPair]:
        
        class_description = f"Class: {cls}\n"
        question_starts = ["how", "why", "what", "when"]
        start = random.choice(question_starts)
        seed_questions = self.random_select_seed_questions_by_category(start, num_questions=10)
        prompt = prompt_template.format(class_description=class_description, seed_questions="\n".join(seed_questions))
        result = self._generate_qa_pairs_with_llm(prompt)
        print(f"Generated {len(result)} questions for class {cls.name}")
        for q in result:
            print(f"Question: {q.question}")
            # 写文件操作，若写同一个文件，建议加锁；这里简化直接写
        self.write_questions_to_file(result, file)
        return result
    
    def generate_questions_by_class_parallel(self, repo_structure: RepositoryStructure, file: str) -> List[QAPair]:
        prompt_template = """
You are an expert software research assistant.

Given:
1. A class description extracted from a software repository.
2. A list of seed questions that are general or vague.

Task:
1. Based on the seed questions and the class description, generate **one single question** that is:
   - As difficult and complex as possible,
   - Requires multi-hop reasoning or deep technical understanding,
   - Not answerable by simple retrieval or direct lookup (i.e., not solvable by basic RAG methods),
   - Clearly related to the class/module description,
   - Technically precise and detailed,
   - Reflects the style and intent of the original seed questions but goes significantly deeper.
   - **Must not be a compound question** (e.g., no use of “and”, “or”, or comma-based subquestions),
   - **Must be not too long and syntactically simple**


2. The question should encourage advanced analysis, integration of multiple concepts, or insight beyond surface-level information.

3. Output only the single refined question without additional explanation or commentary.

Input:
Class Description:
{class_description}

Seed Questions:
{seed_questions}


"""
        questions = []
        sampled_classes = random.sample(repo_structure.classes, k=min(50, len(repo_structure.classes)))

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = [
                executor.submit(self.generate_for_class, cls, prompt_template, file)
                for cls in sampled_classes
            ]
            for future in concurrent.futures.as_completed(futures):
                questions.extend(future.result())

        return questions

    def generate_questions_by_class(self, repo_structure: RepositoryStructure, file: str) -> List[QAPair]:
        prompt_template = """
You are an expert software research assistant.

Given:
1. A class description extracted from a software repository.
2. A list of seed questions that are general or vague.

Task:
1. Based on the seed questions and the class description, generate **one single question** that is:
   - As difficult and complex as possible,
   - Requires multi-hop reasoning or deep technical understanding,
   - Not answerable by simple retrieval or direct lookup (i.e., not solvable by basic RAG methods),
   - Clearly related to the class/module description,
   - Technically precise and detailed,
   - Reflects the style and intent of the original seed questions but goes significantly deeper.
   - **Must not be a compound question** (e.g., no use of “and”, “or”, or comma-based subquestions),
   - **Must be not too long and syntactically simple**

2. The question should encourage advanced analysis, integration of multiple concepts, or insight beyond surface-level information.

3. Output only the single refined question without additional explanation or commentary.

Input:
Class Description:
{class_description}

Seed Questions:
{seed_questions}


"""
        questions = []
        for cls in repo_structure.classes:
            class_description = f"Class: {cls}\n"
            seed_questions = self.random_select_seed_questions()
            prompt = prompt_template.format(class_description=class_description, seed_questions="\n".join(seed_questions))
            result =self._generate_qa_pairs_with_llm(prompt)
            print(f"Generated {len(result)} questions for class {cls.name}")
            for q in result:
                print(f"Question: {q.question}")
            questions.extend(result)
            self.write_questions_to_file(result, file)
        return questions
    
    def generate_for_function(self, func, prompt_template, file) -> List[QAPair]:
        function_description = f"Function: {func}\n"
        question_starts = ["how", "why", "what", "when"]
    
        start = random.choice(question_starts)
        seed_questions = self.random_select_seed_questions_by_category(start, num_questions=10)
        prompt = prompt_template.format(
            function_description=function_description,
            seed_questions="\n".join(seed_questions),
            summary=self.generate_summary_of_repo(self.repo_structure)  # 注意这里self.repo_structure需可用，或者传入
        )
        result = self._generate_qa_pairs_with_llm(prompt)
        # 输出日志
        print(f"Generated {len(result)} questions for function {func.name} with start word '{start}'")
        for q in result:
            print(f"Question: {q.question}")
        # 写文件操作，线程写同一文件请注意同步或改为不同文件
        self.write_questions_to_file(result, file)
        return result

    def generate_questions_by_function_parallel(self, repo_structure: RepositoryStructure, file: str) -> List[QAPair]:
        prompt_template = """
You are an expert software research assistant.

Given:
1. A function description extracted from a software repository.
2. A list of seed questions that are general or vague.

Task:
1. Based on the seed questions and the function description, generate **one single question** that is:
   - As difficult and complex as possible,
   - Requires multi-hop reasoning or deep technical understanding,
   - Not answerable by simple retrieval or direct lookup (i.e., not solvable by basic RAG methods),
   - Clearly related to the function description,
   - Technically precise and detailed,
   - Reflects the style and intent of the original seed questions but goes significantly deeper.

2. The question should encourage advanced analysis, integration of multiple concepts, or insight beyond surface-level information.

3. Output only the single refined question without additional explanation or commentary.

Input:
Function Description:
{function_description}

Seed Questions:
{seed_questions}

"""
        questions = []
        self.repo_structure = repo_structure  # 方便generate_for_function访问，如果需要，也可改成参数传递
        sampled_functions = random.sample(repo_structure.functions, k=min(50, len(repo_structure.functions)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(self.generate_for_function, func, prompt_template, file)
                # for func in repo_structure.functions
                for func in sampled_functions
            ]
            for future in concurrent.futures.as_completed(futures):
                questions.extend(future.result())
        return questions

    def generate_questions_by_function(self, repo_structure: RepositoryStructure, file: str) -> List[QAPair]:
        prompt = """
You are an expert software research assistant.

Given:
1. A function description extracted from a software repository.
2. A list of seed questions that are general or vague.

Task:
1. Based on the seed questions and the function description, generate **one single question** that is:
   - As difficult and complex as possible,
   - Requires multi-hop reasoning or deep technical understanding,
   - Not answerable by simple retrieval or direct lookup (i.e., not solvable by basic RAG methods),
   - Clearly related to the function description,
   - Technically precise and detailed,
   - Reflects the style and intent of the original seed questions but goes significantly deeper.

2. The question should encourage advanced analysis, integration of multiple concepts, or insight beyond surface-level information.

3. Output only the single refined question without additional explanation or commentary.

Input:
Function Description:
{function_description}

Seed Questions:
{seed_questions}

"""
        questions = []
        for func in repo_structure.functions:
            function_description = f"Function: {func}\n"
            seed_questions = self.random_select_seed_questions()
            prompt.format(function_description=function_description, seed_questions="\n".join(seed_questions), summary=self.generate_summary_of_repo(repo_structure))
            result =self._generate_qa_pairs_with_llm(prompt)
            questions.extend(result)
            self.write_questions_to_file(result, file)
        return questions

    def random_select_seed_questions(self, num_questions: int = 10) -> List[str]:
        """Randomly select a subset of seed questions for a specific class"""
        what_questions = self.template_questions.get("what", [])
        how_questions = self.template_questions.get("how", [])
        why_questions = self.template_questions.get("why", [])
        where_questions = self.template_questions.get("where", [])
        seed_questions = what_questions + how_questions + why_questions + where_questions
        return random.sample(seed_questions, min(num_questions, len(seed_questions)))

    def random_select_seed_questions_by_category(self, category: str, num_questions: int = 10) -> List[str]:
        """Randomly select a subset of seed questions for a specific category"""
        if category not in self.template_questions:
            raise ValueError(f"Category '{category}' not found in template questions.")
        questions = self.template_questions.get(category, [])
        return random.sample(questions, min(num_questions, len(questions)))

    def write_questions_to_file(self, questions: List[QAPair], output_file: str):
        """Append QAPair list to a .jsonl file, one JSON per line."""
        with open(output_file, 'a', encoding='utf-8') as f:
            for qa in questions:
                json_line = json.dumps(qa.model_dump(), ensure_ascii=False)
                f.write(json_line + '\n')
           
        
        
    