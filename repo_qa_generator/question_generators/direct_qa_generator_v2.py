from typing import List, Dict, Optional, Set, Tuple, Any
import os
import re
import random
from pydantic import BaseModel, Field
from repo_qa_generator.models.data_models import ClassDefinition, FunctionDefinition, ClassAttribute, RepositoryStructure, QAPair, CodeNode
from repo_qa_generator.question_generators.utils import load_template_questions_v2
# 用于替换标签的Pydantic模型
class TemplateReplacement(BaseModel):
    template: str = Field(..., description="问题模板")
    replacements: Dict[str, str] = Field(..., description="标签替换值")
    
    def apply(self) -> str:
        """应用替换并返回生成的问题"""
        result = self.template
        for tag, value in self.replacements.items():
            result = result.replace(f"<{tag}>", value)
        return result

class DirectQAGeneratorV2:

    def __init__(self, questions_dir: str = None):
        """
        初始化DirectQAGenerator
        
        Args:
            questions_dir: 模板问题目录路径，默认为项目的assets/questions目录
        """
        if questions_dir is None:
            # 使用默认模板问题目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.questions_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'dataset', 'seed_questions_v2', 'direct')
        else:
            self.questions_dir = questions_dir
            
        # 加载模板问题
        self.template_questions = load_template_questions_v2(self.questions_dir, ['how', 'why', 'what', 'where'])
        # template_question是Dict[str, List[str]]
        for key, question_list in self.template_questions.items():
            print(f"Loaded {len(question_list)} questions for category '{key}' from {self.questions_dir}")

    def generate_questions(self, repo_structure: RepositoryStructure, num_questions: int = 10000000) -> List[QAPair]:
        """
        生成问题
        
        Args:
            repo_structure: 代码分析器提取的仓库结构
            num_questions: 要生成的问题数量
            
        Returns:
            List[QAPair]: 生成的问题列表（问题部分）
        """
        questions = []
        
        # 生成种子问题(直接替换标签问题)
        direct_questions = self._generate_direct_tag_questions(repo_structure)
        questions.extend(direct_questions)

        return questions

    def _handle_tag_replacement_questions(self, template:str, repo_structure: RepositoryStructure, questions: List[QAPair]):
     
        tags = re.findall(r'<([^>]+)>', template)
        valid_tags = ['Module', 'Class', 'Function', 'Method', 'Parameter', 'Attribute']
        if all(tag in valid_tags for tag in tags):
            qa_pairs = self._process_template_by_tags(template, tags, repo_structure)
            if qa_pairs is None:
                print(f"No qa pairs found for template: {template}")
                return
            for replacement, code_nodes in qa_pairs:
                questions.append(QAPair(
                    question=self._apply_replacement(template, replacement),
                    answer="",
                    relative_code_list=code_nodes
                ))
            print(len(qa_pairs))
               
    def _apply_replacement(self, template: str, replacements: Dict[str, str]) -> str:
        """应用替换并返回生成的问题"""
        replacement_obj = TemplateReplacement(template=template, replacements=replacements)
        return replacement_obj.apply()

    def _process_template_by_tags(self, template: str, tags: List[str], repo_structure: RepositoryStructure) -> List[Tuple[Dict[str, str], List[CodeNode]]]:
        """根据标签组合处理模板，返回替换数据和代码节点列表的元组列表"""
        print(tags)
        print(template)
        tag_combinations = {
            ('Module',): self._process_module_questions,
            ('Module', 'Class'): self._process_module_questions,
            ('Module', 'Function'): self._process_module_questions,
            ('Module', 'Class', 'Method'): self._process_module_questions,

            ('Class',): self._process_class_only,
            ('Function',): self._process_function_only,

            ('Class', 'Function'): self._process_class_function_combo,
            ('Class', 'Method'): self._process_class_method_combo,
            
            ('Method',): self._process_method_only,
     
            ('Attribute',): self._process_class_attribute_combo,
            ('Class', 'Attribute'): self._process_class_attribute_combo,

            ('Function','Parameter'): self._process_function_parameter_combo,
        }
        
        # 检查匹配的标签组合并调用对应的处理函数
        for tag_combo, handler in tag_combinations.items():
            if all(tag in tags for tag in tag_combo) and len(tag_combo) == sum(1 for tag in tag_combo if tag in tags):
                return handler(template=template, repo_structure=repo_structure)#只有一个template，就只会匹配一个handler
        
        # 如果没有匹配的标签组合，返回空列表
        return []

    def _process_class_function_combo(self, template: str, repo_structure: RepositoryStructure) -> List[Tuple[Dict[str, str], List[CodeNode]]]:
        """处理包含Class和Function标签的模板"""
        result = []
        for cls in repo_structure.classes:
            if not cls.relative_code:
                continue
            # 类的相关code中的函数
            for func in cls.relative_code.relative_functions: #relative_functions是List[str] 
                replacements = {"Class": cls.name, "Function": func}
                code_nodes = [cls.relative_code]
                result.append((replacements, code_nodes))
        return result

    def _process_class_method_combo(self, template: str, repo_structure: RepositoryStructure) -> List[Tuple[Dict[str, str], List[CodeNode]]]:
        """处理包含Class和Method标签的模板"""
        result = []
        for cls in repo_structure.classes:
            if not cls.relative_code:
                continue
            
            for method in cls.methods:
                if not method.relative_code:
                    continue
                replacements = {"Class": cls.name, "Method": method.name}
                code_nodes = [cls.relative_code, method.relative_code]
                result.append((replacements, code_nodes))
                
        return result

    def _process_class_attribute_combo(self, template: str, repo_structure: RepositoryStructure) -> List[Tuple[Dict[str, str], List[CodeNode]]]:
        """处理包含Class和Attribute标签的模板"""
        result = []
        for cls in repo_structure.classes:
            if not cls.relative_code:
                continue
            
            for attr in cls.attributes:
                # 查找包含此属性的代码片段，弃用，直接整个类的代码
                # attribute_line = self._find_attribute_in_code(cls.relative_code.code, attr.name)
                
                replacements = {"Class": cls.name, "Attribute": attr.name}
                code_nodes = [cls.relative_code]
                result.append((replacements, code_nodes))
                
        return result

    def _process_class_only(self, template: str, repo_structure: RepositoryStructure) -> List[Tuple[Dict[str, str], List[CodeNode]]]:
        """处理仅包含Class标签的模板"""
        result = []
        for cls in repo_structure.classes:
            if cls.relative_code:
                replacements = {"Class": cls.name}
                code_nodes = [cls.relative_code]
                result.append((replacements, code_nodes))
        return result

    def _process_function_only(self, template: str, repo_structure: RepositoryStructure) -> List[Tuple[Dict[str, str], List[CodeNode]]]:
        """处理仅包含Function标签的模板"""
        result = []
        for func in repo_structure.functions:
            if not func.is_method and func.relative_code:
                replacements = {"Function": func.name}
                code_nodes = [func.relative_code]
                result.append((replacements, code_nodes))
                
        return result

    def _process_method_only(self, template: str, repo_structure: RepositoryStructure) -> List[Tuple[Dict[str, str], List[CodeNode]]]:
        """处理仅包含Method标签的模板"""
        result = []
        for cls in repo_structure.classes:
            for method in cls.methods:
                if method.relative_code:
                    code_nodes = [method.relative_code]
                    if cls.relative_code:
                        code_nodes.append(cls.relative_code)
                    replacements = {"Method": method.name}
                    if "<Class>" in template:
                        replacements["Class"] = cls.name
                        
                    result.append((replacements, code_nodes))
                    
        return result

    # 可以处理Module、Module+Class、Module+Function
    def _process_module_questions(self, template: str, repo_structure: RepositoryStructure) -> List[Tuple[Dict[str, str], List[CodeNode]]]:
        """处理Module标签的模板"""
        result = []
        for cls in repo_structure.classes:
            if cls.relative_code and cls.relative_code.belongs_to:
                module = cls.relative_code.belongs_to.module or ""
                if '<Class>' in template:
                    # 如果模板中包含Class标签
                    replacements = {"Module": module, "Class": cls.name}
                else:
                    # 如果模板中只包含Module标签
                    replacements = {"Module": module}

                if "<Method>" in template:
                    for method in cls.methods:
                        if method.relative_code:
                            replacements = {"Module": module,"Class": cls.name,"Method": method.name}
                            code_nodes = [cls.relative_code, method.relative_code]
                            result.append((replacements, code_nodes))
                code_nodes = [cls.relative_code]
                result.append((replacements, code_nodes))
        if "<Function>" in template:
            for func in repo_structure.functions:
                if func.relative_code and func.relative_code.belongs_to and func.is_method == False:
                    module = func.relative_code.belongs_to.module or ""
                    replacements = {"Module": module,"Function": func.name}
                    code_nodes = [func.relative_code]
                    result.append((replacements, code_nodes))
        
        seen = set()
        unique_result = []
        for replacements, code_nodes in result:
        # 转成元组排序后字符串，保证顺序无影响
            key = tuple(sorted(replacements.items()))
            if key not in seen:
                seen.add(key)
                unique_result.append((replacements, code_nodes))
        return unique_result
    
    # 处理Function和Parameter标签的模板
    def _process_function_parameter_combo(self, template: str, repo_structure: RepositoryStructure) -> List[Tuple[Dict[str, str], List[CodeNode]]]:
        """处理包含Function和Parameter标签的模板"""
        result = []
        for func in repo_structure.functions:
            if func.relative_code:
                for param in func.parameters:
                    replacements = {"Function": func.name, "Parameter": param}
                    code_nodes = [func.relative_code]
                    result.append((replacements, code_nodes))
        return result

    def _generate_direct_tag_questions(self, repo_structure: RepositoryStructure) -> List[QAPair]:
        """
        生成直接替换标签的问题
        
        Args:
            repo_structure: 代码分析器提取的仓库结构
            
        Returns:
            List[QAPair]: 生成的问题列表
        """
        questions = []

        # Method 1: Directly handle each question type
        # self._handle_where_questions(repo_structure, questions)
        # self._handle_what_questions(repo_structure, questions)
        # self._handle_how_questions(repo_structure, questions)

        # Method 2: Simply replace the tags in the template questions
        question_types = ['where', 'what', 'how', "why"]
        for question_type in question_types:
            for template in self.template_questions.get(question_type, []):#template此处是单个问题模板，类型是str
                self._handle_tag_replacement_questions(template, repo_structure, questions)
        return questions
    

    def _find_attribute_in_code(self, code: str, attribute_name: str) -> Optional[int]:
        """
        在代码中查找属性的定义行
        
        Args:
            code: 类的代码
            attribute_name: 属性名
            
        Returns:
            Optional[int]: 属性定义的行号
        """
        lines = code.split('\n')
        pattern = rf'\s*{attribute_name}\s*='
        
        for i, line in enumerate(lines):
            if re.search(pattern, line):
                return i
                
        return None
    
    def _generate_core_functionality_of_repo(self, relative_docs:list[str]) -> str:
        """生成仓库核心功能描述"""
        return "This codebase is a web application that allows users to manage their projects and tasks."   
    
    def _generate_functional_questions(self, repo_structure: RepositoryStructure) -> List[QAPair]:
        """生成功能性问题，结合文档字符串和摘要"""
        questions = []
        
        # 获取所有带有文档字符串的类和函数
        classes_with_docs = [(cls.name, cls.docstring, cls) 
                        for cls in repo_structure.classes if cls.docstring]
                        
        functions_with_docs = [(func.name, func.docstring, func.class_name, func) 
                            for func in repo_structure.functions if func.docstring]
        
        # 获取所有带有文档字符串的方法
        methods_with_docs = []
        for cls in repo_structure.classes:
            for method in cls.methods:
                if method.docstring:
                    methods_with_docs.append((method.name, method.docstring, cls.name, cls, method))
        
        # 获取仓库核心功能描述
        core_functionality = repo_structure.core_functionality
        if not core_functionality:
            relative_docs = [cls.relative_code.code for cls in repo_structure.classes if cls.relative_code]
            core_functionality = self._generate_core_functionality_of_repo(relative_docs)
        
        # 定义功能性问题的处理逻辑
        class FunctionalReplacement(BaseModel):
            template: str
            logic: str
            class_name: Optional[str] = None
            function_name: Optional[str] = None
            method_name: Optional[str] = None
            
            def apply(self) -> str:
                result = self.template.replace("<Logic>", self.logic)
                if self.class_name and "<Class>" in result:
                    result = result.replace("<Class>", self.class_name)
                if self.function_name and "<Function>" in result:
                    result = result.replace("<Function>", self.function_name)
                if self.method_name and "<Method>" in result:
                    result = result.replace("<Method>", self.method_name)
                return result
        
        # 处理所有包含Logic标签的模板问题
        for templates in self.template_questions.values():
            for template in templates:
                if '<Logic>' not in template:
                    continue
                    
                # 检查是否同时包含Logic和Method标签
                if '<Method>' in template:
                    # 处理方法和逻辑的组合
                    for method_name, docstring, class_name, cls, method in methods_with_docs:
                        if not docstring:
                            continue
                        
                        # 提取文档字符串的第一句作为逻辑描述
                        logic_desc = docstring.split('.')[0] if '.' in docstring else docstring
                        
                        question = FunctionalReplacement(
                            template=template,
                            logic=logic_desc,
                            class_name=class_name,
                            method_name=method_name
                        ).apply()
                        
                        # 添加相关代码节点
                        relative_code_list = []
                        if method.relative_code:
                            relative_code_list.append(method.relative_code)
                        if cls.relative_code:
                            relative_code_list.append(cls.relative_code)
                        
                        questions.append(QAPair(
                            question=question, 
                            answer="",
                            relative_code_list=relative_code_list
                        ))
                    continue  # 已处理完Logic和Method组合，跳过下面的普通Logic处理
                    
                # 使用类的文档字符串生成问题
                for cls_name, docstring, cls in classes_with_docs:
                    if not docstring:
                        continue
                        
                    # 提取文档字符串的第一句作为逻辑描述
                    logic_desc = docstring.split('.')[0] if '.' in docstring else docstring
                    
                    question = FunctionalReplacement(
                        template=template,
                        logic=logic_desc,
                        class_name=cls_name
                    ).apply()
                    
                    relative_code_list = [cls.relative_code] if cls.relative_code else []
                    questions.append(QAPair(
                        question=question, 
                        answer="",
                        relative_code_list=relative_code_list
                    ))
                
                # 使用函数的文档字符串生成问题
                for func_name, docstring, class_name, func in functions_with_docs:
                    if not docstring:
                        continue
                        
                    # 提取文档字符串的第一句作为逻辑描述
                    logic_desc = docstring.split('.')[0] if '.' in docstring else docstring
                    
                    question = FunctionalReplacement(
                        template=template,
                        logic=logic_desc,
                        class_name=class_name,
                        function_name=func_name
                    ).apply()
                    
                    # 添加相关代码节点
                    relative_code_list = [func.relative_code] if func.relative_code else []
                    
                    # 如果是方法，添加所属类的代码节点
                    if class_name:
                        for cls in repo_structure.classes:
                            if cls.name == class_name and cls.relative_code:
                                relative_code_list.append(cls.relative_code)
                                break
                                
                    questions.append(QAPair(
                        question=question, 
                        answer="",
                        relative_code_list=relative_code_list
                    ))
                
                # 使用仓库核心功能描述生成问题
                if core_functionality:
                    sentences = core_functionality.split('.')
                    for sentence in sentences:
                        if sentence.strip():
                            question = FunctionalReplacement(
                                template=template,
                                logic=sentence.strip()
                            ).apply()
                            
                            questions.append(QAPair(
                                question=question, 
                                answer="",
                                relative_code_list=[]
                            ))
        
        return questions