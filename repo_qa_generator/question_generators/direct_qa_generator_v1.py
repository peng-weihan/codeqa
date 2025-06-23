from repo_qa_generator.question_generators.direct_qa_generator import DirectQAGenerator
from repo_qa_generator.models.data_models import RepositoryStructure, QAPair
from repo_qa_generator.models.template_replacement import TemplateReplacement
import re
from typing import List

class DirectQAGeneratorV1:
    def __init__(self, template_questions: dict):
        self.template_questions = template_questions
    def _handle_where_questions(self, repo_structure: RepositoryStructure, questions: List[QAPair]):
        """处理Where类型的问题模板"""
        for template in self.template_questions.get('Where', []):
            # 检查模板中的标签
            tags = re.findall(r'<([^>]+)>', template)
            if all(tag in ['Class', 'Function', 'Method', 'Variable', 'Attribute', 'Parameter', 'Module'] for tag in tags):
                if 'Class' in tags and 'Method' in tags:
                    for cls in repo_structure.classes:
                        if not cls.relative_code:
                            continue
                            
                        class_code_node = cls.relative_code
                        
                        for method in cls.methods:
                            if not method.relative_code:
                                continue
                                
                            method_code_node = method.relative_code
                            
                            replacement = TemplateReplacement(
                                template=template,
                                replacements={"Class": cls.name, "Method": method.name}
                            )
                            question = replacement.apply()
                            
                            # 将类和方法的代码节点添加到QA对中
                            relative_code_list = [class_code_node, method_code_node]
                            
                            questions.append(QAPair(
                                question=question, 
                                answer="",
                                relative_code_list=relative_code_list
                            ))
                            
                elif 'Class' in tags and 'Attribute' in tags:
                    for cls in repo_structure.classes:
                        if not cls.relative_code:
                            continue
                            
                        class_code_node = cls.relative_code
                        
                        for attr in cls.attributes:
                            # 查找包含此属性的代码片段
                            attribute_line = self._find_attribute_in_code(cls.relative_code.code, attr.name)
                            
                            replacement = TemplateReplacement(
                                template=template,
                                replacements={"Class": cls.name, "Attribute": attr.name}
                            )
                            question = replacement.apply()
                            
                            # 将类的代码节点添加到QA对中
                            relative_code_list = [class_code_node]
                            
                            questions.append(QAPair(
                                question=question, 
                                answer="",
                                relative_code_list=relative_code_list
                            ))
                            
                elif 'Class' in tags:
                    for cls in repo_structure.classes:
                        if not cls.relative_code:
                            continue
                            
                        class_code_node = cls.relative_code
                        
                        replacement = TemplateReplacement(
                            template=template,
                            replacements={"Class": cls.name}
                        )
                        question = replacement.apply()
                        
                        # 将类的代码节点添加到QA对中
                        relative_code_list = [class_code_node]
                        
                        questions.append(QAPair(
                            question=question, 
                            answer="",
                            relative_code_list=relative_code_list
                        ))
                        
                elif 'Function' in tags:
                    for func in repo_structure.functions:
                        if not func.is_method and func.relative_code:
                            function_code_node = func.relative_code
                            
                            replacement = TemplateReplacement(
                                template=template,
                                replacements={"Function": func.name}
                            )
                            question = replacement.apply()
                            
                            # 将函数的代码节点添加到QA对中
                            relative_code_list = [function_code_node]
                            
                            questions.append(QAPair(
                                question=question, 
                                answer="",
                                relative_code_list=relative_code_list
                            ))
                            
                elif 'Method' in tags:
                    for cls in repo_structure.classes:
                        for method in cls.methods:
                            if method.relative_code:
                                method_code_node = method.relative_code
                                
                                replacement = TemplateReplacement(
                                    template=template,
                                    replacements={"Method": method.name, "Class": cls.name if 'Class' in tags else ""}
                                )
                                question = replacement.apply()
                                
                                # 将方法的代码节点添加到QA对中
                                relative_code_list = [method_code_node]
                                if cls.relative_code:
                                    relative_code_list.append(cls.relative_code)
                                
                                questions.append(QAPair(
                                    question=question, 
                                    answer="",
                                    relative_code_list=relative_code_list
                                ))
                elif 'Module' in tags:
                    for cls in repo_structure.classes:
                        if cls.relative_code:
                            module = cls.relative_code.belongs_to.module if cls.relative_code.belongs_to else ""
                            replacement = TemplateReplacement(
                                template=template,
                                replacements={"Module": module}
                            )
                            question = replacement.apply()
                            
                            questions.append(QAPair(
                                question=question,
                                answer="",
                                relative_code_list=[cls.relative_code]
                            ))
                elif 'Logic' in tags:
                    for cls in repo_structure.classes:
                        if cls.relative_code:
                            logic = cls.docstring if cls.docstring else ""
                            replacement = TemplateReplacement(
                                template=template,
                                replacements={"Logic": logic}
                            )
                            question = replacement.apply()
                            
                            questions.append(QAPair(
                                question=question,
                                answer="",
                                relative_code_list=[cls.relative_code]
                            ))  

    def _handle_what_questions(self, repo_structure: RepositoryStructure, questions: List[QAPair]):
        """处理What类型的问题模板"""
        for template in self.template_questions.get('What', []):
            # 检查模板中的标签
            tags = re.findall(r'<([^>]+)>', template)
            
            # 处理包含Term标签的问题
            if 'Term' in tags:
                # 收集所有可能的术语（类名、函数名、方法名等）
                all_terms = []
                
                # 添加类名
                for cls in repo_structure.classes:
                    if cls.relative_code:
                        all_terms.append((cls.name, [cls.relative_code]))
                
                # 添加函数名
                for func in repo_structure.functions:
                    if func.relative_code:
                        code_nodes = [func.relative_code]
                        # 如果是方法，添加类的代码节点
                        if func.is_method and func.class_name:
                            for cls in repo_structure.classes:
                                if cls.name == func.class_name and cls.relative_code:
                                    code_nodes.append(cls.relative_code)
                                    break
                        all_terms.append((func.name, code_nodes))
                
                # 为每个术语生成问题
                for term, code_nodes in all_terms:
                    replacement = TemplateReplacement(
                        template=template,
                        replacements={"Term": term}
                    )
                    question = replacement.apply()
                    
                    questions.append(QAPair(
                        question=question,
                        answer="",
                        relative_code_list=code_nodes
                    ))
            
            # 处理类相关问题
            elif 'Class' in tags:
                # 处理类和方法组合
                if 'Method' in tags:
                    for cls in repo_structure.classes:
                        if not cls.relative_code:
                            continue
                        class_code_node = cls.relative_code
                        
                        for method in cls.methods:
                            if not method.relative_code:
                                continue
                            method_code_node = method.relative_code
                            
                            replacement = TemplateReplacement(
                                template=template,
                                replacements={"Class": cls.name, "Method": method.name}
                            )
                            question = replacement.apply()
                            
                            questions.append(QAPair(
                                question=question,
                                answer="",
                                relative_code_list=[class_code_node, method_code_node]
                            ))
                
                # 处理类和属性组合
                elif 'Attribute' in tags or 'Field' in tags:
                    tag = 'Attribute' if 'Attribute' in tags else 'Field'
                    for cls in repo_structure.classes:
                        if not cls.relative_code:
                            continue
                        class_code_node = cls.relative_code
                        
                        for attr in cls.attributes:
                            replacement = TemplateReplacement(
                                template=template,
                                replacements={"Class": cls.name, tag: attr.name}
                            )
                            question = replacement.apply()
                            
                            questions.append(QAPair(
                                question=question,
                                answer="",
                                relative_code_list=[class_code_node]
                            ))
                
                # 处理类和参数组合
                elif 'Parameter' in tags:
                    for cls in repo_structure.classes:
                        if not cls.relative_code:
                            continue
                        class_code_node = cls.relative_code
                        
                        # 找到类的初始化方法
                        init_method = None
                        for method in cls.methods:
                            if method.name == "__init__" and method.relative_code:
                                init_method = method
                                break
                        
                        if init_method and init_method.parameters:
                            for param in init_method.parameters:
                                if param != 'self':  # 排除self参数
                                    replacement = TemplateReplacement(
                                        template=template,
                                        replacements={"Class": cls.name, "Parameter": param}
                                    )
                                    question = replacement.apply()
                                    
                                    questions.append(QAPair(
                                        question=question,
                                        answer="",
                                        relative_code_list=[class_code_node, init_method.relative_code]
                                    ))
                
                # 处理类和模块组合
                elif 'Module' in tags:
                    for cls in repo_structure.classes:
                        if not cls.relative_code:
                            continue
                        class_code_node = cls.relative_code
                        
                        # 获取类所在的模块
                        module = cls.relative_code.belongs_to.module if cls.relative_code.belongs_to else ""
                        
                        if module:
                            replacement = TemplateReplacement(
                                template=template,
                                replacements={"Class": cls.name, "Module": module}
                            )
                            question = replacement.apply()
                            
                            questions.append(QAPair(
                                question=question,
                                answer="",
                                relative_code_list=[class_code_node]
                            ))
                
                # 仅处理类的问题
                else:
                    for cls in repo_structure.classes:
                        if not cls.relative_code:
                            continue
                        class_code_node = cls.relative_code
                        
                        replacement = TemplateReplacement(
                            template=template,
                            replacements={"Class": cls.name}
                        )
                        question = replacement.apply()
                        
                        questions.append(QAPair(
                            question=question,
                            answer="",
                            relative_code_list=[class_code_node]
                        ))
            
            # 处理函数相关问题
            elif 'Function' in tags:
                for func in repo_structure.functions:
                    if not func.relative_code:
                        continue
                    function_code_node = func.relative_code
                    
                    # 处理函数和模块组合
                    if 'Module' in tags:
                        module = func.relative_code.belongs_to.module if func.relative_code.belongs_to else ""
                        
                        if module:
                            replacement = TemplateReplacement(
                                template=template,
                                replacements={"Function": func.name, "Module": module}
                            )
                            question = replacement.apply()
                            
                            questions.append(QAPair(
                                question=question,
                                answer="",
                                relative_code_list=[function_code_node]
                            ))
                    
                    # 处理函数和上下文组合
                    elif 'Context' in tags or 'Logic' in tags:
                        tag = 'Context' if 'Context' in tags else 'Logic'
                        context = func.docstring if func.docstring else "the codebase"
                        
                        replacement = TemplateReplacement(
                            template=template,
                            replacements={"Function": func.name, tag: context}
                        )
                        question = replacement.apply()
                        
                        questions.append(QAPair(
                            question=question,
                            answer="",
                            relative_code_list=[function_code_node]
                        ))
                    
                    # 处理函数和参数组合
                    elif 'Parameter' in tags:
                        for param in func.parameters:
                            replacement = TemplateReplacement(
                                template=template,
                                replacements={"Function": func.name, "Parameter": param}
                            )
                            question = replacement.apply()
                            
                            questions.append(QAPair(
                                question=question,
                                answer="",
                                relative_code_list=[function_code_node]
                            ))
                    
                    # 仅处理函数的问题
                    else:
                        replacement = TemplateReplacement(
                            template=template,
                            replacements={"Function": func.name}
                        )
                        question = replacement.apply()
                        
                        questions.append(QAPair(
                            question=question,
                            answer="",
                            relative_code_list=[function_code_node]
                        ))
            
            # 处理方法相关问题
            elif 'Method' in tags:
                for cls in repo_structure.classes:
                    for method in cls.methods:
                        if not method.relative_code:
                            continue
                        method_code_node = method.relative_code
                        
                        # 处理方法和参数组合
                        if 'Parameter' in tags:
                            for param in method.parameters:
                                if param != 'self':  # 排除self参数
                                    replacement = TemplateReplacement(
                                        template=template,
                                        replacements={"Method": method.name, "Parameter": param}
                                    )
                                    question = replacement.apply()
                                    
                                    questions.append(QAPair(
                                        question=question,
                                        answer="",
                                        relative_code_list=[method_code_node]
                                    ))
                        
                        # 仅处理方法的问题
                        else:
                            replacement = TemplateReplacement(
                                template=template,
                                replacements={"Method": method.name}
                            )
                            question = replacement.apply()
                            
                            questions.append(QAPair(
                                question=question,
                                answer="",
                                relative_code_list=[method_code_node]
                            ))
            
            # 处理变量相关问题
            elif 'Variable' in tags:
                # 查找设置变量属性的函数
                for variable_name in [attr.name for attr in repo_structure.attributes]:
                    related_functions = []
                    
                    for func in repo_structure.functions:
                        if func.relative_code and f"self.{variable_name}" in func.relative_code.code:
                            related_functions.append((func, func.relative_code))
                    
                    if related_functions:
                        replacement = TemplateReplacement(
                            template=template,
                            replacements={"Variable": variable_name}
                        )
                        question = replacement.apply()
                        
                        code_nodes = [func_code for _, func_code in related_functions]
                        
                        questions.append(QAPair(
                            question=question,
                            answer="",
                            relative_code_list=code_nodes
                        ))
    
    def _handle_how_questions(self, repo_structure: RepositoryStructure, questions: List[QAPair]):
        """处理How类型的问题模板"""
        for template in self.template_questions.get('How', []):
            # 检查模板中的标签
            tags = re.findall(r'<([^>]+)>', template)
            
            # 处理类相关问题
            if 'Class' in tags:
                # 处理类和变量组合
                if 'Variable' in tags:
                    for cls in repo_structure.classes:
                        if not cls.relative_code:
                            continue
                        class_code_node = cls.relative_code
                        
                        # 找出类的所有属性
                        for attr in cls.attributes:
                            replacement = TemplateReplacement(
                                template=template,
                                replacements={"Class": cls.name, "Variable": attr.name}
                            )
                            question = replacement.apply()
                            
                            # 找出处理该属性的所有方法
                            method_nodes = []
                            for method in cls.methods:
                                if method.relative_code and f"self.{attr.name}" in method.relative_code.code:
                                    method_nodes.append(method.relative_code)
                            
                            # 构建代码节点列表
                            relative_code_list = [class_code_node]
                            relative_code_list.extend(method_nodes)
                            
                            questions.append(QAPair(
                                question=question,
                                answer="",
                                relative_code_list=relative_code_list
                            ))
                
                # 处理类和方法组合
                elif 'Method' in tags:
                    for cls in repo_structure.classes:
                        if not cls.relative_code:
                            continue
                        class_code_node = cls.relative_code
                        
                        for method in cls.methods:
                            if not method.relative_code:
                                continue
                            method_code_node = method.relative_code
                            
                            replacement = TemplateReplacement(
                                template=template,
                                replacements={"Class": cls.name, "Method": method.name}
                            )
                            question = replacement.apply()
                            
                            # 找出调用此方法的其他方法
                            caller_nodes = []
                            for other_method in cls.methods:
                                if other_method != method and other_method.relative_code and \
                                   f"self.{method.name}" in other_method.relative_code.code:
                                    caller_nodes.append(other_method.relative_code)
                            
                            # 构建代码节点列表
                            relative_code_list = [class_code_node, method_code_node]
                            relative_code_list.extend(caller_nodes)
                            
                            questions.append(QAPair(
                                question=question,
                                answer="",
                                relative_code_list=relative_code_list
                            ))
                
                # 处理类和特性组合
                elif 'Feature' in tags:
                    features = set()
                    
                    # 从文档字符串中提取特性
                    for cls in repo_structure.classes:
                        if cls.docstring:
                            # 简单地从文档中提取可能的特性短语
                            potential_features = re.findall(r'(?:handles|manages|supports|provides) ([a-zA-Z\s]+)', cls.docstring, re.IGNORECASE)
                            features.update(potential_features)
                    
                    # 如果没有找到足够的特性，使用一些通用的特性
                    if len(features) < 3:
                        features.update(["data validation", "error handling", "configuration management", "serialization"])
                    
                    # 为每个类和特性生成问题
                    for cls in repo_structure.classes:
                        if not cls.relative_code:
                            continue
                        class_code_node = cls.relative_code
                        
                        for feature in features:
                            replacement = TemplateReplacement(
                                template=template,
                                replacements={"Class": cls.name, "Feature": feature}
                            )
                            question = replacement.apply()
                            
                            questions.append(QAPair(
                                question=question,
                                answer="",
                                relative_code_list=[class_code_node]
                            ))
                
                # 仅处理类的问题
                else:
                    for cls in repo_structure.classes:
                        if not cls.relative_code:
                            continue
                        class_code_node = cls.relative_code
                        
                        replacement = TemplateReplacement(
                            template=template,
                            replacements={"Class": cls.name}
                        )
                        question = replacement.apply()
                        
                        questions.append(QAPair(
                            question=question,
                            answer="",
                            relative_code_list=[class_code_node]
                        ))
            
            # 处理方法相关问题
            elif 'Method' in tags:
                # 处理方法和逻辑组合
                if 'Logic' in tags:
                    for cls in repo_structure.classes:
                        for method in cls.methods:
                            if not method.relative_code or not method.docstring:
                                continue
                            
                            # 从方法的文档字符串中提取逻辑描述
                            logic = method.docstring.split('.')[0] if '.' in method.docstring else method.docstring
                            
                            replacement = TemplateReplacement(
                                template=template,
                                replacements={"Method": method.name, "Logic": logic}
                            )
                            question = replacement.apply()
                            
                            questions.append(QAPair(
                                question=question,
                                answer="",
                                relative_code_list=[method.relative_code]
                            ))
                            
                # 处理方法与对象组合
                elif 'Object' in tags:
                    for cls in repo_structure.classes:
                        for method in cls.methods:
                            if not method.relative_code:
                                continue
                            
                            # 查找方法中的对象构造语句
                            objects_found = re.findall(r'([A-Z][a-zA-Z0-9_]*)\(', method.relative_code.code)
                            object_names = set(objects_found) - {cls.name}  # 排除当前类名
                            
                            for obj_name in object_names:
                                replacement = TemplateReplacement(
                                    template=template,
                                    replacements={"Method": method.name, "Object": obj_name}
                                )
                                question = replacement.apply()
                                
                                questions.append(QAPair(
                                    question=question,
                                    answer="",
                                    relative_code_list=[method.relative_code]
                                ))
            
            # 处理函数相关问题
            elif 'Function' in tags:
                for func in repo_structure.functions:
                    if not func.relative_code:
                        continue
                    
                    # 处理函数和装饰器组合
                    if 'Decorator' in tags:
                        # 查找函数定义前的装饰器
                        decorators = re.findall(r'@([a-zA-Z_][a-zA-Z0-9_]*)', func.relative_code.code)
                        
                        for decorator in decorators:
                            replacement = TemplateReplacement(
                                template=template,
                                replacements={"Function": func.name, "Decorator": decorator}
                            )
                            question = replacement.apply()
                            
                            questions.append(QAPair(
                                question=question,
                                answer="",
                                relative_code_list=[func.relative_code]
                            ))
                            
                    # 处理函数和上下文/逻辑组合
                    elif any(tag in tags for tag in ['Logic', 'Context']):
                        # 确定要使用的标签
                        context_tag = 'Logic' if 'Logic' in tags else 'Context'
                        
                        # 从函数文档字符串中提取上下文/逻辑
                        context = "this codebase"
                        if func.docstring:
                            context = func.docstring.split('.')[0] if '.' in func.docstring else func.docstring
                            
                        replacement = TemplateReplacement(
                            template=template,
                            replacements={"Function": func.name, context_tag: context}
                        )
                        question = replacement.apply()
                        
                        questions.append(QAPair(
                            question=question,
                            answer="",
                            relative_code_list=[func.relative_code]
                        ))
                        
                    # 处理函数和变量组合
                    elif 'Variable' in tags:
                        # 分析函数代码，寻找变量
                        code_lines = func.relative_code.code.split('\n')
                        var_assignments = []
                        
                        # 简单解析变量赋值语句
                        for line in code_lines:
                            assignment_match = re.search(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?!.*=)', line)
                            if assignment_match:
                                var_assignments.append(assignment_match.group(1))
                        
                        # 如果函数中有参数，也考虑将其作为变量
                        var_assignments.extend(func.parameters)
                        
                        # 从变量列表中选择一个进行问题生成
                        for var_name in var_assignments:
                            replacement = TemplateReplacement(
                                template=template,
                                replacements={"Function": func.name, "Variable": var_name}
                            )
                            question = replacement.apply()
                            
                            questions.append(QAPair(
                                question=question,
                                answer="",
                                relative_code_list=[func.relative_code]
                            ))
                    
                    # 仅处理函数的问题
                    else:
                        replacement = TemplateReplacement(
                            template=template,
                            replacements={"Function": func.name}
                        )
                        question = replacement.apply()
                        
                        questions.append(QAPair(
                            question=question,
                            answer="",
                            relative_code_list=[func.relative_code]
                        ))
            
            # 处理属性相关问题
            elif 'Attribute' in tags and 'Context' in tags:
                for cls in repo_structure.classes:
                    for attr in cls.attributes:
                        # 假设上下文是在方法中
                        for method in cls.methods:
                            if method.relative_code and f"self.{attr.name}" in method.relative_code.code:
                                # 使用方法名作为上下文
                                replacement = TemplateReplacement(
                                    template=template,
                                    replacements={"Attribute": attr.name, "Context": method.name}
                                )
                                question = replacement.apply()
                                
                                questions.append(QAPair(
                                    question=question,
                                    answer="",
                                    relative_code_list=[method.relative_code]
                                ))
            
            # 处理变量相关问题
            elif 'Variable' in tags and not any(tag in tags for tag in ['Class', 'Function']):
                # 收集所有属性作为变量
                for cls in repo_structure.classes:
                    for attr in cls.attributes:
                        # 找出使用该属性的方法
                        method_nodes = []
                        for method in cls.methods:
                            if method.relative_code and f"self.{attr.name}" in method.relative_code.code:
                                method_nodes.append(method.relative_code)
                        
                        if method_nodes:
                            replacement = TemplateReplacement(
                                template=template,
                                replacements={"Variable": attr.name}
                            )
                            question = replacement.apply()
                            
                            questions.append(QAPair(
                                question=question,
                                answer="",
                                relative_code_list=method_nodes
                            ))
            
            # 处理测试相关问题
            elif '测试' in template.lower() or 'test' in template.lower():
                # 尝试找到测试相关的文件
                test_file_nodes = []
                
                for func in repo_structure.functions:
                    if func.relative_code and func.relative_code.belongs_to:
                        file_name = func.relative_code.belongs_to.file_name
                        if 'test' in file_name.lower():
                            test_file_nodes.append(func.relative_code)
                
                # 使用通用的测试相关问题
                if test_file_nodes:
                    questions.append(QAPair(
                        question=template,
                        answer="",
                        relative_code_list=test_file_nodes
                    ))
    
    def _handle_api_questions(self, repo_structure: RepositoryStructure, questions: List[QAPair]):
        """处理API相关的问题"""
        pass