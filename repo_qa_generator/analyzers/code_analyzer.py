import ast
import networkx as nx
import uuid
from typing import List, Dict, Optional, Set, Tuple, Any
import re
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from repo_qa_generator.models.data_models import (
    FileNode, CodeNode, ClassDefinition, FunctionDefinition, 
    ClassAttribute, RepositoryStructure, Repository, 
    ModuleNode, CodeRelationship, VariableDefinition
)

class CodeAnalyzer:
    def __init__(self):
        self.dependency_graph = nx.DiGraph()
        self.repository_structure = RepositoryStructure()
        self.current_content = ""  # 存储当前分析的文件内容
        self.relationships = []  # 存储代码元素间的关系
        self._file_cache: dict[str, FileNode | None] = {}  # 缓存文件分析结果，避免重复分析
        
    def analyze_file(self, file_path: str, repo_root: str) -> Optional[FileNode]:
        """Analyze a single file to extract import relationships and class definitions"""
        if file_path in self._file_cache:
            return self._file_cache[file_path]
        
        print(f"开始分析文件: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            imports = []
            classes = []
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.extend(self._extract_imports(node))
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
            if repo_root:
                upper_path = os.path.relpath(os.path.dirname(file_path), repo_root)
            else:
                upper_path = os.path.dirname(file_path)
                
            node = FileNode(
                file_name=os.path.basename(file_path),
                upper_path=os.path.dirname(file_path),
                module=os.path.basename(os.path.dirname(file_path)),
                define_class=classes,
                imports=imports
            )
            self._file_cache[file_path] = node
            return node
        except SyntaxError:
            print(f"警告：文件 {file_path} 存在语法错误，已跳过")
        except UnicodeDecodeError:
            print(f"警告：文件 {file_path} 编码错误，已跳过")
        except Exception as e:
            print(f"警告：分析文件 {file_path} 时发生错误：{str(e)}")
        return None
    
    def build_dependency_graph(self, files: List[str],repo_root: str):
        """Build dependency graph from files"""
        dependency_graph = {}
        file_nodes = {}
        
        # 第一遍：加载所有文件并创建文件节点
        for file_path in files:
            file_node = self.analyze_file(file_path,repo_root)
            if file_node:
                module_path = os.path.dirname(file_path)
                file_nodes[file_path] = file_node
                if module_path not in dependency_graph:
                    dependency_graph[module_path] = []
        
        # 第二遍：构建依赖关系
        for file_path, file_node in file_nodes.items():
            module_path = os.path.dirname(file_path)
            for imported in file_node.imports:
                # 寻找可能的导入目标
                for potential_source in files:
                    if os.path.basename(potential_source).replace('.py', '') == imported.split('.')[-1]:
                        target_module = os.path.dirname(potential_source)
                        if target_module not in dependency_graph[module_path]:
                            dependency_graph[module_path].append(target_module)
        
        self.repository_structure.dependency_graph = dependency_graph
        return dependency_graph
    
    def _extract_imports(self, node: ast.AST) -> List[str]:
        """Extract import statements"""
        imports = []
        if isinstance(node, ast.Import):
            for name in node.names:
                if hasattr(name, 'name'):
                    imports.append(name.name)
                elif isinstance(name, str):
                    imports.append(name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for name in node.names:
                if hasattr(name, 'name'):
                    imports.append(f"{module}.{name.name}")
                elif isinstance(name, str):
                    imports.append(f"{module}.{name}")
        return imports
    
    def _get_related_functions(self, node: ast.AST) -> List[str]:
        """获取相关的函数调用"""
        # 获取函数体的源代码
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            body = ast.get_source_segment(self.current_content, node)
            if body:
                # 使用增强的调用分析功能
                return self.extract_calls_in_order(body)
        return []

    def simple_extract_calls_in_order(self, body: str) -> List[str]:
        """简单提取函数调用（用于语法错误时的回退方案）"""
        # 提取出函数签名
        func_signature = re.findall(
            r"(([^\s\r\n]+.*?\s*)?\s*def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(.*?\)\s*(?:->\s*['\"a-zA-Z0-9\[\]_.,\s\|]*['\"]*)?:)",
            body, re.DOTALL) 
        if func_signature:
            body = body.removeprefix(func_signature[0][0])
        body_lines = [line for line in body.split("\n") if line.strip() != ""]
        calls = []

        def extract_parts(code_line):
            result = []
            # 匹配()前的部分（函数调用，包括a()、a.b()）
            pattern = r'([a-zA-Z_][\w]*(\.[a-zA-Z_][\w]*)*)(?=\()'
            call_result = re.findall(pattern, code_line)
            if call_result:
                call_result = [item[0] for item in call_result]

            # 匹配没有括号的对象引用 (如 a.b)
            dot_result = re.findall(r'([a-zA-Z_][\w]*(\.[a-zA-Z_][\w]*)*)', code_line)

            # 过滤掉不需要的部分（如关键字、单个字母等）
            dot_result = [item[0] for item in dot_result if len(item[0].split('.')) > 1 and len(item[0]) > 1]
            result.extend(call_result)
            result.extend(dot_result)
            return result

        for line in body_lines:
            calls.extend(extract_parts(line))

        return calls

    def extract_calls_in_order(self, body: str) -> List[str]:
        """从函数体中按顺序提取出所有调用"""
        calls = []
        visited = set()  # 用于存储已经处理的调用路径，避免重复提取

        # 使用AST解析代码
        try:
            tree = ast.parse(body)
        except SyntaxError:
            return self.simple_extract_calls_in_order(body)

        def get_node(node):
            # 处理函数调用
            if isinstance(node, ast.Call):
                # 处理a(参数).b(参数)和a.b.c(参数)
                if isinstance(node.func, ast.Attribute):
                    # a.b.c(参数)的情况，提取a.b.c
                    call = self._get_attribute_call(node.func)
                    if call not in visited:
                        calls.append(call)
                        visited.add(call)
                        # 同时标记它的模块路径（比如jax.random）
                        module_path = '.'.join(call.split('.')[:-1])
                        visited.add(module_path)
                    get_node(node.func.value)

                elif isinstance(node.func, ast.Name):
                    # a(参数)的情况
                    call = node.func.id
                    if call not in visited:
                        calls.append(call)
                        visited.add(call)

                else:
                    get_node(node.func)

                # 递归检查参数中的调用
                for arg in node.args:
                    get_node(arg)

                for keyword in node.keywords:
                    get_node(keyword.value)

            # 处理属性调用
            elif isinstance(node, ast.Attribute):
                # a.b的情况，提取为字段访问
                field_access = self._get_attribute_call(node)
                if field_access not in visited:
                    calls.append(field_access)
                    visited.add(field_access)
                    module_path = '.'.join(field_access.split('.')[:-1])
                    visited.add(module_path)
                get_node(node.value)

            # 处理其他类型的节点
            elif isinstance(node, ast.Expr):
                get_node(node.value)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    get_node(target)
                get_node(node.value)
            elif isinstance(node, ast.Subscript):
                get_node(node.value)
            elif isinstance(node, ast.Name):
                pass  # 简化处理，不考虑全局变量和导入变量
            elif isinstance(node, ast.Starred):
                get_node(node.value)
            elif isinstance(node, (ast.BinOp, ast.UnaryOp)):
                if isinstance(node, ast.BinOp):
                    get_node(node.left)
                    get_node(node.right)
                else:
                    get_node(node.operand)
            elif isinstance(node, ast.BoolOp):
                for value in node.values:
                    get_node(value)
            elif isinstance(node, ast.Compare):
                get_node(node.left)
                for comparator in node.comparators:
                    get_node(comparator)
            elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
                for element in node.elts:
                    get_node(element)
            elif isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if key is not None:  # 处理dict展开的情况
                        get_node(key)
                    get_node(value)
            elif isinstance(node, ast.Lambda):
                get_node(node.body)
            elif isinstance(node, (ast.If, ast.While)):
                get_node(node.test)
                for stmt in node.body:
                    get_node(stmt)
                if node.orelse:
                    for stmt in node.orelse:
                        get_node(stmt)
            elif isinstance(node, ast.For):
                get_node(node.iter)
                get_node(node.target)
                for stmt in node.body:
                    get_node(stmt)
                if node.orelse:
                    for stmt in node.orelse:
                        get_node(stmt)
            elif isinstance(node, ast.Assert):
                get_node(node.test)
            elif isinstance(node, (ast.Try, ast.ExceptHandler, ast.With)):
                if isinstance(node, ast.Try):
                    for stmt in node.body:
                        get_node(stmt)
                    for handler in node.handlers:
                        get_node(handler)
                    if node.finalbody:
                        for stmt in node.finalbody:
                            get_node(stmt)
                elif isinstance(node, ast.With):
                    for item in node.items:
                        get_node(item.context_expr)
                        if item.optional_vars:
                            get_node(item.optional_vars)
                    for stmt in node.body:
                        get_node(stmt)
                elif isinstance(node, ast.ExceptHandler):
                    for stmt in node.body:
                        get_node(stmt)
            elif isinstance(node, ast.Raise):
                if node.exc:
                    get_node(node.exc)
                if node.cause:
                    get_node(node.cause)
            elif isinstance(node, ast.Await):
                get_node(node.value)
            elif isinstance(node, ast.Return):
                if node.value:
                    get_node(node.value)

        # 遍历AST并查找函数调用和方法调用
        for node in ast.walk(tree):
            get_node(node)
            if isinstance(node, ast.Return):
                break

        return calls

    def _get_attribute_call(self, node: ast.Attribute) -> str:
        """提取属性调用的完整路径"""
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return '.'.join(reversed(parts))
    
    def extract_class_docstring_by_pattern(self, code:str, class_name:str) -> Optional[str]:
        match = re.search(rf'class {class_name}\s*(\([^\)]*\))?\s*:\s*"""([\s\S]*?)"""', code, re.DOTALL)

        if match:
            return match.group(2)
        return None
    
    def extract_function_docstring_by_pattern(self, code:str, function_name:str) -> Optional[str]:
        match = re.search(rf'def {function_name}\s*(\([\s\S]*?\))?\s*:\s*"""([\s\S]*?)"""', code, re.DOTALL)

        if match:
            return match.group(2)
        return None
        
    def analyze_repository(self, root_path: str,repo_root: str) -> Repository:
        
        def analyze_wrapper(file_path):
            return self._analyze_file_for_structure(file_path, repo_root)
        
        """分析整个代码仓库，提取关键结构信息"""
        # 创建仓库对象
        repo_id = f"repo-{uuid.uuid4().hex[:8]}"
        repo_name = os.path.basename(os.path.abspath(root_path))
        repository = Repository(
            id=repo_id,
            name=repo_name,
            url=None,  # 可以通过git配置获取
            description=None  # 可以从README或setup.py获取
        )
        
        # 初始化仓库结构和代码节点列表
        self.repository_structure = RepositoryStructure()
        
        # 获取仓库中的所有Python文件
        python_files = self._get_python_files(root_path)
        print(f"找到 {len(python_files)} 个Python文件\n")

        # 分析文件结构并构建模块树
        root_modules = self._build_module_tree(root_path, python_files,repo_root)
        self.repository_structure.root_modules = root_modules
        print(f"模块树结构已构建\n")

        max_workers = min(32, len(python_files))

        # with ThreadPoolExecutor(max_workers=max_workers) as executor:
        #     # 提交所有任务
        #     futures = {executor.submit(analyze_wrapper, file_path): file_path for file_path in python_files}

        #     # 使用 tqdm 跟踪进度
        #     for future in tqdm(as_completed(futures), total=len(futures), desc="Analyzing files (parallel)"):
        #         try:
        #             future.result()  # 如果你希望捕捉异常，可以在这里处理
        #         except Exception as e:
        #             print(f"Error analyzing {futures[future]}: {e}")

        # # 分析每个文件的代码结构
        for file_path in tqdm(python_files, desc="Analyzing files"):
            self._analyze_file_for_structure(file_path, repo_root)
        
        # 构建代码依赖图
        self.build_dependency_graph(python_files,repo_root)
        print(f"代码依赖图已构建，共有 {len(self.repository_structure.dependency_graph)} 个模块\n")

        # 分析代码元素间的关系
        self._extract_code_relationships()      
        print(f"代码元素间的关系已提取，共有 {len(self.repository_structure.relationships)} 个关系\n")  
        
        # 链接类属性与函数
        self._link_attributes_to_functions()
        print(f"类属性与函数的关系已链接，共有 {len(self.repository_structure.attributes)} 个属性\n")

        # # 链接变量与引用它们的函数
        # self._link_variables_to_references()
        # print(f"变量与引用它们的函数的关系已链接，共有 {len(self.repository_structure.variables)} 个变量\n")
        
        # 生成仓库核心功能概述
        self._summarize_core_functionality()
        print(f"仓库核心功能概述已生成\n")
        
        # 将仓库结构添加到仓库对象中
        repository.structure = self.repository_structure
        
        
        return repository
    
    def _get_python_files(self, root_path: str) -> List[str]:
        """获取给定路径下所有Python文件"""
        python_files = []
        for root, _, files in os.walk(root_path):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        return python_files
    
    def _analyze_file_for_structure(self, file_path: str,repo_root: str):
        """分析单个文件，提取类定义、函数定义和类属性"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.current_content = f.read()
                # current_content = f.read()
            
            tree = ast.parse(self.current_content)
            
            # 提取顶级定义
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    # 处理类定义
                    self._extract_class_definition(node, file_path, self.current_content,repo_root)                    # 处理类中的方法和属性
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            self._extract_function_definition(item, file_path,self.current_content, node,repo_root)
                        elif isinstance(item, ast.Assign):
                            self._extract_class_attributes(item, file_path, node)
                            # 提取类变量
                            self._extract_variables(item, file_path, scope="class", class_name=node.name, function_name=None, repo_root=repo_root)
                
                elif isinstance(node, ast.FunctionDef):
                    # 处理函数
                    self._extract_function_definition(node, file_path, self.current_content, None, repo_root)
                    # 提取函数中的变量
                    self._extract_function_variables(node, file_path, repo_root)
                
                elif isinstance(node, ast.Assign):
                    # 提取全局变量
                    self._extract_variables(node, file_path, scope="global", class_name=None, function_name=None, repo_root=repo_root)
        except SyntaxError:
            print(f"警告：文件 {file_path} 存在语法错误，已跳过")
        except UnicodeDecodeError:
            print(f"警告：文件 {file_path} 编码错误，已跳过")
        except Exception as e:
            print(f"警告：分析文件 {file_path} 时发生错误：{str(e)}")
        finally:
            self.current_content = ""  # 清空当前内容

    def _extract_class_definition(self, node: ast.ClassDef, file_path: str, content: str,repo_root: str):
        """提取类定义信息"""
        docstring = ast.get_docstring(node) or ''
        
        # 创建CodeNode
        file_node = self.analyze_file(file_path,repo_root)
        code_node = None
        if file_node:
            code_node = CodeNode(
                start_line=node.lineno,
                end_line=node.end_lineno,
                belongs_to=file_node,
                relative_function=[],
                code=ast.get_source_segment(content, node)
            )
        
        class_def = ClassDefinition(
            name=node.name,
            docstring=docstring,
            relative_code=code_node,
            methods=[],  # 将在后续处理中填充
            attributes=[]  # 将在后续处理中填充
        )
        
        self.repository_structure.classes.append(class_def)
    
    def _extract_function_definition(self, node: ast.FunctionDef, file_path: str, content: str, current_class: Optional[ast.ClassDef] = None,repo_root: str=None):
        """提取函数/方法定义信息"""
        docstring = ast.get_docstring(node) or ''
        
        # 创建CodeNode
        file_node = self.analyze_file(file_path,repo_root)
        code_node = None
        if file_node:
            code_node = CodeNode(
                start_line=node.lineno,
                end_line=node.end_lineno,
                belongs_to=file_node,
                relative_function=[],
                code=ast.get_source_segment(content, node)
            )
        
        # 提取参数列表
        parameters = []
        for arg in node.args.args:
            parameters.append(arg.arg)
        
        # 提取函数调用
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(self._get_attribute_call(child.func))
        is_method = current_class is not None
        class_name = current_class.name if current_class else None
        func_def = FunctionDefinition(
            name=node.name,
            docstring=docstring,
            relative_code=code_node,
            is_method=is_method,
            class_name=class_name,
            parameters=parameters,
            calls=calls
        )
        
        self.repository_structure.functions.append(func_def)
        
        # 如果是方法，将其添加到对应类的方法列表中
        if is_method:
            for cls in self.repository_structure.classes:
                if cls.name == class_name:
                    cls.methods.append(func_def)
                    break

    def _extract_class_attributes(self, node: ast.Assign, file_path: str, current_class: ast.ClassDef):
        """提取类属性信息"""
        for target in node.targets:
            if isinstance(target, ast.Name):
                attr_name = target.id
                
                # 尝试提取类型提示（如果有）
                type_hint = None
                if hasattr(target, 'annotation') and target.annotation:
                    type_hint = ast.unparse(target.annotation)
                
                attr = ClassAttribute(
                    name=attr_name,
                    class_name=current_class.name,
                    file_path=file_path,
                    type_hint=type_hint
                )
                
                self.repository_structure.attributes.append(attr)
                
                # 将属性添加到对应类的属性列表中
                for cls in self.repository_structure.classes:
                    if cls.name == current_class.name:
                        cls.attributes.append(attr)
                        break
    
    def _link_attributes_to_functions(self):
        """将类属性与维护它们的函数关联起来"""
        for attr in self.repository_structure.attributes:
            for func in self.repository_structure.functions:
                if func.class_name == attr.class_name and func.relative_code:
                    # 直接使用函数的代码内容，无需重新打开文件
                    func_code = func.relative_code.code
                    
                    # 简单检查属性名是否在函数代码中出现
                    # 未来可以使用更复杂的AST分析来精确判断属性访问
                    if f"self.{attr.name}" in func_code:
                        attr.related_functions.append(func.name)
    
    def _summarize_core_functionality(self):
        """根据函数注释和代码结构总结仓库核心功能"""
        # 这里是简化的实现，后期可能需要LLM总结
        
        class_summaries = []
        for cls in self.repository_structure.classes:
            if cls.docstring:
                class_summaries.append(f"{cls.name}: {cls.docstring.strip()}")
        
        function_summaries = []
        for func in self.repository_structure.functions:
            if func.docstring and not func.is_method:  # 只考虑顶级函数
                function_summaries.append(f"{func.name}: {func.docstring.strip()}")
        
        # 组合摘要
        summary = "仓库核心功能:\n"
        
        if class_summaries:
            summary += "\n主要类:\n" + "\n".join(class_summaries) + "\n"
            
        if function_summaries:
            summary += "\n主要函数:\n" + "\n".join(function_summaries)
            
        self.repository_structure.core_functionality = summary

    def _build_module_tree(self, root_path: str, python_files: List[str],repo_root: str) -> List[ModuleNode]:
        """构建模块树结构"""
        # 创建模块树的根节点
        root_modules = []
        
        # 创建缓存避免重复创建模块节点
        module_cache = {}
        
        # 规范化根路径
        root_path = os.path.abspath(root_path)
        
        for file_path in python_files:
            # 获取文件相对于根目录的路径
            rel_path = os.path.relpath(file_path, root_path)
            dir_path = os.path.dirname(rel_path)
            
            # 跳过隐藏文件夹
            if any(part.startswith('.') for part in dir_path.split(os.sep)):
                continue
                
            # 分割路径为模块层次
            if dir_path:
                parts = dir_path.split(os.sep)
            else:
                parts = []
                
            # 创建或更新模块树
            current_modules = root_modules
            current_path = ""
            
            for i, part in enumerate(parts):
                # 构建当前模块路径
                if current_path:
                    current_path = os.path.join(current_path, part)
                else:
                    current_path = part
                    
                # 检查模块是否已存在
                module_node = None
                for mod in current_modules:
                    if mod.name == part:
                        module_node = mod
                        break
                        
                # 如果模块不存在，创建新模块节点
                if not module_node:
                    is_package = os.path.exists(os.path.join(root_path, current_path, '__init__.py'))
                    module_node = ModuleNode(
                        name=part,
                        path=os.path.join(root_path, current_path),
                        is_package=is_package
                    )
                    current_modules.append(module_node)
                    module_cache[current_path] = module_node
                    
                # 更新当前模块列表
                current_modules = module_node.sub_modules
                
            # 将文件添加到最后一级模块
            file_node = self.analyze_file(file_path,repo_root)
            if file_node:
                if dir_path and dir_path in module_cache:
                    module_cache[dir_path].files.append(file_node)
                elif not dir_path and os.path.basename(file_path) != '__init__.py':
                    # 处理根目录下的单独文件
                    is_package = False
                    root_module = ModuleNode(
                        name=os.path.basename(file_path).replace('.py', ''),
                        path=root_path,
                        is_package=is_package
                    )
                    root_module.files.append(file_node)
                    root_modules.append(root_module)
                    
        return root_modules
    
    def _extract_code_relationships(self):
        """提取代码元素间的关系"""
        relationships = []
        
        # 提取继承关系
        for cls in tqdm(self.repository_structure.classes, desc="Analyzing class inheritance"):
            # 使用类定义的代码节点中的内容，而不是重新打开文件
            if hasattr(cls, 'relative_code') and cls.relative_code:
                try:
                    content = cls.relative_code.code
                    tree = ast.parse(content)

                    # 查找类定义节点
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef) and node.name == cls.name:
                            # 提取父类
                            for base in node.bases:
                                if isinstance(base, ast.Name):
                                # 简单的父类名称
                                    parent_class = base.id
                                    relationship = CodeRelationship(
                                        source_type="class",
                                        source_id=cls.name,
                                        target_type="class",
                                        target_id=parent_class,
                                        relationship_type="inherits"
                                    )
                                    relationships.append(relationship)
                                elif isinstance(base, ast.Attribute):
                            # 复杂的父类引用，如 module.Class
                                    parent_class = self._get_attribute_call(base)
                                    relationship = CodeRelationship(
                                        source_type="class",
                                        source_id=cls.name,
                                        target_type="class",
                                        target_id=parent_class,
                                        relationship_type="inherits"
                                    )
                                    relationships.append(relationship)
                except Exception as e:
                    print(f"提取类 {cls.name} 继承关系时出错: {str(e)}")
        # for cls in self.repository_structure.classes:
        #     # 使用类定义的代码节点中的内容，而不是重新打开文件
        #     if hasattr(cls, 'relative_code') and cls.relative_code:
        #         try:
        #             content = cls.relative_code.code
        #             tree = ast.parse(content)
                    
        #             # 查找类定义节点
        #             for node in ast.walk(tree):
        #                 if isinstance(node, ast.ClassDef) and node.name == cls.name:
        #                     # 提取父类
        #                     for base in node.bases:
        #                         if isinstance(base, ast.Name):
        #                             # 简单的父类名称
        #                             parent_class = base.id
        #                             relationship = CodeRelationship(
        #                                 source_type="class",
        #                                 source_id=cls.name,
        #                                 target_type="class",
        #                                 target_id=parent_class,
        #                                 relationship_type="inherits"
        #                             )
        #                             relationships.append(relationship)
        #                         elif isinstance(base, ast.Attribute):
        #                             # 复杂的父类引用，如module.Class
        #                             parent_class = self._get_attribute_call(base)
        #                             relationship = CodeRelationship(
        #                                 source_type="class",
        #                                 source_id=cls.name,
        #                                 target_type="class",
        #                                 target_id=parent_class,
        #                                 relationship_type="inherits"
        #                             )
        #                             relationships.append(relationship)
        #         except Exception as e:
        #             print(f"提取类 {cls.name} 继承关系时出错: {str(e)}")
                
        # 提取函数调用关系
        # for func in self.repository_structure.functions:
        for func in tqdm(self.repository_structure.functions, desc="Analyzing function calls"):
            for call in func.calls:
                # 查找调用的函数是否在已知函数列表中
                target_func = next((f for f in self.repository_structure.functions if f.name == call), None)
                if target_func:
                    relationship = CodeRelationship(
                        source_type="function",
                        source_id=func.name,
                        target_type="function",
                        target_id=call,
                        relationship_type="calls"
                    )
                    relationships.append(relationship)
                    
        self.repository_structure.relationships = relationships
        return relationships
    
    def _extract_variables(self, node: ast.Assign, file_path: str, scope: str, class_name: Optional[str], function_name: Optional[str], repo_root: str):
        """提取变量定义信息"""
        file_node = self.analyze_file(file_path, repo_root)
        
        for target in node.targets:
            # 处理简单变量赋值
            if isinstance(target, ast.Name):
                var_name = target.id
                
                # 检查是否是常量（全大写命名）
                is_constant = var_name.isupper() and "_" in var_name
                
                # 获取变量值的字符串表示
                value = None
                try:
                    value = ast.unparse(node.value)
                except:
                    try:
                        value = str(ast.literal_eval(node.value))
                    except:
                        pass

                # 创建CodeNode
                code_node = None
                if file_node:
                    code_segment = ast.get_source_segment(self.current_content, node)
                    if code_segment:
                        code_node = CodeNode(
                            start_line=node.lineno,
                            end_line=node.end_lineno,
                            belongs_to=file_node,
                            relative_function=[function_name] if function_name else [],
                            code=code_segment
                        )
                
                # 尝试提取类型提示（如果有）
                type_hint = None
                if hasattr(target, 'annotation') and target.annotation:
                    try:
                        type_hint = ast.unparse(target.annotation)
                    except:
                        pass
                
                var_def = VariableDefinition(
                    name=var_name,
                    scope=scope,
                    function_name=function_name,
                    class_name=class_name,
                    type_hint=type_hint,
                    value=value,
                    is_constant=is_constant,
                    relative_code=code_node,
                    references=[]  # 将在后续分析中填充
                )
                
                self.repository_structure.variables.append(var_def)

            # 处理元组赋值，如 a, b = 1, 2
            elif isinstance(target, ast.Tuple):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        var_name = elt.id
                        
                        # 检查是否是常量
                        is_constant = var_name.isupper() and "_" in var_name
                        
                        # 创建CodeNode
                        code_node = None
                        if file_node:
                            code_segment = ast.get_source_segment(self.current_content, node)
                            if code_segment:
                                code_node = CodeNode(
                                    start_line=node.lineno,
                                    end_line=node.end_lineno,
                                    belongs_to=file_node,
                                    relative_function=[function_name] if function_name else [],
                                    code=code_segment
                                )
                        
                        var_def = VariableDefinition(
                            name=var_name,
                            scope=scope,
                            function_name=function_name,
                            class_name=class_name,
                            value=None,  # 元组赋值难以确定具体值
                            is_constant=is_constant,
                            relative_code=code_node,
                            references=[]
                        )
                        
                        self.repository_structure.variables.append(var_def)

    def _extract_function_variables(self, node: ast.FunctionDef, file_path: str, repo_root: str):
        """提取函数中的所有变量"""
        function_name = node.name
        class_name = None
        
        # 查找此函数所属的类（如果有）
        for cls in self.repository_structure.classes:
            for method in cls.methods:
                if method.name == function_name and method.is_method:
                    class_name = cls.name
                    break
            if class_name:
                break
        
        # 递归遍历函数体以查找变量定义
        for item in node.body:
            if isinstance(item, ast.Assign):
                self._extract_variables(item, file_path, "local", class_name, function_name, repo_root)
            elif isinstance(item, ast.For):
                # 处理for循环变量
                if isinstance(item.target, ast.Name):
                    var_name = item.target.id
                    self._add_for_loop_variable(var_name, item, file_path, class_name, function_name, repo_root)
    
    def _add_for_loop_variable(self, var_name: str, node: ast.For, file_path: str, class_name: Optional[str], function_name: str, repo_root: str):
        """添加for循环中的变量"""
        file_node = self.analyze_file(file_path, repo_root)
        
        # 创建CodeNode
        code_node = None
        if file_node:
            code_segment = ast.get_source_segment(self.current_content, node)
            if code_segment:
                code_node = CodeNode(
                    start_line=node.lineno,
                    end_line=node.end_lineno,
                    belongs_to=file_node,
                    relative_function=[function_name] if function_name else [],
                    code=code_segment
                )
        
        var_def = VariableDefinition(
            name=var_name,
            scope="local",
            function_name=function_name,
            class_name=class_name,
            relative_code=code_node,
            references=[]
        )
        
        self.repository_structure.variables.append(var_def)

    def process_variable(self, index: int) :
        """处理单个变量，查找引用它的函数"""
        functions = self.repository_structure.functions
        for func in functions:
            if not func.relative_code or not func.relative_code.code:
                continue

            pattern = r'\b' + re.escape(self.repository_structure.variables[index].name) + r'\b'
            if re.search(pattern, func.relative_code.code):
                self.repository_structure.variables[index].references.append(func.name)

    def _link_variables_to_references(self):
        """链接变量与引用它们的函数"""
        # 为每个变量查找引用它的函数
        # 输出变量数和函数数
        variable_num = len(self.repository_structure.variables)
        print(f"链接变量与引用它们的函数，共有 { variable_num } 个变量，{len(self.repository_structure.functions)} 个函数")

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(self.process_variable, idx) for idx in range(variable_num)]
            for future in tqdm(as_completed(futures), total=len(futures), desc="变量引用链接", unit="var"):
                try:
                    _ = future.result()
                except Exception as e:
                    print(f"[ERROR] 任务异常：{e}")

        # for var in self.repository_structure.variables:
        #     for func in self.repository_structure.functions:
        #         if not func.relative_code or not func.relative_code.code:
        #             continue
                
        #         # 查找变量名在函数代码中的出现
        #         # 使用正则表达式匹配整个单词
        #         pattern = r'\b' + re.escape(var.name) + r'\b'
        #         if re.search(pattern, func.relative_code.code):
        #             var.references.append(func.name)
