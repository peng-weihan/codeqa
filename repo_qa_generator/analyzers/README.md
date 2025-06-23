本模块用于预先提取出所有代码仓库的重要信息，提取思路如下：
仓库整体的功能提取
    - 提取重要信息：仓库骨架-调用语句保留-函数级注释
    - 根据函数注释与仓库骨架理解，总结出仓库核心功能
仓库代码信息整理
    - 提取所有文件和Modules相关信息   
    - 提取所有类定义的相对位置
    - 类名+注释+代码begin->end line
提取所有函数/方法定义的相关位置
    - 函数名+注释+{是否属于类，类名}+代码begin->end line
    - 提取所有类下面的Attributes/Fields的维护信息
    - Attribute名+所属类+维护变量相关函数位置
将依赖关系存储到Analyzer的dependency_graph中
将所有信息存储到RepositoryStructure中
# 代码分析器模块

## 功能概述

代码分析器模块负责分析整个代码仓库的结构，提取类、函数、属性等重要元素，并构建它们之间的关系。主要功能包括：

1. **仓库结构分析**：扫描仓库中的所有Python文件，提取类、函数和属性定义
2. **模块结构构建**：构建仓库的模块树结构，识别包和模块之间的层级关系
3. **代码关系提取**：分析代码元素之间的关系，如继承关系和函数调用关系
4. **依赖图构建**：构建模块间的依赖关系图
5. **源代码关联**：将提取的元素与源代码位置（文件路径和行号）关联

## 使用方法

### 基本用法

```python
from src.analyzers.code_analyzer import CodeAnalyzer

# 创建分析器实例
analyzer = CodeAnalyzer()

# 分析仓库并获取仓库结构
repository = analyzer.analyze_repository('/path/to/your/repo')

# 访问仓库结构
repo_structure = repository.structure

# 访问提取的类定义
for cls in repo_structure.classes:
    print(f"类: {cls.name}, 文档: {cls.docstring}")
    
# 访问提取的函数定义
for func in repo_structure.functions:
    print(f"函数: {func.name}, 调用: {func.calls}")
    
# 访问模块结构
for module in repo_structure.root_modules:
    print(f"模块: {module.name}, 路径: {module.path}")
```

### 保存分析结果

可以将分析结果保存为JSON格式，方便后续使用：

```python
import json

# 将整个仓库结构保存为JSON
with open('repo_analysis.json', 'w', encoding='utf-8') as f:
    json.dump(repository.model_dump(), f, ensure_ascii=False, indent=2)
```

### 提取代码节点

可以提取特定文件中的代码节点（类和函数定义）：

```python
# 提取代码节点
code_nodes = analyzer.extract_code_nodes('/path/to/file.py')

for node in code_nodes:
    print(f"代码片段: {node.start_line}-{node.end_line}")
    print(node.code)
```

## 数据模型

代码分析器使用以下主要数据模型：

1. **Repository**：代码仓库主模型，包含仓库基本信息和结构
2. **RepositoryStructure**：仓库结构模型，包含类、函数、属性等代码元素
3. **ClassDefinition**：类定义模型，包含类名、方法、属性等信息
4. **FunctionDefinition**：函数定义模型，包含函数名、参数、调用关系等
5. **ClassAttribute**：类属性模型，包含属性名称和相关函数
6. **ModuleNode**：模块节点模型，构建仓库的模块树结构
7. **CodeRelationship**：代码关系模型，表示代码元素间的关系

## 示例

完整的使用示例可以参考`examples/repo_parser/repository_analysis_example.py`文件，该示例演示了如何分析代码仓库并输出结构信息。

## 注意事项

1. 分析大型仓库可能需要较长时间，请耐心等待
2. 某些复杂的语法结构可能无法完全解析，会输出警告信息
3. 默认只分析Python文件（.py文件），其他类型的文件会被忽略 