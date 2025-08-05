#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import sys
import argparse
from pathlib import Path
import time
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from repo_qa_generator.analyzers.code_analyzer import CodeAnalyzer
from repo_qa_generator.models.data_models import Repository, ModuleNode

def analyze_repository(repo_path: str,repo_root: str):
    """
    分析代码仓库结构并返回Repository对象
    
    Args:
        repo_path: 代码仓库的路径
        
    Returns:
        Repository: 包含仓库结构的对象
    """
    print(f"开始分析仓库: {repo_path}")
    start_time = time.time()
    
    # 创建代码分析器
    analyzer = CodeAnalyzer()
    
    # 分析整个仓库
    repository = analyzer.analyze_repository(repo_path,repo_root)
    
    elapsed_time = time.time() - start_time
    print(f"仓库分析完成，耗时: {elapsed_time:.2f} 秒")
    
    return repository

def save_repository_data(repository: Repository, output_dir: str):
    """
    将仓库数据保存为多种格式
    
    Args:
        repository: 仓库对象
        output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 当前时间作为文件名前缀
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存完整数据为JSON
    full_json_path = os.path.join(output_dir, f"{timestamp}_repo_full.json")
    with open(full_json_path, 'w', encoding='utf-8') as f:
        json.dump(repository.model_dump(), f, ensure_ascii=False, indent=2)
    print(f"完整仓库数据已保存到: {full_json_path}")
    
    # 保存类定义摘要
    classes_json_path = os.path.join(output_dir, f"{timestamp}_classes.json")
    classes_data = [{
        "name": cls.name,
        "docstring": cls.docstring,
        "methods": [m.name for m in cls.methods] if hasattr(cls, 'methods') else [],
        "attributes": [a.name for a in cls.attributes] if hasattr(cls, 'attributes') else [],
        "code_location": {
            "file": cls.relative_code.belongs_to.file_name if cls.relative_code and cls.relative_code.belongs_to else None,
            "path": cls.relative_code.belongs_to.upper_path if cls.relative_code and cls.relative_code.belongs_to else None,
            "start_line": cls.relative_code.start_line if cls.relative_code else None,
            "end_line": cls.relative_code.end_line if cls.relative_code else None
        } if hasattr(cls, 'relative_code') and cls.relative_code else None,
        "code_snippet": cls.relative_code.code[:500] + "..." if hasattr(cls, 'relative_code') and cls.relative_code and len(cls.relative_code.code) > 500 else cls.relative_code.code if hasattr(cls, 'relative_code') and cls.relative_code else None
    } for cls in repository.structure.classes]
    
    with open(classes_json_path, 'w', encoding='utf-8') as f:
        json.dump(classes_data, f, ensure_ascii=False, indent=2)
    print(f"类定义摘要已保存到: {classes_json_path}")
    
    # 保存函数定义摘要
    functions_json_path = os.path.join(output_dir, f"{timestamp}_functions.json")
    functions_data = [{
        "name": func.name,
        "docstring": func.docstring,
        "is_method": func.is_method,
        "class_name": func.class_name,
        "parameters": func.parameters,
        "calls": func.calls,
        "code_location": {
            "file": func.relative_code.belongs_to.file_name if func.relative_code and func.relative_code.belongs_to else None,
            "path": func.relative_code.belongs_to.upper_path if func.relative_code and func.relative_code.belongs_to else None,
            "start_line": func.relative_code.start_line if func.relative_code else None,
            "end_line": func.relative_code.end_line if func.relative_code else None
        } if hasattr(func, 'relative_code') and func.relative_code else None,
        "code_snippet": func.relative_code.code[:500] + "..." if hasattr(func, 'relative_code') and func.relative_code and len(func.relative_code.code) > 500 else func.relative_code.code if hasattr(func, 'relative_code') and func.relative_code else None
    } for func in repository.structure.functions]
    
    with open(functions_json_path, 'w', encoding='utf-8') as f:
        json.dump(functions_data, f, ensure_ascii=False, indent=2)
    print(f"函数定义摘要已保存到: {functions_json_path}")
    
    # 保存代码节点信息
    code_nodes_json_path = os.path.join(output_dir, f"{timestamp}_code_nodes.json")
    code_nodes = []
    
    # 从类和函数中提取代码节点
    for cls in repository.structure.classes:
        if hasattr(cls, 'relative_code') and cls.relative_code:
            code_nodes.append({
                "type": "class",
                "name": cls.name,
                "file": cls.relative_code.belongs_to.file_name if cls.relative_code.belongs_to else None,
                "path": cls.relative_code.belongs_to.upper_path if cls.relative_code.belongs_to else None,
                "start_line": cls.relative_code.start_line,
                "end_line": cls.relative_code.end_line,
                "code": cls.relative_code.code[:1000] + "..." if len(cls.relative_code.code) > 1000 else cls.relative_code.code
            })
    
    for func in repository.structure.functions:
        if hasattr(func, 'relative_code') and func.relative_code:
            code_nodes.append({
                "type": "function",
                "name": func.name,
                "class_name": func.class_name,
                "file": func.relative_code.belongs_to.file_name if func.relative_code.belongs_to else None,
                "path": func.relative_code.belongs_to.upper_path if func.relative_code.belongs_to else None,
                "start_line": func.relative_code.start_line,
                "end_line": func.relative_code.end_line,
                "code": func.relative_code.code[:1000] + "..." if len(func.relative_code.code) > 1000 else func.relative_code.code
            })
    
    with open(code_nodes_json_path, 'w', encoding='utf-8') as f:
        json.dump(code_nodes, f, ensure_ascii=False, indent=2)
    print(f"代码节点信息已保存到: {code_nodes_json_path}")
    
    # 保存模块结构
    modules_json_path = os.path.join(output_dir, f"{timestamp}_modules.json")
    modules_data = _extract_module_structure(repository.structure.root_modules)
    
    with open(modules_json_path, 'w', encoding='utf-8') as f:
        json.dump(modules_data, f, ensure_ascii=False, indent=2)
    print(f"模块结构已保存到: {modules_json_path}")
    
    # 保存代码关系
    relationships_json_path = os.path.join(output_dir, f"{timestamp}_relationships.json")
    with open(relationships_json_path, 'w', encoding='utf-8') as f:
        json.dump([rel.model_dump() for rel in repository.structure.relationships], f, ensure_ascii=False, indent=2)
    print(f"代码关系已保存到: {relationships_json_path}")

def _extract_module_structure(modules: list):
    """递归提取模块结构"""
    result = []
    for module in modules:
        module_data = {
            "name": module.name,
            "is_package": module.is_package,
            "files": [f.file_name for f in module.files],
            "sub_modules": _extract_module_structure(module.sub_modules)
        }
        result.append(module_data)
    return result

def print_repository_summary(repository: Repository):
    """
    打印仓库结构摘要
    
    Args:
        repository: 仓库对象
    """
    print("\n===== 仓库结构摘要 =====")
    print(f"仓库名称: {repository.name}")
    print(f"仓库ID: {repository.id}")
    
    # 输出结构统计
    structure = repository.structure
    print(f"\n总计:")
    print(f"  类定义: {len(structure.classes)}个")
    print(f"  函数定义: {len(structure.functions)}个")
    print(f"  类属性: {len(structure.attributes)}个")
    print(f"  代码关系: {len(structure.relationships)}个")
    
    # 输出模块结构
    print("\n模块结构:")
    for module in structure.root_modules:
        _print_module(module, "  ")
    
    # 输出主要类
    print("\n主要类:")
    for cls in sorted(structure.classes, key=lambda c: len(c.methods) if hasattr(c, 'methods') else 0, reverse=True)[:5]:
        method_count = len(cls.methods) if hasattr(cls, 'methods') else 0
        attr_count = len(cls.attributes) if hasattr(cls, 'attributes') else 0
        print(f"  {cls.name}: {method_count}个方法, {attr_count}个属性")
        if cls.docstring:
            print(f"    文档: {cls.docstring[:100]}..." if len(cls.docstring) > 100 else f"    文档: {cls.docstring}")
    
    # 输出核心功能概述
    if structure.core_functionality:
        print("\n核心功能概述:")
        print(f"  {structure.core_functionality[:500]}..." if len(structure.core_functionality) > 500 else structure.core_functionality)

def _print_module(module, indent=""):
    """递归打印模块结构"""
    print(f"{indent}模块: {module.name} ({'包' if module.is_package else '模块'})")
    
    if module.files:
        print(f"{indent}  {len(module.files)}个文件")
    
    for sub_module in module.sub_modules:
        _print_module(sub_module, indent + "  ")

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="分析代码仓库结构")
    parser.add_argument("--repo_path","-r",default="/data3/pwh/sympy", help="代码仓库的路径")
    parser.add_argument("--output_dir", "-o", default="repo_analysis", help="输出目录")
    
    args = parser.parse_args()
    
    # 分析仓库
    repository = analyze_repository(repo_path=args.repo_path, repo_root="/data3/pwh/sympy")
    
    # 打印摘要
    print_repository_summary(repository)
    
    # 保存数据
    save_repository_data(repository, args.output_dir)

if __name__ == "__main__":
    main() 