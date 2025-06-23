#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
提取仓库中的所有代码节点
这个脚本会分析代码仓库并提取所有的代码节点（类和函数定义），
并将它们保存为JSON文件，同时按类型分别保存
"""

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

def extract_code_nodes(repo_path: str, output_dir: str):
    """
    提取仓库中的所有代码节点
    
    Args:
        repo_path: 代码仓库的路径
        output_dir: 输出目录
    """
    print(f"开始分析仓库: {repo_path}")
    start_time = time.time()
    
    # 创建代码分析器
    analyzer = CodeAnalyzer()
    
    # 分析整个仓库
    repository = analyzer.analyze_repository(repo_path,repo_path)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 当前时间作为文件名前缀
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 分别提取类定义和函数定义
    class_nodes = []
    function_nodes = []
    
    for cls in repository.structure.classes:
        if hasattr(cls, 'relative_code') and cls.relative_code:
            class_nodes.append({
                "type": "class",
                "name": cls.name,
                "file": cls.relative_code.belongs_to.file_name if cls.relative_code.belongs_to else None,
                "path": cls.relative_code.belongs_to.upper_path if cls.relative_code.belongs_to else None,
                "module": cls.relative_code.belongs_to.module if cls.relative_code.belongs_to else None,
                "start_line": cls.relative_code.start_line,
                "end_line": cls.relative_code.end_line,
                "docstring": cls.docstring,
                "methods_count": len(cls.methods) if hasattr(cls, 'methods') else 0,
                "attributes_count": len(cls.attributes) if hasattr(cls, 'attributes') else 0,
                "code_length": len(cls.relative_code.code),
                "code": cls.relative_code.code[:500] + "..." if len(cls.relative_code.code) > 500 else cls.relative_code.code
            })
    
    for func in repository.structure.functions:
        if hasattr(func, 'relative_code') and func.relative_code:
            function_nodes.append({
                "type": "function",
                "name": func.name,
                "is_method": func.is_method,
                "class_name": func.class_name,
                "file": func.relative_code.belongs_to.file_name if func.relative_code.belongs_to else None,
                "path": func.relative_code.belongs_to.upper_path if func.relative_code.belongs_to else None,
                "module": func.relative_code.belongs_to.module if func.relative_code.belongs_to else None,
                "start_line": func.relative_code.start_line,
                "end_line": func.relative_code.end_line,
                "docstring": func.docstring,
                "parameters": func.parameters,
                "calls": func.calls,
                "code_length": len(func.relative_code.code),
                "code": func.relative_code.code[:500] + "..." if len(func.relative_code.code) > 500 else func.relative_code.code
            })
    
    # 保存类定义
    classes_path = os.path.join(output_dir, f"{timestamp}_classes.json")
    with open(classes_path, 'w', encoding='utf-8') as f:
        json.dump(class_nodes, f, ensure_ascii=False, indent=2)
    print(f"类定义已保存到: {classes_path}")
    
    # 保存函数定义
    functions_path = os.path.join(output_dir, f"{timestamp}_functions.json")
    with open(functions_path, 'w', encoding='utf-8') as f:
        json.dump(function_nodes, f, ensure_ascii=False, indent=2)
    print(f"函数定义已保存到: {functions_path}")
    
    elapsed_time = time.time() - start_time
    print(f"代码节点提取完成，耗时: {elapsed_time:.2f} 秒")

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="提取代码仓库中的所有代码节点")
    parser.add_argument("--repo_path","-r",default="/Users/xinyun/Programs/django/django/core", help="代码仓库的路径")
    parser.add_argument("--output-dir", "-o", default="code_nodes", help="输出目录")
    
    args = parser.parse_args()
    
    # 执行提取
    extract_code_nodes(args.repo_path, args.output_dir)

if __name__ == "__main__":
    main() 