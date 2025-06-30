#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
示例脚本：演示如何使用增强的仓库索引功能
"""

import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from moatless_qa.index import CodeIndex
from moatless_qa.benchmark.swebench import create_repository, create_index


def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='代码仓库索引工具')
    parser.add_argument('--repo-url', help='Git仓库URL')
    parser.add_argument('--repo-path', help='本地仓库路径')
    parser.add_argument('--instance-id', help='SWE-bench实例ID')
    parser.add_argument('--commit', help='Git提交哈希')
    parser.add_argument('--index-name', help='索引名称')
    parser.add_argument('--index-dir', help='索引存储目录')
    parser.add_argument('--repo-dir', help='仓库基础目录')
    parser.add_argument('--force-rebuild', action='store_true', help='强制重建索引')
    parser.add_argument('--query', help='执行语义搜索的查询')
    parser.add_argument('--flag',help='索引是否已经存在')
    args = parser.parse_args()
    
    # 设置环境变量
    if args.index_dir:
        os.environ['MOATLESS_INDEX_DIR'] = args.index_dir
    if args.repo_dir:
        os.environ['REPO_DIR'] = args.repo_dir
    
    # 确定索引方法
    if args.flag == 'True' and args.index_name and args.index_dir:
        # 如果索引已经存在，直接加载索引
        print(f"从本地索引目录 {args.index_dir} 加载索引 {args.index_name}")
        code_index = CodeIndex.from_index_name(
            index_name=args.index_name,
            index_store_dir=args.index_dir,
            file_repo=create_repository(repo_path=args.repo_path)
        )
    if args.instance_id:
        # 使用SWE-bench实例ID
        print(f"从SWE-bench实例 {args.instance_id} 创建索引")
        code_index = CodeIndex.from_index_name(
            index_name=args.instance_id,
            file_repo=create_repository(instance_id=args.instance_id)
        )
    elif args.repo_url or args.repo_path:
        # 使用仓库URL或路径
        source = args.repo_url or args.repo_path
        print(f"从仓库源 {source} 创建索引")
        code_index = CodeIndex.from_repository(
            repo_url=args.repo_url,
            repo_path=args.repo_path,
            commit=args.commit,
            index_name=args.index_name,
            force_rebuild=args.force_rebuild
        )
    else:
        parser.error("必须提供--repo-url、--repo-path或--instance-id参数之一")
        return
    
    # 如果提供了查询，执行语义搜索
    if args.query:
        print(f"执行查询: {args.query}")
        # results = code_index.find_by_name(args.query)
        results = code_index.find_class(class_name=args.query)
        # results = code_index.find_function(function_name=args.query)
        print("\n查询结果:")
        print(results)
        # for i, hit in enumerate(results.hits[:5], 1):  # 只显示前5个结果
        #     print(f"\n{i}. 文件: {hit.file_path}")
        #     # print(f"   类型: {hit.match_type}")
        #     # print(f"   相似度: {hit.score:.4f}")
        #     print(f"   代码片段:")
        #     for span in hit.spans:
        #         print(f"  {span.span_id, span.rank, span.tokens}")
        #     # print("   " + "\n   ".join(hit.content.split("\n")[:5]) + "...")  # 只显示前5行
    
    print("\n索引信息:")
    print(f"索引名称: {code_index._index_name}")
    print(f"仓库目录: {code_index._file_repo.repo_dir}")


if __name__ == "__main__":
    setup_logging()
    sys.argv = [
    'example_repo_index.py',
    '--repo-path', '/home/stu/Desktop/my_codeqa/djongo',
    '--index-dir', '/home/stu/Desktop/my_codeqa/codeqa/dataset/index_store',
    '--index-name', 'djongo',
    '--query', 'ModelAdmin',
    ]
    # sys.argv = [
    # 'example_repo_index.py',
    # '--index-dir', '/home/stu/Desktop/my_codeqa/codeqa/dataset/index_store',
    # '--index-name', 'sphinx-doc__sphinx-8551',
    # '--repo-path', '/home/stu/Desktop/my_codeqa/codeqa/dataset/repos/swe-bench_sphinx-doc__sphinx-8551',
    # '--flag', 'True',
    # '--query', 'verify_needs_extensions',
    # ]
    main()