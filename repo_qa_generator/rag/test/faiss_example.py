#!/usr/bin/env python3
"""
FAISS优化的RAG系统使用示例
"""

import os
import json
import sys
sys.path.append("/data3/pwh/codeqa")

from repo_qa_generator.rag.func_chunk_rag import RAGFullContextCodeQA

def main():
    """主函数，演示FAISS优化的RAG系统使用"""
    
    # 配置路径
    code_nodes_file = "/data3/pwh/repo_analysis/full_code_for_embedding/flask/flask_code_nodes.json"  # 替换为实际的代码节点文件路径
    save_path = "/data3/pwh/embeddings/tmp/code_nodes_faiss.json"  # 替换为实际的保存路径
    
    # 检查文件是否存在
    if not os.path.exists(code_nodes_file):
        print(f"错误：代码节点文件不存在: {code_nodes_file}")
        return
    
    try:
        # 初始化RAG系统（使用FAISS）
        print("初始化FAISS优化的RAG系统...")
        rag_system = RAGFullContextCodeQA(
            filepath=code_nodes_file,
            save_path=save_path,
            mode="external"  # 使用外部embedding模型
        )
        
        # 获取索引统计信息
        stats = rag_system.get_index_stats()
        print(f"FAISS索引统计信息: {stats}")
        
        # 示例查询
        test_queries = [
            "如何实现用户认证功能？",
            "数据库连接是如何配置的？",
            "API路由是如何定义的？",
            "错误处理机制是什么？"
        ]
        
        for query in test_queries:
            print(f"\n查询: {query}")
            print("-" * 50)
            
            # 查找相关代码
            relevant_code = rag_system.find_relevant_code(query, top_k=3)
            
            if relevant_code:
                print(f"找到 {len(relevant_code)} 个相关代码片段:")
                for i, code_node in enumerate(relevant_code, 1):
                    print(f"\n{i}. 文件: {code_node.belongs_to.file_name}")
                    print(f"   路径: {code_node.belongs_to.upper_path}")
                    print(f"   行数: {code_node.start_line}-{code_node.end_line}")
                    print(f"   代码预览: {code_node.code[:100]}...")
            else:
                print("未找到相关代码")
        
        # 演示添加新代码到索引
        print("\n" + "="*60)
        print("演示：添加新代码到FAISS索引")
        
        # 示例新代码节点
        new_code_nodes = [
            {
                "code": "def new_function():\n    return 'This is a new function'",
                "path": "/example/path",
                "file": "new_file.py",
                "start_line": 1,
                "end_line": 2,
                "type": "function",
                "class_name": None
            }
        ]
        
        # 添加新代码到索引
        rag_system.add_code_to_index(new_code_nodes)
        
        # 再次获取统计信息
        updated_stats = rag_system.get_index_stats()
        print(f"更新后的索引统计信息: {updated_stats}")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
