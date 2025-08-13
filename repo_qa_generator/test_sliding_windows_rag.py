#!/usr/bin/env python3
"""
SlidingWindowsRAG 系统测试脚本

这个脚本用于测试SlidingWindowsRAG系统的基本功能
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag.sliding_windows_rag import SlidingWindowsRAG
from dotenv import load_dotenv

load_dotenv()

def test_basic_functionality():
    """测试基本功能"""
    print("=== 测试SlidingWindowsRAG基本功能 ===")
    
    # 检查环境变量
    voyage_api_key = os.getenv("VOYAGE_API_KEY")
    if not voyage_api_key:
        print("❌ 错误：未设置VOYAGE_API_KEY环境变量")
        print("请运行：export VOYAGE_API_KEY='your_api_key'")
        return False
    
    print("✅ Voyage API密钥已设置")
    
    # 测试仓库路径
    repo_path = "/data3/pwh/swebench-repos/flask"
    if not os.path.exists(repo_path):
        print(f"❌ 错误：仓库路径不存在: {repo_path}")
        return False
    
    print(f"✅ 仓库路径存在: {repo_path}")
    
    try:
        # 初始化RAG系统（使用较小的chunk_size进行快速测试）
        print("\n🔄 初始化SlidingWindowsRAG...")
        rag = SlidingWindowsRAG(
            repo_path=repo_path,
            voyage_api_key=voyage_api_key,
            chunk_size=500,  # 较小的chunk_size用于快速测试
            overlap=100,
            embedding_model="voyage-code-3",
            faiss_index_path="/data3/pwh/embeddings"
        )
        
        print("✅ SlidingWindowsRAG初始化成功")
        
        # 获取统计信息
        stats = rag.get_index_stats()
        print(f"✅ 索引统计信息:")
        print(f"   - 总chunks数量: {stats['total_chunks']}")
        print(f"   - 向量维度: {stats['dimension']}")
        print(f"   - 处理的文件数: {stats['files_processed']}")
        
        # 测试搜索功能
        print("\n🔄 测试搜索功能...")
        test_query = "sliding windows"
        search_results = rag.search(test_query, top_k=5)
        
        if search_results:
            print(f"✅ 搜索成功，找到 {len(search_results)} 个结果")
            for i, result in enumerate(search_results, 1):
                print(f"   结果 {i}: 相似度 {result['similarity']:.4f}")
        else:
            print("⚠️  搜索未返回结果")
        
        # 测试问答功能
        print("\n🔄 测试问答功能...")
        from repo_qa_generator.models.data_models import QAPair
        
        test_question = "What is Flask's blueprint system and how does it help organize large applications into modular components？"
        qa_pair = QAPair(question=test_question)
        
        result = rag.process_qa_pair(qa_pair)
        if result.answer:
            print(f"✅ 问答功能正常，生成了回答")
            print(f"   问题: {test_question}")
            print(f"   回答: {result.answer[:200]}...")
        else:
            print("⚠️  问答功能未生成回答")
        
        print("\n✅ 所有基本功能测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_index_persistence():
    """测试索引持久化功能"""
    print("\n=== 测试索引持久化功能 ===")
    
    voyage_api_key = os.getenv("VOYAGE_API_KEY")
    if not voyage_api_key:
        print("❌ 跳过持久化测试：未设置API密钥")
        return False
    
    repo_path = "/data3/pwh/codeqa"
    index_path = "test_persistence_index"
    
    try:
        # 第一次初始化（构建索引）
        print("🔄 第一次初始化（构建索引）...")
        rag1 = SlidingWindowsRAG(
            repo_path=repo_path,
            voyage_api_key=voyage_api_key,
            chunk_size=300,
            overlap=50,
            faiss_index_path=index_path
        )
        
        stats1 = rag1.get_index_stats()
        print(f"✅ 第一次构建完成，chunks数量: {stats1['total_chunks']}")
        
        # 第二次初始化（加载索引）
        print("🔄 第二次初始化（加载索引）...")
        rag2 = SlidingWindowsRAG(
            repo_path=repo_path,
            voyage_api_key=voyage_api_key,
            chunk_size=300,
            overlap=50,
            faiss_index_path=index_path
        )
        
        stats2 = rag2.get_index_stats()
        print(f"✅ 第二次加载完成，chunks数量: {stats2['total_chunks']}")
        
        # 验证一致性
        if stats1['total_chunks'] == stats2['total_chunks']:
            print("✅ 索引持久化功能正常")
            return True
        else:
            print("❌ 索引持久化功能异常，chunks数量不一致")
            return False
            
    except Exception as e:
        print(f"❌ 持久化测试失败: {e}")
        return False

def main():
    """主函数"""
    print("开始SlidingWindowsRAG系统测试...\n")
    
    # 测试基本功能
    basic_test_passed = test_basic_functionality()
    
    # 测试持久化功能
    persistence_test_passed = test_index_persistence()
    
    # 总结
    print("\n=== 测试总结 ===")
    print(f"基本功能测试: {'✅ 通过' if basic_test_passed else '❌ 失败'}")
    print(f"持久化测试: {'✅ 通过' if persistence_test_passed else '❌ 失败'}")
    
    if basic_test_passed and persistence_test_passed:
        print("\n🎉 所有测试通过！SlidingWindowsRAG系统工作正常。")
    else:
        print("\n⚠️  部分测试失败，请检查配置和依赖。")

if __name__ == "__main__":
    main()
