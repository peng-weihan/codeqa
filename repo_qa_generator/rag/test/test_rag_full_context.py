import os
from ..rag_full_context import RAGFullContextCodeQA
from repo_qa_generator.models.data_models import QAPair  # 替换为实际模块路径

def test_rag_full_context_code_qa():
    # 1. 初始化（确保 test_code_nodes.json 路径正确）
    rag_qa = RAGFullContextCodeQA(filepath="/data3/pwh/repo_analysis/flask/flask_code_nodes.json", mode="external")
    
    # 2. 测试查询相关代码查找
    query = "有没有什么类是处理UnexpectedUnicode异常的?"
    relevant_codes = rag_qa.find_relevant_code(query, top_k=5)
    print(f"找到的相关代码数量: {len(relevant_codes)}")
    for code_node in relevant_codes:
        print("相关代码内容:\n", code_node.code)
    
    # 3. 测试生成回答提示
    prompt = rag_qa.make_question_prompt("有没有什么类是处理UnexpectedUnicode异常的?")
    print("生成的提示内容:\n", prompt)
    
    # 4. 测试QA Pair处理
    qa_pair = QAPair(question="有没有什么类是处理UnexpectedUnicode异常的?", answer="", relative_code_list=[])
    updated_qa_pair = rag_qa.process_qa_pair(qa_pair)
    print("生成的回答:\n", updated_qa_pair.answer)

if __name__ == "__main__":
    test_rag_full_context_code_qa()
