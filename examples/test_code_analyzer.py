from repo_qa_generator.analyzers.code_analyzer import CodeAnalyzer

def main():
    analyzer = CodeAnalyzer()
    
    # 分析文件
    test_file = "src/examples/test_code.py"
    file_node = analyzer.analyze_file(test_file)
    
    print("文件分析结果:")
    print(f"文件名: {file_node.file_name}")
    print(f"路径: {file_node.upper_path}")
    print(f"模块: {file_node.module}")
    print(f"定义的类: {file_node.define_class}")
    print(f"导入: {file_node.file_imports}")
    
    print("\n代码节点分析:")
    code_nodes = analyzer.extract_code_nodes(test_file)
    for node in code_nodes:
        print(f"\n代码块 ({node.start_line}-{node.end_line}):")
        print(f"相关函数调用: {node.relative_function}")
        print("代码片段:")
        print(node.code)
        print("-" * 50)
    
    # 构建依赖图
    analyzer.build_dependency_graph([test_file])
    print("\n依赖图节点:", analyzer.dependency_graph.nodes())
    print("依赖图边:", analyzer.dependency_graph.edges())

if __name__ == "__main__":
    main() 