import os
import sys
import json
from pathlib import Path
import time
# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))
from repo_qa_generator.question_generators.direct_qa_generator_v2 import DirectQAGeneratorV2
from repo_qa_generator.question_generators.qa_generate_agent_v2 import AgentQAGeneratorV2
from repo_qa_generator.analyzers.code_analyzer import CodeAnalyzer

import argparse
def main():
    parser = argparse.ArgumentParser(description="提取代码仓库中的所有代码节点")
    # parser.add_argument("--repo_path","-r",default="/Users/xinyun/Programs/django/django/core", help="代码仓库的路径")
    parser.add_argument("--repo_path", "-r",default="/home/stu/Desktop/my_codeqa/codeqa/dataset/repos/swe-bench_sphinx-doc__sphinx-8551", help="代码仓库的路径")
    parser.add_argument("--output-dir", "-o", default="/home/stu/Desktop/my_codeqa/codeqa/dataset/concrete_questions", help="输出目录")
    parser.add_argument("--batch-size", "-b", type=int, default=20, help="每批写入的问题数量")
    
    args = parser.parse_args()
    path = args.repo_path
    output_dir = args.output_dir
    batch_size = args.batch_size

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    output_path_direct = os.path.join(output_dir, "generated_questions_moatless_direct_sphinx-8551.json")
    output_path_agent = os.path.join(output_dir, "generated_questions_moatless_agent.jsonl")
    # 分析代码仓库
    start_time = time.perf_counter()
    analyzer = CodeAnalyzer()
    repo = analyzer.analyze_repository(path, project_root) 
    end_time = time.perf_counter()

    # 获取具体的问题
    qa_generator = DirectQAGeneratorV2()
    qa_pairs = qa_generator.generate_questions(repo.structure)
    print(f"问题生成完成，共生成 {len(qa_pairs)} 个问题\n")
    
    # 分批写入文件
    batch_count = 0
    qa_batch = []
    
    # 打开文件一次，使用列表方式写入JSON
    with open(output_path_direct, 'w', encoding='utf-8') as f:
        # 写入JSON数组开始
        f.write("[\n")
        
        first_item = True
        for qa in qa_pairs:
            qa_batch.append(qa.model_dump())
            batch_count += 1
            
            # 当达到批次大小时，写入文件
            if batch_count >= batch_size:
                for item in qa_batch:
                    if not first_item:
                        f.write(",\n")
                    else:
                        first_item = False
                    json.dump(item, f, ensure_ascii=False)
                
                qa_batch = []
                batch_count = 0
        
        # 处理最后一批数据
        for item in qa_batch:
            if not first_item:
                f.write(",\n")
            else:
                first_item = False
            json.dump(item, f, ensure_ascii=False)
        
        # 写入JSON数组结束
        f.write("\n]")
    
    print(f"基于规则的问题生成完成，已保存到 {output_path_direct}")
    print(f"代码仓库分析用时 {end_time - start_time:.2f} 秒\n")

    # qa_agent = AgentQAGeneratorV2()

    # qa_pairs_llm = qa_agent.generate_questions(repo.structure)

    # print(f"基于LLM的问题生成完成，已保存到 {output_path_agent}")
    
if __name__ == "__main__":
    main()