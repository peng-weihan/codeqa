import os
import sys
import json
from pathlib import Path
# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))
from repo_qa_generator.question_generators.direct_qa_generator import DirectQAGenerator
from repo_qa_generator.analyzers.code_analyzer import CodeAnalyzer

import argparse
def main():
    parser = argparse.ArgumentParser(description="提取代码仓库中的所有代码节点")
    # parser.add_argument("--repo_path","-r",default="/Users/xinyun/Programs/django/django/core", help="代码仓库的路径")
    parser.add_argument("--repo_path", "-r",default="/home/stu/Desktop/my_codeqa/codeqa/moatless_qa", help="代码仓库的路径")
    parser.add_argument("--output-dir", "-o", default="/home/stu/Desktop/my_codeqa/codeqa/dataset/seed_questions", help="输出目录")
    parser.add_argument("--batch-size", "-b", type=int, default=100, help="每批写入的问题数量")
    
    args = parser.parse_args()
    path = args.repo_path
    output_dir = args.output_dir
    batch_size = args.batch_size

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "generated_questions_moatless.json")

    # 分析代码仓库
    analyzer = CodeAnalyzer()
    repo = analyzer.analyze_repository(path, project_root) 

    # 获取具体的问题
    qa_generator = DirectQAGenerator()
    qa_pairs = qa_generator.generate_questions(repo.structure)
    
    # 分批写入文件
    batch_count = 0
    qa_batch = []
    
    # 打开文件一次，使用列表方式写入JSON
    with open(output_path, 'w', encoding='utf-8') as f:
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
    
    print(f"问题生成完成，已保存到 {output_path}")


if __name__ == "__main__":
    main()