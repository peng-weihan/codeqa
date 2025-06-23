from tqdm import tqdm
from repo_qa_generator import CodeAnalyzer, DirectQAGenerator, AgentQAGenerator
import os
import json

def generate_questions(repo_path: str, repo_root: str, question_store_dir: str, batch_size: int = 100):
    """
    Analyze repository structure and generate questions
    """
    print("Analyzing repository structure...")
    analyzer = CodeAnalyzer()
    repository = analyzer.analyze_repository(repo_path, repo_root)
    print(f"Repository analysis complete. Found {len(repository.structure.classes)} classes and {len(repository.structure.functions)} functions")
    
    print("\nStarting question generation...")
    direct_qa_generator = DirectQAGenerator(questions_dir=question_store_dir)
    agent_qa_generator = AgentQAGenerator(questions_dir=question_store_dir)
    
    # 确保输出目录存在
    os.makedirs(question_store_dir, exist_ok=True)
    question_store_path = os.path.join(question_store_dir, "generated_questions.json")
    
    # 使用列表方式写入JSON，避免一次性加载所有问题到内存
    with open(question_store_path, 'w', encoding='utf-8') as f:
        # 写入JSON数组开始
        f.write("[\n")
        
        # 跟踪是否是第一个元素
        first_item = True
        total_questions = 0
        
        # 使用tqdm显示生成器进度
        generators = [
            ("Direct Question Generator", lambda: direct_qa_generator.generate_questions(analyzer.repository_structure)),
            ("Agent Question Generator", lambda: agent_qa_generator.generate_questions(analyzer.repository_structure))
        ]
        
        for gen_name, gen_func in tqdm(generators, desc="Question Generator Progress"):
            print(f"\nRunning {gen_name}...")
            
            # 分批处理生成的问题
            batch_count = 0
            qa_batch = []
            
            for question in gen_func():
                # 准备写入当前问题
                if not first_item:
                    f.write(",\n")
                else:
                    first_item = False
                
                # 直接写入文件，不存储在内存中
                f.write(question.model_dump_json())
                
                total_questions += 1
                
                # 每批次后刷新到磁盘
                if total_questions % batch_size == 0:
                    f.flush()
                    print(f"已写入 {total_questions} 个问题到文件")
            
            print(f"{gen_name}: Generated and wrote questions")
        
        # 写入JSON数组结束
        f.write("\n]")
            
    print(f"\n总共生成并保存了 {total_questions} 个问题到 {question_store_path}")
    print("问题生成和保存完成！")