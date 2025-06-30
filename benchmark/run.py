from dotenv import load_dotenv
import sys
import os
import json
import asyncio


# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.getcwd(), ''))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"已添加 {project_root} 到Python路径")

import sys
from finetune.create_data import DataCreator
from score.evaluator import QAEvaluator
from repo_qa_generator import generate_questions
from repo_qa_generator.models.data_models import QAPair
from repo_qa_generator.rag.code_qa import RecordedRAGCodeQA
from moatless_qa.moatless_solve import MoatlessSolve
from code_aot.solve import atom_solve
from utils.load import stream_read_qa_json_file
# from finetune.train import ModelFinetuner
model_list_qwen = ["Qwen/Qwen2.5-Coder-0.5B-Instruct","Qwen/Qwen2.5-Coder-1.5B-Instruct","Qwen/Qwen2.5-Coder-3B-Instruct","Qwen/Qwen2.5-Coder-7B-Instruct"]
model_list_llama = ["meta-llama/Llama-3.2-1B-Instruct","meta-llama/Llama-3.2-3B-Instruct","meta-llama/Llama-3.2-70B-Instruct"]

async def main():
    repo_path = sys.argv[1] if len(sys.argv) > 1 else "./dataset/repos/swe-bench_sphinx-doc__sphinx-8551"
    repo_root = sys.argv[2] if len(sys.argv) > 2 else "./dataset/repos/swe-bench_sphinx-doc__sphinx-8551"
    question_store_dir = sys.argv[3] if len(sys.argv) > 3 else "./dataset/seed_questions"
    generate_questions(repo_path, repo_root, question_store_dir)

    # Read questions from question_store_path
    question_store_path = os.path.join(question_store_dir, "generated_questions.json")
    
    # 初始化评估器和模型
    moatless_solve_instance = MoatlessSolve(repo_name=repo_root, repo_path=repo_path)
    rag = RecordedRAGCodeQA(repo_path, repo_root)
    evaluator = QAEvaluator()
    updated_qa_pairs_path = os.path.join(question_store_dir, "updated_questions.json")
    
    # 统计变量
    moatless_score_all = 0
    atom_score_all = 0
    rag_score_all = 0
    question_count = 0
    select_rag_count = 0
    select_atom_count = 0
    select_moatless_count = 0
    
    # 文件模式 - 'w'表示每次覆盖写入，如需追加可改为'a'
    file_mode = 'w'

    # 定义批处理函数
    async def process_batch(batch):
        nonlocal moatless_score_all, atom_score_all, rag_score_all
        nonlocal question_count, select_rag_count, select_atom_count, select_moatless_count
        nonlocal file_mode
        
        processed_batch = []
        
        for qa_pair in batch:
            print(f"处理问题: {qa_pair.question}")
            # 从qa_pair中提取问题(或者其他必要信息)获得三者方式的答案
            moatless_answer = moatless_solve_instance.moatless_solve(qa_pair.question)
            atom_answer = await atom_solve(qa_pair)
            rag_answer = rag.process_qa_pair(qa_pair)
            # 评估每个答案
            moatless_score = evaluator.evaluate_qa(qa_pair, moatless_answer)
            atom_score = evaluator.evaluate_qa(qa_pair, atom_answer)
            rag_score = evaluator.evaluate_qa(qa_pair, rag_answer)

            moatless_score_all += moatless_score
            atom_score_all += atom_score
            rag_score_all += rag_score
            question_count += 1

            if rag_score > moatless_score and rag_score > atom_score:
                select_rag_count += 1
                chosen_answer = rag_answer
            elif atom_score > moatless_score and atom_score > rag_score:
                select_atom_count += 1
                chosen_answer = atom_answer
            else:
                select_moatless_count += 1
                chosen_answer = moatless_answer
                
            updated_qa_pair = QAPair(
                question=qa_pair.question,
                answer=chosen_answer
            )
            processed_batch.append(updated_qa_pair)
        
        # 将当前批次写入文件
        with open(updated_qa_pairs_path, file_mode) as f:
            if file_mode == 'w':
                # 首次写入，创建新文件并开始JSON数组
                json.dump([qa_pair.model_dump() for qa_pair in processed_batch], f, indent=4)
                # 后续追加
                file_mode = 'a'
            else:
                # 追加模式下，需要先读取现有内容，添加新内容后重写
                f.seek(0, 0)  # 回到文件开头
                try:
                    existing_data = json.load(f)
                    combined_data = existing_data + [qa_pair.model_dump() for qa_pair in processed_batch]
                    f.seek(0, 0)  # 回到文件开头
                    f.truncate()  # 清空文件
                    json.dump(combined_data, f, indent=4)
                except json.JSONDecodeError:
                    # 如果文件为空或不是有效JSON
                    json.dump([qa_pair.model_dump() for qa_pair in processed_batch], f, indent=4)
        
        print(f"已将{len(processed_batch)}个问答对写入文件: {updated_qa_pairs_path}")
    # stream_read_and_process_qa_file(question_store_path, await process_batch)
    # 使用自定义处理流程
    with open(question_store_path, 'rb') as f:
        batch = []
        for item in ijson.items(f, 'item'):
            try:
                qa_pair = QAPair.model_validate(item)
                batch.append(qa_pair)
                
                # 当批次达到100个时处理
                if len(batch) >= 100:
                    await process_batch(batch)
                    batch = []
                    
            except Exception as e:
                print(f"验证问题失败: {str(e)[:100]}...")
        
        # 处理最后剩余的批次
        if batch:
            await process_batch(batch)

    # 打印最终统计信息
    if question_count > 0:
        print(f"Moatless 评分: {moatless_score_all / question_count}")
        print(f"Atom 评分: {atom_score_all / question_count}")
        print(f"RAG 评分: {rag_score_all / question_count}")
        print(f"选择 RAG 次数: {select_rag_count}")
        print(f"选择 Atom 次数: {select_atom_count}")
        print(f"选择 Moatless 次数: {select_moatless_count}")
    else:
        print("没有处理任何问题。")

if __name__ == "__main__":
    asyncio.run(main())


