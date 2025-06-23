from dotenv import load_dotenv
import sys
import os
import json

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.getcwd(), ''))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"已添加 {project_root} 到Python路径")

from score.evaluator import QAEvaluator
from repo_qa_generator.models.data_models import QAPair,ResultPair
from repo_qa_generator.rag.code_qa import RecordedRAGCodeQA
import logging
from datetime import datetime

# 配置日志记录
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"run_rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

import ijson  # 需要先安装: pip install ijson

def stream_read_and_process_qa_file(file_path, process_func, batch_size=100):
    """
    流式读取大型JSON数组文件并批量处理
    """
    batch = []
    batch_count = 0
    total_processed = 0
    
    try:
        with open(file_path, 'rb') as f:
            # 使用ijson流式解析JSON数组
            for item in ijson.items(f, 'item'):
                try:
                    qa_pair = QAPair.model_validate(item)
                    batch.append(qa_pair)
                    
                    # 当批次达到指定大小时处理
                    if len(batch) >= batch_size:
                        batch_count += 1
                        print(f"处理第{batch_count}批数据，共{len(batch)}个问题...")
                        process_func(batch)
                        total_processed += len(batch)
                        batch = []
                        
                except Exception as e:
                    print(f"验证问题失败: {str(e)[:100]}...")
            
            # 处理最后剩余的批次
            if batch:
                batch_count += 1
                print(f"处理最后一批数据，共{len(batch)}个问题...")
                process_func(batch)
                total_processed += len(batch)
                
    except Exception as e:
        print(f"读取文件出错: {e}")
        
    print(f"成功处理了 {total_processed} 个问答对")
    return total_processed

def process_batch_rag(batch, rag: RecordedRAGCodeQA, evaluator: QAEvaluator, results_file):
    """
    处理一批问题，只使用RAG解决并评估
    """
    records = []
    for qa_pair in batch:
        print(f"处理问题: {qa_pair.question}")
        rag_qa_pair = rag.process_qa_pair(qa_pair)
        rag_score,score_reasoning = evaluator.evaluate_qa(qa_pair, rag_qa_pair)
        result_json = json.loads(rag_qa_pair.answer)
        result_pair = ResultPair.model_validate(result_json)


        result = {
            "question": qa_pair.question,
            "rag_answer": result_pair.answer,
            "rag_ground_truth": result_pair.ground_truth,
            "rag_thought": result_pair.thought,
            "rag_score": rag_score
        }
        
        # 将结果写入文件
        with open(results_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        records.append({
            "question": qa_pair.question,
            "rag_score": rag_score,
            "score_reasoning": score_reasoning
        })
    return records
def main():
    repo_path = sys.argv[1] if len(sys.argv) > 1 else "./dataset/repos/swe-bench_sphinx-doc__sphinx-8551"
    repo_root = sys.argv[2] if len(sys.argv) > 2 else "./dataset/repos/swe-bench_sphinx-doc__sphinx-8551"
    question_store_dir = sys.argv[3] if len(sys.argv) > 3 else "./dataset/seed_questions"
    res_store_dir = sys.argv[4] if len(sys.argv) > 4 else "./dataset/generated_qa"
    # 初始化评估器和模型
    rag = RecordedRAGCodeQA(repo_path, repo_root)
    evaluator = QAEvaluator()
    
    # 设置输入和输出文件路径
    questions_path = os.path.join(question_store_dir, "generated_questions.json")
    results_file = os.path.join(res_store_dir, "rag_results.jsonl")
    
    # 清空结果文件
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("")
    
    # 统计变量
    question_count = 0
    rag_score_all = 0
    
    # 定义批处理函数包装器
    def process_batch_wrapper(batch):
        nonlocal rag_score_all, question_count
        
        records = process_batch_rag(batch, rag, evaluator, results_file)
        for record in records:
            rag_score_all += record["rag_score"]
            question_count += 1
        with open("records.json", "a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    # 处理问题文件
    stream_read_and_process_qa_file(questions_path, process_batch_wrapper)
    
    # 打印并记录最终统计信息
    if question_count > 0:
        avg_score = rag_score_all / question_count
        print(f"RAG 平均评分: {avg_score}")
        
        # 将总结果写入文件
        summary = {
            "total_questions": question_count,
            "rag_average_score": avg_score
        }
        
        with open(os.path.join(question_store_dir, "rag_summary.json"), 'w', encoding='utf-8') as f:
            json.dump(summary, ensure_ascii=False, indent=2, fp=f)
    else:
        print("没有处理任何问题。")

if __name__ == "__main__":
    main()


