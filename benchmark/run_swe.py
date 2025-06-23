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
from repo_qa_generator.models.data_models import QAPair
from moatless_qa.moatless_solve import MoatlessSolve

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
            try:
                # 使用ijson流式解析JSON数组
                parser = ijson.items(f, 'item')
                for item in parser:
                    try:
                        qa_pair = QAPair.model_validate(item)
                        batch.append(qa_pair)
                        
                        # 当批次达到指定大小时处理
                        if len(batch) >= batch_size:
                            batch_count += 1
                            print(f"处理第{batch_count}批数据，共{len(batch)}个问题...")
                            try:
                                process_func(batch)
                                total_processed += len(batch)
                            except Exception as e:
                                print(f"处理批次时出错: {e}")
                                # 继续处理下一批，而不是完全终止
                            finally:
                                batch = []  # 无论处理成功与否，都清空当前批次
                            
                    except Exception as e:
                        print(f"验证问题失败: {str(e)[:100]}...")
                
                # 处理最后剩余的批次
                if batch:
                    batch_count += 1
                    print(f"处理最后一批数据，共{len(batch)}个问题...")
                    try:
                        process_func(batch)
                        total_processed += len(batch)
                    except Exception as e:
                        print(f"处理最后批次时出错: {e}")
            except ijson.JSONError as e:
                print(f"JSON解析错误: {e}")
            except Exception as e:
                print(f"读取JSON时发生错误: {e}")
    except FileNotFoundError:
        print(f"文件不存在: {file_path}")
    except PermissionError:
        print(f"没有权限读取文件: {file_path}")
    except Exception as e:
        print(f"读取文件出错: {e}")
        
    print(f"成功处理了 {total_processed} 个问答对")
    return total_processed

def process_batch_swe(batch, moatless_solve_instance, evaluator, results_file):
    """
    处理一批问题，只使用moatless解决并评估
    """
    processed_batch = []
    total_score = 0
    count = 0
    
    for qa_pair in batch:
        print(f"处理问题: {qa_pair.question}")
        moatless_answer = moatless_solve_instance.moatless_solve(qa_pair.question)
        moatless_score = evaluator.evaluate_qa(qa_pair, moatless_answer)
        total_score += moatless_score
        count += 1
        
        result = {
            "question": qa_pair.question,
            "moatless_answer": moatless_answer,
            "moatless_score": moatless_score
        }
        
        # 将结果写入文件
        with open(results_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        print(f"问题评分: {moatless_score}")
        
        processed_batch.append(QAPair(
            question=qa_pair.question,
            answer=moatless_answer
        ))
    
    return processed_batch, total_score, count

def main():
    repo_path = sys.argv[1] if len(sys.argv) > 1 else "./dataset/repos/swe-bench_sphinx-doc__sphinx-8551"
    repo_root = sys.argv[2] if len(sys.argv) > 2 else "./dataset/repos/swe-bench_sphinx-doc__sphinx-8551"
    question_store_dir = sys.argv[3] if len(sys.argv) > 3 else "./dataset/seed_questions"
    
    # 设置输入和输出文件路径
    questions_path = os.path.join(question_store_dir, "generated_questions.json")
    results_file = os.path.join(question_store_dir, "moatless_results.jsonl")
    
    # 初始化评估器和模型
    moatless_solve_instance = MoatlessSolve(repo_name=repo_root, repo_path=repo_path)
    evaluator = QAEvaluator()
    
    # 清空结果文件
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("")
    
    # 统计变量
    question_count = 0
    moatless_score_all = 0
    
    # 定义批处理函数包装器
    def process_batch_wrapper(batch):
        nonlocal moatless_score_all, question_count
        
        processed_batch, batch_score, batch_count = process_batch_swe(batch, moatless_solve_instance, evaluator, results_file)
        
        # 直接累加分数和问题数
        moatless_score_all += batch_score
        question_count += batch_count
    
    # 处理问题文件
    stream_read_and_process_qa_file(questions_path, process_batch_wrapper)
    
    # 打印并记录最终统计信息
    if question_count > 0:
        avg_score = moatless_score_all / question_count
        print(f"Moatless 平均评分: {avg_score}")
        
        # 将总结果写入文件
        summary = {
            "total_questions": question_count,
            "moatless_average_score": avg_score
        }
        
        with open(os.path.join(question_store_dir, "moatless_summary.json"), 'w', encoding='utf-8') as f:
            json.dump(summary, ensure_ascii=False, indent=2, fp=f)
    else:
        print("没有处理任何问题。")

if __name__ == "__main__":
    main()


