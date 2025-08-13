import os
import openai
import json
import concurrent.futures
from typing import Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("AIHUBMIX_API_KEY")

from openai import OpenAI

client = OpenAI(api_key=api_key, base_url="https://aihubmix.com/v1")

def score_answer(question, reference, candidate):
    # ... existing code ...
    prompt = f"""You are a professional evaluator. Please rate the candidate answer against the reference answer based on the following five criteria:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋  YOUR TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Evaluation Criteria and Scoring Guidelines (each scored 0 to 10):

1. Accuracy:
  9-10 — Completely accurate; core points and details are correct with no ambiguity.
  7-8  — Mostly accurate; only minor details are slightly incorrect or loosely expressed.
  5-6  — Partially accurate; some errors or omissions, but main points are generally correct.
  3-4  — Several errors or ambiguities that affect the understanding of core information.
  0-2  — Serious errors; misleading or fails to convey key information.

2. Completeness:
  9-10 — Covers all key points from the reference answer without omission.
  7-8  — Covers most key points; only minor non-critical information missing.
  5-6  — Missing several key points; content is somewhat incomplete.
  3-4  — Important information largely missing; content is one-sided.
  0-2  — Covers very little or irrelevant information; seriously incomplete.

3. Clarity of Expression:
  9-10 — Fluent language; clear and precise expression; easy to understand.
  7-8  — Mostly fluent; some expressions slightly unclear or not concise.
  5-6  — Expression somewhat awkward; some ambiguity or lack of fluency.
  3-4  — Language obscure; sentences are not smooth; hinders understanding.
  0-2  — Expression confusing; very difficult to understand.

4. Relevance:
  9-10 — Content fully focused on the question topic; no irrelevant information.
  7-8  — Mostly focused; only minor irrelevant or peripheral information.
  5-6  — Topic not sufficiently focused; contains considerable off-topic content.
  3-4  — Content deviates from topic; includes excessive irrelevant information.
  0-2  — Majority of content irrelevant to the question.

5. Logical Structure:
  9-10 — Clear and reasonable structure; well-organized and coherent.
  7-8  — Generally reasonable structure; mostly clear organization; minor misarrangements.
  5-6  — Loose structure; logical jumps; organization is average.
  3-4  — Disorganized structure; lacks logical order; difficult to follow.
  0-2  — No clear structure; logic is chaotic.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📥  INPUT STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Question:
{question}

Reference Answer:
{reference}

Candidate Answer:
{candidate}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📤  OUTPUT REQUIREMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Please output only one integer score between 0 and 50 representing the sum score of the 5 angles. No explanation or other content is needed.
"""

    try:
        response = client.chat.completions.create(
            model="DeepSeek-V3",
            messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
            ],
            stream=False
        )
        score_str = response.choices[0].message.content.strip()
        print(f"评分结果：{score_str}")
        try:
            score = int(score_str)
            if 0 <= score <= 50:
                return score
            else:
                return None
        except:
            return None
    except Exception as e:
        print(f"评分出错: {e}")
        return None

def compare_answers_pairwise(question, reference, rag, mcts):
    # ... existing code ...
    prompt = f"""
你是一个专业评审专家，需要根据问题和参考答案，判断两个候选答案中哪个更好。

请严格依据以下评估维度进行判断：

1. 准确性：是否准确传达参考答案的核心信息
2. 完整性：是否覆盖所有关键信息点
3. 表达清晰度：语言是否清晰、通顺
4. 相关性：是否聚焦问题主题
5. 逻辑结构：是否结构清晰、条理清楚

---
问题：
{question}

参考答案：
{reference}

候选答案 A（来自 RAG）：
{rag}

候选答案 B（来自 MCTS）：
{mcts}

请从 A 和 B 中选择一个**更优**的答案（即更符合上述标准），或者在两者质量非常接近、难以分出优劣时，选择 "Same"。

只输出一个单词："A"、"B" 或 "Same"，不要输出其他内容或解释。
"""

    try:
        response = client.chat.completions.create(
            model="DeepSeek-V3",
            messages=[
                {"role": "system", "content": "You are a helpful evaluator."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )

        choice = response.choices[0].message.content.strip().upper()
        print(f"比较结果：{choice}")
        return choice
    except Exception as e:
        print(f"比较出错: {e}")
        return None

def process_single_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """处理单个记录的函数，用于并行执行"""
    try:
        question = record.get("question", "")
        reference = record.get("ground_truth", "")
        
        # 获取各种答案
        direct_answer = record.get("direct_answer", "")
        rag_answer = record.get("rag_answer", "")
        mcts_answer = record.get("mcts_answer", "")
        
        if not reference:
            return None

        # 直接在原记录中添加评分
        if direct_answer:
            direct_score = score_answer(question, reference, direct_answer)
            record["score"] = direct_score
            print(f"Direct答案评分: {direct_score}")

        if rag_answer:
            rag_score = score_answer(question, reference, rag_answer)
            record["score"] = rag_score
            print(f"RAG答案评分: {rag_score}")

        if mcts_answer:
            mcts_score = score_answer(question, reference, mcts_answer)
            record["score"] = mcts_score
            print(f"MCTS答案评分: {mcts_score}")

        print(f"已评分问题: {question}")
        return record
        
    except Exception as e:
        print(f"处理记录时出错: {e}")
        return None

def evaluate_jsonl_parallel(input_jsonl_path, output_jsonl_path, max_workers=16):
    """并行处理 JSONL 文件"""
    # 读取所有记录
    records = []
    with open(input_jsonl_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            try:
                record = json.loads(line)
                records.append(record)
            except Exception as e:
                print(f"[跳过] 无效JSON行: {e}")
                continue
    
    print(f"总共读取到 {len(records)} 条记录，开始并行处理...")
    
    # 使用线程池并行处理
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_record = {executor.submit(process_single_record, record): record for record in records}
        
        # 收集结果
        for future in concurrent.futures.as_completed(future_to_record):
            record = future_to_record[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"处理记录时出错: {e}")
    
    print("prepare to write results...")
    # 写入结果
    with open(output_jsonl_path, 'w', encoding='utf-8') as fout:
        for result in results:
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
    

if __name__ == "__main__":
    repos = [
        'astropy',
        # 'flask',
        # 'matplotlib',
        # 'pylint',
        # 'pytest',
        # 'requests',
        # 'scikit-learn',
        # 'sphinx',
        # 'sqlfluff',
        # 'xarray',
        # 'django',
        # 'sympy',
    ]
    subtype = "rag_doc"
    type = "rag"
    # type = "mcts"
    for repo in repos:
        # input_path = f"/data3/pwh/answers/direct/{repo}_direct.jsonl"
        # output_path = f"/data3/pwh/answers/score/direct/{repo}_score.jsonl"
        input_path = f"/data3/pwh/answers/{subtype}/{repo}_{type}.jsonl"
        output_path = f"/data3/pwh/answers/score/{subtype}/{repo}_score.jsonl"
        # 使用并行处理
        evaluate_jsonl_parallel(input_path, output_path, max_workers=16)