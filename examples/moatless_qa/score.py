import os
import openai
import json

from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

from openai import OpenAI

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

def score_answer(question, reference, candidate, model="deepseek/deepseek-reasoner"):

    prompt = f"""You are a professional evaluator. Please rate the candidate answer against the reference answer based on the following five criteria:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋  YOUR TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Evaluation Criteria and Scoring Guidelines (each scored 1 to 5):

1. Accuracy:
  5 — Completely accurate; core points and details are correct with no ambiguity.
  4 — Mostly accurate; only minor details are slightly incorrect or loosely expressed.
  3 — Partially accurate; some errors or omissions, but main points are generally correct.
  2 — Several errors or ambiguities that affect the understanding of core information.
  1 — Serious errors; misleading or fails to convey key information.

2. Completeness:
  5 — Covers all key points from the reference answer without omission.
  4 — Covers most key points; only minor non-critical information missing.
  3 — Missing several key points; content is somewhat incomplete.
  2 — Important information largely missing; content is one-sided.
  1 — Covers very little or irrelevant information; seriously incomplete.

3. Clarity of Expression:
  5 — Fluent language; clear and precise expression; easy to understand.
  4 — Mostly fluent; some expressions slightly unclear or not concise.
  3 — Expression somewhat awkward; some ambiguity or lack of fluency.
  2 — Language obscure; sentences are not smooth; hinders understanding.
  1 — Expression confusing; very difficult to understand.

4. Relevance:
  5 — Content fully focused on the question topic; no irrelevant information.
  4 — Mostly focused; only minor irrelevant or peripheral information.
  3 — Topic not sufficiently focused; contains considerable off-topic content.
  2 — Content deviates from topic; includes excessive irrelevant information.
  1 — Majority of content irrelevant to the question.

5. Logical Structure:
  5 — Clear and reasonable structure; well-organized and coherent.
  4 — Generally reasonable structure; mostly clear organization; minor misarrangements.
  3 — Loose structure; logical jumps; organization is average.
  2 — Disorganized structure; lacks logical order; difficult to follow.
  1 — No clear structure; logic is chaotic.

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

Please output only one integer score between 0 and 25 representing the overall quality of the candidate answer. No explanation or other content is needed.
"""

    response = client.chat.completions.create(
    model="deepseek-chat",
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
        if 0 <= score <= 100:
            return score
        else:
            return None
    except:
        return None

def compare_answers_pairwise(question, reference, rag, mcts):
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

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful evaluator."},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )

    choice = response.choices[0].message.content.strip().upper()
    print(f"比较结果：{choice}")
    return choice

def evaluate_jsonl(input_jsonl_path, output_jsonl_path):
    with open(input_jsonl_path, 'r', encoding='utf-8') as fin, \
        open(output_jsonl_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            try:
                record = json.loads(line)
            except Exception as e:
                print(f"[跳过] 无效JSON行: {e}")
                continue

            reference = record.get("wiki_answer", "")
            rag = record.get("rag_answer", "")
            mcts = record.get("mcts_answer", "")
            question = record.get("question", "")
            if not reference or not rag or not mcts:
                continue


            # winner = compare_answers_pairwise(question, reference, rag, mcts)
            # result = {
            #     "question": question,
            #     "winner": winner,  # 'A' for RAG, 'B' for MCTS
            #     "rag_answer": rag,
            #     "mcts_answer": mcts,
            #     "reference": reference
            # }
            # fout.write(json.dumps(result, ensure_ascii=False) + "\n")


            rag_score = score_answer(question, reference, rag)
            mcts_score = score_answer(question, reference, mcts)

            result = {
                "question": record.get("question", ""),
                "rag_score": rag_score,
                "mcts_score": mcts_score
            }

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")


            print(f"已评分问题: {result['question']}, RAG分数: {rag_score}, MCTS分数: {mcts_score}")

if __name__ == "__main__":
    input_path = "/data3/pwh/codeqa/dataset/generated_answers/generated_answers_rag_agent_flask_new.jsonl"    # 你的输入jsonl文件路径
    output_path = "/data3/pwh/codeqa/dataset/score/result.jsonl"  # 输出评分结果文件路径
    evaluate_jsonl(input_path, output_path)
