import json
import re

def filter_null_rag_answers(input_path: str, output_path: str):
    """
    从 input_path 读取 JSONL，每行包含 'rag_answer' 字段，
    过滤掉 rag_answer 为 None 或 "null" 的记录，并将其余写入 output_path。
    """
    removed = 0
    total = 0

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        for line in fin:
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"⚠️ 第 {total} 行非JSON，跳过：{line.strip()[:50]}")
                continue

            val = obj.get("rag_answer", None)
            # rag_answer 字段为 None 或字符串 "null" 都视为无效过滤掉
            if val is None or val == "null" or val == "None":
                removed += 1
                continue

            # 否则写入输出文件
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"✅ 共读取 {total} 条，删除 rag_answer 为 null 的 {removed} 条，保留 {total-removed} 条")

def rename_answer_to_mcts(input_path: str, output_path: str):
    """
    从 input_path 读取 JSONL，重命名每条记录的 "answer" 为 "mcts_answer"，
    并写入 output_path。
    """
    total, renamed = 0, 0
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[跳过] 第 {total} 行无效 JSON")
                continue

            # 如果存在 answer 字段则重命名
            if "answer" in obj:
                obj["mcts_answer"] = obj.pop("answer")  # 重命名字段 :contentReference[oaicite:1]{index=1}
                renamed += 1

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"✅ 处理完成：共 {total} 行，成功重命名 {renamed} 条")

def clean_mcts_answer(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            try:
                data = json.loads(line)

                mcts_raw = data.get("mcts_answer", "")
                if mcts_raw:
                    # 解析 mcts_answer 中的 JSON 字符串
                    mcts_obj = json.loads(mcts_raw)
                    # 只保留 answer 部分
                    data["mcts_answer"] = mcts_obj.get("answer", "")
                else:
                    data["mcts_answer"] = ""

                fout.write(json.dumps(data, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"[跳过] 错误行: {e}")

def build_jsonl_from_txt(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            line = line.strip()
            if not line or line.startswith("//"):
                continue  # 跳过空行和注释行
            
            obj = {
                "question": line,
                "answer": None,
                "relative_code_list": None,
                "ground_truth": None,
                "score": None
            }
            json_line = json.dumps(obj, ensure_ascii=False)
            outfile.write(json_line + "\n")

def build_jsonl_from_django_md(input_path: str, output_path: str):
    """
    从带编号的问题中提取问题内容，并以指定格式写入 JSONL 文件。
    """
    with open(input_path, 'r', encoding='utf-8') as infile:
        content = infile.read()
    
    # 匹配以“数字. 空格”开头的问题
    pattern = re.compile(r'^\s*\d+\.\s+(.*)', re.MULTILINE)
    questions = pattern.findall(content)

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for question in questions:
            obj = {
                "question": question.strip(),
                "answer": None,
                "relative_code_list": None,
                "ground_truth": None,
                "score": None
            }
            json_line = json.dumps(obj, ensure_ascii=False)
            outfile.write(json_line + "\n")
    
    print(f"共提取 {len(questions)} 个问题，写入到 {output_path}")

if __name__ == "__main__":
    # filter_null_rag_answers("/data3/pwh/codeqa/dataset/generated_answers/generated_answers_rag_agent_flask.jsonl.bak", "/data3/pwh/codeqa/dataset/generated_answers/generated_answers_rag_agent_flask.jsonl")
    # filter_null_rag_answers(
    #     input_path="/data3/pwh/codeqa/dataset/generated_answers/generated_answers_rag_agent_flask_2_rename.jsonl",
    #     output_path="/data3/pwh/codeqa/dataset/generated_answers/generated_answers_rag_agent_flask_2.jsonl"
    # )
    build_jsonl_from_txt("/data3/pwh/codeqa/dataset/tmp.txt", "/data3/pwh/codeqa/dataset/generated_questions/sympy_questions.jsonl")
    # build_jsonl_from_django_md("/data3/pwh/codeqa/dataset/tmp.txt","/data3/pwh/codeqa/dataset/generated_questions/django_questions.jsonl")