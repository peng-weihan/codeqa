import json

def read_jsonl(file_path):
    result = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                result.append(obj)
    return result

def write_to_jsonl(data_list, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def summary_score(repo,input_dir:str,output_dir:str):  
    # 读取数据
    mcts_data = read_jsonl(f'{input_dir}/mcts/{repo}_score.jsonl')
    rag_func_data = read_jsonl(f'{input_dir}/rag_func/{repo}_score.jsonl')
    rag_doc_data = read_jsonl(f'{input_dir}/rag_doc/{repo}_score.jsonl')
    direct_data = read_jsonl(f'{input_dir}/direct/{repo}_score.jsonl')

    # 提取 question 集合
    mcts_questions = set(item['question'] for item in mcts_data)
    rag_func_questions = set(item['question'] for item in rag_func_data)
    rag_doc_questions = set(item['question'] for item in rag_doc_data)
    direct_questions = set(item['question'] for item in direct_data)

    # 取交集
    common_questions = mcts_questions & rag_func_questions & direct_questions & rag_doc_questions

    # 过滤数据，只保留交集的问题
    mcts_common = [item for item in mcts_data if item['question'] in common_questions]
    rag_func_common = [item for item in rag_func_data if item['question'] in common_questions]
    rag_doc_common = [item for item in rag_doc_data if item['question'] in common_questions]
    direct_common = [item for item in direct_data if item['question'] in common_questions]

    # 转成字典，方便按 question 快速查找
    def list_to_dict(data_list):
        return {item['question']: item for item in data_list}

    mcts_dict = list_to_dict(mcts_common)
    rag_func_dict = list_to_dict(rag_func_common)
    rag_doc_dict = list_to_dict(rag_doc_common)
    direct_dict = list_to_dict(direct_common)

    # 汇总分数，注意变量初始化
    new_data = []

    for q in common_questions:
        mcts_score = mcts_dict.get(q, {}).get('score', None)
        rag_func_score = rag_func_dict.get(q, {}).get('score', None)
        rag_doc_score = rag_doc_dict.get(q, {}).get('score', None)
        direct_score = direct_dict.get(q, {}).get('score', None)

        new_data.append({
            "question": q,
            "mcts_score": mcts_score,
            "rag_func_score": rag_func_score,
            "rag_doc_score": rag_doc_score,
            "direct_score": direct_score
        })

        # 写入文件
        write_to_jsonl(new_data, f"{output_dir}/{repo}_summary_score.jsonl")

if __name__ == "__main__":
    repos = [
        "astropy",
        "flask", 
        "matplotlib",
        "pylint",
        "pytest",
        "requests",
        "scikit-learn",
        "sphinx",
        "sqlfluff",
        "xarray",
        # "django",
        # "sympy"
    ]
    
    input_dir = "/data3/pwh/answers/score"
    output_dir = "/data3/pwh/answers/score/summary"
    for repo in repos:
        summary_score(repo,input_dir,output_dir)










