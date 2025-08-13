import json

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                data.append(obj)
    return data

def summarize_scores(data_list):
    if not data_list:
        return {}

    score_keys = [k for k in data_list[0].keys() if k.endswith('_score')]

    summary = {'total_questions': len(data_list)}
    for key in score_keys:
        scores = [item[key] for item in data_list if item.get(key) is not None]

        if not scores:
            summary[key] = {
                'avg': None,
            }
        else:
            summary[key] = {
                'avg': sum(scores) / len(scores),
            }
    
    # 统计mcts_score是最高分的比例
    mcts_highest_count = 0
    total_count = 0
    for item in data_list:
        # 过滤掉没有分数的情况
        scores = [(k, item.get(k)) for k in score_keys if item.get(k) is not None]
        if not scores:
            continue
        total_count += 1
        # 找最大分的字段名和分数
        max_score = max(s[1] for s in scores)
        # 看 mcts_score 是否为最大分
        if item.get('mcts_score') == max_score:
            mcts_highest_count += 1
    
    mcts_highest_ratio = mcts_highest_count / total_count if total_count > 0 else None
    summary['mcts_highest_ratio'] = mcts_highest_ratio

    return summary

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
    
    for repo in repos:
        file_path = f"/data3/pwh/answers/score/summary_three/{repo}_summary_score.jsonl"
        data = read_jsonl(file_path)
        result = summarize_scores(data)
        print(f"仓库: {repo}")
        print(json.dumps(result, indent=4, ensure_ascii=False))
