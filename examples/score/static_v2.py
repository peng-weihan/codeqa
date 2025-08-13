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
    scores_avg = {}

    # 需要替代 direct_score 的字段列表
    replace_with_direct = {'rag_func_score', 'rag_doc_score'}

    # 初始化存储各分数的列表
    adjusted_scores = {key: [] for key in score_keys}

    for item in data_list:
        for key in score_keys:
            val = item.get(key)
            # 如果是需要替代的字段且值为0或None，尝试用direct_score代替
            if key in replace_with_direct and (val == 0 or val is None):
                direct_val = item.get('direct_score')
                if direct_val is not None:
                    adjusted_scores[key].append(direct_val)
                else:
                    # direct_score也无效则跳过
                    continue
            else:
                if val is not None:
                    adjusted_scores[key].append(val)

    # 计算平均值
    for key in score_keys:
        scores = adjusted_scores.get(key, [])
        if not scores:
            avg = None
        else:
            avg = sum(scores) / len(scores)
        scores_avg[key] = avg
        summary[key] = {'avg': avg}

    # 统计mcts_score是最高分的比例，计算时也要替代 rag_func_score 和 rag_doc_score
    mcts_highest_count = 0
    total_count = 0
    for item in data_list:
        scores = []
        for key in score_keys:
            val = item.get(key)
            if key in replace_with_direct and (val == 0 or val is None):
                val = item.get('direct_score')
            if val is not None:
                scores.append((key, val))

        if not scores:
            continue

        total_count += 1
        max_score = max(s[1] for s in scores)
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
        file_path = f"/data3/pwh/answers/score/summary/{repo}_summary_score.jsonl"
        data = read_jsonl(file_path)
        result = summarize_scores(data)
        print(f"仓库: {repo}")
        print(json.dumps(result, indent=4, ensure_ascii=False))
