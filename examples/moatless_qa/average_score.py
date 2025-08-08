import json

def analyze_scores(jsonl_file_path):
    total_rag_score = 0
    total_mcts_score = 0
    count = 0

    rag_better = 0
    mcts_better = 0
    tie_count = 0
    tie_rag_total = 0
    tie_mcts_total = 0

    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                rag_score = data.get('rag_score', 0)
                mcts_score = data.get('mcts_score', 0)
                total_rag_score += rag_score
                total_mcts_score += mcts_score
                count += 1

                if rag_score > mcts_score:
                    rag_better += 1
                elif mcts_score > rag_score:
                    mcts_better += 1
                else:
                    tie_count += 1
                    tie_rag_total += rag_score
                    tie_mcts_total += mcts_score

    if count == 0:
        return None

    avg_rag = total_rag_score / count
    avg_mcts = total_mcts_score / count

    rag_percent = rag_better / count * 100
    mcts_percent = mcts_better / count * 100
    tie_percent = tie_count / count * 100

    avg_tie_rag = tie_rag_total / tie_count if tie_count else 0
    avg_tie_mcts = tie_mcts_total / tie_count if tie_count else 0

    return {
        'avg_rag': avg_rag,
        'avg_mcts': avg_mcts,
        'rag_better_percent': rag_percent,
        'mcts_better_percent': mcts_percent,
        'tie_percent': tie_percent,
        'avg_tie_score': avg_tie_rag,
    }

# Run analysis on the provided file path
print(analyze_scores('/data3/pwh/codeqa/dataset/score/django_result.jsonl'))
