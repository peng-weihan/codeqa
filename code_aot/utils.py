import re
import os
import json
import re

def extract_json(string):
    try:
        string = string.strip("```json").strip("```")
        start, end = string.find("{"), string.rfind("}")
        if start != -1 and end != -1:
            string = string[start : end + 1]
        json_data = json.loads(string)
        return json_data
    except Exception as e:
        return str(e)

def extract_xml(string):
    try:
        # Remove any leading/trailing whitespace
        string = string.strip()

        # Use regex to find all tag-content pairs
        pattern = r"<([\w-]+)>(.*?)</\1>"
        matches = re.finditer(pattern, string)

        result = {}

        # Process each match, later matches will overwrite earlier ones
        for match in matches:
            tag = match.group(1)
            content = match.group(2).strip()

            # Try to convert content to number if possible
            try:
                if content.isdigit():
                    value = int(content)
                else:
                    value = float(content)
            except:
                value = content

            # Simply update the value, overwriting any previous value
            result[tag] = value

        return result
    except Exception as e:
        return {}


def save_json(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return {}

def duration_formatter(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    if hours > 0:
        return f"{int(hours):02d}h:{int(minutes):02d}m:{int(seconds):02d}s"
    elif minutes > 0:
        return f"{int(minutes):02d}m:{int(seconds):02d}s"
    else:
        return f"{int(seconds):02d}s"

def calculate_depth(sub_questions: list):
    try:
        n = len(sub_questions)

        # Initialize distances matrix with infinity
        distances = [[float("inf")] * n for _ in range(n)]

        # Set direct dependencies
        for i, sub_q in enumerate(sub_questions):
            # Distance to self is 0
            distances[i][i] = 0
            # Set direct dependencies with distance 1
            for dep in sub_q.get("depend", []):
                distances[dep][i] = 1

        # Floyd-Warshall algorithm to find shortest paths
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if distances[i][k] != float("inf") and distances[k][j] != float("inf"):
                        distances[i][j] = min(
                            distances[i][j], distances[i][k] + distances[k][j]
                        )

        # Find maximum finite distance
        max_depth = 0
        for i in range(n):
            for j in range(n):
                if distances[i][j] != float("inf"):
                    max_depth = max(max_depth, distances[i][j])

        return int(max_depth)
    except:
        return 3

def get_next_log_file(log_dir, size, dataset):
    directory = log_dir.format(dataset=dataset, size=size)
    os.makedirs(directory, exist_ok=True)
    
    # 只计算数字命名的json文件，排除score.json
    files = [f for f in os.listdir(directory) if f.endswith('.json') and f != 'score.json']
    
    # 找出最大的数字编号
    max_num = 0
    for f in files:
        try:
            num = int(f.split('.')[0])
            max_num = max(max_num, num)
        except ValueError:
            continue
    
    return os.path.join(directory, f"{max_num + 1}.json")

def get_file_count(log_dir, interval, dataset, exclude_score=False):
    directory = log_dir.format(dataset=dataset, size=interval)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        return 0
    
    files = os.listdir(directory)
    if exclude_score:
        # 排除score.json，只计算数字命名的json文件
        files = [f for f in files if f != "score.json"]
    
    return len(files)

def save_question_answer(question, answer, filepath = "./patched_question_answer.json"):
    with open(filepath, "w") as f:
        json.dump({"question": question, "answer": answer}, f, indent=2)