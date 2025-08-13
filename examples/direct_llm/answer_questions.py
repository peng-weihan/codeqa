import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm
import threading


AIHUBMIX_API_KEY="sk-oskggzxhO6jon2ZQC946Bc8c5e664f28B505422811A39c42"
# Initialize OpenAI client
# AIHUBMIX_API_KEY = os.getenv("AIHUBMIX_API_KEY")

client = OpenAI(
  base_url="https://aihubmix.com/v1",
  api_key= AIHUBMIX_API_KEY
)

def load_questions_from_file(file_path):
    """Load questions from JSONL file with existing format"""
    questions_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    if 'question' in data:
                        questions_data.append(data)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return questions_data

def get_llm_answer(question):
    """Get direct answer from LLM for a question"""
    try:
        completion = client.chat.completions.create(
            model="DeepSeek-V3",
            messages=[
                {
                    "role": "system",
                    "content": "You are a direct answer generator. Provide ONLY the direct answer to the question. Do not include explanations, citations, references, or any additional content. Give the most concise and direct response possible. If the question asks for code, provide only the code. If it asks for a definition, provide only the definition. Be brief and to the point."
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            temperature=0.3,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error getting answer: {e}")
        return f"Error: {e}"

def process_single_question(question_data):
    """Process a single question and return the result"""
    try:
        question = question_data['question']
        direct_answer = get_llm_answer(question)
        
        # Add direct_answer to existing data structure
        question_data['direct_answer'] = direct_answer
        return question_data
    except Exception as e:
        print(f"Error processing question: {e}")
        question_data['direct_answer'] = f"Error: {e}"
        return question_data

def process_repo_parallel(repo, max_workers=5):
    """Process all questions in a repository using parallel execution"""
    input_file = f"/data3/pwh/questions/{repo}.jsonl"
    output_dir = "/data3/pwh/answers/direct"
    output_file = os.path.join(output_dir, f"{repo}_direct.jsonl")
    
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return
    
    questions_data = load_questions_from_file(input_file)
    if not questions_data:
        print(f"No questions found in {input_file}")
        return
    
    print(f"Processing {repo}: Found {len(questions_data)} questions")
    
    # Process questions in parallel
    questions_with_answers = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_question = {
            executor.submit(process_single_question, question_data): i 
            for i, question_data in enumerate(questions_data)
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(questions_data), desc=f"Processing {repo}", unit="question") as pbar:
            for future in as_completed(future_to_question):
                question_idx = future_to_question[future]
                try:
                    result = future.result()
                    questions_with_answers.append(result)
                except Exception as e:
                    print(f"Error processing question {question_idx + 1}: {e}")
                finally:
                    pbar.update(1)
    
    # Sort results to maintain original order
    questions_with_answers.sort(key=lambda x: questions_data.index(x))
    
    # Save answers for this file
    if questions_with_answers:
        save_answers_to_file(questions_with_answers, output_file)
        print(f"Saved {len(questions_with_answers)} answers to: {output_file}")
    else:
        print(f"No questions processed for {repo}")

def save_answers_to_file(questions_with_answers, output_file):
    """Save questions and answers to file"""
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in questions_with_answers:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        print(f"Answers saved to: {output_file}")
    except Exception as e:
        print(f"Error saving file: {e}")

def main():
    # Define input file paths (modify as needed)
    repos = [
        # "pylint",
        # "pytest",
        # "requests",
        # "matplotlib", 
        # "sphinx",
        "sqlfluff",
        # "xarray",
        # "scikit-learn",
        # "flask",
        # "django",
        # "sympy",
        # "astropy",
        # 添加更多仓库名
    ]
    
    output_dir = "/data3/pwh/answers/direct"
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration for parallel processing
    repo_max_workers = 1  # Number of repositories to process simultaneously
    question_max_workers = 32  # Number of questions to process simultaneously per repo
    
    print(f"Starting parallel processing with {repo_max_workers} repos and {question_max_workers} questions per repo")
    
    # Process repositories in parallel
    with ThreadPoolExecutor(max_workers=repo_max_workers) as executor:
        # Submit all repository processing tasks
        future_to_repo = {
            executor.submit(process_repo_parallel, repo, question_max_workers): repo 
            for repo in repos
        }
        
        # Process completed repositories
        for future in as_completed(future_to_repo):
            repo = future_to_repo[future]
            try:
                future.result()
                print(f"Completed processing repository: {repo}")
            except Exception as e:
                print(f"Error processing repository {repo}: {e}")
    
    print(f"\n{'='*50}")
    print("All files processed!")

if __name__ == "__main__":
    main()
