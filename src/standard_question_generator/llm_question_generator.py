"""
Script to generate questions for SWE-bench_Lite statements using OpenAI API,
and categorize them into respective result files.
"""
import json
import os
import re
import time
from datasets import load_dataset
from openai import AzureOpenAI
from prompt_template.question_generator import generate_questions
from src.assets.questions.standard_question_generator.llm_models_config import PerflabLLMProxyConfig

# Global variables that can be modified by external scripts
total_count_limit = 100
split_name = "test"

# Default configuration for OpenAI API
config = PerflabLLMProxyConfig()
client = AzureOpenAI(
    azure_deployment=config.deployment_name,
    azure_endpoint=config.azure_endpoint,
    api_key=os.environ.get("LLM_API_KEY"),
    api_version=config.openai_api_version
)

def parse_questions_from_response(response_text):
    """
    Parse the JSON list of questions from the LLM response.
    
    Args:
        response_text (str): The text response from the LLM
        
    Returns:
        list: A list of dictionaries with 'dimension' and 'question' keys
    """
    try:
        # First try to find JSON array pattern using regex
        match = re.search(r'\[\s*{.*}\s*\]', response_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        
        # If regex failed, try parsing the whole response as JSON
        return json.loads(response_text)
    except json.JSONDecodeError:
        print(f"Failed to parse response as JSON: {response_text[:100]}...")
        return []

def generate_questions_with_openai(prompt):
    """
    Generate questions using OpenAI API.
    
    Args:
        prompt (str): The prompt to send to the OpenAI API
        config (PerflabLLMProxyConfig, optional): Configuration for the OpenAI API call
        
    Returns:
        list: A list of dictionaries with 'dimension' and 'question' keys
    """
    try:
        # Use provided config or default
        
        response = client.completions.create(
            model=config.deployment_name,
            prompt=prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
        print(response)
        response_text = response.choices[0].text.strip()
        print(response_text)
        return parse_questions_from_response(response_text)
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return []

def generate_questions_with_openai_chat(prompt):
    """
    Generate questions using OpenAI Chat API.
    
    Args:
        prompt (str): The prompt to send to the OpenAI API
        config (PerflabLLMProxyConfig, optional): Configuration for the OpenAI API call
        
    Returns:
        list: A list of dictionaries with 'dimension' and 'question' keys
    """
    try:
        # Format messages for chat API
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        response = client.chat.completions.create(
            model=config.deployment_name,
            messages=messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
        
        response_text = response.choices[0].message.content
        return parse_questions_from_response(response_text)
    except Exception as e:
        print(f"Error calling OpenAI Chat API: {e}")
        return []

def generate_dummy_questions(problem_statement, index):
    """
    Generate dummy questions for testing without using the API.
    
    Args:
        problem_statement (str): The problem statement
        index (int): The example index
        
    Returns:
        list: A list of dictionaries with 'dimension' and 'question' keys
    """
    return [
        {"dimension": "where", "question": f"Example {index}: Where in the codebase is the functionality related to {problem_statement[:50]}...?"},
        {"dimension": "what", "question": f"Example {index}: What does the error message in '{problem_statement[:50]}...' mean?"},
        {"dimension": "how", "question": f"Example {index}: How can I fix the issue described in '{problem_statement[:50]}...'?"},
        {"dimension": "relation", "question": f"Example {index}: What's the relationship between components mentioned in '{problem_statement[:50]}...'?"},
        {"dimension": "api", "question": f"Example {index}: What API parameters should I use to resolve '{problem_statement[:50]}...'?"}
    ]

def main():
    print("Loading SWE-bench_Lite dataset...")
    # Use global variables that can be set from external scripts
    global total_count_limit
    global split_name
    
    try:
        # Load the dataset
        ds = load_dataset("princeton-nlp/SWE-bench_Lite")
        print(f"Successfully loaded dataset with {len(ds)} splits")
        
        # Create or clear result files
        categories = ["where", "what", "how", "relation", "api"]
        for category in categories:
            with open(os.path.join("results", f"{category}.txt"), "w", encoding="utf-8") as f:
                pass  # Just create or clear the file
        
        # Process each example in the dataset
        total_count = 0
        
        if split_name in ds:
            split = ds[split_name]
            print(f"Processing {len(split)} examples from the {split_name} split (limit: {total_count_limit})...")
            
            for i, example in enumerate(split):
                problem_statement = example.get('problem_statement', '')
                if not problem_statement:
                    continue
                
                print(f"Processing example {i+1}/{len(split)}")
                
                # Generate the prompt for question generation
                prompt = generate_questions(problem_statement)

                # Check if OpenAI API key is available
                if os.environ.get("LLM_API_KEY"):
                    questions = generate_questions_with_openai_chat(prompt)
                    time.sleep(1)
                else:
                    print("Using dummy questions (OpenAI API key not provided)...")
                    questions = generate_dummy_questions(problem_statement, i+1)
                
                # If no questions were generated, use dummy questions as fallback
                if not questions:
                    print("Fallback to dummy questions...")
                    questions = generate_dummy_questions(problem_statement, i+1)
                
                # Save questions to their respective category files
                for question_data in questions:
                    category = question_data["dimension"].lower()
                    question = question_data["question"]
                    
                    # Make sure the category is valid
                    if category not in categories:
                        print(f"Warning: Skipping question with unknown category '{category}'")
                        continue
                    
                    with open(os.path.join("results", f"{category}.txt"), "a", encoding="utf-8") as f:
                        f.write(question + "\n\n")
                
                total_count += 1
                
                # Limit the number of examples to process
                if total_count >= total_count_limit:
                    break
        
        print(f"Successfully processed {total_count} examples and categorized questions.")
        print("Results saved in the 'results' directory.")
    
    except Exception as e:
        print(f"Error processing dataset: {e}")
        print("\nIf you're seeing authentication errors, please login using:")
        print("huggingface-cli login")
        print("or")
        print("python -c 'from huggingface_hub import login; login()'")

if __name__ == "__main__":
    main() 