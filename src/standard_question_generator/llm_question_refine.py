"""
Script to refine questions from the results directory and remove duplicates.
The refined questions are stored in a new 'refined_results' directory.
"""
import os
import json
import ast
import re
from collections import defaultdict
from prompt_template.question_refine import problem_refine
from src.assets.questions.standard_question_generator.llm_models_config import PerflabLLMProxyConfig
from openai import AzureOpenAI

# Configuration for OpenAI API
config = PerflabLLMProxyConfig()
client = AzureOpenAI(
    azure_deployment=config.deployment_name,
    azure_endpoint=config.azure_endpoint,
    api_key=os.environ.get("LLM_API_KEY"),
    api_version=config.openai_api_version
)

def refine_questions_with_openai(prompt):
    """
    Use OpenAI to refine the questions.
    
    Args:
        prompt: The prompt to send to the OpenAI API
        
    Returns:
        Refined questions as a list
    """
    try:
        response = client.chat.completions.create(
            model=config.deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that refines questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=4000
        )
        
        refined_text = response.choices[0].message.content.strip()
        
        # Extract the JSON list from the response
        try:
            # Find the JSON array in the response
            match = re.search(r'\[\s*".*"\s*\]', refined_text, re.DOTALL)
            if match:
                json_str = match.group(0)
                refined_questions = json.loads(json_str)
            else:
                # Try to find any list-like structure
                match = re.search(r'\[(.*)\]', refined_text, re.DOTALL)
                if match:
                    list_str = match.group(0)
                    refined_questions = ast.literal_eval(list_str)
                else:
                    # Fall back to splitting by lines
                    refined_questions = [line.strip().strip('"') for line in refined_text.split('\n') 
                                        if line.strip() and not line.startswith('#') and not line.startswith('[')]
        except Exception as e:
            print(f"Error parsing refined questions: {e}")
            refined_questions = [line.strip() for line in refined_text.split('\n') 
                               if line.strip() and not line.startswith('#')]
        
        return refined_questions
    except Exception as e:
        print(f"Error refining questions with OpenAI: {e}")
        return []

def remove_duplicates(questions):
    """
    Remove duplicate questions by comparing them after normalization.
    
    Args:
        questions: List of questions to deduplicate
        
    Returns:
        List of unique questions
    """
    # Normalize questions by converting to lowercase and removing punctuation
    normalized_questions = {}
    for q in questions:
        normalized = q.lower().strip()
        normalized = ''.join(c for c in normalized if c.isalnum() or c.isspace())
        normalized_questions[normalized] = q
    
    # Return the original form of unique questions
    return list(normalized_questions.values())

def main():
    # Create refined_results directory if it doesn't exist
    if not os.path.exists('refined_results'):
        os.makedirs('refined_results')
    
    # Get all files in the results directory
    result_files = [f for f in os.listdir('results') if f.endswith('.txt')]
    
    # Process each file
    for file_name in result_files:
        print(f"Processing {file_name}...")
        
        # Read questions from the file
        with open(os.path.join('results', file_name), 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"Found {len(questions)} questions in {file_name}")
        
        # Refine the questions
        if questions:
            prompt = problem_refine(questions)
            refined_questions = refine_questions_with_openai(prompt)
            
            # Remove duplicates
            unique_questions = remove_duplicates(refined_questions)
            
            print(f"Refined to {len(unique_questions)} unique questions")
            
            # Save refined questions to new file
            with open(os.path.join('refined_results', file_name), 'w', encoding='utf-8') as f:
                for question in unique_questions:
                    f.write(question + '\n')
        
        print(f"Completed processing {file_name}")

if __name__ == "__main__":
    main() 