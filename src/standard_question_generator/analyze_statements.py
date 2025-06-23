"""
Script to analyze statements from SWE-bench_Lite using the question generator,
and categorize questions into respective result files.
"""
import json
import os
from datasets import load_dataset
from prompt_template.question_generator import generate_questions

# Global variables that can be modified by external scripts
total_count_limit = 10
split_name = "test"

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
            with open(os.path.join("results", f"{category}.txt"), "w") as f:
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
                
                # In a real scenario, you would call an LLM API here to generate questions
                # For demonstration, we'll parse the prompt and generate sample questions
                
                # This is a placeholder for actual LLM generated questions
                # In reality, you would need to call an API with the prompt and parse the response
                sample_questions = [
                    {"dimension": "where", "question": f"Example {i+1}: Where in the codebase is the functionality related to {problem_statement[:50]}...?"},
                    {"dimension": "what", "question": f"Example {i+1}: What does the error message in '{problem_statement[:50]}...' mean?"},
                    {"dimension": "how", "question": f"Example {i+1}: How can I fix the issue described in '{problem_statement[:50]}...'?"},
                    {"dimension": "relation", "question": f"Example {i+1}: What's the relationship between components mentioned in '{problem_statement[:50]}...'?"},
                    {"dimension": "api", "question": f"Example {i+1}: What API parameters should I use to resolve '{problem_statement[:50]}...'?"}
                ]
                
                # Save questions to their respective category files
                for question_data in sample_questions:
                    category = question_data["dimension"]
                    question = question_data["question"]
                    
                    with open(os.path.join("results", f"{category}.txt"), "a", encoding="utf-8") as f:
                        f.write(question + "\n\n")
                
                total_count += 1
                
                # Limit to the specified number of examples
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