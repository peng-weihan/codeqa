import os
from typing import List
from analyzers.code_analyzer import CodeAnalyzer
from evaluators.qa_evaluator import QAEvaluator
from models.data_models import QAPair, EvaluationResult

def process_codebase(codebase_path: str, output_path: str):
    """Process codebase and generate Q&A pairs"""
    # Initialize components
    analyzer = CodeAnalyzer()
    generator = QAGenerator()
    evaluator = QAEvaluator()
    
    # Collect all Python files
    python_files = []
    for root, _, files in os.walk(codebase_path):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    # Build dependency graph
    analyzer.build_dependency_graph(python_files)
    
    all_qa_pairs: List[QAPair] = []
    all_evaluations: List[EvaluationResult] = []
    
    # Process each file
    for file_path in python_files:
        # Analyze file
        file_node = analyzer.analyze_file(file_path)
        code_nodes = analyzer.extract_code_nodes(file_path)
        
        # Generate dependency-related Q&A pairs
        dependency_qa_pairs = generator.generate_dependency_questions(file_node)
        
        # Generate code-related Q&A pairs
        code_qa_pairs = []
        for code_node in code_nodes:
            code_qa_pairs.extend(generator.generate_code_questions(code_node))
        
        # Evaluate Q&A pairs
        all_pairs = dependency_qa_pairs + code_qa_pairs
        evaluation_results = evaluator.batch_evaluate(all_pairs)
        all_evaluations.extend(evaluation_results)
        
        # Filter low-quality pairs
        filtered_pairs = evaluator.filter_low_quality(evaluation_results)
        all_qa_pairs.extend(filtered_pairs)
    
    # Save results
    save_qa_pairs(all_qa_pairs, output_path)
    save_evaluation_report(all_evaluations, output_path + '.report.txt')

def save_qa_pairs(qa_pairs: List[QAPair], output_path: str):
    """Save Q&A pairs to file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in qa_pairs:
            f.write(f"Question: {pair.question}\n")
            f.write(f"Answer: {pair.answer}\n")
            if pair.related_code:
                f.write(f"Related Code:\n{pair.related_code}\n")
            f.write("\n---\n\n")

def save_evaluation_report(evaluations: List[EvaluationResult], output_path: str):
    """Save detailed evaluation report"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Q&A Evaluation Report\n")
        
        total_score = sum(eval.score for eval in evaluations)
        avg_score = total_score / len(evaluations) if evaluations else 0
        
        f.write(f"Total Q&A Pairs: {len(evaluations)}\n")
        f.write(f"Average Score: {avg_score:.2f}\n\n")
        
        for i, eval in enumerate(evaluations, 1):
            f.write(f"Q&A Pair #{i}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Question: {eval.qa_pair.question}\n")
            f.write(f"Answer: {eval.qa_pair.answer}\n")
            if eval.qa_pair.related_code:
                f.write(f"Related Code:\n{eval.qa_pair.related_code}\n")
            f.write(f"\nScore: {eval.score}\n")
            f.write(f"Reasoning: {eval.reasoning}\n")
            if eval.suggestions:
                f.write("Suggestions for improvement:\n")
                for suggestion in eval.suggestions:
                    f.write(f"- {suggestion}\n")
            f.write("\n" + "=" * 40 + "\n\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Code Repository Q&A Generator")
    parser.add_argument("--codebase", required=True, help="Path to codebase")
    parser.add_argument("--output", required=True, help="Path to output file")
    
    args = parser.parse_args()
    process_codebase(args.codebase, args.output) 
import os
from typing import List
from analyzers.code_analyzer import CodeAnalyzer
from src.generators.qa_generate_agent import QAGenerator
from evaluators.qa_evaluator import QAEvaluator
from models.data_models import QAPair, EvaluationResult

def process_codebase(codebase_path: str, output_path: str):
    """Process codebase and generate Q&A pairs"""
    # Initialize components
    analyzer = CodeAnalyzer()
    generator = QAGenerator()
    evaluator = QAEvaluator()
    
    # Collect all Python files
    python_files = []
    for root, _, files in os.walk(codebase_path):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    # Build dependency graph
    analyzer.build_dependency_graph(python_files)
    
    all_qa_pairs: List[QAPair] = []
    all_evaluations: List[EvaluationResult] = []
    
    # Process each file
    for file_path in python_files:
        # Analyze file
        file_node = analyzer.analyze_file(file_path)
        code_nodes = analyzer.extract_code_nodes(file_path)
        
        # Generate dependency-related Q&A pairs
        dependency_qa_pairs = generator.generate_dependency_questions(file_node)
        
        # Generate code-related Q&A pairs
        code_qa_pairs = []
        for code_node in code_nodes:
            code_qa_pairs.extend(generator.generate_code_questions(code_node))
        
        # Evaluate Q&A pairs
        all_pairs = dependency_qa_pairs + code_qa_pairs
        evaluation_results = evaluator.batch_evaluate(all_pairs)
        all_evaluations.extend(evaluation_results)
        
        # Filter low-quality pairs
        filtered_pairs = evaluator.filter_low_quality(evaluation_results)
        all_qa_pairs.extend(filtered_pairs)
    
    # Save results
    save_qa_pairs(all_qa_pairs, output_path)
    save_evaluation_report(all_evaluations, output_path + '.report.txt')

def save_qa_pairs(qa_pairs: List[QAPair], output_path: str):
    """Save Q&A pairs to file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in qa_pairs:
            f.write(f"Question: {pair.question}\n")
            f.write(f"Answer: {pair.answer}\n")
            if pair.related_code:
                f.write(f"Related Code:\n{pair.related_code}\n")
            f.write("\n---\n\n")

def save_evaluation_report(evaluations: List[EvaluationResult], output_path: str):
    """Save detailed evaluation report"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Q&A Evaluation Report\n")
        
        total_score = sum(eval.score for eval in evaluations)
        avg_score = total_score / len(evaluations) if evaluations else 0
        
        f.write(f"Total Q&A Pairs: {len(evaluations)}\n")
        f.write(f"Average Score: {avg_score:.2f}\n\n")
        
        for i, eval in enumerate(evaluations, 1):
            f.write(f"Q&A Pair #{i}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Question: {eval.qa_pair.question}\n")
            f.write(f"Answer: {eval.qa_pair.answer}\n")
            if eval.qa_pair.related_code:
                f.write(f"Related Code:\n{eval.qa_pair.related_code}\n")
            f.write(f"\nScore: {eval.score}\n")
            f.write(f"Reasoning: {eval.reasoning}\n")
            if eval.suggestions:
                f.write("Suggestions for improvement:\n")
                for suggestion in eval.suggestions:
                    f.write(f"- {suggestion}\n")
            f.write("\n" + "=" * 40 + "\n\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Code Repository Q&A Generator")
    parser.add_argument("--codebase", required=True, help="Path to codebase")
    parser.add_argument("--output", required=True, help="Path to output file")
    
    args = parser.parse_args()
    process_codebase(args.codebase, args.output) 