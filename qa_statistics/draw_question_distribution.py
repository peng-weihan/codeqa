import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class CodeQuestionAnalyzer:
    def __init__(self, questions_dir):
        """
        Initialize code question analyzer
        
        Parameters:
            questions_dir: Directory path containing question files
        """
        self.questions_dir = questions_dir
        self.question_types = []
        self.tags = []
        self.data = None
        self.high_level_count = 0
        # Define tags to be categorized as Undirect
        self.undirect_tags = [
            'Decorator', 'Condition', 'Component', 'Model', 'BaseClass', 
            'Object', 'Model1', 'Model2', 'Parameter', 'Field'
        ]
        
    def load_data(self):
        """Load and parse all question files"""
        # Question type mapping based on filenames
        question_files = [f for f in os.listdir(self.questions_dir) 
                         if os.path.isfile(os.path.join(self.questions_dir, f)) 
                         and not f.endswith('.json') and not f == 'code_qa_categorized.txt'
                         and not f == 'Rule.txt' and os.path.getsize(os.path.join(self.questions_dir, f)) > 0]
        
        # Extract all question types except _Undirect files
        self.question_types = [f.replace('.txt', '') for f in question_files 
                              if not f.endswith('_Undirect.txt') and f != 'High-Level.txt']
        
        # Add High-Level as a question type
        self.question_types.append('High-Level')
        
        # Extract all tags
        tags_set = set()
        pattern = r'`<([^>]+)>`'
        
        # Record all tags and their occurrences
        tag_occurrences = {}
        
        for file_name in question_files:
            if file_name != 'High-Level.txt':  # Process normal questions
                file_path = os.path.join(self.questions_dir, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Extract all tags
                    tags = re.findall(pattern, content)
                    for tag in tags:
                        if tag in tag_occurrences:
                            tag_occurrences[tag] += 1
                        else:
                            tag_occurrences[tag] = 1
        
        # Filter out tags to be categorized as Undirect
        for tag in tag_occurrences:
            if tag not in self.undirect_tags:
                tags_set.add(tag)
        
        # Add Undirect tag
        tags_set.add('Undirect')
        self.tags = sorted(list(tags_set))
        
        # Create statistics matrix
        data = np.zeros((len(self.question_types), len(self.tags)))
        
        # Fill statistics matrix for normal question types
        for i, q_type in enumerate(self.question_types[:-1]):  # Exclude High-Level
            # Process regular files
            file_path = os.path.join(self.questions_dir, f"{q_type}.txt")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.strip():
                            found_tags = re.findall(pattern, line)
                            if found_tags:
                                for tag in found_tags:
                                    if tag in self.undirect_tags:
                                        # Categorize to Undirect
                                        undirect_index = self.tags.index('Undirect')
                                        data[i, undirect_index] += 1
                                    elif tag in self.tags:
                                        tag_index = self.tags.index(tag)
                                        data[i, tag_index] += 1
            
            # Process Undirect files
            undirect_file = os.path.join(self.questions_dir, f"{q_type}_Undirect.txt")
            if os.path.exists(undirect_file):
                with open(undirect_file, 'r', encoding='utf-8') as f:
                    undirect_count = sum(1 for line in f if line.strip())
                    undirect_index = self.tags.index('Undirect')
                    data[i, undirect_index] += undirect_count
        
        # Process High-Level questions - all are categorized as Undirect
        high_level_index = self.question_types.index('High-Level')
        high_level_file = os.path.join(self.questions_dir, "High-Level.txt")
        if os.path.exists(high_level_file):
            with open(high_level_file, 'r', encoding='utf-8') as f:
                self.high_level_count = sum(1 for line in f if line.strip())
                if self.high_level_count > 0:
                    undirect_index = self.tags.index('Undirect')
                    data[high_level_index, undirect_index] = self.high_level_count
                    print(f"Added {self.high_level_count} High-Level questions (all in Undirect category)")
        
        # Ensure data is integer type
        data = data.astype(int)
        self.data = pd.DataFrame(data, index=self.question_types, columns=self.tags)
        return self.data
    
    def plot_heatmap(self, figsize=(12, 10), cmap="YlGnBu", annot=True, fmt=".0f"):
        """
        Plot question distribution heatmap
        
        Parameters:
            figsize: Figure size
            cmap: Color map
            annot: Whether to show numbers in cells
            fmt: Number format, using .0f instead of d to accommodate possible float values
        """
        if self.data is None:
            self.load_data()
        
        plt.figure(figsize=figsize)
        ax = sns.heatmap(self.data, annot=annot, fmt=fmt, cmap=cmap, linewidths=.5)
        plt.title('Code Question Distribution Heatmap', fontsize=15)
        plt.xlabel('Question Tags', fontsize=12)
        plt.ylabel('Question Types', fontsize=12)
        
        # Adjust layout
        plt.tight_layout()
        return ax
    
    def save_plot(self, filename='code_questions_heatmap.png', dpi=300):
        """
        Save heatmap to file
        
        Parameters:
            filename: File name to save
            dpi: Resolution
        """
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Heatmap saved to {filename}")
    
    def get_statistics(self):
        """Get statistics summary"""
        if self.data is None:
            self.load_data()
        
        # Calculate total count for each question type
        type_counts = self.data.sum(axis=1).sort_values(ascending=False)
        
        # Calculate total count for each tag
        tag_counts = self.data.sum(axis=0).sort_values(ascending=False)
        
        # Calculate total questions
        total_questions = self.data.values.sum()
        
        return {
            'total_questions': total_questions,
            'question_type_counts': type_counts,
            'tag_counts': tag_counts,
            'high_level_count': self.high_level_count
        }

# Usage example
def analyze_code_questions(questions_dir, output_file=None):
    """
    Analyze code questions and generate heatmap
    
    Parameters:
        questions_dir: Directory path containing question files
        output_file: File path to save heatmap (optional)
    
    Returns:
        analyzer: CodeQuestionAnalyzer instance
    """
    analyzer = CodeQuestionAnalyzer(questions_dir)
    
    # Load data
    analyzer.load_data()
    
    # Plot heatmap
    analyzer.plot_heatmap()
    
    # Save heatmap (if output file specified)
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Ensure output directory exists
        analyzer.save_plot(output_file)
    else:
        plt.show()
    
    # Print statistics
    stats = analyzer.get_statistics()
    print(f"Total questions: {stats['total_questions']}")
    print(f"High-Level questions (in Undirect): {stats['high_level_count']}")
    print("\nQuestion type statistics:")
    print(stats['question_type_counts'])
    print("\nTag statistics:")
    print(stats['tag_counts'])
    
    return analyzer

if __name__ == "__main__":
    # Specify directory containing question files
    questions_dir = "./dataset/seed_questions"
    
    # Analyze questions and generate heatmap
    analyzer = analyze_code_questions(questions_dir, "./statistics/code_questions_heatmap.png")