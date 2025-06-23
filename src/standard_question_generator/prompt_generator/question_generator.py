QUESTION_GENERATOR_PROMPT = """
Suppose you are a professional programmer tasked with fixing a bug in a code repository.
You have limited knowledge of the codebase, you need to ask questions to an instructor who is familiar with the codebase to better understand the problem and find a solution.
Your questions can address different types of issues such as _where_ to find relevant files or functions, _what_ certain terms or components mean, _how_ to perform specific tasks, or _relationships_ between different parts of the code (e.g., functions, variables, classes).

The instuctor only knows the information about the codebase.
It can only help you understand the repository better, but cannot help you fix the problem.
So donnot ask the instructor about 'How to fix the problem' or 'What is the solution.

# Problem statement:
{problem_statement}

# Instructions:
Generate questions covering these five dimensions of code understanding:
1. WHERE: Questions about locating relevant files, functions, or code sections
2. WHAT: Questions about terminology, component purposes, or concept explanations
3. HOW: Questions about implementing specific operations or processes
4. RELATION: Questions about relationships between different code elements (functions, variables, classes)
5. API: Questions about API usage, parameters, return values, or behavior

Your questions should be specific, clear, and directly related to understanding or fixing the problem described above.

"""

FOMMATTING_PROMPT = """
# Format:
[
    {"dimension": "where", "question": "Question 1"},
    {"dimension": "what", "question": "Question 2"},
    {"dimension": "how", "question": "Question 3"},
    {"dimension": "relation", "question": "Question 4"},
    {"dimension": "api", "question": "Question 5"}
]
"""
def generate_questions(problem_statement):
    prompt = QUESTION_GENERATOR_PROMPT.format(problem_statement=problem_statement)
    prompt += FOMMATTING_PROMPT
    return prompt
