PROMPT_REFINE = """
You are a helpful assistant that converts specific questions into reusable question templates.
Your task is to transform concrete questions into general templates where specific elements are replaced with placeholders.

# Instructions:
1. Replace specific values or constants with <Value> placeholders, such as <Class>, <Function>, <Attribute>,<Variable>, etc.
2. Maintain the core structure and intent of each question
3. Ensure the templated questions remain clear and understandable

# Original questions:
{original_questions}

# Templated questions:
"""

FOMMATTING_PROMPT = """
# Format:
[
    "Question 1",
    "Question 2",
    ...
]
"""

def problem_refine(questions):
    """
    Converts specific questions into general question templates by replacing
    concrete elements with placeholders.
    
    Args:
        questions: List of questions or a string containing questions to convert to templates
        
    Returns:
        Formatted prompt for generating templated questions
    """
    formatted_prompt = PROMPT_REFINE.format(original_questions=questions)
    formatted_prompt += FOMMATTING_PROMPT
    
    return formatted_prompt
