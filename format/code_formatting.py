from repo_qa_generator.models.data_models import QAPair, CodeNode
CODE_FORMATITING_PROMPT = """
## Code Information
- File Name: {file_name}
- Belongs to the moddule: {module}
- The code is part of the definition of the class: {define_class}
- Imports: {imports}
- Code Snippet: 
```
{code}
```
"""

CODE_INSTRUCTION = """
These are the code snippets that are related to the question.
Please answer your question based on the code snippets.
"""

#目前为止swe bench问题难以解决是states及上下文不够
#定位不到想要的信息: Where catagory: {category}
#感觉qa应该是帮助agent进行粗粒度定位的

# 需要修改
ground_truth_field_prompt = "\\nYour answer must be based SOLELY on verified information directly from the repository.\\nInclude all supporting evidence, such as relevant code snippets and repository-specific knowledge.\\nDO NOT introduce any external information, guesses, or assumptions.\\nIf no supporting information can be found within the repository for the query, respond with the exact tag: \\\"None\\\".\\nWhen including code snippets that informed your answer, enclose them in triple backticks like this:\\n```python\\n# Your code snippet here\\n```\\nEnsure the code language is correctly specified if it's not Python.\\n"
ANSWER_FORMAT_INSTRUCTION = f"""
Return the answer field in the following json format:
{{
    "thought": "The thought process of the answer.",
    "ground_truth": "{ground_truth_field_prompt}",
    "answer": "The answer to the question."
}}
"""

def format_code_from_list(relative_code_list: list[CodeNode]):
    code_prompt = CODE_INSTRUCTION
    for code in relative_code_list:
        code_prompt += format_code_from_code_node(code)
    
    code_prompt += ANSWER_FORMAT_INSTRUCTION
    return code_prompt

def format_code_from_code_node(code_node: CodeNode):
    code_prompt = CODE_FORMATITING_PROMPT.format(
        file_name=code_node.belongs_to.file_name,
        module=code_node.belongs_to.module,
        define_class=code_node.belongs_to.define_class,
        imports=code_node.belongs_to.imports,
        code=code_node.code
     )
    return code_prompt

def format_context(qa_pair: QAPair):
    context_prompt = """
    ## Related Context
    Supporting Ground Truth:
    {ground_truth}
    {code_snippets}
    """

    prompt = context_prompt.format(
        ground_truth=qa_pair.ground_truth,
        code_snippets=("\n".join([format_code_from_code_node(code) for code in qa_pair.relative_code_list]))
    )
    return prompt

