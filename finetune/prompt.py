instruction="""
You are an expert in answering questions about code in the {repository} repository.
You are given a question about the code repository. You need to answer the question based on your understanding of the code repository.
Return "Unknown" if you don't know the answer.
"""


def format_data_item(input:str, output:str):
    return {
        "instruction": instruction,
        "input": input,
        "output": output
    }
