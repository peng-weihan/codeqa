import json

def check_json(json_obj, keys: list):
    if not isinstance(json_obj, dict):
        return False
    for key in keys:
        if key not in json_obj.keys():
            return False
    return True

def cot(question: str, contexts: str = None):
    instruction = """
        Please solve the multi-hop question below based on the following contexts step by step:

        QUESTION: 
        {question}

        CONTEXTS: 
        {contexts}
    """
    formatter = """
        Provide your response in this JSON format:
        {{
            "thought": "Give your step-by-step reaoning process",
            "answer": "Your precise answer"
        }}
    """
    prompt = (instruction + formatter).format(question=question, contexts=contexts)
    return prompt

def direct(question: str,context:str=None):
    instruction = """      
        You are a professional code question solver.
        Answer the following code problem as precisely using only the provided contexts as you can, step by step:

        QUESTION: {question}
        CONTEXT: {context}
        
        INSTRUCTIONS:
        1. Analysis Rules:
           a) Carefully analyze all provided code segments
           b) Consider class definitions, function implementations, and variable references
           c) Pay attention to dependencies between components and inheritance relationships
           d) Look for potential side effects or behavior modifiers (hooks, middleware, etc.)
        
        2. Reasoning Process:
           a) Break down the problem into smaller, manageable parts
           b) Track variable values, function parameters, and return types
           c) Consider different execution paths and edge cases
           d) Verify assumptions with evidence from the code
        
        3. Answer Format:
           a) Provide a clear, concise, and precise answer
           b) Reference specific parts of the code to support your conclusion
           c) Address all aspects of the original question

        Please extend your chain of thought as much as possible; 
        The longer the chain of thought, the better.
        Make sure to use the context to solve the question.
    """

    formatter = """
    Provide your response in this JSON format:
    {{
        "question": {question},
        "thought": "give your step by step thought process here",
        "supporting_sentences": [
            "Include ALL sentences needed to justify your answer",
            "Use ... for long sentences when appropriate"
        ],
        "answer": "Your precise answer following the instructions above" or "none" if no answer can be found
    }}
    """
    prompt = (instruction + formatter).format(question=question, context=context)
    return prompt

def multistep(question: str):
    instruction = """
        You are a professional code question solver. Answer the following as precisely as you can step by step:

        QUESTION: {question}
        
        Please extend your chain of thought as much as possible; the longer the chain of thought, the better.
        
        You can freely reason in your response, but please enclose the final answer within <answer></answer> tags (pure number without units and explanations)
    """
    prompt = instruction.format(question=question)
    return prompt

def multistep_with_context(question: str, contexts: str):
    instruction = """
        You are a precise question-answering solver.
        Breaks down multi-hop questions into single-hop sub-questions to answer the following question using only the provided contexts:

        QUESTION: {question}
        CONTEXT: {contexts}

        INSTRUCTIONS:
        1. Answer Selection Rules:
           a) Use ONLY information from the given contexts
           b) Extract a precise answer that is:
              - CONTINUOUS: Must be an unbroken segment from the text
              - EXACT: Use the original text without modifications
              - MINIMAL: Include only the essential information
        2. Supporting Evidence:
           - Select ALL relevant sentences that lead to your answer
           - Include complete context when needed
           - You may use ellipsis (...) to connect relevant parts of long sentences
        
           EXAMPLE:
           Question: "Which function calculates the total price?"
           Supporting Code Lines: 
           ✓ Good: "const calculateTotal = (items) => items.reduce((sum, item) => sum + item.price, 0);"
           × Bad: "calculateTotal calculates the price" (not exact code)

        3. Answer Extraction Guidelines:
           a) CONTINUOUS code segment only:
              Question: "What is the key for the user ID in the session storage?"
              Code: "sessionStorage.setItem('userId', user.id);"
              ✓ CORRECT: "'userId'"
              × WRONG: "setItem('userId'" (incomplete segment)

           b) EXACT code text:
              Question: "What is the type hint for the `config` parameter?"
              Code: "def load_settings(config: dict):"
              ✓ CORRECT: "dict"
              × WRONG: "config: dict" (not minimal type)

           c) MINIMAL answer:
              Question: "What module is imported for path manipulation?"
              Code: "import os.path as path"
              ✓ CORRECT: "os.path" 
              ✓ Also potentially correct depending on nuance: "path" (if asking about the alias)
              × WRONG: "import os.path as path" (not minimal)
        
        4. Important:
           - Handle unclear questions by focusing on the main intent
           - Avoid common pitfalls like combining disconnected information
           - Prioritize precision over completeness
           
        5. Robustness:
            Sometimes the question may have some errors, leading to a situation where there is actually no answer in the context.
            I hope you can infer what the questioner is actually asking and then respond according to the above process.
    """
    formatter = """
    Provide your response in this JSON format:
    {{
        "question": {question},
        "thought": "give your step by step thought process here",
        "sub-questions": [
            {{
                "description": "the description of the sub-question",
                "supporting_contexts": [
                    "Include ALL contexts needed to justify your answer to this sub-question",
                    "Use ... for long contexts when appropriate"
                ],
                "answer": "Answer to this sub-question"
            }},
            ...more sub-questions as needed
        ],
        "conclusion": "Explain how the sub-answers combine to answer the main question",
        "answer": "Your precise answer to the main question" or "none" if no answer can be found
        }}
    """
    prompt = (instruction + formatter).format(question=question, contexts=contexts)
    return prompt

def label(question: str, trajectory: str, answer: str):
    instruction = """
        You are tasked with breaking down a **code-related reasoning process** into sub-questions.

        Original Code Question: {question}
        Complete Reasoning Process for the Code Question: {trajectory}

        Instructions:
        1. Break down the reasoning process into a series of **code-specific** sub-questions (e.g., about function parameters, variable values, control flow, dependencies, code snippets).
        2. Each sub-question should:
           - Be written in clear, interrogative form.
           - Have a clear, concise answer derived from the reasoning process (can be numerical, text, boolean, or a simple structure like a list).
           - List the indices (0-based) of other sub-questions it depends on. Dependencies arise when information needed is NOT directly from the original question/code but MUST come from the answers of previous sub-questions. An empty list indicates no dependencies.
        3. Ensure the final answer provided aligns with the overall goal.
    """
    formatter = """
        Format your response as the following JSON object:
        {{
            "sub-questions": [
                {{
                    "description": "<clear interrogative code-related question>",
                    "answer": <concise answer - can be string, boolean, list, etc.>,
                    "dependencies": [<indices of prerequisite sub-questions>]
                }},
                ...
            ],
            "answer": {answer} // This should represent the final conclusion/answer derived from the sub-questions.
        }}
    """
    return (instruction + formatter).format(question=question, trajectory=trajectory, answer=repr(answer))

def contract(question: str, decompose_result: dict, independent_subquestions: list, dependent_subquestions: list, contexts: str=None):
    instruction = """
        You are an AI assistant specializing in optimizing **code question answering** processes.
        Your task is to use the results of previously solved sub-questions to formulate a more efficient, self-contained follow-up question or analysis task.
        Also, extract the relevant code contexts that may be relevant to the new created question or analysis task.
        
        Original Code Question: {question}
        Contexts: {contexts}

        Here is the breakdown of the reasoning process and its sub-questions:
        {response} // This contains the sub-questions list from the 'label' step.

        {sub_questions_guidance}

        Key Concepts:
        1. **Self-contained:** The new optimized question/task should incorporate the knowledge gained from the 'independent' sub-questions and be solvable without referring back to their *reasoning*, only their *answers*.
        2. **Efficient:** The new question/task should be simpler or more focused than the original, leveraging the resolved sub-problems to target the remaining uncertainty or the next logical step in understanding the code. It might involve asking for specific code modifications, explanations based on established facts, or verification.

    """
    independent_sub_questions_text = """
        The following sub-questions and their answers are now considered **known facts or established context**:
        {independent_subquestions}
    """
    dependent_sub_questions_text = """
        The descriptions and potentially the answers of the following **dependent** sub-questions should guide the formulation of the next step:
        {dependent_subquestions}
    """

    # Prepare the sub-questions details for the prompt
    sub_questions_guidance = ""
    if independent_subquestions: # Assuming independent is a list of sub-question dicts
        # Format independent questions for clarity in the prompt
        formatted_independent = "\\n".join(["- Q: {sub_q.get('description', 'N/A')} | A: {sub_q.get('answer', 'N/A')}" for sub_q in independent_subquestions])
        sub_questions_guidance += independent_sub_questions_text.format(independent_subquestions=formatted_independent)

    if dependent_subquestions: # Assuming dependent is a list of sub-question dicts
        # Format dependent questions for clarity in the prompt
        formatted_dependent = "\\n".join(["- Q: {sub_q.get('description', 'N/A')} (Depends on: {sub_q.get('dependencies', [])}) | A: {sub_q.get('answer', 'N/A')}" for sub_q in dependent_subquestions])
        sub_questions_guidance += dependent_sub_questions_text.format(dependent_subquestions=formatted_dependent)

    if not independent_subquestions and not dependent_subquestions:
         sub_questions_guidance = "No specific sub-questions were marked as independent or dependent, consider the full decomposition."

    # Let's pass the whole decompose_result dictionary as a string for context.
    response_str = json.dumps(decompose_result, indent=2)

    formatter = """
        Based on the original question and the provided sub-question analysis, formulate the **next logical question or code analysis task**.
        Provide your response in this JSON format:
        {{
            "thought": "give your step by step thought process here",
            "question": {question},
            "contexts": {contexts},
            "response": {response}
        }}
    """
    # Format the final prompt
    prompt = (instruction + formatter).format(
        question=question,
        response=response_str, # Pass the full decomposition result stringified
        sub_questions_guidance=sub_questions_guidance,
        contexts=contexts
    )
    return prompt

def ensemble(question: str, solutions: list, contexts: str = None):
    instruction = """
        You are a precise code question solver. Compare then synthesize the best answer from multiple solutions to solve the following question according to the provided score instructions.

        QUESTION: {question}

        SOLUTIONS:
        {solutions}

        CONTEXT:
        {contexts}

        # Instructions for Selecting the Best Solution:
        Compare the provided SOLUTIONS based on the following criteria to determine the single best answer for the QUESTION, considering the CONTEXT:

        1.  **Correctness and Completeness:** How accurately and fully does the solution solve the problem stated in the QUESTION? Are all necessary parts (imports, configurations, steps) included?
        2.  **Relevance:** Does the solution directly address the specific aspect of the QUESTION?
        3.  **Clarity and Quality:** Is the code well-structured, readable, and the explanation clear and concise?
        4.  **Adherence to Best Practices:** Does the code follow standard conventions and safety guidelines?

        **Your Task:** Identify and output the solution that best meets these criteria overall. Prioritize accuracy and completeness. If no single solution is ideal, you may synthesize an improved answer by combining the strengths of multiple solutions, but clearly indicate this.
    """
    
    formatter = """
        Format your response as the following JSON object:
        {{
            "question": "{question}",
            "thought": "Explain your analysis of the different results and why you chose the final answer",
            "supporting_contexts": [
                "Include ALL contexts needed to justify your answer",
                "Use ... for long contexts when appropriate"
            ],
            "answer": "The most reliable answer following the answer instructions"
        }}
    """
    solutions_str = ""
    for i, solution in enumerate(solutions):
        solutions_str += f"solution {i}: {solution}\\n"
    prompt = (instruction + formatter).format(question=question, solutions=solutions_str,contexts=contexts)
    return prompt

def code_multihop_with_context(question: str, contexts: str = None):
    """
    Approach 1: Solving code problems with provided code context as multihop questions
    This function generates a prompt that breaks down code problems into specific types of sub-questions
    when the relevant code snippets are already provided as context.
    """
    instruction = """
        You are a professional code problem solver. Please solve the following code problem by breaking it down into structured sub-questions:

        QUESTION: 
        {question}

        CODE CONTEXTS: 
        {contexts}
        
        INSTRUCTIONS:
        1. Carefully analyze the problem and the provided code contexts
        2. Break down the complex problem into multiple sub-questions
        3. Each sub-question must be one of the following types:
           - Where: Where is the definition of class/function/variable `<X>`?
           - What: What are the attributes (properties) of class `<X>`?
           - How: How is `<Variable>`'s `<Attribute>` dependent on any specific `<Model>` layer?
           - API: Are there other `<API>` or `<Hook>` that can affect the behavior of `<Class>`'s `<Variable>`?
        4. Solve each sub-question step by step, using the provided code contexts
        5. Combine the answers to solve the original problem
    """
    formatter = """
        Provide your response in this JSON format:
        {{
            "thought": "Your detailed step-by-step reasoning process",
            "sub-questions": [
                {{
                    "type": "Where/What/How/API",
                    "description": "Sub-question description (e.g., 'Where is the definition of class X?')",
                    "answer": "Answer to this sub-question based on code context",
                    "dependencies": [indices of sub-questions this one depends on, empty list if none]
                }},
                // more sub-questions as needed
            ],
            "answer": "Your final answer to the original question"
        }}
    """
    
    prompt = (instruction + formatter).format(question=question, contexts=contexts)
    return prompt

def code_multihop_with_search(question: str):
    """
    Approach 2: Solving code problems without initial context, requiring search
    This function generates a prompt that breaks down code problems into searchable sub-questions
    when no code contexts are initially provided. After decomposition, contexts will be searched.
    """
    instruction = """
        You are an expert code analyzer. Your task is to decompose a complex code question into searchable sub-questions without having the code context yet.
        
        ORIGINAL QUESTION:
        {question}
        
        INSTRUCTIONS:
        1. First, decompose the question into specific sub-questions that follow these formats:
           - Where is the definition of <class/function/variable>?
           - What are the attributes of <class>?
           - How is <variable>'s <attribute> dependent on <model> layer <variable>?
           - Are there other <API> or <Hook> that can affect the behavior of <class>'s <variable>?
           
        2. Each sub-question should be concrete enough that it could be answered by searching for specific code segments
        
        3. After this decomposition, we will search for relevant code contexts for each sub-question
    """
    
    formatter = """
        Provide your response in this JSON format:
        {{
            "thought": "Your analysis of how to break down this code problem",
            "sub-questions": [
                {{
                    "type": "Where/What/How/API",
                    "description": "Specific sub-question that can be used to search for relevant code"
                }},
                // more sub-questions as needed
            ]
        }}
    """
    prompt = (instruction + formatter).format(question=question)
    return prompt

def code_multihop_with_searched_context(question: str, decomposed_questions: list, contexts: dict):
    """
    Follow-up function for code_multihop_with_search
    This function generates a prompt that solves the original question after contexts have been found
    for the decomposed sub-questions.
    
    Parameters:
    - question: Original code question
    - decomposed_questions: List of sub-questions from code_multihop_with_search
    - contexts: Dictionary mapping sub-question index to found code context
    """
    instruction = """
        You are a professional code problem solver. Now that we have searched for relevant code contexts for each sub-question,
        please solve the original code problem.
        
        ORIGINAL QUESTION:
        {question}
        
        DECOMPOSED SUB-QUESTIONS AND THEIR CONTEXTS:
        {contexts_info}
        
        INSTRUCTIONS:
        1. Analyze each sub-question and its associated code context
        2. Answer each sub-question based on its context
        3. Identify dependencies between sub-questions
        4. Combine all insights to solve the original question
    """
    contexts_info = ""
    for i, q in enumerate(decomposed_questions):
        contexts_info += f"Sub-question {i+1}: {q['description']}\n"
        contexts_info += f"Related Context: {contexts.get(i, 'No context found')}\n\n"
    
    formatter = """
        Provide your response in this JSON format:
        {{
            "thought": "Your detailed step-by-step reasoning process",
            "sub-questions": [
                {{
                    "description": "Sub-question description",
                    "context": "Related code context",
                    "answer": "Answer to this sub-question",
                    "dependencies": [indices of sub-questions this one depends on]
                }},
                // more sub-questions as needed
            ],
            "answer": "Your final answer to the original question"
        }}
    """
    
    prompt = (instruction + formatter).format(question=question, contexts_info=contexts_info)
    return prompt


def search_context(question_str: str):
    """
    An api reserved for code repository search though a question.
    """
    #TODO: fetch contexts from code repository
    #Method 1: Direct Search and return the context
    #Method 2: Search and answer the question directly as the context
    instruction = """
        You are a professional code question solver. You are given a question and a code repository.
        You need to search the code repository for the context that is most relevant to the question.

    """
    context = f"Searching context for: {question_str}"
    return context

def check(name: str, result: dict, *args):
    if name == "cot":
        if not check_json(result, ["thought", "answer"]):
            return False
        if not isinstance(result["answer"], str) or result["answer"].lower() in ["null", "none", ""]:
            return False
    elif name == "direct":
        print(result)
        if not check_json(result, ["question", "thought", "supporting_sentences", "answer"]):
            return False
        if not isinstance(result["supporting_sentences"], list) or not all(isinstance(s, str) for s in result["supporting_sentences"]):
            return False
        if not isinstance(result["answer"], str) or result["answer"].lower() in ["null", "none", ""]:
            return False
    elif name == "contract":
        if not check_json(result, ["question", "thought"]):
            return False
    elif name == "code_multihop_with_context":
        if not check_json(result, ["thought", "sub-questions", "answer"]):
            return False
        for sub_q in result["sub-questions"]:
            if not check_json(sub_q, ["type", "description", "answer", "dependencies"]):
                return False
    elif name == "code_multihop_with_search":
        if not check_json(result, ["thought", "sub-questions"]):
            return False
        for sub_q in result["sub-questions"]:
            if not check_json(sub_q, ["type", "description"]):
                return False
    return True

