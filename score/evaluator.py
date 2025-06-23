import os
import json
import openai
from typing import List
from dotenv import load_dotenv
from repo_qa_generator.models.data_models import (
    QAPair, 
    EvaluationResult, 
    GPTEvaluationResponse, 
    EvaluationScore
)
from repo_qa_generator.core.generator import BaseGenerator

load_dotenv()
SYSTEM_PROMPT = """
You are a professional Q&A quality evaluation expert.

Below is an instruction from an user and a candidate answer. Evaluate whether or not the answer is a good example of how AI Assistant should respond to the user's instruction. Please assign a score using the following 5-point scale:
1: It means the answer is severely flawed and largely unhelpful. It may be incomplete, critically vague, significantly off-topic, contain factual inaccuracies, or directly contradict the user's request. It might also include harmful, biased, or controversial content. The answer fails to demonstrate a basic understanding of the user's needs and may contain irrelevant promotional text, navigation elements, or be written from an inappropriate perspective (e.g., a forum post).
2: It means the answer attempts to address the user's query but does so poorly or superficially. While it might contain a fragment of relevant information, it largely misses the core of the question, provides a significantly oversimplified or high-level methodology where a specific solution is needed, or is poorly structured and difficult to understand. It may still contain some irrelevant information or minor inaccuracies.
3: It means the answer provides a generally correct and helpful response to the user's basic asks but has noticeable shortcomings. It might be from an AI assistant but lacks polish, clarity, or conciseness. For example, it could be overly verbose, poorly organized, or require a fair amount of interpretation by the user. Alternatively, if not from an AI perspective, it might resemble an excerpt from a blog post or web page, containing personal opinions or non-assistant-like phrasing, even if the core information is useful.
4: It means the answer is good, clearly written from an AI assistant's perspective, and effectively addresses the user's instruction. It is complete, accurate, and well-organized. However, it's not perfect and has slight room for improvement. For example, it could be more insightful, more concise, better anticipate user needs, or the explanation could be slightly clearer or more direct. The quality is high, but not exceptional.
5: It means the answer is exceptional and exemplifies a gold-standard AI assistant response. It is perfectly tailored to the user's instruction, demonstrating deep expert knowledge with outstanding clarity and precision. The content is insightful, highly valuable, and presented in a logical, easy-to-follow, and engaging manner. It is flawlessly written, concise, and may proactively offer additional, highly relevant information or considerations. There is no discernible room for improvement.

"""
EVALUATION_PROMPT = """
Evaluate the quality of the following Q&A pair:
Question: {qa_pair.question}
Answer: {answer}
Ground Truth: {ground_truth}

Here're information about you can relate to when you evaluate the answer:
Related Code: {computed_relative_code_list}
Ground Truth: {computed_ground_truth}

Provide your evaluation in the following JSON format:
{{
    "reasoning": "<detailed explanation for the score>",
    "score": <score>,
}}
"""

class QAEvaluator(BaseGenerator):
    def __init__(self):
        super().__init__()
    
    def evaluate_qa(self, qa_pair: QAPair,answer: str) -> float:
        # Prepare values for the new placeholders in the prompt
        # Safely access attributes and provide 'None' string if attribute is falsy (None, empty string, empty list etc.)
        _relative_code_list = qa_pair.relative_code_list
        computed_relative_code_list = _relative_code_list if _relative_code_list else 'None'
        
        _ground_truth = qa_pair.ground_truth
        computed_ground_truth = _ground_truth if _ground_truth else 'None'

        prompt = EVALUATION_PROMPT.format(
            qa_pair=qa_pair,  # Used for {qa_pair.question}
            answer=answer,    # Used for {answer}
            ground_truth=_ground_truth, # For the first {ground_truth}, pass the actual value or None
            computed_relative_code_list=computed_relative_code_list,
            computed_ground_truth=computed_ground_truth
        )
        print(prompt)
        response = self._call_llm(system_prompt=SYSTEM_PROMPT,user_prompt=prompt)

        try:

            gpt_response = GPTEvaluationResponse(**json.loads(response))
            print(gpt_response)
            
            return float(gpt_response.score),gpt_response.reasoning
        except Exception as e:
            return EvaluationScore.BASIC,f"Failed to parse GPT response: {str(e)}"

    def evaluate_qa_pair(self, qa_pair: QAPair) -> EvaluationResult:
        """Evaluate the quality of a single Q&A pair"""
        prompt = f"""
        Evaluate the quality of the following Q&A pair:
        Question: {qa_pair.question}
        Answer: {qa_pair.answer}
        Related Code: {qa_pair.relative_code_list if qa_pair.relative_code_list else 'None'}
        
        Please rate according to the following criteria (1-5 points):
        1: Answer is incomplete, vague, or off-topic
        2: Answer addresses the question but lacks accuracy or detail
        3: Answer is complete and helpful but could be improved
        4: Answer is very good, accurate, and comprehensive
        5: Answer is perfect, accurate, comprehensive, and easy to understand

        Provide your evaluation in the following JSON format:
        {
            "score": <score>,
            "reasoning": "<detailed explanation for the score>",
            "suggestions": ["suggestion1", "suggestion2", ...]  // Optional list of improvement suggestions
        }
        """
        
        response = self._call_llm(system_prompt=SYSTEM_PROMPT,user_prompt=prompt)
        
        # Parse evaluation results using Pydantic model
        try:
            gpt_response = GPTEvaluationResponse(**json.loads(response.choices[0].message.content))
            
            return EvaluationResult(
                qa_pair=qa_pair,
                score=float(gpt_response.score),
                reasoning=gpt_response.reasoning,
            )
        except Exception as e:
            # Fallback to a default evaluation if parsing fails
            return EvaluationResult(
                qa_pair=qa_pair,
                score=EvaluationScore.BASIC,
                reasoning=f"Failed to parse GPT response: {str(e)}",
                suggestions=["Review and re-evaluate this Q&A pair manually"]
            )
    
    def batch_evaluate(self, qa_pairs: List[QAPair]) -> List[EvaluationResult]:
        """Evaluate multiple Q&A pairs in batch"""
        return [self.evaluate_qa_pair(pair) for pair in qa_pairs]
    
    def filter_low_quality(self, evaluation_results: List[EvaluationResult], 
                          threshold: float = EvaluationScore.GOOD) -> List[QAPair]:
        """Filter out low-quality Q&A pairs"""
        return [result.qa_pair for result in evaluation_results 
                if result.score >= threshold] 