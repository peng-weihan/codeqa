import os
import json
import openai
from typing import List
from dotenv import load_dotenv
from ..models.data_models import (
    QAPair, 
    EvaluationResult, 
    GPTEvaluationResponse, 
    EvaluationScore
)

load_dotenv()

class QAEvaluator:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def evaluate_qa_pair(self, qa_pair: QAPair) -> EvaluationResult:
        """Evaluate the quality of a single Q&A pair"""
        prompt = f"""
        Evaluate the quality of the following Q&A pair:
        Question: {qa_pair.question}
        Answer: {qa_pair.answer}
        Related Code: {qa_pair.related_code if qa_pair.related_code else 'None'}
        
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
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional Q&A quality evaluation expert. Always respond in the requested JSON format."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" }
        )
        
        # Parse evaluation results using Pydantic model
        try:
            gpt_response = GPTEvaluationResponse(**json.loads(response.choices[0].message.content))
            
            return EvaluationResult(
                qa_pair=qa_pair,
                score=float(gpt_response.score),
                reasoning=gpt_response.reasoning,
                suggestions=gpt_response.suggestions
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