import logging
from typing import List, Optional, Any, Dict

from litellm.types.llms.openai import ChatCompletionUserMessage
from pydantic import Field

from moatless_qa.actions.model import ActionArguments
from moatless_qa.completion.completion import CompletionModel
from moatless_qa.completion.model import StructuredOutput
from moatless_qa.feedback.feedback import FeedbackGenerator
from moatless_qa.node import Node, FeedbackData, generate_ascii_tree

logger = logging.getLogger(__name__)


class GroundTruthResponse(StructuredOutput):
    """Schema for ground truth feedback response"""

    action_name: str = "provide_ground_truth_feedback"

    analysis: str = Field(
        ...,
        description="Detailed analysis of whether node response is supported by ground truth",
    )
    has_ground_truth: bool = Field(
        ..., description="Whether the response has ground truth support"
    )
    feedback: str = Field(
        ..., description="Detailed feedback on response quality"
    )
    suggested_node_id: Optional[int] = Field(
        None, description="ID of the node that should be expanded next (optional)"
    )
    
    @classmethod
    def anthropic_schema(cls) -> Dict[str, Any]:
        """Provide schema in format expected by Anthropic's tool calling"""
        return {
            "type": "custom",
            "name": "provide_ground_truth_feedback",
            "description": "Provide feedback on ground truth support",
            "input_schema": {
                "type": "object",
                "properties": {
                    "analysis": {
                        "type": "string",
                        "description": "Detailed analysis of whether node response is supported by ground truth",
                    },
                    "has_ground_truth": {
                        "type": "boolean",
                        "description": "Whether the response has ground truth support",
                    },
                    "feedback": {
                        "type": "string",
                        "description": "Detailed feedback on response quality",
                    },
                    "suggested_node_id": {
                        "type": ["integer", "null"],
                        "description": "ID of the node that should be expanded next (optional)",
                    },
                },
                "required": ["analysis", "has_ground_truth", "feedback"],
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert Message objects to dictionaries"""
        return {
            "role": self.role if hasattr(self, "role") else "assistant",
            "content": self.content if hasattr(self, "content") else str(self),
        }


class GroundTruthFeedbackGenerator(FeedbackGenerator):
    """Feedback generator that uses LLM to evaluate if responses have ground truth support"""
    
    completion_model: CompletionModel = Field(
        ..., description="The completion model to use"
    )
    include_tree: bool = Field(True, description="Whether to include tree visualization")
    include_node_suggestion: bool = Field(True, description="Whether to include node suggestions")
    
    class Config:
        arbitrary_types_allowed = True
    
    def generate_feedback(
        self, node: Node, actions: List[ActionArguments] | None = None
    ) -> FeedbackData | None:
        """Generate feedback based on the node"""
        if not node.parent:
            logger.info(
                f"Node {node.node_id} has no parent node, skipping feedback generation"
            )
            return None
        
        messages = self._create_analysis_messages(node)
        system_prompt = self._create_system_prompt()
        
        try:
            completion_response = self.completion_model.create_completion(
                messages=messages,
                system_prompt=system_prompt,
                response_model=GroundTruthResponse,
            )
            
            # Store the completion in the node
            node.completions["ground_truth_feedback"] = completion_response.completion
            
            logger.debug(f"Raw completion content: {completion_response.completion}")
            feedback_response: GroundTruthResponse = completion_response.structured_output
            
            # If node suggestions are disabled, set to None
            if not self.include_node_suggestion:
                feedback_response.suggested_node_id = None
            
            feedback_message = (
                "System Analysis: I've analyzed your response and related nodes.\n\n"
                f"Feedback: {feedback_response.feedback}\n\n"
                f"Response has ground truth support: {'Yes' if feedback_response.has_ground_truth else 'No'}\n\n"
                "Note: This feedback is based on analysis of response content and factual evidence. "
                "Please consider the feedback carefully to improve response quality."
            )
            
            return FeedbackData(
                analysis=feedback_response.analysis,
                feedback=feedback_message,
                suggested_node_id=feedback_response.suggested_node_id,
            )
        
        except Exception as e:
            logger.exception(f"Error generating feedback: {e}")
            return None
    
    def _create_analysis_messages(
        self, current_node: Node
    ) -> List[ChatCompletionUserMessage]:
        """Create messages for analysis"""
        messages = []
        
        # Only get siblings that have been run (have actions set)
        sibling_nodes = [
            s for s in current_node.get_sibling_nodes() if s.action is not None
        ]
        
        # Format tree visualization section
        if self.include_tree:
            tree_message = "# Search Tree Visualization\n"
            tree_message += "<search_tree>\n"
            tree_message += generate_ascii_tree(
                current_node.get_root(),
                current=current_node,
                include_explanation=True,
                use_color=False,
                include_diffs=True,
                include_action_details=False,
                include_file_context=False,
                show_trajectory=True,
            )
            tree_message += "\n</search_tree>\n\n"
            messages.append(
                ChatCompletionUserMessage(role="user", content=tree_message)
            )
        
        # Format node relationships section
        relationship_message = "# Node Relationships\n"
        relationship_message += "<relationships>\n"
        relationship_message += f"Current Node: {current_node.node_id}\n"
        relationship_message += f"Parent Node: {current_node.parent.node_id if current_node.parent else 'None'}\n"
        relationship_message += (
            f"Sibling Nodes: {[n.node_id for n in current_node.get_sibling_nodes()]}\n"
        )
        relationship_message += (
            f"Child Nodes: {[n.node_id for n in current_node.children]}\n"
        )
        relationship_message += "</relationships>\n\n"
        messages.append(
            ChatCompletionUserMessage(role="user", content=relationship_message)
        )
        
        # Format root task section
        root_node = current_node.get_root()
        first_message = "# Original Task\n"
        first_message += root_node.message
        messages.append(ChatCompletionUserMessage(role="user", content=first_message))
        
        # Format current node response section
        if current_node.action and current_node.action.name == "Finish" and current_node.observation:
            answer_message = "# Current Node Response\n"
            answer_message += "<current_answer>\n"
            answer_message += current_node.observation.message
            answer_message += "\n</current_answer>\n\n"
            messages.append(ChatCompletionUserMessage(role="user", content=answer_message))
        
        # Format alternative attempts section
        if sibling_nodes:
            analysis_message = "# Alternative Solution Attempts\n"
            
            for sibling in sibling_nodes:
                if not sibling.action:
                    continue
                
                if sibling.action.name == "Finish" and sibling.observation:
                    analysis_message += f"<attempt_{sibling.node_id}>\n"
                    analysis_message += f"Node {sibling.node_id} (Parent: {sibling.parent.node_id if sibling.parent else 'None'})\n"
                    analysis_message += f"Action: {sibling.action.name}\n"
                    analysis_message += sibling.action.to_prompt()
                    
                    if sibling.observation:
                        analysis_message += "\nObservation:\n"
                        analysis_message += sibling.observation.message
                    
                    analysis_message += f"\n</attempt_{sibling.node_id}>\n\n"
            
            messages.append(
                ChatCompletionUserMessage(role="user", content=analysis_message)
            )
        
        return messages
    
    def _create_system_prompt(self) -> str:
        """Create system prompt"""
        system_prompt = """You are a feedback agent responsible for evaluating the quality of AI assistant responses.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋  YOUR TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Analyze whether the current node's response has reliable ground truth support
   • Evaluate if the response is based on reliable facts
   • Check for unsupported claims
   • Analyze the completeness and correctness of the response
   • Determine if the response truly addresses the problem posed in the original task

2. Evaluate the quality of the response
   • Whether it comprehensively answers the question
   • Whether there are errors or misleading information
   • Whether sufficient context and explanation is provided
   • Whether there are better alternative ways to respond

3. Provide specific feedback
   • Point out strengths in the response
   • Identify weaknesses in the response
   • Provide improvement suggestions
   • If necessary, suggest possible follow-up questions or exploration directions

4. Optional: Suggest which node to expand
   • Explain why this node is promising
   • If there is no strong preference, leave as null

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📥  INPUT STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Search Tree Visualization: ASCII representation showing:
   • Node IDs and relationships
   • Action types at each node
   • Key outcomes and observations

2. Original Task: The problem to solve

3. Current Node Response: The response to evaluate

4. Alternative Solution Attempts:
   • Other node responses
   • Their outcomes (from separate, independent trajectories)

Remember: Your primary goal is to evaluate whether the response has reliable ground truth support, not whether unit tests pass.
"""
        return system_prompt 