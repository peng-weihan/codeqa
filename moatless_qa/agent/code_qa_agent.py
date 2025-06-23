import json
import logging
from typing import List

from moatless_qa.actions import (
    FindClass,
    FindFunction,
    FindCodeSnippet,
    SemanticSearch,
    ViewCode,
)
from moatless_qa.actions.action import Action
from moatless_qa.actions.finish import Finish
from moatless_qa.actions.list_files import ListFiles
from moatless_qa.actions.reject import Reject
from moatless_qa.agent.agent import ActionAgent
from moatless_qa.agent.code_qa_prompts import (
    QA_AGENT_ROLE,
    REACT_GUIDELINES,
    REACT_CORE_OPERATION_RULES,
    ADDITIONAL_NOTES,
    generate_workflow_prompt,
    generate_guideline_prompt,
)
from moatless_qa.completion.completion import (
    LLMResponseFormat,
    CompletionModel,
)

from moatless_qa.index import CodeIndex
from moatless_qa.repository.repository import Repository
from moatless_qa.schema import MessageHistoryType
from moatless_qa.message_history import MessageHistoryGenerator
logger = logging.getLogger(__name__)


class CodeQAAgent(ActionAgent):
    @classmethod
    def create(
        cls,
        repository: Repository,
        completion_model: CompletionModel,
        preset_actions: List[Action] | None = None,
        code_index: CodeIndex | None = None,
        edit_completion_model: CompletionModel | None = None,
        message_history_type: MessageHistoryType | None = None,
        thoughts_in_action: bool = False,
        **kwargs,
    ):
        # Clone the completion model to ensure we have our own instance
        completion_model = completion_model.clone()

        if message_history_type is None:
            if completion_model.response_format == LLMResponseFormat.TOOLS:
                message_history_type = MessageHistoryType.MESSAGES
            else:
                message_history_type = MessageHistoryType.REACT

        action_completion_format = completion_model.response_format
        if action_completion_format != LLMResponseFormat.TOOLS:
            logger.info(
                "Default to JSON as Response format for action completion model"
            )
            action_completion_format = LLMResponseFormat.JSON

        # Create action completion model by cloning the input model with JSON response format
        action_completion_model = completion_model.clone(
            response_format=action_completion_format
        )

        if preset_actions:
            actions = preset_actions
        else:
            actions = create_all_actions(
                repository=repository,code_index=code_index,completion_model=action_completion_model,
            )
        action_type = "standard edit code understanding actions"
        use_few_shots = True

        # Generate workflow prompt based on available actions
        workflow_prompt = generate_workflow_prompt(actions)

        # Compose system prompt based on model type and format
        system_prompt = QA_AGENT_ROLE
        if completion_model.response_format == LLMResponseFormat.REACT:
            system_prompt += REACT_CORE_OPERATION_RULES
        elif completion_model.response_format == LLMResponseFormat.TOOLS:
            system_prompt += REACT_GUIDELINES

        # Add workflow and guidelines
        system_prompt += workflow_prompt + generate_guideline_prompt() + ADDITIONAL_NOTES

        message_generator = MessageHistoryGenerator(
            message_history_type=message_history_type,
            include_file_context=True,
            thoughts_in_action=thoughts_in_action,
        )
        config = {
            "completion_model": completion_model.__class__.__name__,
            "code_index_enabled": code_index is not None,
            "edit_completion_model": edit_completion_model.__class__.__name__
            if edit_completion_model
            else None,
            "action_type": action_type,
            "actions": [a.__class__.__name__ for a in actions],
            "message_history_type": message_history_type.value,
            "thoughts_in_action": thoughts_in_action,
            "file_context_enabled": True,
        }

        logger.info(
            f"Created CodingAgent with configuration: {json.dumps(config, indent=2)}"
        )

        return cls(
            completion=completion_model,
            actions=actions,
            system_prompt=system_prompt,
            use_few_shots=use_few_shots,
            message_generator=message_generator,
            thoughts_in_action=thoughts_in_action,
            **kwargs,
        )


def create_base_actions(
    repository: Repository,
    code_index: CodeIndex | None = None,
    completion_model: CompletionModel | None = None,
) -> List[Action]:
    """Create the common base actions used across all action creators."""
    return [
        SemanticSearch(
            code_index=code_index,
            repository=repository,
            completion_model=completion_model,
        ),
        FindClass(
            code_index=code_index,
            repository=repository,
            completion_model=completion_model,
        ),
        FindFunction(
            code_index=code_index,
            repository=repository,
            completion_model=completion_model,
        ),
        FindCodeSnippet(
            code_index=code_index,
            repository=repository,
            completion_model=completion_model,
        ),
        ViewCode(repository=repository, completion_model=completion_model),
    ]


def create_all_actions(
    repository: Repository,
    code_index: CodeIndex | None = None,
    completion_model: CompletionModel | None = None,
) -> List[Action]:
    actions = create_base_actions(repository, code_index, completion_model)
    actions.append(Finish())
    return actions
