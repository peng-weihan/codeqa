import importlib
import json
import logging
import traceback
from typing import List, Type, Dict, Any

from pydantic import BaseModel, Field, PrivateAttr, model_validator, ValidationError

from moatless_qa.actions.action import Action
from moatless_qa.actions.model import (
    ActionArguments,
    Observation,
)
from moatless_qa.agent.settings import AgentSettings
from moatless_qa.completion.completion import CompletionModel, LLMResponseFormat
from moatless_qa.completion.model import Completion
from moatless_qa.exceptions import RuntimeError, CompletionRejectError
from moatless_qa.index.code_index import CodeIndex
from moatless_qa.message_history import MessageHistoryGenerator
from moatless_qa.node import Node, ActionStep
from moatless_qa.repository.repository import Repository
from moatless_qa.agent.code_qa_prompts import RESPONSE_FORMAT

logger = logging.getLogger(__name__)


class ActionAgent(BaseModel):
    system_prompt: str = Field(
        ..., description="System prompt to be used for generating completions"
    )
    use_few_shots: bool = Field(
        True, description="Whether to use few-shot examples for generating completions"
    )
    thoughts_in_action: bool = Field(True, description="")
    actions: List[Action] = Field(default_factory=list)
    message_generator: MessageHistoryGenerator = Field(
        default_factory=lambda: MessageHistoryGenerator(),
        description="Generator for message history",
    )

    _completion: CompletionModel = PrivateAttr()
    _action_map: dict[Type[ActionArguments], Action] = PrivateAttr(default_factory=dict)

    def __init__(
        self,
        completion: CompletionModel,
        system_prompt: str | None = None,
        actions: List[Action] | None = None,
        message_generator: MessageHistoryGenerator | None = None,
        **data,
    ):
        actions = actions or []
        message_generator = message_generator or MessageHistoryGenerator()
        super().__init__(
            actions=actions,
            system_prompt=system_prompt,
            message_generator=message_generator,
            **data,
        )
        self.set_actions(actions)
        self._completion = completion

    @classmethod
    def from_agent_settings(
        cls, agent_settings: AgentSettings, actions: List[Action] | None = None
    ):
        if agent_settings.actions:
            actions = [
                action
                for action in actions
                if action.__class__.__name__ in agent_settings.actions
            ]

        return cls(
            completion=agent_settings.completion_model,
            system_prompt=agent_settings.system_prompt,
            actions=actions,
        )

    def set_actions(self, actions: List[Action]):
        self.actions = actions
        self._action_map = {action.args_schema: action for action in actions}

    @model_validator(mode="after")
    def verify_actions(self) -> "ActionAgent":
        for action in self.actions:
            if not isinstance(action, Action):
                raise ValidationError(
                    f"Invalid action type: {type(action)}. Expected Action subclass."
                )
            if not hasattr(action, "args_schema"):
                raise ValidationError(
                    f"Action {action.__class__.__name__} is missing args_schema attribute"
                )
        return self

    def run(self, node: Node, experience):
        """Run the agent on a node to generate and execute an action."""

        if node.action:
            logger.info(f"Node{node.node_id}: Resetting node")
            node.reset()

        node.possible_actions = [action.name for action in self.actions]
        system_prompt = self.generate_system_prompt()
        action_args = [action.args_schema for action in self.actions]

        messages = self.message_generator.generate(node)
        # add experience
        if experience:
            messages.append({'role': 'user',  'content': [{'type': 'text',
                                                           'text': experience}]})
        
        logger.info(f"Node{node.node_id}: Build action with {len(messages)} messages")
        try:
            completion_response = self._completion.create_completion(
                messages, system_prompt=system_prompt, response_model=action_args
            )
            print('Response Structured Output: \n', completion_response.structured_outputs,'\n')

            if completion_response.structured_outputs:
                node.action_steps = [
                    ActionStep(action=action)
                    for action in completion_response.structured_outputs
                ]

            node.assistant_message = completion_response.text_response

            node.completions["build_action"] = completion_response.completion
        except Exception as e:
            node.terminal = True
            node.error = traceback.format_exc()

            if hasattr(e, "messages") and hasattr(e, "last_completion"):
                # TODO: Move mapping to completion.py
                node.completions["build_action"] = Completion.from_llm_completion(
                    input_messages=e.messages,
                    completion_response=e.last_completion,
                    model=self.completion.model,
                )
                logger.warning(
                    f"Node{node.node_id}: Build action failed with error: {e}"
                )
                return
            else:
                raise e

        if node.action_steps is None:
            return

        duplicate_node = node.find_duplicate()
        if duplicate_node:
            node.is_duplicate = True
            logger.info(
                f"Node{node.node_id} is a duplicate to Node{duplicate_node.node_id}. Skipping execution."
            )
            return

        logger.info(f"Node{node.node_id}: Execute {len(node.action_steps)} actions")
        print(f"Node{node.node_id}: Execute {len(node.action_steps)} actions")

        for action_step in node.action_steps:
            self._execute(node, action_step)

    def _execute(self, node: Node, action_step: ActionStep):
        action = self._action_map.get(type(action_step.action))
        if not action:
            logger.error(
                f"Node{node.node_id}: Action {node.action.name} not found in action map. "
                f"Available actions: {self._action_map.keys()}"
            )
            raise RuntimeError(f"Action {type(node.action)} not found in action map.")

        try:
            action_step.observation = action.execute(
                action_step.action, node.file_context
            )
            if not action_step.observation:
                print(
                    f"Node{node.node_id}: Action {action_step.action.name} returned no observation"
                )
            else:
                print('Observation: ', action_step.observation)
                node.terminal = action_step.observation.terminal
                if action_step.observation.execution_completion:
                    action_step.completion = (
                        action_step.observation.execution_completion
                    )

            logger.info(
                f"Executed action: {action_step.action.name}. "
                f"Terminal: {action_step.observation.terminal if node.observation else False}. "
                f"Output: {action_step.observation.message if node.observation else None}"
            )

        except CompletionRejectError as e:
            logger.warning(f"Node{node.node_id}: Action rejected: {e.message}")
            action_step.completion = e.last_completion
            action_step.observation = Observation(
                message=e.message,
                is_terminal=True,
            )

    def generate_system_prompt(self) -> str:
        """Generate a system prompt for the agent."""

        system_prompt = self.system_prompt
        if self.use_few_shots:
            system_prompt += "\n\n" + self.generate_few_shots()
        
        system_prompt+=f"\n{RESPONSE_FORMAT}"
        

        return system_prompt

    def generate_few_shots(self) -> str:
        few_shot_examples = []
        for action in self.actions:
            examples = action.get_few_shot_examples()
            if examples:
                few_shot_examples.extend(examples)

        prompt = ""
        if few_shot_examples:
            prompt += "\n\n# Examples\nHere are some examples of how to use the available actions:\n\n"
            for i, example in enumerate(few_shot_examples):
                if self.completion.response_format == LLMResponseFormat.REACT:
                    prompt += f"\n**Example {i + 1}**"
                    action_data = example.action.model_dump()
                    thoughts = action_data.pop("thoughts", "")

                    # Use standard JSON format for all actions
                    prompt += (
                        f"\nTask: {example.user_input}"
                        f"\nThought: {thoughts}\n"
                        f"Action: {str(example.action.name)}\n"
                        f"{json.dumps(action_data)}\n\n"
                    )

                elif self.completion.response_format == LLMResponseFormat.JSON:
                    action_json = {
                        "action": example.action.model_dump(),
                        "action_type": example.action.name,
                    }
                    prompt += f"User: {example.user_input}\nAssistant:\n```json\n{json.dumps(action_json, indent=2)}\n```\n\n"

                elif self.completion.response_format == LLMResponseFormat.TOOLS:
                    tools_json = {"tool": example.action.name}
                    if self.thoughts_in_action:
                        tools_json.update(example.action.model_dump())
                    else:
                        tools_json.update(
                            example.action.model_dump(exclude={"thoughts"})
                        )

                    prompt += f"Task: {example.user_input}\n"
                    if not self.thoughts_in_action:
                        prompt += f"<thoughts>{example.action.thoughts}</thoughts>\n"
                    prompt += json.dumps(tools_json)
                    prompt += "\n\n"

        return prompt

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        dump = super().model_dump(**kwargs)
        dump["completion"] = self._completion.model_dump(**kwargs)
        dump["actions"] = []
        dump["agent_class"] = f"{self.__class__.__module__}.{self.__class__.__name__}"
        for action in self.actions:
            dump["actions"].append(action.model_dump(**kwargs))
        return dump

    @classmethod
    def model_validate(
        cls,
        obj: Any,
        repository: Repository = None,
        runtime: Any = None,
        code_index: CodeIndex = None,
    ) -> "ActionAgent":
        if isinstance(obj, dict):
            obj = obj.copy()
            completion_data = obj.pop("completion", None)
            agent_class_path = obj.pop("agent_class", None)

            message_generator_data = obj.get("message_generator", {})
            if message_generator_data:
                obj["message_generator"] = MessageHistoryGenerator.model_validate(
                    message_generator_data
                )

            if completion_data:
                obj["completion"] = CompletionModel.model_validate(completion_data)
            else:
                obj["completion"] = None

            if repository:
                obj["actions"] = [
                    Action.model_validate(
                        action_data,
                        repository=repository,
                        runtime=runtime,
                        code_index=code_index,
                    )
                    for action_data in obj.get("actions", [])
                ]
            else:
                logger.info(f"No repository provided, skip initiating actions")
                obj["actions"] = []

            if agent_class_path:
                module_name, class_name = agent_class_path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                agent_class = getattr(module, class_name)
                instance = agent_class(**obj)
            else:
                instance = cls(**obj)

            return instance

        return super().model_validate(obj)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        repository: Repository | None = None,
        code_index: CodeIndex | None = None,
        runtime: Any | None = None,
    ) -> "ActionAgent":
        """Create an ActionAgent from a dictionary, properly handling dependencies."""
        data = data.copy()

        # Handle completion model
        if "completion" in data and isinstance(data["completion"], dict):
            data["completion"] = CompletionModel.model_validate(data["completion"])

        # Handle actions with dependencies
        if repository and "actions" in data and isinstance(data["actions"], list):
            data["actions"] = [
                Action.model_validate(
                    action_data,
                    repository=repository,
                    runtime=runtime,
                    code_index=code_index,
                )
                for action_data in data["actions"]
            ]

        # Handle message generator
        if "message_generator" in data and isinstance(data["message_generator"], dict):
            data["message_generator"] = MessageHistoryGenerator.model_validate(
                data["message_generator"]
            )

        # Handle agent class if specified
        if "agent_class" in data:
            module_name, class_name = data["agent_class"].rsplit(".", 1)
            module = importlib.import_module(module_name)
            agent_class = getattr(module, class_name)
            return agent_class(**data)

        return cls.model_validate(data)

    @property
    def completion(self) -> CompletionModel:
        return self._completion

    @completion.setter
    def completion(self, value: CompletionModel):
        self._completion = value
