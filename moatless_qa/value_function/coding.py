import logging
from typing import Optional, Tuple

from moatless_qa.actions.search_base import SearchBaseArgs
from moatless_qa.completion.model import Completion
from moatless_qa.node import Node
from moatless_qa.value_function.base import ValueFunction
from moatless_qa.value_function.model import Reward
from moatless_qa.value_function.terminal import TerminalValueFunction

logger = logging.getLogger(__name__)

FAILURE_REWARDS = {
    "MAJOR": {
        "file_not_found": "Requested file does not exist in repository",
        "is_directory": "Requested path is a directory, not a file",
        "invalid_file": "File exists but could not be parsed or is empty",
        "no_spans_found": "No code spans found matching the request",
        "string_not_found": "String to replace was not found in file",
        "string_already_exists": "New string already exists in file",
        "no_changes": "Old and new strings are identical - no changes needed",
    },
    "MINOR": {
        "too_many_tokens": "Requested context exceeds token limit, needs to be more specific",
        "no_spans_added": "Requested spans were already in context",
        "no_search_hits": "Search returned no results",
        "file_exists": "Cannot create file - it already exists",
        "multiple_occurrences": "Multiple occurrences of string found - need more context",
        "indentation_differs": "Content matches but indentation is incorrect",
        "line_breaks_differs": "Content matches but line breaks are incorrect",
        "multiple_format_matches": "Multiple potential matches with different formatting found",
    },
}

FAILURE_VALUES = {"MAJOR": -50, "MINOR": -25}


class CodingValueFunction(ValueFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._terminal_function = TerminalValueFunction(
            completion_model=self.completion_model
        )

    def get_reward(self, node: Node) -> Tuple[Reward, Optional[Completion]]:
        if node.observation.expect_correction and self.correction_award is not None:
            # Start with the base correction award
            correction_penalty = 0
            current_node = node.parent

            # Check parent nodes for expect_correction
            while (
                current_node
                and current_node.observation
                and current_node.observation.expect_correction
            ):
                if (
                    current_node.observation
                    and current_node.observation.expect_correction
                ):
                    correction_penalty += 25
                current_node = current_node.parent

            # Calculate final reward with penalty, minimum of -100
            final_reward = max(-100, self.correction_award - correction_penalty)
            logger.info(
                f"Expecting a correction, base reward {self.correction_award}, penalty {correction_penalty}, final reward {final_reward}"
            )
            return Reward(value=final_reward, explanation="Expects a correction"), None

        if node.action.name == "Finish":
            logger.info(f"Run finish function")
            return self._terminal_function.get_reward(node)

        if node.action.name == "Reject":
            logger.info(f"Reject action, assigning reward -100")
            return Reward(value=-100, explanation="Reject action"), None

        if node.observation.properties:
            fail_reason = node.observation.properties.get("fail_reason")
            if fail_reason:
                logger.info(f"Action failed with reason: {fail_reason}")

                # Find the severity and explanation for the failure
                for severity, failures in FAILURE_REWARDS.items():
                    if fail_reason in failures:
                        return Reward(
                            value=FAILURE_VALUES[severity],
                            explanation=failures[fail_reason],
                        ), None

                # Default case for unknown failures
                return Reward(value=-100, explanation="Action failed"), None

            if isinstance(node.action, SearchBaseArgs):
                if not node.observation.properties.get(
                    "new_span_ids"
                ):  # TODO Use fail reason?
                    return Reward(
                        value=0,
                        explanation="Search returned results but did not add any new spans to the context",
                    ), None
                return Reward(
                    value=100,
                    explanation="Search returned results and added new spans to the context",
                ), None

        if node.action.name == "ViewCode":
            # Check files in properties for new span IDs
            files_info = node.observation.properties.get("files", {}).values()
            files_with_new_spans = [
                file for file in files_info if file.get("new_span_ids")
            ]

            if len(files_with_new_spans) == len(files_info) and files_with_new_spans:
                return Reward(
                    value=50,
                    explanation="Successfully added relevant code context for all requested files",
                ), None
            elif files_with_new_spans:
                return Reward(
                    value=25,
                    explanation="Successfully added some new code context, but not for all requested files",
                ), None

            return Reward(
                value=25, explanation="Request completed but no new context was needed"
            ), None

        logger.warning("Return default reward")
        return Reward(value=50, explanation=""), None
