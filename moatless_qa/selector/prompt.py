def format_examples(examples):
    formatted = []
    for i, example in enumerate(examples, 1):
        header = f"Example {i}:\n\n"
        node_id = f"<node_id>: {i}\n"
        feedback = f"<feedback>: {example}"
        formatted.append(header + node_id + feedback)
    return "\n\n".join(formatted)


SYSTEM_PROMPT = """You are an AI tasked with analyzing a Monte Carlo Tree Search (MCTS) tree and selecting the most promising node for expansion.
The agent starts with searching through the codebase to find the most promising approach to the problem, and then continues by implementing code changes and tests to validate the approach, before concluding by reaching a finish state.
By choosing a node, you are selecting the state at which the agent will continue from.
Be reasonable and think step-by-step about which node will best continue the search.

When analyzing nodes:
- Consider reward, visits, and action potential
- Prioritize nodes with higher rewards
- Provide specific code context for the software developer agent
- Describe nodes in relation to others (siblings, parents, etc.), not by numbers
- Focus on context and actions, not rewards or visit counts
- Aim for diverse 'finished' nodes through depth-wise expansion
- Avoid loops with repetitive actions
- Try completely new approaches by expanding nodes earlier in the tree if current paths aren't working, or if the current trajectories have already found solutions. For example:
    - Working on a new file from scratch
    - Working on a new class from scratch
    - Working on a new function from scratch
- Don't allude to node numbers or IDs but rather describe the node in relation to others, since the agent will not see the tree structure
- Only select nodes that are "expandable" (do not select nodes that are "not-expandable")
- Keep the feedback specific and to the point. In it just include useful information that can be used to generate a better next step.

The goal is to help the software engineer *efficiently* find diverse, high-quality code analysis results through effective tree exploration."""

EXAMPLES_1 = """The trajectory has used FindFunction 4 times in a row examining Django's query compiler:
- django/db/models/sql/compiler.py
- django/db/models/sql/query.py
- django/db/models/query.py

While we've found several query compilation methods, continuing with function searches is becoming redundant.

Recommended Next Step:
Use SemanticSearch with:
- Query: "django subquery optimization compiler"
- Category: "implementation"
- File pattern: "django/db/models/sql/*.py"

This will help identify optimization opportunities in the query compilation process rather than finding more compiler functions.
"""

EXAMPLES_2 = """
The trajectory shows 3 consecutive RequestMoreContext actions examining scikit-learn's gradient boosting implementation:
- sklearn/ensemble/_gb.py
- sklearn/ensemble/gradient_boosting.py
- sklearn/tree/_tree.py

We now have extensive context about the gradient boosting internals, but more context requests aren't yielding new insights.

Recommended Next Step:
Use FindClass with:
- Class name: "BaseGradientBoosting"
- File pattern: "sklearn/ensemble/*.py"
To focus on the base implementation rather than gathering more peripheral context.
"""

EXAMPLES_3 = """
The trajectory has used RequestMoreContext and FindFunction actions repeatedly to understand Django's URL resolver:
- django/urls/resolvers.py (via RequestMoreContext)
- django/urls/conf.py (via RequestMoreContext)
- django/urls/base.py (via FindFunction)

Key signs we're stuck in a search loop:
1. We've gathered context about URLResolver, URLPattern, and RegexPattern classes
2. We understand the resolve() method implementation
3. Additional context requests are returning overlapping information
4. We have a clear picture of where changes need to be made

This pattern of repeatedly gathering context without moving to implementation is a common trap.

Recommended Next Step:
Use RequestCodeChange with:
- File: "django/urls/resolvers.py"
- Change: "Add support for custom path converters in URLResolver.resolve() method:
```python
def resolve(self, path):
    # Add converter handling
    if hasattr(path, 'converter'):
        path = path.converter.to_url(path)
    return super().resolve(path)
```"

Remember: When you have sufficient understanding of the codebase through searching and context gathering, the next step should be to request concrete code changes rather than continuing to search.
"""

# Store all examples in a list
EXAMPLE_LIST = [
    EXAMPLES_1,
    EXAMPLES_2,
    EXAMPLES_3,
]

# Generate the combined examples
ALL_EXAMPLES = format_examples(EXAMPLE_LIST)
