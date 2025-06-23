import asyncio
from repo_qa_generator.models.data_models import QAPair
from code_aot.atom import atom_with_context
from format.code_formatting import format_context

async def atom_solve(question: QAPair):
    context = format_context(question)
    result = await atom_with_context(question.question, context)
    return result
