from moatless_qa.codeblocks.codeblocks import CodeBlock, CodeBlockType
from moatless_qa.codeblocks.parser.create import create_parser
from moatless_qa.codeblocks.parser.java import JavaParser
from moatless_qa.codeblocks.parser.parser import CodeParser
from moatless_qa.codeblocks.parser.python import PythonParser


def supports_codeblocks(path: str):
    return path.endswith(".py")


def get_parser_by_path(file_path: str) -> CodeParser | None:
    if file_path.endswith(".py"):
        return PythonParser()
    elif file_path.endswith(".java"):
        return JavaParser()
    else:
        return None
