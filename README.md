# CoSQA
Repository-level  CodeScope q&amp;a standardized dataset generation

# 问答对生成

self-alignment 训练：采用大模型+长Context 增强作为教师模型，产生规范问答对用于训练小模型

- 生成问题
	- AI自动生成
	- 人类补全问题库
- 生成Q&A对
	- 分析代码库间依赖关系，生成代码依赖表格或依赖图，从表格/图中提取信息作为上下文
	- 采用强模型的API生成问答对
- Q&A对清洗
	- 利用LLM-As-A-Judge进行评分，评分较低的重新生成
	- 人类观察Q&A对质量

现有问题：
- 依赖关系
- 类定义,类结构
- 函数定义和实现

上层依赖关系：
查找所有图中的关系，生成依赖关系，产生Q&A对
```
You are an Assistant to create question answer pairs for a programming repository.
You will receive a table with information about all used imports and files of one file of a programming repository.
Your task is create a short question and answer pair about the table. Vary the question so that you are ask for only one specific row sometimes about the whole table.
Please either ask about imported libraries or imported files, orientate on the category column.
Also write questions where the answer is No or the questions ask for a library that does not exist.
If you ask multiple question in one prompt always provide the file name.

Example Question could be (FILL <<>> with data):
- Which libraries are used in the file <<FILE_NAME>>?
- What libraries are imported directly in the file <<FILE_NAME>>?
- Does the file <<FILE_NAME>> also uses the library <<LIBRARY_NAME>>?
- Is the <<module>> part of the the file <<FILE_NAME>>?
- Are the files <<FILE_NAME>> and <<FILE_NAMES_2>> highly coupled?
- What library does the function <<FUNCTION_NAME>> belong to in the file <<FILE_NAME >> within the programming repository?
- Is the file <<FILE_NAME>> depending on the module <<module_name>>?


Please only provide questions and Answer in this format:
Question:
<<CREATED_QUESTION>>
Answer:
<<CREATED_ANSWER>>
Keep your questions and answer short.
```


类和函数关系问题：
问题生成阶段，对相关代码与类提出问题，并记录问题涉及到的代码段。
- 如，对于一个类的问题，加入类定义



生成答案前，采用代码分析工具/RAG方法分析本仓库内的函数及类的相关依赖关系，

Example:
File Node:
```
{"meta_data": 
{
"file_name": "config.py", 
"upper_path":"django/apps"
"module": "apps", 
"define_class": ["AppConfig"], 
"file_imports": ["import inspect","import os","from importlib import import_module"]
}
```

Code Node
```
"start_line": 13, "end_line": 42, 
"relative_function": [], 
"code": 
"""
class AppConfig:
    """Class representing a Django application and its configuration."""

    def __init__(self, app_name, app_module):
        # Full Python path to the application e.g. 'django.contrib.admin'.
        self.name = app_name

        # Root module for the application e.g. <module 'django.contrib.admin'
        # from 'django/contrib/admin/__init__.py'>.
        self.module = app_module

        # Reference to the Apps registry that holds this AppConfig. Set by the
        # registry when it registers the AppConfig instance.
        self.apps = None

        # The following attributes could be defined at the class level in a
        # subclass, hence the test-and-set pattern.

        # Last component of the Python path to the application e.g. 'admin'.
        # This value must be unique across a Django project.
        if not hasattr(self, "label"):
            self.label = app_name.rpartition(".")[2]
        if not self.label.isidentifier():
            raise ImproperlyConfigured(
                "The app label '%s' is not a valid Python identifier." % self.label
            )

        # Human-readable name for the application e.g. "Admin".
        if not hasattr(self, "verbose_name"):
            self.verbose_name = self.label.title()
"""}
```

上下文查找方式:

Class/Function Extraction->Code Node->Relative Context
#TODO
跟据问题改一下：
- 对着SWE-Bench构造问题
增加定制的Question,仓库定位问题：
- Where 类型的问题
- 是不是本仓库的类
- 这个名字的类有几个，有哪些实现？ Feedback
- 是类还是函数
- 根据关系找问题
- text folder


```
You are an AI programming assistant that is an expert in extracting relative class and function from a question for the {{project_name}} Git repository.
Your task is to extract the name of the class and function that is mentioned or possbly needed in the question.

Please only give truthful answers, and if you don't know an answer, don't hallucinate, but write that you don't know it.
Question:
<<QUESTION>>
```

将涉及到的代码查询出来后加入到Prompt中
```
# Used in <<Function/Class name + Relative Line>>
<<LINKED CODE>>
```

之后，利用LLM生成答案：
```
You are an AI programming assistant that is an expert in the {{project_name}} Git repository.
Your task to answer questions about this repository as good as possible. Consider the following information about the repository.
The repository is open-source and hosted on GitHub. Anybody can contribute to the codebase.
Please only give truthful answers, and if you don't know an answer, don't hallucinate, but write that you don't know it.

Question:
<<QUESTION>>
Relative Code:
<<RELATIVE CODE>>
Relative Context:
<<RELATIVE CONTEXT>>
```

问答对筛选
- 利用LLM-As-Judge进行评分，评分较低的问题重新生成问答对
```
Below is an instruction from an user and a candidate answer. Evaluate whether or not the answer is a good example of how AI Assistant should respond to the user's instruction. Please assign a score using the following 5-point scale:
1: It means the answer is incomplete, vague, off-topic, controversial, or not exactly what the user asked for. For example, some content seems missing, numbered list does not start from the beginning, the opening sentence repeats user's question. Or the response is from another person’s perspective with their personal experience (e.g. taken from blog posts), or looks like an answer from a forum. Or it contains promotional text, navigation text, or other irrelevant information.
2: It means the answer addresses most of the asks from the user. It does not directly address the user's question. For example, it only provides a high-level methodology instead of the exact solution to user's question.
3: It means the answer is helpful but not written by an AI Assistant. It addresses all the basic asks from the user. It is complete and self contained with the drawback that the response is not written from an AI assistant's perspective, but from other people's perspective. The content looks like an excerpt from a blog post, web page, or web search results. For example, it contains personal experience or opinion, mentions comments section, or share on social media, etc.
4: It means the answer is written from an AI assistant's perspective with a clear focus of addressing the instruction. It provide a complete, clear, and comprehensive response to user’s question or instruction without missing or irrelevant information. It is well organized, self-contained, and written in a helpful tone. It has minor room for improvement, e.g. more concise and focused.
5: It means it is a perfect answer from an AI Assistant. It has a clear focus on being a helpful AI Assistant, where the response looks like intentionally written to address the user's question or instruction without any irrelevant sentences. The answer provides high quality content, demonstrating expert knowledge in the area, is very well written, logical, easy-to-follow, engaging and insightful.

Please first provide a brief reasoning you used to derive the rating score, and then write "Score: <rating>" in the last line.

{generated_instruction}
{response}
```
