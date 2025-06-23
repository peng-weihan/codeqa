# Standard Question Generator

## 简介

Standard Question Generator 是一个用于自动生成针对软件工程问题的标准化问题集的工具。该模块主要基于 LLM（大型语言模型）技术，能够从问题描述中生成不同维度的相关问题，帮助开发者更全面地理解和解决代码问题。

## 功能特点

- 基于问题陈述自动生成多维度的相关问题
- 支持五种问题维度：WHERE（位置）、WHAT（概念）、HOW（方法）、RELATION（关系）和 API（接口）
- 使用 OpenAI API 进行问题生成
- 支持问题解析和分类
- 提供问题优化和精炼功能

## 目录结构

```
standard_question_generator/
├── prompt_generator/           # 提示词生成模块
│   ├── question_generator.py   # 问题生成提示词
│   └── question_refine.py      # 问题精炼提示词
├── llm_question_generator.py   # LLM 问题生成主要实现
├── llm_question_refine.py      # LLM 问题精炼功能
├── analyze_statements.py       # 问题陈述分析工具
└── llm_models_config.py        # LLM 模型配置
```

## 使用方法

### 基本用法

```python
from src.standard_question_generator.llm_question_generator import generate_questions_with_openai_chat
from src.standard_question_generator.prompt_generator.question_generator import generate_questions

# 准备问题陈述
problem_statement = "在用户登录时，系统无法正确验证用户凭证，导致授权失败。"

# 生成提示词
prompt = generate_questions(problem_statement)

# 使用 OpenAI API 生成问题
questions = generate_questions_with_openai_chat(prompt)

# 打印生成的问题
for q in questions:
    print(f"{q['dimension']}: {q['question']}")
```

### 问题精炼

```python
from src.standard_question_generator.llm_question_refine import refine_questions_with_openai_chat
from src.standard_question_generator.prompt_generator.question_refine import generate_refine_prompt

# 准备待精炼的问题
questions = [
    {"dimension": "where", "question": "在代码库中哪里实现了用户凭证验证的功能？"},
    # 更多问题...
]

# 生成精炼提示词
prompt = generate_refine_prompt(questions)

# 使用 OpenAI API 精炼问题
refined_questions = refine_questions_with_openai_chat(prompt)
```

## 配置

在 `llm_models_config.py` 中可以配置 LLM API 的各项参数：

- `deployment_name`：部署名称
- `azure_endpoint`：Azure 端点 URL
- `openai_api_version`：OpenAI API 版本
- `max_tokens`：最大生成 token 数
- `temperature`：生成多样性

## 环境变量

使用前需要设置以下环境变量：

- `LLM_API_KEY`：OpenAI API 密钥

## 贡献

欢迎提交 Issues 或 Pull Requests 来改进此模块。在提交代码前，请确保代码符合项目的编码规范并通过所有测试。 