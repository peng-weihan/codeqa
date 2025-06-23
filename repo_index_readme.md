# 代码仓库索引工具

此工具用于对任意Git仓库或本地代码库建立语义搜索索引，并提供快速的代码检索功能。

## 功能特点

- 支持多种仓库源：
  - Git仓库URL
  - 本地仓库路径
  - SWE-bench实例ID
- 自动索引管理：
  - 自动派生索引名称
  - 索引持久化和加载
  - 远程索引下载
- 高级语义搜索：
  - 基于自然语言的代码搜索
  - 代码结构感知（函数、类等）
  - 精确匹配与模糊匹配

## 使用方法

### 通过命令行工具

```bash
# 从Git仓库URL创建索引
python example_repo_index.py --repo-url https://github.com/username/repo.git

# 从本地仓库路径创建索引
python example_repo_index.py --repo-path /path/to/local/repo

# 指定特定提交
python example_repo_index.py --repo-url https://github.com/username/repo.git --commit abc1234

# 强制重建索引
python example_repo_index.py --repo-path /path/to/local/repo --force-rebuild

# 执行语义搜索
python example_repo_index.py --repo-path /path/to/local/repo --query "查找处理用户登录的函数"
```

### 通过Python API

```python
# 从Git仓库URL创建索引
from moatless_qa.index import CodeIndex

# 方法1：使用新增的from_repository方法
code_index = CodeIndex.from_repository(
    repo_url="https://github.com/username/repo.git",
    commit="main",  # 可选
    index_name="my_custom_index"  # 可选
)

# 方法2：使用底层的create_repository和create_index函数
from moatless_qa.benchmark.swebench import create_repository, create_index

repository = create_repository(
    repo_url="https://github.com/username/repo.git"
)
code_index = create_index(repository=repository)

# 执行语义搜索
results = code_index.semantic_search(
    query="查找处理用户登录的函数",
    max_results=10
)

# 显示结果
for hit in results.hits:
    print(f"文件: {hit.file_path}")
    print(f"得分: {hit.score}")
    print(f"代码片段:\n{hit.content}")
    print("-" * 50)
```

## 环境变量配置

工具支持通过环境变量进行配置：

- `MOATLESS_INDEX_DIR`: 索引存储目录，默认为 `/tmp/index_store`
- `REPO_DIR`: 仓库基础目录，默认为 `/tmp/repos`
- `INDEX_STORE_URL`: 远程索引存储URL，用于下载预构建的索引

## 高级用法

### 使用预构建索引

对于常用的开源项目，我们提供预构建的索引以加快初始化：

```python
# 通过环境变量设置索引存储URL
os.environ["INDEX_STORE_URL"] = "https://your-server.com/indices/"

# 系统将尝试从远程下载预构建的索引
code_index = CodeIndex.from_repository(
    repo_url="https://github.com/popular/project.git"
)
```

### 自定义索引参数

```python
from moatless_qa.index.settings import IndexSettings

settings = IndexSettings(
    dimensions=1536,  # 向量维度
    embed_model="openai/text-embedding-3-large",  # 嵌入模型
    # 其他参数...
)

# 创建自定义索引
code_index = CodeIndex(
    file_repo=repository,
    settings=settings,
    max_results=50
)
code_index.run_ingestion(repo_path=repository.repo_dir)
``` 