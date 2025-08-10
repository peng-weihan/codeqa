import json
from typing import Type, TypeVar
from pathlib import Path
from pydantic import BaseModel
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
# 定义泛型类型变量（用于支持任意Pydantic模型）
T = TypeVar('T', bound=BaseModel)

def load_repository_from_json(
    json_path: str | Path,
    model_class: Type[T],
    encoding: str = 'utf-8'
) -> T:
    """
    从JSON文件加载并还原Repository对象
    
    Args:
        json_path: JSON文件路径
        model_class: 目标Pydantic模型类（如Repository）
        encoding: 文件编码（默认utf-8）
    
    Returns:
        还原后的模型实例
    
    Raises:
        FileNotFoundError: 文件不存在时抛出
        json.JSONDecodeError: JSON解析失败时抛出
        pydantic.ValidationError: 数据验证失败时抛出
    """
    try:
        with open(json_path, 'r', encoding=encoding) as f:
            data = json.load(f)
            return model_class(**data)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON文件不存在: {json_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON解析失败: {e}")
    except Exception as e:
        raise RuntimeError(f"加载失败: {e}")


# 使用示例
if __name__ == "__main__":
    from datetime import datetime
    from repo_qa_generator.models.data_models import Repository

    full_json_path = "/data3/pwh/repo_analysis/20250802_213947_repo_full.json"
    
    # 加载示例
    try:
        repo = load_repository_from_json(full_json_path, Repository)
        print(f"成功加载Repository对象")
        print(f"对象类型: {type(repo)}")
    except Exception as e:
        print(f"加载失败: {e}")