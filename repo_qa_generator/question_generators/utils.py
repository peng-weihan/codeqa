import os
from typing import Dict, List
from repo_qa_generator.models.data_models import CodeRelationship

def load_template_questions(questions_dir:str,name_list:list[str]) -> Dict[str, List[str]]:
    """
    加载所有模板问题文件
        
    Returns:
        Dict[str, List[str]]: 按类别分组的模板问题
    """
    templates = {}
    for filename in os.listdir(questions_dir):
        category = filename.split('.')[0]
        if filename.endswith('.txt') and category in name_list:
            file_path = os.path.join(questions_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                questions = [q.strip() for q in f.readlines() if q.strip()]
            templates[category] = questions
    return templates


def load_template_questions_v2(questions_dir:str, name_list:list[str]) -> Dict[str, List[str]]:
    """
    加载所有子目录（如 how、why、what、where）下的模板问题文件
    
    Args:
        questions_dir (str): 根目录路径，包含多个分类子目录
        name_list (List[str]): 允许加载的子目录名列表，如 ['how', 'why', 'what', 'where']

    Returns:
        Dict[str, List[str]]: 按子目录分类的模板问题字典，键为子目录名，值为所有该目录下问题的列表
    """
    templates = {}
    for category in name_list:
        category_path = os.path.join(questions_dir, category)
        if not os.path.isdir(category_path):
            continue  # 如果子目录不存在，跳过
        
        questions = []
        for filename in os.listdir(category_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(category_path, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                     for line in f:
                        line = line.strip()
                        if not line or line.startswith('//'):
                            continue  # 跳过空行和注释行
                        questions.append(line)
        templates[category] = questions
    return templates

def format_code_relationship_list(relationship: list[CodeRelationship]) -> list[str]:
    """
    格式化代码关系列表，每10个关系分成一组
    
    Args:
        relationship: 关系列表
        
    Returns:
        字符串列表，每个字符串包含最多30个关系的描述
    """
    result = []
    current_group = ""
    
    for i, rel in enumerate(relationship):
        current_group += f"{rel.source_type} {rel.source_id} {rel.relationship_type} {rel.target_type} {rel.target_id}\n"
        # 每10个关系或到达列表末尾时，添加到结果并重置当前组
        if (i + 1) % 10 == 0 or i == len(relationship) - 1:
            result.append(current_group)
            current_group = ""
            
    return result