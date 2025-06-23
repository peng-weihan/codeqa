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