from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field, validator
from enum import IntEnum
import json

class EvaluationScore(IntEnum):
    """Evaluation score enum"""
    INCOMPLETE = 1  # Answer is incomplete, vague, or off-topic
    BASIC = 2       # Answer addresses the question but lacks accuracy or detail
    GOOD = 3        # Answer is complete and helpful but could be improved
    VERY_GOOD = 4   # Answer is very good, accurate, and comprehensive
    PERFECT = 5     # Answer is perfect, accurate, comprehensive, and easy to understand

class GPTEvaluationResponse(BaseModel):
    """Model for parsing GPT's evaluation response"""
    score: EvaluationScore
    reasoning: str

    @validator('score')
    def validate_score(cls, v):
        if not 1 <= v <= 5:
            raise ValueError('Score must be between 1 and 5')
        return v

class FileNode(BaseModel):
    """Data model for file node"""
    file_name: str
    upper_path: str
    module: str
    define_class: List[str]
    imports: List[str]

class CodeNode(BaseModel):
    """Data model for code node"""
    start_line: int
    end_line: int
    belongs_to: FileNode
    relative_function: List[str]
    code: str

class ResultPair(BaseModel):
    """Data model for result pair"""
    answer: str
    ground_truth: Optional[str] = Field(default=None,description="ground truth of the question")
    thought: str

class QAPair(BaseModel):
    """Data model for question-answer pair"""
    question: str
    answer: Optional[str] = Field(default=None,description="answer of the question")
    relative_code_list: Optional[List[CodeNode]] = Field(default=None,description="code list of the question")
    ground_truth: Optional[str] = Field(default=None,description="ground truth of the question")
    score: Optional[float] = None

class QAGeneratorResponse(BaseModel):
    """Data model for question-answer pair generator response"""
    question: str
    ground_truth: str

class QAGeneratorResponseList(BaseModel):
    """Data model for question-answer pair generator response list"""
    qa_pairs: List[QAGeneratorResponse]

class QAPairListResponse(BaseModel):
    """Data model for question-answer pair list response"""
    qa_pairs: List[QAPair]
    
class EvaluationResult(BaseModel):
    """Data model for evaluation result"""
    qa_pair: QAPair
    score: float
    reasoning: str
    suggestions: Optional[List[str]] = Field(default_factory=list)

    @validator('score')
    def validate_score(cls, v):
        if not 1 <= v <= 5:
            raise ValueError('Score must be between 1 and 5')
        return v

class ClassAttribute(BaseModel):
    """数据模型用于类属性/字段信息"""
    name: str
    class_name: str = Field(...,description="所属的类名")
    related_functions: List[str] = Field(default_factory=list,description="修改Attribute的值所涉及到的方法名")
    type_hint: Optional[str] = None

class VariableDefinition(BaseModel):
    """数据模型用于变量定义信息"""
    name: str
    docstring: Optional[str] = None
    relative_code: Optional[CodeNode] = Field(default=None, description="变量定义涉及到的代码段")
    scope: str = Field("global", description="变量作用域: global/local/class")
    function_name: Optional[str] = Field(default=None, description="如果是局部变量，所属的函数名")
    class_name: Optional[str] = Field(default=None, description="如果是类变量，所属的类名")
    type_hint: Optional[str] = Field(default=None, description="变量类型提示")
    value: Optional[str] = Field(default=None, description="变量值的字符串表示")
    is_constant: bool = Field(default=False, description="是否是常量（全大写命名）")
    references: List[str] = Field(default_factory=list, description="引用该变量的函数/方法")

class FunctionDefinition(BaseModel):
    """数据模型用于函数/方法定义信息"""
    name: str
    docstring: Optional[str] = None
    relative_code: Optional[CodeNode] = Field(..., description="函数定义涉及到的代码段")
    is_method: bool = False
    class_name: Optional[str] = None
    parameters: List[str] = Field(default_factory=list)
    calls: List[str] = Field(default_factory=list)

class ClassDefinition(BaseModel):
    """数据模型用于类定义信息"""
    name: str
    docstring: Optional[str] = None
    relative_code: Optional[CodeNode] = Field(..., description="仓库定义涉及到的代码段")
    methods: List[FunctionDefinition] = Field(default_factory=list,description="仓库包含的方法")
    attributes: List[ClassAttribute] = Field(default_factory=list)

class ModuleNode(BaseModel):
    """模块节点模型"""
    name: str
    path: str
    files: List[FileNode] = Field(default_factory=list)
    sub_modules: List["ModuleNode"] = Field(default_factory=list)
    is_package: bool = False
    
    class Config:
        arbitrary_types_allowed = True

class CodeRelationship(BaseModel):
    """代码关系模型"""
    source_type: str = Field(..., description="源类型: class/function/attribute")
    source_id: str = Field(..., description="源标识符")
    target_type: str = Field(..., description="目标类型: class/function/attribute")
    target_id: str = Field(..., description="目标标识符")
    relationship_type: str = Field(..., description="关系类型: inherits/calls/uses/implements")

class RepositoryStructure(BaseModel):
    """仓库结构模型"""
    root_modules: List[ModuleNode] = Field(default_factory=list)
    classes: List[ClassDefinition] = Field(default_factory=list)
    functions: List[FunctionDefinition] = Field(default_factory=list)
    attributes: List[ClassAttribute] = Field(default_factory=list)
    core_functionality: Optional[str] = None
    variables: List[VariableDefinition] = Field(default_factory=list)
    
    # 添加依赖图
    dependency_graph: Dict[str, List[str]] = Field(
        default_factory=dict, 
        description="模块间依赖关系图: {模块路径: [依赖模块路径...]}"
    )
    relationships: List[CodeRelationship] = Field(default_factory=list, description="代码元素间的关系")

class Repository(BaseModel):
    """代码仓库主模型"""
    id: str = Field(..., description="仓库唯一标识符")
    name: str = Field(..., description="仓库名称")
    url: Optional[str] = Field(None, description="仓库URL")
    description: Optional[str] = Field(None, description="仓库描述")
    structure: RepositoryStructure = Field(default_factory=RepositoryStructure, description="仓库结构")
    qa_pairs: List[QAPair] = Field(default_factory=list, description="与仓库关联的问答对")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "repo-123",
                "name": "example-repo",
                "url": "https://github.com/user/example-repo"
            }
        } 

def load_repository_from_json(file_path: str) -> Repository:
    """从JSON文件加载并重建Repository实例"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 使用Pydantic的model_validate方法（v2）或parse_obj方法（v1）
    try:
        # Pydantic v2 方式（推荐）
        return Repository.model_validate(data)
    except AttributeError:
        print("Unable to use model_validate, falling back to parse_obj")
        return Repository.parse_obj(data)

# class CodeReference(BaseModel):
#     """代码引用模型"""
#     class_name: Optional[str] = Field(None, description="引用的类名")
#     function_name: Optional[str] = Field(None, description="引用的函数名")
#     attribute_name: Optional[str] = Field(None, description="引用的属性名")
#     file_path: str = Field(..., description="文件路径")
#     line_range: Tuple[int, int] = Field(..., description="代码行范围")
#     snippet: str = Field(..., description="代码片段")
#     reference_type: str = Field(..., description="引用类型: 定义/调用/使用")

# class EnhancedQAPair(BaseModel):
#     """增强版问答对模型"""
#     id: str = Field(..., description="问答对唯一标识符")
#     question: str = Field(..., description="问题")
#     answer: str = Field(..., description="回答")
#     references: List[CodeReference] = Field(default_factory=list, description="代码引用列表")
#     related_classes: List[str] = Field(default_factory=list, description="相关类名称列表")
#     related_functions: List[str] = Field(default_factory=list, description="相关函数名称列表")
#     score: Optional[float] = None
#     tags: List[str] = Field(default_factory=list, description="问题标签")
    
#     class Config:
#         schema_extra = {
#             "example": {
#                 "id": "qa-456",
#                 "question": "类X的主要功能是什么?",
#                 "answer": "类X主要负责数据处理...",
#                 "references": [
#                     {
#                         "class_name": "X",
#                         "file_path": "src/models/x.py",
#                         "line_range": [10, 50],
#                         "snippet": "class X:\n    def __init__(self)...",
#                         "reference_type": "definition"
#                     }
#                 ]
#             }
#         } 