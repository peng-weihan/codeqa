import ijson  # 需要先安装: pip install ijson
from repo_qa_generator.models.data_models import QAPair
def stream_read_qa_json_file(file_path):
    """
    流式读取大型JSON数组文件
    """
    qa_pairs = []
    
    try:
        with open(file_path, 'rb') as f:
            # 使用ijson流式解析JSON数组
            for item in ijson.items(f, 'item'):
                try:
                    qa_pair = QAPair.model_validate(item)
                    qa_pairs.append(qa_pair)
                    
                    # 每处理100个问题输出一次进度
                    if len(qa_pairs) % 10000 == 0:
                        print(f"已读取 {len(qa_pairs)} 个问题...")
                except Exception as e:
                    print(f"验证问题失败: {str(e)[:100]}...")
                    
    except Exception as e:
        print(f"读取文件出错: {e}")
        
    print(f"成功读取了 {len(qa_pairs)} 个问答对")
    return qa_pairs