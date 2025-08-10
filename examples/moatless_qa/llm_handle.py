import json
import os
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm  # 进度条工具
from openai import OpenAI
from typing import List, Dict, Union

from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

class BatchAnswerOptimizer:

    def process_jsonl_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        use_llm: bool = True,
        batch_size: int = 100,
        overwrite: bool = False
    ) -> Dict[str, int]:
        """
        批量处理JSONL文件
        
        Args:
            input_path: 输入JSONL文件路径
            output_path: 输出JSONL文件路径
            use_llm: 是否使用LLM优化
            batch_size: 批处理大小(用于进度显示)
            overwrite: 是否覆盖已存在输出文件
            
        Returns:
            处理统计信息: {'total': 总数量, 'processed': 成功处理数量}
        """
        # 路径处理
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        # 检查输出文件是否存在
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"输出文件已存在: {output_path}")
        
        # 统计数据
        stats = {'total': 0, 'processed': 0}
        
        with (
            open(input_path, 'r', encoding='utf-8') as f_in,
            open(output_path, 'w', encoding='utf-8') as f_out,
            tqdm(desc="处理进度") as pbar
        ):
            for line in f_in:
                stats['total'] += 1
                try:
                    data = json.loads(line.strip())
                    optimized_data = self._process_single_item(data, use_llm)
                    f_out.write(json.dumps(optimized_data, ensure_ascii=False) + '\n')
                    stats['processed'] += 1
                    
                    # 更新进度条
                    if stats['processed'] % batch_size == 0:
                        pbar.update(batch_size)
                except Exception as e:
                    print(f"\n处理第{stats['total']}行时出错: {str(e)}")
                    continue
        
        return stats
    
    def _process_single_item(
        self,
        data: Dict,
        use_llm: bool
    ) -> Dict:
        """处理单个JSON对象"""
        # 提取问题
        question = data.get('question', '')
        print(f"处理问题: {question}")
        
        # 提取原始答案
        answer = data.get('mcts_answer', '')
        rag_answer = data.get('rag_answer', '')
        
        # 生成优化答案
        if use_llm and client:
            try:
                optimized_answer = self._optimize_with_llm(question, answer, rag_answer)
                data['ground_truth'] = optimized_answer
            except Exception as e:
                print(f"LLM处理失败: {str(e)}")
                data['ground_truth'] = self._merge_answers(answer, rag_answer)

        else:
            data['ground_truth'] = self._merge_answers(answer, rag_answer)
        
        return data
    
    def _optimize_with_llm(
        self,
        question: str,
        answer: str,
        rag_answer: str
    ) -> str:
        """使用LLM优化答案"""
        prompt = f"""
        [任务说明]
        你是一位资深技术专家，需要整合以下两个答案，生成更优质的版本：
        
        [原始问题]
        {question}
        
        [现有答案1]
        {answer}
        
        [现有答案2]
        {rag_answer}
        
        [输出要求]
        1. 合并相同观点，消除冗余
        2. 补充缺失的技术细节
        3. 按逻辑组织内容结构
        4. 保持技术准确性
        5. 使用中文回答(除非问题本身是英文)
        
        请直接返回优化后的答案，不要包含额外说明。
        """
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        reference = response.choices[0].message.content.strip()
        
        return reference
    
    def _merge_answers(
        self,
        answer: str,
        rag_answer: str
    ) -> str:
        """简单合并策略"""
        if not (answer or rag_answer):
            return "无可用答案"
            
        if answer and rag_answer:
            if rag_answer.lower() in answer.lower():
                return answer
            if answer.lower() in rag_answer.lower():
                return rag_answer
            return f"合并答案:\n[答案1]\n{answer}\n\n[答案2]\n{rag_answer}"
        
        return answer or rag_answer

# 使用示例
if __name__ == "__main__":
    # 初始化优化器
    optimizer = BatchAnswerOptimizer()
    
    # 输入输出路径
    input_file = "/data3/pwh/codeqa/dataset/generated_answers/django_answers_mcts.jsonl"
    output_file = "/data3/pwh/codeqa/dataset/generated_answers/django_answers_ref.jsonl"
    
    # 处理文件
    try:
        print(f"开始处理文件: {input_file}")
        stats = optimizer.process_jsonl_file(
            input_path=input_file,
            output_path=output_file,
            use_llm=True,  # 启用LLM优化
            overwrite=True
        )
        
        print(f"\n处理完成: 共处理{stats['total']}条，成功{stats['processed']}条")
        print(f"结果已保存到: {output_file}")
    except Exception as e:
        print(f"处理失败: {str(e)}")