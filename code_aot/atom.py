import asyncio
import logging
from contextlib import contextmanager
from functools import wraps

import code_aot.prompt_generator as code
from code_aot.utils import extract_json, save_question_answer
from code_aot.llm import gen
from code_aot.score import score_code

# 配置日志
logger = logging.getLogger("code_aot")

RETRIES_TIMES = 3
ATOM_DEPTH = 3
MAX_RETRIES = 5
logger = logging.getLogger("code_aot")
logger.setLevel(logging.DEBUG)
def retry(func_name):
    # function wrapped with LLM calls and retries to get the result using the prompt from atom.py
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            global MAX_RETRIES
            retries = MAX_RETRIES
            logger.debug("开始执行 %s 函数，最大重试次数: %s", func_name, retries)
            while retries >= 0:
                logger.debug("%s 剩余重试次数: %s", func_name, retries)
                prompt = getattr(code, func_name)(*args, **kwargs)
                # if func_name == "contract":
                #     logger.debug("调用 LLM 处理 %s 请求 (xml格式)", func_name)
                #     response = await gen(prompt, response_format="text")
                #     logger.debug("LLM 返回结果: %s", response)
                #     result = extract_xml(response)
                # else:
                logger.debug("调用 LLM 处理 %s 请求 (文本格式)", func_name)
                response = await gen(prompt, response_format="text")
                result = extract_json(response)
                if isinstance(result, dict):
                    result["response"] = response
                if code.check(func_name, result):
                    logger.debug("%s 检查通过，返回结果", func_name)
                    return result
                logger.warning("%s 检查未通过，进行重试", func_name)
                retries -= 1
            logger.warning("%s 已达到最大重试次数，返回最后结果", func_name)
            return result if isinstance(result, dict) else {}
        return wrapper
    return decorator


def calculate_depth(sub_questions: list):
    try:
        logger.debug("计算问题深度，共 %s 个子问题", len(sub_questions))
        n = len(sub_questions)
        
        #intialize distances matrix with inf
        distances = [[float('inf') for _ in range(n)] for _ in range(n)]

        #Set direct dependencies
        for i, sub_question in enumerate(sub_questions):
            # Distance to self is 0
            distances[i][i] = 0

            # Set direct dependencies 1
            for dep in sub_question.get("dependencies", []):
                distances[dep][i] = 1

        #Floyd-Warshall algorithm to find shortest paths
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if distances[i][k] + distances[k][j] < distances[i][j]:
                        distances[i][j] = distances[i][k] + distances[k][j]
        
        #Find max distance
        max_distance = 0
        for i in range(n):
            for j in range(n):
                if distances[i][j] != float("inf"):
                    max_distance = max(max_distance, distances[i][j])
        logger.debug("计算得到问题深度为: %s", int(max_distance))
        return int(max_distance)
    except Exception as e:
        logger.error("计算深度过程中发生错误: %s", str(e), exc_info=True)
        return 0
                    
async def decompose(question: str, contexts: str=None):
    logger.info("开始分解问题: %s...", question[:50])
    retries = RETRIES_TIMES
    if contexts:
        logger.debug("使用上下文进行多步分解")
        try:
            multistep_result = await multistep_with_context(question, contexts)
            logger.debug("使用上下文的多步分解成功")
        except Exception as e:
            logger.error("使用上下文的多步分解初次尝试失败: %s", str(e), exc_info=True)
            multistep_result = None
            
        while retries > 0:
            try:
                logger.debug("尝试标注问题")
                multistep_result = await label(question, multistep_result["response"], multistep_result["answer"])
                break
            except Exception as e:
                logger.error("标注问题失败: %s", str(e), exc_info=True)
                await asyncio.sleep(1)
                retries -= 1
                continue
    else:
        logger.debug("不使用上下文进行多步分解")
        multistep_result = await multistep(question)
        while retries > 0:
            try:
                logger.debug("尝试标注问题")
                multistep_result = await label(question, multistep_result["response"], multistep_result["answer"])
            except Exception as e:
                logger.error("标注问题失败: %s", str(e), exc_info=True)
                await asyncio.sleep(1)
                retries -= 1
                continue

    try:
        logger.debug("计算问题深度")
        depth = calculate_depth(multistep_result["sub-questions"])
        if depth > 0:
            logger.info("问题分解完成，深度为 %s", depth)
            return multistep_result
    except Exception as e:
        logger.error("计算深度过程中发生错误: %s", str(e), exc_info=True)
        await asyncio.sleep(1)
        retries -= 1
    return multistep_result

async def merging(question: str, decompose_result: dict, independent_sub_questions: list, dependent_sub_questions: list, contexts: str=None):
    logger.info("开始合并问题结果，独立子问题数量: %s，依赖子问题数量: %s", 
                len(independent_sub_questions), len(dependent_sub_questions))
    contract_args = {
        "question": question,
        "decompose_result": decompose_result,
        "independent_subquestions": independent_sub_questions,
        "dependent_subquestions": dependent_sub_questions,
        "contexts": contexts
    }
    try:
        logger.debug("调用合约函数处理问题")
        contracted_result = await contract(**contract_args)
        contracted_result["response"] = contracted_result["thought"]
        
        # Thought and question after contraction
        contracted_thought = contracted_result["response"]
        contracted_question = contracted_result["question"]
        logger.debug("收缩后的问题: %s...", contracted_question[:50])

        # Solve the optimized question
        direct_args = (
            contracted_question,
            contracted_result.get("contexts", None)
        )
        logger.debug("调用直接回答函数处理收缩后的问题")
        contraction_result = await direct(*direct_args)
        logger.info("合并问题结果完成")
        return contracted_thought, contracted_question, contraction_result
    except Exception as e:
        logger.error("合并问题过程中发生错误: %s", str(e), exc_info=True)
        raise

async def atom_with_context(question: str, contexts: str=None, direct_result=None, decompose_result=None, depth=None):
    logger.info("使用上下文开始处理问题: %s...", question[:50])
    if depth == 0:
        logger.warning("深度为0，不执行处理")
        return None
    
    # 第一步先进行direct answer,获取足够长的参考reasoning
    try:
        logger.debug("开始直接回答问题")
        direct_args = (question, contexts)
        direct_result = direct_result if direct_result else await direct(*direct_args)
        logger.info("直接回答完成")
    except Exception as e:
        logger.error("直接回答过程中发生错误: %s", str(e), exc_info=True)
        return None

    # 第二步进行decompose,根据问题和给出答案的reasoning的过程拆解出多个decompose_result
    try:
        logger.debug("开始分解问题")
        decompose_args = {"question": question, "contexts": contexts}
        decompose_result = decompose_result if decompose_result else await decompose(**decompose_args)
        logger.info("问题分解完成")
    except Exception as e:
        logger.error("问题分解过程中发生错误: %s", str(e), exc_info=True)
        return {
            "method": "direct",
            "response": direct_result.get("response"),
            "answer": direct_result.get("answer"),
        }
    logger.debug("decompose_result: %s", decompose_result)
    try:
        depth = depth if depth else min(ATOM_DEPTH, calculate_depth(decompose_result["sub-questions"]))
        logger.debug("问题深度: %s", depth)
        
        independent_sub_questions = [sub_question for sub_question in decompose_result["sub-questions"] if sub_question["dependencies"] == []]
        dependent_sub_questions = [sub_question for sub_question in decompose_result["sub-questions"] if sub_question["dependencies"] != []]
        logger.debug("独立子问题数量: %s，依赖子问题数量: %s", 
                    len(independent_sub_questions), len(dependent_sub_questions))
        
        # Get contraction result
        merging_args = {
            "question": question,
            "decompose_result": decompose_result,
            "independent_sub_questions": independent_sub_questions,
            "dependent_sub_questions": dependent_sub_questions,
            "contexts": contexts
        }

        logger.debug("开始合并问题结果")
        contracted_thought, contracted_question, contraction_result = await merging(**merging_args)
        
        # Update contraction result with additional information
        # For all the independent sub-questions, add all the independent sub-questions
        # For all the dependent sub-questions, add the contraction thought and the sub-question description
        contraction_result["contraction_thought"] = contracted_thought
        contraction_result["sub-questions"] = independent_sub_questions + [{
            "description": contracted_question,
            "answer": contraction_result.get("answer", ""),
            "response": contraction_result.get("response", ""),
            "dependencies": []
        }]

        for question in contraction_result["sub-questions"]:
            save_question_answer(question["description"], question["answer"])
        
        logger.debug("准备进行集成")
        ensemble_args = {
            "question": question,
            "solutions": [direct_result["response"], decompose_result["response"], contraction_result["response"]],
            "contexts": contexts
        }
        
        ensemble_result = await ensemble(**ensemble_args)
        ensemble_answer = ensemble_result.get("answer", "")
        logger.info("集成完成")

        # Calculate the score of the ensemble answer
        logger.debug("计算回答得分")
        scores = []
        if all(result["answer"] == ensemble_answer for result in [direct_result, decompose_result, contraction_result]):
            scores = [1, 1, 1]
            logger.debug("所有回答一致，得分相同")
        else:
            for result in [direct_result, decompose_result, contraction_result]:
                score = score_code(result["answer"], ensemble_answer)
                scores.append(score)
            logger.debug("回答得分: 直接=%s, 分解=%s, 合约=%s", scores[0], scores[1], scores[2])

        # Select best method based on scores
        methods = {
            2: ("contract", contraction_result),
            0: ("direct", direct_result),
            1: ("decompose", decompose_result),
            -1: ("ensemble", ensemble_result)
        }
        max_score_index = scores.index(max(scores))
        method, result = methods.get(max_score_index, methods[-1])
        logger.info("选择最佳方法: %s，分数: %s", method, 
                   scores[max_score_index] if max_score_index >= 0 else 'N/A')
        
        # Return appropriate result format
        return {
            "method": method,
            "response": result.get("response"),
            "answer": result.get("answer"),
        }
    except Exception as e:
        logger.error("处理过程中发生错误: %s", str(e), exc_info=True)
        # 发生错误时，返回直接答案作为备选
        return {
            "method": "direct",
            "response": direct_result.get("response"),
            "answer": direct_result.get("answer"),
        }

@retry("direct")
async def direct(question: str, contexts: str=None):
    if isinstance(question, (list, tuple)):
        question = ''.join(map(str, question))
    pass

@retry("multistep")
async def multistep(question: str, contexts: str=None):
    pass

@retry("multistep_with_context")
async def multistep_with_context(question: str, contexts: str=None):
    pass

@retry("label")
async def label(question: str, sub_questions: str, answer: str=None):
    pass

@retry("contract")
async def contract(question: str, sub_result: dict, independent_subquestions: list, dependent_subquestions: list, contexts: str=None):
    pass

@retry("ensemble")
async def ensemble(question: str, solutions: list, contexts: str=None):
    pass

@contextmanager
def temporary_retries(value):
    global MAX_RETRIES
    original = MAX_RETRIES
    logger.debug("临时修改最大重试次数从 %s 到 %s", original, value)
    MAX_RETRIES = value
    try:
        yield
    finally:
        logger.debug("恢复最大重试次数为 %s", original)
        MAX_RETRIES = original