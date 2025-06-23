import random
import asyncio
import openai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
url = os.getenv("OPENAI_URL")
model_name = os.getenv("OPENAI_MODEL")

client = openai.AsyncClient(base_url=url, api_key=api_key)

MAX_RETRIES = 3
total_prompt_tokens, total_completion_tokens, call_count, cost = 0, 0, 0, 0
current_prompt_tokens, current_completion_tokens = 0, 0

def set_model(model):
    global model_name
    model_name = model

async def gen(msg, model=None, temperature=None, response_format="json_object"):
    
    global call_count, cost, current_prompt_tokens, current_completion_tokens, model_name
    if not model:
        model = model_name
    errors = []
    call_count += 1

    DEFAULT_RETRY_AFTER = random.uniform(0.1, 2)
    for retry in range(MAX_RETRIES):
        try:
            async with asyncio.timeout(120):
                response = await client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "user", "content": msg},
                        ],
                        temperature=temperature,
                        stop=None,
                        max_tokens=8192,
                        response_format={"type": response_format}
                        )
                content = response.choices[0].message.content
                
                usage = response.usage
                current_prompt_tokens = usage.prompt_tokens
                current_completion_tokens = usage.completion_tokens
                update_token()
                
                return content
        except asyncio.TimeoutError:
            errors.append("Request timeout")
        except openai.RateLimitError:
            errors.append("Rate limit error")
        except openai.APIError as e:
            errors.append(f"API error: {str(e)}")
        except Exception as e:
            errors.append(f"Error: {type(e).__name__}, {str(e)}")
        
        await asyncio.sleep(DEFAULT_RETRY_AFTER * (2 ** retry))

    print(f"Error log: {errors}")


def get_cost():
    return cost

def update_token():
    global total_prompt_tokens, total_completion_tokens, current_completion_tokens, current_prompt_tokens
    total_prompt_tokens += current_prompt_tokens
    total_completion_tokens += current_completion_tokens

def reset_token():
    global total_prompt_tokens, total_completion_tokens, call_count
    total_prompt_tokens = 0
    total_completion_tokens = 0
    call_count = 0

def get_token():
    return total_prompt_tokens, total_completion_tokens

def get_call_count():
    return call_count