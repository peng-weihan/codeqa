import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
AIHUBMIX_API_KEY = os.getenv("AIHUBMIX_API_KEY")

client = OpenAI(
  base_url="https://aihubmix.com/v1",
  api_key= AIHUBMIX_API_KEY
)

completion = client.chat.completions.create(
  model="DeepSeek-V3", # 替换模型 id
  messages=[
    {
      "role": "developer",
      "content": "总是用中文回复"
    },    
    {
      "role": "user",
      "content": "What is the meaning of life?"
    }
  ],
  temperature=0.8,
  max_tokens=1024,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  seed=random.randint(1, 1000000000),
)

print(completion.choices[0].message.content)