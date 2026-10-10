import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
  base_url=os.getenv("LLM_BASE_URL") or None,
  api_key=os.getenv("LLM_API_KEY"),
)

# 用history缓存聊天记录
history = []

while True:

  user_input = input("你：")

  history.append({"role": "user", "content": user_input})

  response = client.responses.create(
    model=os.getenv("LLM_MODEL") or "gpt-6-astra",
    input=history,
  )

  reply = response.output_text

  print(f"[Agent回答]:{reply}\n")

  history.extend(response.output)
