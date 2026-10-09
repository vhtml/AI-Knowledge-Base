import json
import os
import subprocess

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
  base_url=os.getenv("LLM_BASE_URL") or None,
  api_key=os.getenv("LLM_API_KEY"),
)

SYSTEM_PROMPT = """
你是大内太监总管，侍奉皇上多年，忠心耿耿。
说话风格符合宫廷太监，语气恭敬谦卑。
你必须尊称用户为皇上。
每次回复前必须加上固定前缀“奉天承运皇帝诏曰”，然后再给出回答。
使用中文回复。
"""

history = []

while True:

  user_input = input("你：")

  history.append({"role": "user", "content": user_input})

  response = client.responses.create(
    model=os.getenv("LLM_MODEL") or "gpt-6-astra",
    instructions=SYSTEM_PROMPT,
    input=history,
  )

  reply = response.output_text

  print(f"[Agent回答]:{reply}\n")

  history.append({"role":"assistant", "content": reply})
