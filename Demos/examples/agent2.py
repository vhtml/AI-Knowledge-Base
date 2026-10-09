import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
  base_url=os.getenv("LLM_BASE_URL") or None,
  api_key=os.getenv("LLM_API_KEY"),
)

while True:

  user_input = input("你：")

  response = client.responses.create(
    model=os.getenv("LLM_MODEL") or "gpt-6-astra",
    input=[{"role": "user", "content": user_input}],
  )

  print(f"[Agent回答]:{response.output_text}\n")
