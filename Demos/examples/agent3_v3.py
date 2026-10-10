import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
  base_url=os.getenv("LLM_BASE_URL") or None,
  api_key=os.getenv("LLM_API_KEY"),
)

# 使用previous_response_id

previousResponseId = None

while True:

  user_input = input("你：")

  response = client.responses.create(
    model=os.getenv("LLM_MODEL") or "gpt-6-astra",
    previous_response_id=previousResponseId,
    input=user_input,
  )

  previousResponseId = response.id

  reply = response.output_text

  print(f"[Agent回答]:{reply}\n")
