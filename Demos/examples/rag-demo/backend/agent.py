# backend/agent.py
import json
import os
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL") or None,
    api_key=os.getenv("LLM_API_KEY"),
)

client = OpenAI()

def calculator(a, b, operation):
 if operation == "add": return str(a+b)
 if operation == "multiply": return str(a*b)
 if operation == "divide": return "除数不能为0" if b==0 else str(a/b)
 return "不支持"

def current_time(): return datetime.now().astimezone().isoformat()
FUNCTIONS={"calculator":calculator,"current_time":current_time}
TOOLS=[
 {"type":"function","name":"calculator","description":"精确计算",
  "parameters":{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"},"operation":{"type":"string","enum":["add","multiply","divide"]}},"required":["a","b","operation"],"additionalProperties":False},"strict":True},
 {"type":"function","name":"current_time","description":"获取当前时间",
  "parameters":{"type":"object","properties":{},"additionalProperties":False},"strict":True}
]

def run_agent(task, max_steps=8):
 items=[{"role":"user","content":task}]
 for _ in range(max_steps):
  r=client.responses.create(model=os.getenv("LLM_MODEL") or "gpt-5.4-mini",tools=TOOLS,input=items)
  items += r.output
  calls=[x for x in r.output if x.type=="function_call"]
  if not calls: return r.output_text
  for call in calls:
   try: result=FUNCTIONS[call.name](**json.loads(call.arguments))
   except Exception as e: result=f"工具错误：{e}"
   print("调用",call.name,call.arguments,"=>",result)
   items.append({"type":"function_call_output","call_id":call.call_id,"output":result})
 return "超过最大步骤，已停止"
