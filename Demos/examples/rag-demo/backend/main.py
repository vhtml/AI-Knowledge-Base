# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .agent import run_agent

app=FastAPI(title="My Agent API")
app.add_middleware(CORSMiddleware,allow_origins=["http://localhost:5173"],allow_methods=["*"],allow_headers=["*"])

class ChatRequest(BaseModel):
 message: str = Field(min_length=1,max_length=4000)

class ChatResponse(BaseModel):
 answer: str

@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/chat",response_model=ChatResponse)
def chat(body: ChatRequest):
 try: return {"answer":run_agent(body.message)}
 except Exception as e:
  # 生产环境记录详细日志，但不要把密钥等细节返回前端
  raise HTTPException(500,"Agent 暂时不可用") from e