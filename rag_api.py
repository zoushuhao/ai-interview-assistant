
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"   # 模型下载走国内镜像
from fastembed import TextEmbedding
from langchain_openai import ChatOpenAI
import numpy as np
from fastapi import FastAPI      # 从 fastapi 工具箱里拿 FastAPI 工具
from pydantic import BaseModel   # 从 pydantic 工具箱里拿 BaseModel 工具
import uvicorn                   # 整个 uvicorn 工具箱都拿来（因为要用 uvicorn.run）
from rag_embedding import search,llm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles   # 让 FastAPI 顺便托管网页文件
from pathlib import Path                      # 处理文件路径
from job_match_agent import app as match_app, SYSTEM_PROMPT
from langchain_core.messages import SystemMessage, HumanMessage

class ChatRequest(BaseModel):
    question: str
class MatchRequest(BaseModel):
    text: str

app = FastAPI()
app.add_middleware(CORSMiddleware,allow_origins = ["*"],allow_methods=["*"],allow_headers=["*"])

@app.post("/chat")
def chat(req: ChatRequest):
    question = req.question
    context = "\n\n".join(search(question))
    prompt = (
        "根据下面的资料回答问题。如果资料里没有答案，直接说'资料里没有'，不要编。\n\n"
        f"资料：\n{context}\n\n问题：{question}\n回答："
    )
    resp = llm.invoke(prompt)
    return {"answer" : resp.content}

@app.post("/match")
def match(req :MatchRequest):
    messages = [SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=req.text)]
    result = match_app.invoke({"messages": messages})
    messages = result["messages"]
    return {"answer":messages[-1].content}

# 把 static 文件夹里的网页交给 FastAPI 托管：
# 浏览器访问 http://localhost:8080 就会打开里面的 index.html
app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "static"), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0",port = 8080)