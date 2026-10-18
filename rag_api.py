
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"   
from fastembed import TextEmbedding
from langchain_openai import ChatOpenAI
import numpy as np
from fastapi import FastAPI      
from pydantic import BaseModel   # 从 pydantic 工具箱里拿 BaseModel 工具
import uvicorn                   # 整个 uvicorn 工具箱都拿来（因为要用 uvicorn.run）
from rag_embedding import search,llm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles   # 让 FastAPI 顺便托管网页文件
from pathlib import Path                      # 处理文件路径
from job_match_agent import app as match_app, SYSTEM_PROMPT
from langchain_core.messages import SystemMessage, HumanMessage
from fastapi.responses import StreamingResponse   # 流式响应工具
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class ChatRequest(BaseModel):
    question: str
class MatchRequest(BaseModel):
    text: str

def gen(chain, question):
    for chunk in chain.stream(question):
        yield chunk
        

def retrieve(question):
    context = "\n\n".join(search(question))
    return context

app = FastAPI()
app.add_middleware(CORSMiddleware,allow_origins = ["*"],allow_methods=["*"],allow_headers=["*"])
prompt = ChatPromptTemplate.from_template(
        "根据下面的资料回答问题。如果资料里没有答案，直接说'资料里没有'，不要编。\n\n"
        "资料：\n{context}\n\n问题：{question}\n回答："
    )
parsers = StrOutputParser()
chain = {"context": retrieve, "question": RunnablePassthrough()} | prompt | llm | parsers

@app.post("/chat/stream")
def chat(req: ChatRequest):
    question = req.question
    return StreamingResponse(gen(chain, question), media_type="text/plain;charset=utf-8")

# ===== 全局记忆：跨请求记住对话历史 =====
# 只记"用户说的话 + AI 的最终回答"（干净的对话层）。
# 不记 Agent 内部的工具调用消息（ToolMessage）：它们只在当轮执行中有用，
# 跨轮带回去会触发模型兼容问题（之前 500 就是这原因），而且没必要。
# 历史存在"函数外"，请求之间就不会丢（内存级；重启服务会清空，
# 生产级做法是存到 Redis/数据库，面试可以这样答）
session_history = []

@app.post("/match")
def match(req :MatchRequest):
    global session_history
    # 每次请求 = 系统提示词 + 历史对话 + 本次输入
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + session_history + [HumanMessage(content=req.text)]
    result = match_app.invoke({"messages": messages})
    # 记忆里只存本轮"用户输入 + AI最终回答"，不存工具中间消息
    session_history.append(HumanMessage(content=req.text))
    session_history.append(result["messages"][-1])
    return {"answer":result["messages"][-1].content}

# 把 static 文件夹里的网页交给 FastAPI 托管：
# 浏览器访问 http://localhost:8080 就会打开里面的 index.html
app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "static"), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0",port = 8080)