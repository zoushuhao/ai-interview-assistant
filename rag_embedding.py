# ============================================================
# 和 rag_demo.py 的区别：词频向量 → 语义向量
# 骨架（切块→检索→生成）完全一样
# ============================================================

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"   # 模型下载走国内镜像
from fastembed import TextEmbedding
from langchain_openai import ChatOpenAI
import numpy as np



KNOWLEDGE = """
RAG

RAG（检索增强生成）是让大模型基于外部资料回答问题的技术。因为大模型的知识有截止日期，还会编造不存在的答案（幻觉），所以先检索外部资料，再把资料和问题一起给模型生成回答。它分五步：资料切块、文本向量化、向量入库、相似度检索、生成回答。检索的核心是向量相似度：把问题和资料块都转成向量，算余弦相似度，取最高的几块作为上下文。
===
为什么要切块

大模型能接受的输入长度有限，而且相关资料可能很长。把资料按章节或段落切成小块，检索时只找最相关的几块，既能放进模型上下文，又避免无关内容干扰回答。切得太碎会丢失上下文，切得太大检索不够精准，所以块大小要权衡。
===
向量化与嵌入模型

向量化是把文字变成一串数字（向量）的过程，类似给每段文字做一个"语义指纹"。做向量化的是嵌入模型（embedding model），比如我用的 BAAI/bge-small-zh-v1.5，是专门训练的中文模型，把文字映射成 512 维向量。语义相近的文字，向量在空间里距离也近。
===
余弦相似度

余弦相似度用来衡量两个向量方向的接近程度，取值范围是 -1 到 1，越接近 1 越相似。公式是向量的点积除以两个向量长度的乘积。它只关心方向不关心长度，所以即使两段文字长度不同，只要语义相近，相似度也高。比关键词匹配强的地方是：能命中同义词和改写表达。
===
top_k 的作用

top_k 是检索时返回的最相关资料块数量。设置太小可能漏掉正确答案（召回不足），设置太大又会把无关内容带进来干扰回答（引入噪声）。调 top_k 是 RAG 最常见的调优手段。
===
Function Calling（工具调用）

Function Calling 是让大模型在回答时选择调用外部工具的能力。模型不真正执行代码，它只输出"要调用哪个函数、传什么参数"的结构化结果，真正执行的是程序，执行完把结果回传给模型，模型再基于结果组织最终回答。这样模型就能获取实时数据（天气、时间）或执行程序能力。
===
多轮对话记忆

大模型本身没有记忆，每次对话都是独立的。要实现多轮对话，需要把历史消息都放进 messages 列表，每次都把完整的对话历史带给模型。messages 里每条消息有 role（user/assistant/tool）和 content，工具调用结果用 role=tool 的消息回传。
===
Agent 与 ReAct

Agent 是能自己决定调用什么工具的智能体。ReAct 模式是"思考-行动-观察"循环：模型先思考需要什么信息，决定调用工具，程序执行工具得到结果，模型再基于结果继续思考，直到认为信息足够才直接回答。手写 Agent 就是实现这个循环：调模型、看是否要调用工具、执行工具、把结果喂回去、再调模型。
===
LangGraph

LangGraph 是用图结构组织 Agent 流程的框架。节点是处理步骤（调用模型、执行工具），边是步骤之间的连接，条件边根据上一节点的输出决定走哪条路。State 是各节点共享的状态，保存对话历史等数据。用 LangGraph 的好处是把 Agent 的流程画成图，结构清晰、好调试，还能实现循环。
===
流式输出

流式输出（stream）是让模型边生成边返回文字，而不是等全部生成完再一次性返回。用户体验是字一个个蹦出来，响应更快。实现方式是调用接口时打开 stream 参数，逐块接收内容并实时显示。
===
FastAPI

FastAPI 是 Python 的 Web 框架，用来把代码变成可以被网页或其他程序调用的 HTTP 接口。用 @app.post('/chat') 定义接口路径和请求方式，用 pydantic 的 BaseModel 定义请求体的数据格式（类似 SpringBoot 的 DTO），uvicorn 是启动服务的工具。浏览器访问接口文档地址 /docs 可以交互测试。
===
CORS 跨域

浏览器出于安全，默认禁止网页向"不同来源"的服务器发请求（跨域）。解决方法是服务器加 CORS 中间件，声明允许哪些来源访问。最简单的是允许所有来源（allow_origins=['*']）。网页和接口放在同一个服务里（同源）也能彻底避免跨域。
===
OpenAI 兼容协议

很多大模型服务商提供和 OpenAI 一样的接口格式（路径、请求体、返回值都相同），只要换 base_url、api_key、model 三个配置就能切换供应商，代码不用改。这就是为什么 DeepSeek 宕机时能快速换成通义千问。
===
幻觉问题

幻觉是模型编造不存在的答案。缓解方法：RAG 给模型提供真实资料并要求"资料里没有就说没有，不要编"；降低温度参数；用检索结果约束回答范围。RAG 不能完全消除幻觉，但能大幅减少。
===
API 故障排查

遇到 API 卡住，按顺序排查：先确认自己的服务活着（访问其他接口），再排除环境问题（换个终端试），再确认网络通不通（浏览器能访问官方域名），再看是不是 key 问题（真无效会秒回 401 而不是超时），最后查官方状态页确认是不是服务商宕机。判断标准：连接成功但读取超时，多半是服务商服务端问题。
===
本地嵌入模型

本地嵌入模型（如 fastembed 加载的 bge-small-zh）在电脑本地跑，不联网、不花钱、没有速率限制。首次使用要下载模型文件（国内用 HF_ENDPOINT 镜像加速），之后秒加载。适合对隐私和成本敏感的场景。
"""

chunks = []
for c in KNOWLEDGE.split("==="):
    c = c.strip()
    if c:
        chunks.append(c)
print(f"共 {len(chunks)} 块")


def to_embed(texts):
    return list(embed_model.embed(texts))


embed_model = TextEmbedding("BAAI/bge-small-zh-v1.5")   # 上次下载过，这次秒加载
chunk_vecs = to_embed(chunks)

def cosine_sim(v1, v2):
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def search(query, top_k=6):
    qv = to_embed([query])[0]
    # 和每一块算相似度，记下 (分数, 块序号)
    scored = []
    for i, v in enumerate(chunk_vecs):
        scored.append((cosine_sim(qv, v), i))
    scored.sort(reverse=True)
    # 取分数最高的 top_k 块原文返回
    result = []
    for _, i in scored[:top_k]:
        result.append(chunks[i])
    return result

# ===== 第 5 步：生成 =====
from config import QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL  # 密钥从 config.py 读（不上传 GitHub）

llm = ChatOpenAI(
    model=QWEN_MODEL,
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
)

if __name__ == "__main__":

    print("\n===== 语义版面试知识库问答 =====")
    while True:
        question = input("\n你：")
        if question == "退出":
            break
        context = "\n\n".join(search(question))
        prompt = (
            "根据下面的资料回答问题。如果资料里没有答案，直接说'资料里没有'，不要编。\n\n"
            f"资料：\n{context}\n\n问题：{question}\n回答："
        )
        resp = llm.invoke(prompt)
        print("AI：", resp.content)
