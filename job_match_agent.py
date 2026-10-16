# ============================================================
# 项目一：岗位匹配 Agent（LangGraph 版）
# 复用天气 Agent 的同一张图，只换了【工具】+【提示词】
# ============================================================
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
import requests   # 发 HTTP 请求（联网用）

# ===================== 岗位库（先内置 4 个示例，后续可换真实数据） =====================
JOBS = {
    "AI应用开发工程师": {
        "skills": ["Python", "LangChain", "LangGraph", "RAG", "Function Calling", "FastAPI"],
        "salary": "15-25K",
        "desc": "负责大模型应用开发：Agent 搭建、RAG 问答系统、工具调用集成",
    },
    "Java后端开发工程师": {
        "skills": ["Java", "SpringBoot", "MySQL", "Redis", "微服务"],
        "salary": "12-20K",
        "desc": "负责后端接口开发、数据库设计、微服务架构",
    },
    "Python数据分析师": {
        "skills": ["Python", "pandas", "SQL", "数据可视化"],
        "salary": "10-18K",
        "desc": "负责数据分析、报表制作、业务指标监控",
    },
    "算法工程师（NLP方向）": {
        "skills": ["Python", "PyTorch", "NLP", "Transformer", "大模型微调"],
        "salary": "20-35K",
        "desc": "负责 NLP 模型训练、微调、推理优化",
    },
}

# ===================== 工具 1：搜索岗位 =====================
@tool
def search_jobs(keyword: str) -> str:
    """在岗位库中搜索与关键词相关的岗位。keyword 是关键词，如：AI、Java、Python、算法。返回岗位名称、技能要求和职责描述。"""
    results = []
    for name, info in JOBS.items():
        # 关键词出现在岗位名 或 技能列表里就算命中
        if (keyword in name) or any(keyword in s for s in info["skills"]):
            results.append(
                f"【{name}】薪资{info['salary']}，技能要求：{'、'.join(info['skills'])}，职责：{info['desc']}"
            )
    return "\n".join(results) if results else f"没有找到与'{keyword}'相关的岗位"

# ===================== 工具 2：计算匹配度 =====================
@tool
def calculate_match(skills: str, job_title: str) -> str:
    """计算某人的技能与某岗位的匹配情况。skills 是用顿号分隔的技能列表，如：Python、LangChain、Java。job_title 是岗位名称。返回匹配百分比、匹配上的技能、缺失的技能。"""
    if job_title not in JOBS:
        return f"岗位库中没有'{job_title}'"
    my_skills = {s.strip() for s in skills.split("、")}       # 我的技能（集合）
    need_skills = set(JOBS[job_title]["skills"])              # 岗位要求（集合）
    matched = my_skills & need_skills                          # 交集 = 匹配上的
    missing = need_skills - my_skills                          # 差集 = 缺失的
    percent = round(len(matched) / len(need_skills) * 100)    # 匹配百分比
    return (
        f"岗位'{job_title}'匹配度：{percent}%\n"
        f"匹配上的技能：{'、'.join(matched) if matched else '无'}\n"
        f"缺失的技能：{'、'.join(missing) if missing else '无（全部满足）'}"
    )

# ===================== 解析器（供"智能工具"内部调用的模型，不绑定工具） =====================
from config import QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL  # 密钥从 config.py 读（不上传 GitHub）

extractor = ChatOpenAI(
    model=QWEN_MODEL,
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
)

# ===================== 工具 3：解析简历（智能工具：内部调 LLM） =====================
@tool
def parse_resume(resume_text: str) -> str:
    """从简历原文中提取技能列表。resume_text 是简历的完整文字（可以很长）。返回提取出的技能，用顿号分隔。"""
    resp = extractor.invoke(
        "从下面的简历文本中，提取这个人掌握的所有技术技能"
        "（编程语言、框架、数据库、工具等）。只输出技能清单，用顿号分隔，不要任何解释。\n\n"
        f"简历：\n{resume_text[:2000]}"
    )
    return resp.content

# ===================== 工具 4：生成面试题（智能工具） =====================
@tool
def generate_interview_questions(job_title: str, skills: str) -> str:
    """根据岗位和候选人的技能生成面试题。job_title 是岗位名称，skills 是用顿号分隔的技能。返回 5 道面试题。"""
    resp = extractor.invoke(
        f"岗位：{job_title}\n候选人技能：{skills}\n\n"
        "针对这个岗位出 5 道技术面试题（2 道基础 + 2 道进阶 + 1 道项目深挖），每题后面标注考察点。"
    )
    return resp.content

# ===================== 工具 5：联网搜真实岗位（联网工具：内部发 HTTP 请求） =====================
@tool
def search_remote_jobs(keyword: str) -> str:
    """在真实的远程工作平台上联网搜索岗位。keyword 是关键词，如：python、java、ai、data。返回真实岗位的标题、公司、地区。"""
    try:
        resp = requests.get(                    # 发 HTTP 请求
            "https://remotive.com/api/remote-jobs",
            params={"search": keyword},         # 关键词作为查询参数
            timeout=10,                         # 10 秒超时，防止卡死
        )
        data = resp.json()                      # 把返回的 JSON 转成 Python 数据
        jobs = data.get("jobs", [])[:5]         # 取前 5 个岗位
        if not jobs:
            return f"没搜到 '{keyword}' 的真实岗位"
        lines = []
        for j in jobs:
            lines.append(
                f"【{j['title']}】{j['company_name']} | {j['candidate_required_location']} | {j.get('salary', '薪资面议')}"
            )
        return "\n".join(lines)
    except Exception as e:
        # 联网失败时回退，不让整个 Agent 崩掉
        return f"联网失败（{e}），请用 search_jobs 搜内置岗位库"

# ===================== 模型（绑定工具） =====================
tools = [search_jobs, calculate_match, parse_resume, generate_interview_questions, search_remote_jobs]

llm = ChatOpenAI(
    model=QWEN_MODEL,
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
).bind_tools(tools)

# ===================== 图（和天气 Agent 一模一样，没改） =====================
class State(TypedDict):
    messages: Annotated[list, add_messages]

def call_model(state):
    msg = llm.invoke(state["messages"])
    return {"messages": [msg]}

def call_tool(state):
    last_msg = state["messages"][-1]
    new_msgs = []
    for tc in last_msg.tool_calls:
        result = {t.name: t for t in tools}[tc["name"]].invoke(tc["args"])
        new_msgs.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": new_msgs}

def should_continue(state):
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"
    return "end"

graph = StateGraph(State)
graph.add_node("model", call_model)
graph.add_node("tools", call_tool)
graph.add_edge(START, "model")
graph.add_conditional_edges("model", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "model")
app = graph.compile()

# ===================== 对话（提示词换成了"求职顾问"，并支持多轮记忆） =====================
SYSTEM_PROMPT = (
    "你是一个求职顾问 Agent。用户可能给你：技能清单，或者一整段简历原文。你的任务："
    "1) 如果输入像简历（一大段文字），先调 parse_resume 提取技能；"
    "2) 优先用 search_remote_jobs 联网搜真实岗位；如果用户想对比内置岗位，用 search_jobs；"
    "3) 用 calculate_match 算匹配度；"
    "4) 综合结果告诉用户：最匹配的岗位、匹配度、差距在哪、怎么补。"
    "如果用户追问（比如要面试题、要学习计划），用 generate_interview_questions 等工具回答。回答简洁。"
)

# 多轮记忆：messages 在循环外创建，每轮把完整对话历史带回下一轮

if __name__ == '__main__':

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    while True:
        question = input("你（输入你的技能/简历/追问）：")
        if question == "退出":
            break
        messages.append(HumanMessage(content=question))
        result = app.invoke({"messages": messages})
        messages = result["messages"]          # 把这一轮的全部对话历史存起来
        print("AI：", messages[-1].content)
