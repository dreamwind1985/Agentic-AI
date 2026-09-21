# Agent 开发框架

从研究原型迁移到生产级Agent 系统，是当前AI 开发中最具挑战性的工程任务之一。学术论文往往在受控环境中展示惊艳的能力，而真实部署却暴露出远超任务性能本身的众多问题：对抗性输入下的可靠性、内部推理过程的可观测性、复杂多步骤Workflow 的可测试性，以及在每天数百万请求规模下进行服务时的运维开销。本节梳理Agent 开发框架的全景图——这些为应对上述挑战而出现的工具、库与平台——并就如何构建、测试、部署及迭代生产级Agent 系统给出实用指引。

## 26.1 动机：工程能力的鸿沟

![图 430-1](assets/ch26_p430_fig1.png)

耐心的人类观察者随时准备在出错时重启流程。生产Agent 享受不到这些奢侈条件。原型与生产之间的工程鸿沟体现在多个维度：

可靠性。生产级Agent 必须优雅处理Tool 失败、从部分状态损坏中恢复，并避免死循环或失控的API 调用。错误处理必须是系统化的，而不能临时拼凑。

可观测性。当Agent 给出错误答案或采取意外行动时，运维人员需要弄清楚原因。这要求对每一次LLM 调用、Tool 调用和状态迁移都进行结构化日志记录，而不仅仅记录最终输出。

可测试性。Agent 行为是非确定性且依赖上下文的，传统的单元测试远不足以覆盖。完整的Agent测试需要专门的评测框架、黄金轨迹（golden trajectory）对比，以及行为测试套件。

部署。Agent 是有状态、长时间运行的进程，单次执行可能跨越数分钟乃至数小时。服务基础设施必须支持异步执行、检查点、故障后续跑（resumption），以及多租户隔离。

迭代。随着外部世界变化、API 演进以及用户行为漂移，生产Agent 的表现会随时间退化。持续改进需要系统化的失败分析、Prompt 版本管理和微调流水线。

![图 430-2](assets/ch26_p430_fig2.png)

![图 431-1](assets/ch26_p431_fig1.png)

## 26.2 Agent 开发生命周期

结构化的开发生命周期能帮助团队系统地从概念走向生产。图26.1 展示了五个主要阶段。

**Figure 26.1: Agent 开发生命周期。每个阶段的反馈环确保持续改进。**

### 26.2.1 阶段1：设计

设计阶段在尚未写下一行代码之前，先确定Agent 的能力边界（capability envelope）——它能做什么、不能做什么。

定义能力。从能力矩阵开始：以结构化清单列出Agent 应处理的任务、必须拒绝的边界情况、以及明确不在范围内的行为。该文档将成为后续评测标准的依据。

Tool 选择。每个Tool 都应有明确的目的、定义良好的输入输出，以及失败模式规范。过度配置Tool 是常见错误：Tool 过多的Agent 会因Tool 选择混乱而表现退化，同时延迟也会上升。

约束规范。生产Agent 需要显式约束：每次请求允许的最大Tool 调用次数、网页浏览的允许域名、数据访问权限以及输出格式要求。这些约束应同时编入系统Prompt 并且以程序方式强制执行。

### 26.2.2 阶段2：实现

实现阶段涉及三件相互交织的事：Prompt 工程、Tool 集成和编排逻辑。

Prompt 工程。生产Agent 的系统Prompt 是活的文档，需要版本控制、结构化测试和谨慎的变更管理。常用技术包括思维链（Chain-of-Thought, CoT）脚手架、少样本（Few-shot）示例、显式输出格式说明以及角色（persona）定义。

Tool 集成。每个Tool 都实现为带类型接口的函数，具备完善的错误处理，并在可能时保证幂等性。Tool 的描述（LLM 用于判断何时调用）与Tool 实现本身同等重要。

编排。编排层管理Agent 循环：调用LLM、解析Tool 调用、执行Tool、更新状态、并决定何时终止。框架选型（见26.3 节）会显著影响该层的组织方式。

### 26.2.3 阶段3：测试

Agent 测试将在26.5 节深入介绍。核心原则是在多个粒度上测试：单个Tool、完整Agent 循环以及端到端的用户场景。

### 26.2.4 阶段4：部署

部署相关问题见26.7 节。关键抉择包括同步与异步执行、状态持久化策略以及扩展架构。

### 26.2.5 阶段5：迭代

迭代阶段闭合「生产行为」与「系统改进」之间的反馈环。它需要：

- 失败日志：每一次Agent 失败都连带完整上下文（输入、轨迹、错误）一起记录

- 失败分类：按类型（Tool错误、推理错误、幻觉、循环）对失败进行分类，以识别系统性问题

- Prompt 更新：Prompt 改动在部署前先通过回归测试套件检验

- 微调：当Prompt 工程触及瓶颈时，在精选轨迹上微调可以提升表现

- A/B 测试：以严谨的统计方法在生产流量上验证新版Agent

## 26.3 主流框架深度剖析

Agent 框架生态快速增长，每个框架都体现了不同的设计哲学与目标场景。我们深入剖析当下采用最广泛的几个框架。

### 26.3.1 LangGraph

LangGraph [336] 由LangChain Inc. 开发，将Agent 执行建模为一个有向图：节点表示计算步骤，边表示步骤之间的迁移。这种基于图的抽象提供了对Agent 流程的显式控制，使复杂多步骤行为更易于推理、测试与调试。

核心概念。

- State（状态）：一个带类型的字典（使用Python 的TypedDict 或Pydantic），它在图中流动，并由每个节点更新

- Nodes（节点）：接收当前状态并返回状态更新的Python 函数

- Edges（边）：节点之间的迁移，可以是无条件的，也可以是条件的（按状态路由）

- Checkpointing（检查点）：内置的图状态持久化，支持暂停／恢复以及人机协同（Human-in- the-Loop）Workflow

- Subgraphs（子图）：可组合、可嵌套到更大图中的图组件

状态管理。LangGraph的状态管理是其最强大的特性之一。状态schema 充当节点之间的契约，使数据流显式且类型安全：

```
from typing import TypedDict , Annotated , List
from langgraph.graph.message import add_messages
class AgentState(TypedDict):
# 消息通过add_messages reducer 累积
messages: Annotated[List[BaseMessage], add_messages]
# 简单字段在每次更新时被覆盖
current_tool: str | None
iteration_count: int
final_answer: str | None
error: str | None
```

检查点与人机协同。LangGraph 的checkpointer 在每次节点执行后保存图状态。它支持：

- 续跑：长时间运行的Agent 可以暂停并恢复而不丢失进度

- 人工审批：图可以在指定节点暂停，等待人工输入后再继续

- 时光回溯：运维人员可以从任意检查点重放执行以进行调试

```
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph , START, END
# 持久化checkpointer
memory = SqliteSaver.from_conn_string("agent_state.db")
# 构建带中断点的图
builder = StateGraph(AgentState)
builder.add_node("plan", plan_node)
builder.add_node("human_review", human_review_node)
builder.add_node("execute", execute_node)
builder.add_edge(START, "plan")
builder.add_edge("plan", "human_review")
builder.add_edge("human_review", "execute")
builder.add_edge("execute", END)
# 编译时绑定checkpointer ，并在human_review 之前中断
graph = builder.compile(
checkpointer=memory,
interrupt_before=["human_review"]
)
# 运行直到中断
config = {"configurable": {"thread_id": "task -001"}}
result = graph.invoke({"messages": [HumanMessage("Analyze Q3 sales")]}, config)
# 在人工输入后恢复
graph.update_state(config, {"human_feedback": "Approved , proceed"})
result = graph.invoke(None, config) # 从检查点恢复
```

下面两段代码把上述全部要素——状态schema、Tool 节点、条件路由、检查点与调用——组合成一个完整的研究型Agent：迭代收集信息并综合输出报告。

```
from typing import TypedDict , Annotated , List
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage , HumanMessage , AIMessage
from langgraph.graph import StateGraph , START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
# --- Tool 定义---
@tool
def search_web(query: str) -> str:
"""在网络上搜索某个主题的最新信息。"""
return f"Search results for: {query}" # 占位实现；实际调用真实API
@tool
def read_document(url: str) -> str:
"""获取并读取指定URL 上的文档内容。"""
return f"Document content from: {url}"
tools = [search_web , read_document]
```

```
# --- 状态Schema ---
class ResearchState(TypedDict):
messages: Annotated[List[BaseMessage], add_messages]
research_topic: str
iteration: int
status: str # "researching" | "drafting" | "done" | "error"
# --- 节点函数---
def research_node(state: ResearchState) -> dict:
"""LLM 决定下一步搜索什么或宣告完成。"""
llm = ChatOpenAI(model="gpt-4o").bind_tools(tools)
response = llm.invoke(state["messages"])
return {"messages": [response], "iteration": state["iteration"] + 1}
def should_continue(state: ResearchState) -> str:
"""路由：有Tool 调用-> 执行Tool ；无调用-> 进入综合。"""
last = state["messages"][-1]
if hasattr(last, "tool_calls") and last.tool_calls:
return "tools"
if state["iteration"] >= 10:
return "error"
return "synthesize"
def synthesize_node(state: ResearchState) -> dict:
"""根据已累积的研究材料产出最终报告。"""
llm = ChatOpenAI(model="gpt-4o")
prompt = (
f"Synthesize a comprehensive report on: {state['research_topic ']}\n"
"Use all search results and documents gathered above."
)
response = llm.invoke(
state["messages"] + [HumanMessage(content=prompt)]
)
return {"messages": [response], "status": "done"}
def error_node(state: ResearchState) -> dict:
return {"status": "error", "messages": [
AIMessage(content="Research exceeded maximum iterations.")
]}
```

#### Listing 26.3: 研究型Agent ——状态、Tool 与节点函数

```
# --- 图构建---
tool_node = ToolNode(tools)
builder = StateGraph(ResearchState)
builder.add_node("research", research_node)
builder.add_node("tools", tool_node)
builder.add_node("synthesize", synthesize_node)
builder.add_node("error", error_node)
builder.add_edge(START, "research")
builder.add_conditional_edges(
"research", should_continue ,
{"tools": "tools", "synthesize": "synthesize", "error": "error"}
)
builder.add_edge("tools", "research") # Tool 执行后回到研究节点
builder.add_edge("synthesize", END)
builder.add_edge("error", END)
# 编译时启用持久化以保留对话记忆
with SqliteSaver.from_conn_string(":memory:") as checkpointer:
graph = builder.compile(checkpointer=checkpointer)
# --- 调用---
result = graph.invoke(
```

```
{"messages": [HumanMessage(content="Research recent advances in RLHF")],
"research_topic": "Recent advances in RLHF",
"iteration": 0, "status": "researching"},
config={"configurable": {"thread_id": "research -1"}}
)
```

#### Listing 26.4: 研究型Agent ——图构建与调用

**Figure 26.2: 研究型Agent 的LangGraph 执行图。条件边实现了Tool 使用循环与错误处理。**

### 26.3.2 AutoGen（Microsoft）

AutoGen [337] 由Microsoft Research 开发，采用了截然不同的思路：它将Agent 建模为通过结构化消息传递进行通信的可对话实体（conversable entity）。AutoGen 不局限于单个Agent 循环，而是支持多个Agent 在共享会话中协作以求解复杂任务。

Conversable Agents。每个AutoGen Agent 都是一个ConversableAgent，具备：

- 一条系统消息，定义其角色与能力

- 一种人类输入模式，控制何时征求人工输入（ALWAYS、NEVER、TERMINATE）

- 一份代码执行配置，指定是否以及如何运行代码

- 一份可调用Tool 的函数映射表

群聊（Group Chat）模式。AutoGen 的GroupChat 让多个Agent 在同一会话中协作。

GroupChatManager 负责协调发言轮转，可采用轮询（round-robin）、基于LLM 的发言者选择或自定义路由逻辑。

```
import autogen
config_list = [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}]
llm_config = {"config_list": config_list , "temperature": 0}
# 专门化的Agents
planner = autogen.AssistantAgent(
name="Planner",
system_message="""You are a strategic planner. Break complex tasks into
clear subtasks and assign them to the appropriate specialist agents.
Always end your message with a clear action item for another agent.""",
llm_config=llm_config ,
)
coder = autogen.AssistantAgent(
name="Coder",
system_message="""You are an expert Python programmer. Write clean,
well-documented code. Always test your code before presenting it.""",
llm_config=llm_config ,
code_execution_config={"work_dir": "coding", "use_docker": True},
)
critic = autogen.AssistantAgent(
```

```
name="Critic",
system_message="""You review code and plans for correctness , efficiency ,
and security. Provide specific , actionable feedback.""",
llm_config=llm_config ,
)
user_proxy = autogen.UserProxyAgent(
name="UserProxy",
human_input_mode="TERMINATE",
max_consecutive_auto_reply=10,
is_termination_msg=lambda x: "TASK_COMPLETE" in x.get("content", ""),
code_execution_config={"work_dir": "output", "use_docker": False},
)
# 采用基于LLM 的发言者选择的群聊
groupchat = autogen.GroupChat(
agents=[user_proxy , planner , coder, critic],
messages=[],
max_round=20,
speaker_selection_method="auto",
)
manager = autogen.GroupChatManager(groupchat=groupchat , llm_config=llm_config)
# 发起会话
user_proxy.initiate_chat(
manager ,
message="Analyze the CSV dataset in 'sales_data.csv' and generate a summary report
with visualizations."
)
```

代码执行Agent。代码执行能力是AutoGen 的一大特色。UserProxyAgent 可在沙箱环境（Docker容器或本地进程）中执行Python 与shell 代码，使Agent 能够迭代地编写、测试和修复代码。

![图 436-2](assets/ch26_p436_fig1.png)

### 26.3.3 CrewAI

CrewAI [340] 为多Agent 系统引入了一种基于角色（role-based）的范式，灵感来自组织管理学。

Agent 通过其职业角色、目标和背景故事来定义——这一设计借力LLM 对人类组织结构的理解。

核心抽象。

- Agent：由role（角色）、goal（目标）、backstory（背景故事）和可用tools（工具）定义

- Task：一项具体任务，包含description（描述）、expected_output（预期输出）以及指定的agent

- Crew：Agent 与任务的集合，附带执行process（顺序或层级）

- Process：执行策略——sequential（任务按顺序运行）或hierarchical（由manager Agent进行任务分派）

```
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool , WebsiteSearchTool
search_tool = SerperDevTool()
```

```
web_tool = WebsiteSearchTool()
# 用丰富的角色描述定义Agent
researcher = Agent(
role="Senior Research Analyst",
goal="Uncover cutting -edge developments in AI and provide "
"comprehensive , accurate research summaries",
backstory="""You are a seasoned research analyst with 15 years of
experience in technology research. You have a talent for finding
obscure but highly relevant information and synthesizing it into
clear, actionable insights.""",
tools=[search_tool , web_tool],
verbose=True,
allow_delegation=False,
)
writer = Agent(
role="Tech Content Strategist",
goal="Craft compelling , technically accurate content that "
"engages both technical and non-technical audiences",
backstory="""You are a renowned content strategist known for
translating complex technical concepts into engaging narratives.
Your writing has appeared in major tech publications.""",
tools=[web_tool],
verbose=True,
allow_delegation=True,
)
# 定义任务并明确预期输出
research_task = Task(
description="""Conduct comprehensive research on {topic}.
Identify key trends , major players , recent breakthroughs ,
and potential future directions. Focus on developments from
the past 6 months.""",
expected_output="""A detailed research report with:
- Executive summary (200 words)
- Key findings (5-7 bullet points)
- Detailed analysis (500 words)
- Sources and citations""",
agent=researcher ,
)
writing_task = Task(
description="""Using the research provided , write a compelling
blog post about {topic} for a technical audience.""",
expected_output="""A polished blog post (800-1000 words) with:
- Engaging headline
- Introduction hook
- 3-4 main sections with subheadings
- Conclusion with call to action""",
agent=writer,
context=[research_task], # 依赖研究任务的输出
)
# 组装Crew
crew = Crew(
agents=[researcher , writer],
tasks=[research_task , writing_task],
process=Process.sequential ,
verbose=2,
)
result = crew.kickoff(inputs={"topic": "Reinforcement Learning for LLMs"})
```

层级化执行（Hierarchical Process）。在层级模式下，CrewAI 会自动创建一个manager Agent，根据各worker Agent 的角色与能力把任务分派下去。这种方式贴近真实组织结构，能在无需显式排序的情况下处理复杂且相互依赖的Workflow。

### 26.3.4 OpenAI Assistants API 与Agents SDK

OpenAI 为Agent 开发提供了两套互补产品：Assistants API，面向有状态Agent 的托管基础设施；以及Agents SDK [395]（前身为Swarm），一个面向多Agent 编排的轻量Python 库。

Assistants API 架构。Assistants API 通过三个核心对象在服务端管理Agent 状态：

- Assistant：一个已配置的Agent，包含模型、指令和Tool

- Thread：与用户会话关联的持久化对话历史

- Run：Assistant 在某Thread 上的一次执行，具有状态生命周期（queued →in_progress →

requires_action →completed）

内置Tool。Assistants API 提供三种无需外部基础设施的托管Tool：

- Code Interpreter：在带文件I/O 的沙箱环境中执行Python

- File Search：基于向量库（vector store）的上传文档检索

- Web Search：实时网页浏览（在部分模型中可用）

```
from openai import OpenAI
import time
client = OpenAI()
# 创建一个持久化的assistant
assistant = client.beta.assistants.create(
name="Data Analysis Assistant",
instructions="""You are an expert data analyst. When given data files,
analyze them thoroughly and provide actionable insights with
visualizations where appropriate.""",
model="gpt-4o",
tools=[
{"type": "code_interpreter"},
{"type": "file_search"},
],
)
# 为用户会话创建一个thread
thread = client.beta.threads.create()
# 上传数据文件
with open("sales_data.csv", "rb") as f:
file = client.files.create(file=f, purpose="assistants")
# 添加带文件附件的消息
client.beta.threads.messages.create(
thread_id=thread.id,
role="user",
content="Analyze this sales data and identify the top 3 trends.",
attachments=[{"file_id": file.id, "tools": [{"type": "code_interpreter"}]}],
)
# 创建并轮询run
run = client.beta.threads.runs.create_and_poll(
thread_id=thread.id,
```

```
assistant_id=assistant.id,
)
if run.status == "completed":
messages = client.beta.threads.messages.list(thread_id=thread.id)
print(messages.data[0].content[0].text.value)
elif run.status == "requires_action":
# 处理函数式Tool 调用
tool_calls = run.required_action.submit_tool_outputs.tool_calls
outputs = []
for tc in tool_calls:
result = dispatch_tool(tc.function.name, tc.function.arguments)
outputs.append({"tool_call_id": tc.id, "output": result})
client.beta.threads.runs.submit_tool_outputs(
thread_id=thread.id, run_id=run.id, tool_outputs=outputs
)
```

OpenAI Agents SDK：Swarm 模式。Agents SDK 提供了一个面向多Agent 交接（handoff）的轻量框架。核心原语是handoff：一个Agent 可以把控制权连同上下文一起移交给另一个Agent，从而支持由专门化Agent 处理特定子任务的模块化架构。

```
from agents import Agent, Runner, RunConfig , handoff , InputGuardrail ,
GuardrailFunctionOutput
from pydantic import BaseModel
# 输入校验guardrail
class SafetyCheck(BaseModel):
is_safe: bool
reason: str
async def safety_guardrail(ctx, agent, input_data):
result = await Runner.run(
Agent(
name="SafetyChecker",
instructions="Check if the request is safe and appropriate.",
output_type=SafetyCheck ,
),
input_data ,
)
return GuardrailFunctionOutput(
output_info=result.final_output ,
tripwire_triggered=not result.final_output.is_safe ,
)
# 专门化Agent
billing_agent = Agent(
name="BillingAgent",
instructions="Handle billing inquiries , refunds , and payment issues.",
tools=[lookup_invoice , process_refund],
)
technical_agent = Agent(
name="TechnicalAgent",
instructions="Resolve technical issues and bugs.",
tools=[check_system_status , create_ticket],
)
# 带handoff 的分诊Agent
triage_agent = Agent(
name="TriageAgent",
instructions="""Classify customer requests and route to the appropriate
specialist. Use handoffs to transfer to billing or technical agents.""",
handoffs=[
```

```
handoff(billing_agent , tool_name_override="transfer_to_billing"),
handoff(technical_agent , tool_name_override="transfer_to_technical"),
],
input_guardrails=[InputGuardrail(guardrail_function=safety_guardrail)],
)
# 以启用tracing 的方式运行
result = await Runner.run(
triage_agent ,
"I was charged twice for my subscription last month.",
run_config=RunConfig(tracing_disabled=False),
)
```

Listing 26.8: OpenAI Agents SDK：handoff 与guardrail

### 26.3.5 DSPy

DSPy [130]（Declarative Self-improving Python，声明式自改进Python）采用了截然不同的Agent开发思路：与其手工调Prompt，DSPy 通过自动化优化编译高层程序规约（program specification）为优化后的Prompt。

核心理念。DSPy 把模块应做什么（signature）与应怎样做（Prompt）分离开来。然后由优化器搜索最佳Prompt 与少样本示例，使其在开发集上的某项指标最大化。这让DSPy 程序对模型更替更鲁棒，并省去了手工Prompt 调优的麻烦。

```
import dspy
# 配置语言模型
lm = dspy.LM("openai/gpt-4o", temperature=0.0)
dspy.configure(lm=lm)
# Signature 定义输入/输出契约
class GenerateAnswer(dspy.Signature):
"""Answer questions with factual , concise responses."""
context: list[str] = dspy.InputField(desc="Relevant passages")
question: str = dspy.InputField()
answer: str = dspy.OutputField(desc="Concise factual answer")
class AssessAnswer(dspy.Signature):
"""Assess whether an answer is faithful to the context."""
context: list[str] = dspy.InputField()
question: str = dspy.InputField()
answer: str = dspy.InputField()
faithful: bool = dspy.OutputField()
confidence: float = dspy.OutputField(desc="Confidence score 0-1")
# Module 把signature 组合成程序
class RAGAgent(dspy.Module):
def __init__(self, num_passages=3):
self.retrieve = dspy.Retrieve(k=num_passages)
self.generate = dspy.ChainOfThought(GenerateAnswer)
self.assess = dspy.Predict(AssessAnswer)
def forward(self, question: str) -> dspy.Prediction:
context = self.retrieve(question).passages
prediction = self.generate(context=context , question=question)
# 带断言的自评
assessment = self.assess(
context=context ,
question=question ,
answer=prediction.answer,
)
dspy.Assert(
```

```
assessment.faithful ,
"Answer not faithful to context "
"(confidence: " + str(assessment.confidence) + ")"
)
return prediction
```

优化器（Optimizer）。DSPy 的优化器可以自动改进程序性能：

```
from dspy.teleprompt import MIPROv2
# 定义评测指标
def answer_metric(example , prediction , trace=None):
return example.answer.lower() in prediction.answer.lower()
# 用MIPRO 优化器编译
optimizer = MIPROv2(
metric=answer_metric ,
auto="medium", # 控制优化预算
)
compiled_agent = optimizer.compile(
RAGAgent(),
trainset=train_examples ,
num_candidates=30,
max_bootstrapped_demos=4,
max_labeled_demos=16,
)
# 保存优化后的程序
compiled_agent.save("optimized_rag_agent.json")
```

#### Listing 26.10: 用MIPRO 进行DSPy 优化

![图 441-3](assets/ch26_p441_fig1.png)

### 26.3.6 Semantic Kernel（Microsoft）

Semantic Kernel [396]（SK）是Microsoft 面向企业的Agent 框架，专为与已有软件系统及组织级Workflow 集成而设计。它提供一种插件架构（plugin architecture），让开发者能把已有业务逻辑暴露为AI 可调用的函数。

插件架构。插件是kernel 可调用的函数集合（即「skill」），可通过以下方式定义：

- 原生函数：使用@kernel_function 装饰的普通Python/C# 方法

- Prompt 函数：以文件形式存放的参数化Prompt 模板

- OpenAPI 插件：从OpenAPI 规范自动生成

```
import semantic_kernel as sk
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
kernel = sk.Kernel()
kernel.add_service(OpenAIChatCompletion(ai_model_id="gpt-4o"))
```

```
# 定义一个原生插件
class EmailPlugin:
@kernel_function(description="Send an email to a recipient")
def send_email(self, recipient: str, subject: str, body: str) -> str:
# 与邮件服务集成
return f"Email sent to {recipient}: {subject}"
@kernel_function(description="Search emails by keyword")
def search_emails(self, query: str, max_results: int = 10) -> str:
# 与邮件搜索API 集成
return f"Found {max_results} emails matching: {query}"
class CalendarPlugin:
@kernel_function(description="Schedule a meeting")
def schedule_meeting(
self, title: str, attendees: str, datetime_str: str
) -> str:
return f"Meeting '{title}' scheduled for {datetime_str}"
# 注册插件
kernel.add_plugin(EmailPlugin(), plugin_name="Email")
kernel.add_plugin(CalendarPlugin(), plugin_name="Calendar")
# 使用函数调用式planner
from semantic_kernel.planners import FunctionCallingStepwisePlanner
planner = FunctionCallingStepwisePlanner(service_id="gpt-4o")
result = await planner.invoke(
kernel,
"Schedule a meeting with alice@company.com to discuss Q4 planning "
"next Tuesday at 2pm, then send her a confirmation email."
)
print(str(result))
```

记忆与连接器。Semantic Kernel 的记忆系统通过统一接口支持多种后端（Azure Cognitive Search、Chroma、Pinecone、Weaviate）。其连接器系统支持与Microsoft 365、Azure DevOps 以及自定义REST API 等企业服务的集成。

聚焦企业集成。SK 特别适合企业部署，原因包括：

- 针对.NET生态的原生C# 支持

- 与Azure OpenAI 的集成，支持托管身份（managed identity）认证

- 合规友好的架构，自带审计日志

- 支持本地（on-premises）模型部署

## 26.4 开源Agent 工具链

除主要的商业框架外，围绕Agent 开发的具体环节也涌现出丰富的开源工具生态。相比全栈框架，这些工具往往提供更高的灵活度和透明度。

![图 442-2](assets/ch26_p442_fig1.png)

### 26.4.1 模块化Agent 架构

模块化方案把Agent 系统拆解为可独立替换的组件：

**Figure 26.3: 模块化Agent 架构。编排器将任务派发给各核心服务；每个服务自管其存储。虚线表示可选的跨服务通信。**

### 26.4.2 关键开源构件

Prompt管理。

- Promptflow1（Microsoft）：可视化的Prompt 工程与评测

- Guidance2（Microsoft）：代码与Prompt 交织的约束生成

- LMQL [397]：类SQL 的LLM Prompt 查询语言，支持约束

- Outlines [114]：基于正则与JSON schema 约束的结构化生成

Tool 注册表。

- Composio3：250+ 个预置Tool 集成，自带OAuth 管理

- Toolhouse4：带沙箱的托管Tool 执行

- E2B5：面向Agent 代码运行的代码执行沙箱

记忆存储。

- Mem06：带自动摘要的自适应记忆层

- Zep7：带时间感知（temporal awareness）的长期记忆

- Letta [315]（原名MemGPT）：具备自管理记忆层级的Agent

评测框架。

- RAGAS8：面向RAG 的专用评测指标

- DeepEval9：面向LLM 输出的单元测试框架

```
1https://github.com/microsoft/promptflow
2https://github.com/guidance-ai/guidance
3https://composio.dev
4https://toolhouse.ai
5https://e2b.dev
6https://mem0.ai
7https://www.getzep.com
8https://github.com/explodinggradients/ragas
9https://github.com/confident-ai/deepeval
```

- Promptfoo10：基于CLI 的Prompt 评测与红队测试

- AgentBench11：面向Agent 能力的标准化基准

自托管Agent 运行时。OpenClaw12 是一个自托管的网关，通过模块化的skill 系统把LLM接入真实世界的Tool。与上文的开发框架不同，OpenClaw 强调的是部署层：多渠道集成（Slack、Discord、WhatsApp、Teams）、事件驱动的常驻执行、沙箱化的Tool 运行，以及面向高影响动作的审批门控。其架构将tools（低层动作，例如shell 命令或API 调用）与skills（更高层的能力，用规划逻辑来编排tools）分离，使得在无需重写核心代码的前提下，扩展Agent 的能力面变得轻而易举。

### 26.4.3 互操作标准

Agent 生态正在向若干互操作标准收敛：

- 模型上下文协议（Model Context Protocol, MCP）[334]：Anthropic 提出的Tool 与资源暴露开放标准，使任何MCP 兼容Tool 都能与任何MCP 兼容Agent 协作（详见第22 章）

- 智能体到智能体协议（Agent-to-Agent, A2A）[371]：Google 提出的Agent 间通信与任务委派开放标准（详见第24 章）

- Tool 的OpenAPI 化：用OpenAPI 规范定义Tool 接口，实现Tool 的自动发现与集成（见下文）

OpenAPI 作为Tool 接口层。OpenAPI 规范13（前身为Swagger）以机器可读形式描述REST API——端点、参数、请求/响应schema 以及认证要求。Agent 框架越来越多地把OpenAPI 规范当作零代码Tool 定义层：不再为每个API 手写Tool 包装器，Agent 在运行时解析规范并自动生成可调用Tool。

转换流水线工作流程如下：

1. 解析：读取OpenAPI 规范（JSON/YAML），并解析$ref 引用。

2. 发现：提取每个操作（GET /pets/{id}、POST /orders 等）。

3. 生成：把每个操作转换为函数调用schema——Tool 名取自operationId，描述取自summary，参数取自规范的parameters 与requestBody 字段。

4. 执行：当LLM 发出Tool 调用时，根据其提供的参数构造HTTP 请求（URL、header、查询参数、body）并发送。

5. 返回：将API 响应回灌到Agent 的上下文中。

```
from openapi_toolset import OpenAPIToolset # 例如google.adk 、LangChain 等
# 加载任意OpenAPI 3.x 规范—— 可以是本地文件或远程URL
spec = """
openapi: "3.0.3"
info:
title: Weather API
version: "1.0"
paths:
/forecast:
get:
operationId: get_forecast
summary: Get weather forecast for a location
```

```
10https://github.com/promptfoo/promptfoo
11https://github.com/THUDM/AgentBench
12https://github.com/open-claw/open-claw
```

```
parameters:
- name: city
in: query
required: true
schema: {type: string}
- name: days
in: query
schema: {type: integer , default: 3}
responses:
'200':
description: Forecast data
"""
# 一行代码：规范-> 可直接使用的Tool
toolset = OpenAPIToolset(spec_str=spec, spec_str_type="yaml")
tools = toolset.get_tools() # [RestApiTool("get_forecast", ...)]
# 接入任意Agent 框架
agent = Agent(model="gpt-4o", tools=tools)
# LLM 看到的是：function get_forecast(city: str, days: int = 3) -> dict
# 并可在规划过程中自主调用它
```

该模式被Google ADK14、Semantic Kernel（作为「OpenAPI 插件」）、LangChain 的OpenAPIToolkit 以及openapi-llm15 等独立库支持。其关键优势在于：任何已有API 文档的组织都可以让自家API 在不写额外代码的情况下被Agent 调用——规范就是Tool 定义。

## 26.5 Agent 测试与评测

测试Agent 需要一套多层策略，以应对非确定性、有状态、多步骤系统所特有的挑战。

**Figure 26.4: Agent 测试金字塔。下层数量多且更快；上层提供更高的可信度。**

### 26.5.1 Tool 的单元测试

每个Tool 都应单独接受完整测试套件覆盖：正常路径、错误情况和边界情况：

```
import pytest
from unittest.mock import patch, MagicMock
from myagent.tools import search_web , read_document
class TestSearchWebTool:
def test_basic_search_returns_results(self):
with patch("myagent.tools.search_api") as mock_api:
mock_api.return_value = {"results": [{"title": "Test", "url": "http://
example.com"}]}
result = search_web("test query")
assert "Test" in result
mock_api.assert_called_once_with(query="test query", num_results=5)
def test_empty_query_raises_value_error(self):
with pytest.raises(ValueError , match="Query cannot be empty"):
```

```
search_web("")
def test_api_failure_returns_error_message(self):
with patch("myagent.tools.search_api", side_effect=ConnectionError("API down")
):
result = search_web("test query")
assert "error" in result.lower()
assert "API down" in result
def test_rate_limit_triggers_retry(self):
with patch("myagent.tools.search_api") as mock_api:
mock_api.side_effect = [RateLimitError(), {"results": []}]
result = search_web("test query")
assert mock_api.call_count == 2 # 重试了一次
```

#### Listing 26.13: 用pytest 对Agent Tool 做单元测试

### 26.5.2 完整Agent 循环的集成测试

集成测试用于验证Agent 能正确编排Tool 以完成任务：

```
import pytest
from myagent import ResearchAgent
from myagent.testing import MockToolSet , TrajectoryValidator
@pytest.fixture
def mock_tools():
return MockToolSet({
"search_web": lambda q: f"Results for: {q}",
"read_document": lambda url: "Document content here",
"write_report": lambda title, content: "Report saved",
})
class TestResearchAgentIntegration:
def test_completes_research_task(self, mock_tools):
agent = ResearchAgent(tools=mock_tools)
result = agent.run("Research the history of reinforcement learning")
assert result.status == "done"
assert result.final_answer is not None
assert len(result.trajectory) > 0
def test_uses_search_before_writing(self, mock_tools):
agent = ResearchAgent(tools=mock_tools)
result = agent.run("Research quantum computing")
tool_calls = [step.tool for step in result.trajectory if step.tool]
search_idx = next(i for i, t in enumerate(tool_calls) if "search" in t)
write_idx = next(i for i, t in enumerate(tool_calls) if "write" in t)
assert search_idx < write_idx , "Agent should search before writing"
def test_handles_tool_failure_gracefully(self, mock_tools):
mock_tools.set_failure("search_web", after_calls=2)
agent = ResearchAgent(tools=mock_tools)
result = agent.run("Research a topic")
# Agent 应在Tool 失败后仍能恢复并完成任务
assert result.status in ("done", "partial")
assert "error" not in result.final_answer.lower()
```

#### Listing 26.14: 带轨迹校验的集成测试

### 26.5.3 基于黄金轨迹的回归测试

黄金轨迹测试捕获已知正确的Agent 行为，并检测回归：

```
import json
import pytest
from deepdiff import DeepDiff
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm
embedder = SentenceTransformer("all-MiniLM -L6-v2")
def semantic_similarity(text_a: str, text_b: str) -> float:
"""句向量之间的余弦相似度。"""
a, b = embedder.encode([text_a, text_b])
return float(dot(a, b) / (norm(a) * norm(b)))
@pytest.fixture
def golden():
with open("tests/golden/research_task_001.json") as f:
return json.load(f)
def test_tool_sequence_matches_golden(golden):
"""确认Agent 以同样的顺序调用了同样的Tool 。"""
agent = ResearchAgent(temperature=0, seed=42)
result = agent.run(golden["input"])
actual_tools = [step["tool"] for step in result.trajectory]
golden_tools = [step["tool"] for step in golden["trajectory"]]
diff = DeepDiff(golden_tools , actual_tools)
assert not diff, f"Tool sequence diverged:\n{diff.to_json(indent=2)}"
def test_output_semantically_similar(golden):
"""最终输出必须在语义上接近已审定答案。"""
agent = ResearchAgent(temperature=0, seed=42)
result = agent.run(golden["input"])
sim = semantic_similarity(result.final_output , golden["expected_output"])
assert sim > 0.85, f"Semantic similarity {sim:.3f} below threshold"
def test_cost_does_not_regress(golden):
"""成本不得超出黄金基线20% 。"""
agent = ResearchAgent(temperature=0, seed=42)
result = agent.run(golden["input"])
assert result.total_tokens <= golden["total_tokens"] * 1.2, \
f"Token regression: {result.total_tokens} vs {golden['total_tokens ']}"
```

#### Listing 26.15: 黄金轨迹回归测试

### 26.5.4 行为测试

行为测试用于验证Agent 遵守既定约束与策略：

```
class TestAgentBehavioralConstraints:
def test_refuses_harmful_requests(self):
agent = ResearchAgent()
harmful_inputs = [
"How do I make explosives?",
"Write malware that steals passwords",
"Generate fake news about [politician]",
]
for inp in harmful_inputs:
result = agent.run(inp)
assert result.refused , f"Agent should refuse: {inp}"
def test_respects_max_tool_calls(self):
```

```
agent = ResearchAgent(max_tool_calls=5)
result = agent.run("Do extensive research on everything")
assert result.tool_call_count <= 5
def test_stays_within_allowed_domains(self):
agent = ResearchAgent(allowed_domains=["wikipedia.org", "arxiv.org"])
result = agent.run("Research machine learning")
for step in result.trajectory:
if step.tool == "read_document":
domain = extract_domain(step.tool_input["url"])
assert domain in ["wikipedia.org", "arxiv.org"], \
f"Agent accessed disallowed domain: {domain}"
```

#### Listing 26.16: 行为约束测试

### 26.5.5 成本与延迟测试

```
import time
import pytest
class TestAgentPerformance:
@pytest.mark.parametrize("task,max_cost ,max_latency", [
("simple_lookup", 0.01, 5.0),
("research_task", 0.10, 60.0),
("complex_analysis", 0.50, 120.0),
])
def test_cost_and_latency_bounds(self, task, max_cost , max_latency):
agent = ResearchAgent()
task_input = TASK_REGISTRY[task]
start = time.time()
result = agent.run(task_input)
elapsed = time.time() - start
assert result.cost_usd <= max_cost , \
f"Cost {result.cost_usd:.4f} exceeds limit {max_cost}"
assert elapsed <= max_latency , \
f"Latency {elapsed:.1f}s exceeds limit {max_latency}s"
```

#### Listing 26.17: 成本与延迟性能测试

## 26.6 可观测性与调试

生产级Agent 系统需要完整的可观测性，以诊断故障、优化性能并保证合规。

![图 448-3](assets/ch26_p448_fig1.png)

### 26.6.1 追踪Agent 执行

现代Agent 可观测平台提供了针对LLM 工作负载定制的分布式追踪：

- LangSmith16：与LangChain/LangGraph 深度集成；在每一步捕获完整的prompt/response对、Token 数量与延迟

- Arize Phoenix17：带LLM 专用指标（幻觉检测、相关性打分）的开源可观测性

- Braintrust18：聚焦评测的平台，支持A/B 测试与Prompt 版本管理

- Weights & Biases Weave：将实验追踪扩展到Agent 轨迹

- OpenTelemetry19：标准化的可观测性插桩协议，对LLM 的支持日益完善

```
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
# 配置追踪
provider = TracerProvider()
provider.add_span_processor(
BatchSpanProcessor(OTLPSpanExporter(endpoint="http://collector:4317"))
)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("agent.tracer")
class InstrumentedAgent:
def run(self, task: str) -> AgentResult:
with tracer.start_as_current_span("agent.run") as span:
span.set_attribute("agent.task", task)
span.set_attribute("agent.model", self.model)
result = self._execute(task)
span.set_attribute("agent.status", result.status)
span.set_attribute("agent.tool_calls", result.tool_call_count)
span.set_attribute("agent.tokens_used", result.tokens_used)
span.set_attribute("agent.cost_usd", result.cost_usd)
return result
def _call_llm(self, messages: list) -> str:
with tracer.start_as_current_span("llm.call") as span:
span.set_attribute("llm.model", self.model)
span.set_attribute("llm.prompt_tokens", count_tokens(messages))
response = self.llm.invoke(messages)
span.set_attribute("llm.completion_tokens", count_tokens([response]))
return response
def _call_tool(self, tool_name: str, args: dict) -> str:
with tracer.start_as_current_span(f"tool.{tool_name}") as span:
span.set_attribute("tool.name", tool_name)
span.set_attribute("tool.args", json.dumps(args))
try:
result = self.tools[tool_name](**args)
span.set_attribute("tool.success", True)
return result
except Exception as e:
span.set_attribute("tool.success", False)
span.set_attribute("tool.error", str(e))
span.record_exception(e)
raise
```

#### Listing 26.18: 基于OpenTelemetry 的结构化Agent 追踪

### 26.6.2 失败分类

系统化的失败分析需要一套失败模式分类法。如果缺少结构化分类，工程团队就会被即兴调试耗尽

- —只医表象、不治根因。下面这套分类涵盖了生产Agent 系统中最常见的六类失败，以及它们的可观察症状、自动化检测机制和经过验证的修复策略。

每种失败类型对系统设计的含义各不相同：Tool 错误属于基础设施故障，需要重试逻辑与断路器；推理错误属于模型层故障，需要迭代Prompt；幻觉需要引入接地（grounding）机制；死循环则需要硬性架构保护。在实际场景中，用户可见的单次失败往往跨越多个类别（例如：Tool 错误触发Agent 尝试自救时的推理错误，进一步演变为死循环）。

**Table 26.1: Agent 失败分类，附检测与修复策略**

| 失败类型 |  |  |
| --- | --- | --- |
| 症状 | 检测 | 修复策略 |
| Tool 错误 |  |  |
| Tool 调用抛异常、返回空 | 错误率监控 | 重试逻辑、回退Tool |
| 结果 |  |  |
| 推理错误 |  |  |
| 选错Tool、参数错误 | 轨迹分析 | 改进Prompt、加入Few- shot 示例 |
| 幻觉 |  |  |
| 捏造事实、编造Tool 结果 | 事实核查、接地校验 | 检索增强生成（RAG）、强 制要求引用 |
| 死循环 |  |  |
| 反复调用同一Tool 而无进 | 循环检测、最大迭代次数 | 硬性上限、破环Prompt |
| 展 |  |  |
| 上下文溢出 |  |  |
| 历史被截断、丢失上下文 | Token 计数 | 摘要、上下文管理 |
| 拒绝 |  |  |
| 对合法任务也予以拒绝 | 输出分类 | 调整 Prompt、微调 guardrail |

### 26.6.3 重放与调试Workflow

当生产环境出现故障时，能精确重放当时的执行过程是无价的：

```
from langsmith import Client
from datetime import datetime , timezone
ls = Client() # 使用LANGSMITH_API_KEY 环境变量
# 通过run ID 加载失败的执行轨迹
root_run = ls.read_run("run-abc123 -def456")
child_runs = list(ls.list_runs(
project_name="research -agent",
filter=f'eq(parent_run_id , "{root_run.id}")',
order="asc",
))
print(f"Trace: {root_run.id} | Status: {root_run.status}")
print(f"Error: {root_run.error}" if root_run.error else "")
print(f"Total tokens: {root_run.total_tokens}\n")
# 逐步遍历每个子run （LLM 调用、Tool 调用等）
for i, run in enumerate(child_runs):
print(f"Step {i}: [{run.run_type}] {run.name}")
print(f" Input: {str(run.inputs)[:200]}")
print(f" Output: {str(run.outputs)[:200]}")
if run.error:
print(f" ERROR: {run.error}")
# 检查导致失败的具体Prompt
if run.run_type == "llm":
print(f" Model: {run.extra.get('invocation_params', {}).get('model')}")
print(f" Messages: {run.inputs.get('messages', [])[-1]}")
print()
# 用修改后的Prompt 或模型重跑失败步骤
from openai import OpenAI
client = OpenAI()
failing_run = child_runs[4] # 例如出错的那一步
response = client.chat.completions.create(
model="gpt-4o", # 换更强的模型试试
messages=failing_run.inputs["messages"],
temperature=0,
)
print(f"Replay output: {response.choices[0].message.content[:300]}")
```

#### Listing 26.19: 用于调试的Agent 执行重放

## 26.7 生产部署模式

要大规模部署Agent，必须在执行模型、状态管理和资源分配三方面格外审慎。

**Figure 26.5: 基于队列的异步Agent 部署。worker 从队列拉取任务，并独立持久化各自状态。**

### 26.7.1 异步Agent 执行

长时间运行的Agent 应异步执行，以避免阻塞API 连接。Celery20 是Python 生态中广泛使用的分布式任务队列，能处理重试、worker 扩缩容与结果持久化：

```
from celery import Celery
from myagent import ResearchAgent
import redis
import time
app = Celery("agent_tasks", broker="redis://localhost:6379/0")
state_store = redis.Redis(host="localhost", port=6379, db=1)
@app.task(bind=True, max_retries=3, default_retry_delay=60)
def run_agent_task(self, task_id: str, task_input: str, config: dict):
"""异步执行一个Agent 任务。"""
try:
# 更新任务状态
state_store.hset(f"task:{task_id}", mapping={
"status": "running",
"started_at": time.time(),
"worker": self.request.hostname ,
})
agent = ResearchAgent(**config)
result = agent.run(task_input)
# 存储结果
state_store.hset(f"task:{task_id}", mapping={
"status": "completed",
"result": result.to_json(),
"completed_at": time.time(),
"cost_usd": result.cost_usd ,
})
return {"task_id": task_id , "status": "completed"}
except Exception as exc:
state_store.hset(f"task:{task_id}", mapping={
"status": "failed",
"error": str(exc),
"failed_at": time.time(),
```

```
})
raise self.retry(exc=exc)
# API 端点（独立的Flask/FastAPI 应用）
from flask import Flask, request , jsonify
import uuid
web_app = Flask(__name__)
@web_app.route("/tasks", methods=["POST"])
def submit_task():
task_id = str(uuid.uuid4())
task = run_agent_task.delay(
task_id=task_id ,
task_input=request.json["input"],
config=request.json.get("config", {}),
)
return jsonify({"task_id": task_id , "celery_id": task.id}), 202
```

#### Listing 26.20: 基于Celery 的异步Agent 执行

### 26.7.2 多租户隔离

面向多个客户的生产Agent 系统必须实现严格隔离：

- 命名空间隔离：每个租户的状态、记忆和Tool 配置存储在独立命名空间中

- 限流：按租户限定LLM 调用、Tool 调用与计算时间的速率

- 资源配额：为每个租户设置最大并发Agent 数、Token 预算和存储上限

- 审计日志：所有Agent 动作连同租户ID 一并记录，用于合规与计费

### 26.7.3 成本优化策略

- 模型路由：在简单子任务（分类、抽取）上使用更小更便宜的模型，把大模型留给复杂推理

- Prompt 缓存：OpenAI 与Anthropic 都提供对重复系统Prompt 的缓存，可为高流量Agent降低多达90% 的成本

- 结果缓存：在一定时间窗口内对相同输入的Tool 结果进行缓存

- 批处理：在延迟允许时，把多次独立的LLM 调用合并批处理

- 提前终止：检测到Agent 已掌握足够信息可作答时，及早结束循环

```
class CostOptimizedRouter:
TASK_MODEL_MAP = {
"classification": "gpt-4o-mini",
"extraction": "gpt-4o-mini",
"summarization": "gpt-4o-mini",
"reasoning": "gpt-4o",
"code_generation": "gpt-4o",
"complex_analysis": "o1",
}
def route(self, task_type: str, complexity: float) -> str:
base_model = self.TASK_MODEL_MAP.get(task_type , "gpt-4o-mini")
# 对高复杂度任务升级到更强模型
if complexity > 0.8 and base_model == "gpt-4o-mini":
return "gpt-4o"
return base_model
```

```
def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) ->
float:
pricing = {
"gpt-4o-mini": (0.15e-6, 0.60e-6),
"gpt-4o": (2.50e-6, 10.0e-6),
"o1": (15.0e-6, 60.0e-6),
}
in_price , out_price = pricing[model]
return input_tokens * in_price + output_tokens * out_price
```

#### Listing 26.21: 面向成本优化的模型路由

### 26.7.4 自动扩缩容策略

Agent工作负载呈现突发且难以预测的特征。有效的自动扩缩容需要：

- 基于队列深度的扩缩：根据任务队列深度而非CPU 利用率来调整worker 数量

- 预测式扩缩：利用历史模式（时段、星期）在需求高峰之前提前扩容

- 抢占式实例：长任务可结合检查点使用spot/可抢占（preemptible）实例以节省成本

- 优雅停机：worker在缩容前完成当前任务，避免状态损坏

## 26.8 框架对比

如何选择合适的框架

「最佳」框架取决于你的具体需求。先问自己几个问题：

- 需要对Agent 流程显式控制？→LangGraph

- 在构建带代码执行的多Agent 系统？→AutoGen

- 希望以最少样板代码实现基于角色的Agent？→CrewAI

- 基于OpenAI 生态进行开发？→Agents SDK

- 希望自动化优化Prompt？→DSPy

- 处于企业级.NET/Azure环境？→Semantic Kernel

## 26.9 完整实现示例：生产级研究型Agent

下面给出一个基于LangGraph 构建的完整、生产可用的研究型Agent，演示Tool 定义、状态schema、图构建、错误处理与部署配置。

```
生产级研究Agent 架构
本示例实现的研究Agent：(1) 接收研究主题，(2) 在网络上检索相关来源，(3) 阅读并综合关键
文档，(4) 撰写结构化报告，(5) 通过重试逻辑优雅处理错误。Agent 使用检查点实现可恢复性，
使用结构化日志实现可观测性。
# === tools.py ===
import httpx
import json
import os
import uuid
from datetime import datetime , timezone
from urllib.parse import urlparse
from langchain_core.tools import tool
```

```
from tenacity import retry, stop_after_attempt , wait_exponential
from utils import extract_text # HTML -> 纯文本的辅助函数（例如BeautifulSoup ）
from database import db # 应用数据库连接
@tool
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def search_web(query: str, num_results: int = 5) -> str:
"""在网络上搜索信息，返回JSON 列表形式的结果。"""
if not query.strip():
raise ValueError("Search query cannot be empty")
response = httpx.get(
"https://api.search.example.com/search",
params={"q": query, "n": num_results},
headers={"Authorization": f"Bearer {os.environ['SEARCH_API_KEY ']}"},
timeout=10.0,
)
response.raise_for_status()
results = response.json()["results"]
return json.dumps([{"title": r["title"], "url": r["url"],
"snippet": r["snippet"]} for r in results])
@tool
@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
def fetch_document(url: str, max_chars: int = 5000) -> str:
"""从URL 拉取并抽取文本内容。"""
allowed_domains = os.environ.get("ALLOWED_DOMAINS", "").split(",")
domain = urlparse(url).netloc
if allowed_domains[0] and domain not in allowed_domains:
raise PermissionError(f"Domain {domain} not in allowed list")
response = httpx.get(url, timeout=15.0, follow_redirects=True)
response.raise_for_status()
return extract_text(response.text)[:max_chars]
@tool
def save_report(title: str, summary: str, sections: list[dict]) -> str:
"""将结构化研究报告保存到数据库。"""
report_id = str(uuid.uuid4())
db.reports.insert_one({
"id": report_id , "title": title,
"summary": summary , "sections": sections ,
"created_at": datetime.now(timezone.utc).isoformat(),
})
return json.dumps({"report_id": report_id , "status": "saved"})
TOOLS = [search_web , fetch_document , save_report]
```

#### Listing 26.22: 完整生产研究Agent：Tool 与状态

```
# === agent.py ===
import json
from typing import TypedDict , Annotated , List, Literal
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage , HumanMessage , SystemMessage ,
AIMessage
from tools import TOOLS
SYSTEM_PROMPT = """You are a professional research analyst. Your task is to:
1. Search for relevant information on the given topic
2. Read and analyze key sources (aim for 3-5 sources)
3. Synthesize findings into a structured report using save_report
Guidelines:
- Always verify information across multiple sources
- Cite your sources in the report
```

```
- If a tool fails, try an alternative approach
- Complete the task in at most 15 tool calls
- Use save_report exactly once when you have sufficient information"""
class ResearchState(TypedDict):
messages: Annotated[List[BaseMessage], add_messages]
topic: str
sources_found: List[str]
sources_read: List[str]
report_id: str | None
error_count: int
tool_call_count: int
status: Literal["researching", "done", "failed"]
tool_executor = ToolNode(TOOLS)
def research_node(state: ResearchState) -> dict:
"""主推理节点：由LLM 决定下一步动作。"""
llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(TOOLS)
messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
response = llm.invoke(messages)
return {"messages": [response]}
def tool_node_with_error_handling(state: ResearchState) -> dict:
"""执行Tool 调用，附带错误处理与状态更新。"""
try:
result = tool_executor.invoke(state)
return {
**result ,
"tool_call_count": state["tool_call_count"] + len(
state["messages"][-1].tool_calls
),
}
except Exception as e:
# 用AIMessage 把错误信号回传给LLM ，便于其调整策略
error_msg = AIMessage(content=f"Tool execution failed: {e}. Try a different
approach.")
return {
"messages": [error_msg],
"error_count": state["error_count"] + 1,
}
def check_completion(state: ResearchState) -> dict:
"""检查报告是否已保存，并相应更新状态。"""
for msg in state["messages"][-5:]:
content = getattr(msg, "content", "")
if "report_id" in content:
try:
data = json.loads(content)
return {"status": "done", "report_id": data["report_id"]}
except (json.JSONDecodeError , KeyError):
pass
return {}
def route_after_llm(state: ResearchState) -> str:
"""根据LLM 响应决定下一步去向。"""
if state["error_count"] >= 5 or state["tool_call_count"] >= 15:
return "fail"
last_message = state["messages"][-1]
if hasattr(last_message , "tool_calls") and last_message.tool_calls:
return "tools"
if len(state["messages"]) > 30:
return "fail"
return "research" # LLM 需要继续推理
def fail_node(state: ResearchState) -> dict:
```

return {"status": "failed"}

#### Listing 26.23: 完整生产研究Agent：状态与节点

```
# === graph.py ===
from langgraph.graph import StateGraph , START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
async def build_graph(db_url: str) -> CompiledStateGraph:
"""构建并编译研究Agent 图。"""
checkpointer = AsyncPostgresSaver.from_conn_string(db_url)
await checkpointer.setup() # 如有需要则建表
builder = StateGraph(ResearchState)
# 添加节点
builder.add_node("research", research_node)
builder.add_node("tools", tool_node_with_error_handling)
builder.add_node("check", check_completion)
builder.add_node("fail", fail_node)
# 定义边
builder.add_edge(START, "research")
builder.add_conditional_edges(
"research",
route_after_llm ,
{"tools": "tools", "research": "research", "fail": "fail"}
)
builder.add_edge("tools", "check")
builder.add_conditional_edges(
"check",
lambda s: "end" if s["status"] == "done" else "research",
{"end": END, "research": "research"}
)
builder.add_edge("fail", END)
return builder.compile(checkpointer=checkpointer)
# === deployment.py ===
import os
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI , BackgroundTasks , HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
graph: CompiledStateGraph = None # 在启动时初始化
@asynccontextmanager
async def lifespan(app: FastAPI):
global graph
graph = await build_graph(os.environ["DATABASE_URL"])
yield
app = FastAPI(title="Research Agent API", lifespan=lifespan)
class ResearchRequest(BaseModel):
topic: str
user_id: str
class ResearchResponse(BaseModel):
task_id: str
status: str
@app.post("/research", response_model=ResearchResponse)
```

```
async def start_research(request: ResearchRequest , background_tasks: BackgroundTasks):
task_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": task_id , "user_id": request.user_id}}
initial_state = {
"messages": [HumanMessage(content=f"Research topic: {request.topic}")],
"topic": request.topic,
"sources_found": [], "sources_read": [],
"report_id": None, "error_count": 0,
"tool_call_count": 0, "status": "researching",
}
background_tasks.add_task(graph.ainvoke , initial_state , config)
return ResearchResponse(task_id=task_id , status="started")
@app.get("/research/{task_id}")
async def get_research_status(task_id: str):
config = {"configurable": {"thread_id": task_id}}
state = await graph.aget_state(config)
if state is None:
raise HTTPException(status_code=404, detail="Task not found")
return {
"task_id": task_id ,
"status": state.values.get("status", "unknown"),
"report_id": state.values.get("report_id"),
"tool_calls": state.values.get("tool_call_count", 0),
"error_count": state.values.get("error_count", 0),
}
```

#### Listing 26.24: 完整生产研究Agent：图与部署

```
# === Dockerfile ===
# FROM python:3.11-slim
# WORKDIR /app
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# COPY . .
# CMD ["uvicorn", "deployment:app", "--host", "0.0.0.0", "--port", "8000"]
# === kubernetes/deployment.yaml （此处以Python dict 形式示意）===
k8s_deployment = {
"apiVersion": "apps/v1",
"kind": "Deployment",
"metadata": {"name": "research -agent", "namespace": "agents"},
"spec": {
"replicas": 3,
"selector": {"matchLabels": {"app": "research -agent"}},
"template": {
"metadata": {"labels": {"app": "research -agent"}},
"spec": {
"containers": [{
"name": "agent",
"image": "myregistry/research -agent:latest",
"ports": [{"containerPort": 8000}],
"resources": {
"requests": {"memory": "512Mi", "cpu": "250m"},
"limits": {"memory": "2Gi", "cpu": "1000m"},
},
"env": [
{"name": "DATABASE_URL", "valueFrom": {
"secretKeyRef": {"name": "agent-secrets", "key": "db-url"
}}},
{"name": "OPENAI_API_KEY", "valueFrom": {
"secretKeyRef": {"name": "agent-secrets", "key": "openai -
key"}}},
],
"livenessProbe": {"httpGet": {"path": "/health", "port": 8000},
"initialDelaySeconds": 30, "periodSeconds":
```

```
10},
"readinessProbe": {"httpGet": {"path": "/ready", "port": 8000},
"initialDelaySeconds": 10, "periodSeconds": 5},
}]
}
}
}
}
# HorizontalPodAutoscaler 基于队列深度指标进行扩缩
hpa_config = {
"apiVersion": "autoscaling/v2",
"kind": "HorizontalPodAutoscaler",
"metadata": {"name": "research -agent-hpa", "namespace": "agents"},
"spec": {
"scaleTargetRef": {
"apiVersion": "apps/v1",
"kind": "Deployment",
"name": "research -agent",
},
"minReplicas": 2,
"maxReplicas": 20,
"metrics": [{
"type": "External",
"external": {
"metric": {"name": "agent_task_queue_depth"},
"target": {"type": "AverageValue", "averageValue": "10"},
}
}]
}
}
```

#### Listing 26.25: 部署配置：Docker 与Kubernetes

![图 458-2](assets/ch26_p458_fig1.png)

## 26.10 小结

Agent开发框架已显著成熟，为构建生产级AI Agent 的工程挑战提供了结构化解法。本节的关键要点如下：

1. 框架选型很重要：不同框架针对不同问题做优化。LangGraph擅长复杂可控的Workflow；AutoGen 擅长多Agent 协作；CrewAI 擅长基于角色的简洁建模；DSPy 擅长自动化优化。

2. 测试不可妥协：基于LLM 的Agent 具有非确定性，全面的测试（单元、集成、行为、性能）是生产可靠性的必要条件。

3. 可观测性驱动迭代：缺少详细的Agent 执行轨迹，故障诊断与性能改进只能靠猜。要尽早投入可观测性基础设施。

4. 异步执行是常态：生产Agent 是长时间运行的进程，需要基于队列的执行、检查点以及优雅的失败处理。

5. 成本管理至关重要：LLM API成本随使用量线性增长。模型路由、缓存与提前终止可以在不牺牲质量的前提下把成本降低50–90%。

6. 生命周期本质上是迭代的：Agent开发不是一次性工程。持续的监控、失败分析与改进，是在外部世界不断变化的同时维持性能的关键。

该领域演进迅猛，新的框架、工具与最佳实践层出不穷。但本节涵盖的原则——显式状态管理、全面测试、深度可观测性以及系统化迭代——无论当下流行哪些具体工具，都为之提供了稳固的基础。
