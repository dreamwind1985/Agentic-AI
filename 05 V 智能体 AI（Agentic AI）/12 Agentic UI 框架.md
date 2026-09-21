# Agentic UI 框架

随着大语言模型从被动的文本生成器演进为能够规划、调用工具与进行多步推理的主动Agent，人类与之交互的界面也必须同步演化。传统聊天界面——为单轮或短上下文对话而设计——无法满足Agent 化工作流的要求：长时间运行的任务、分支决策树、并行的工具调用，以及对有效人类监督的需求。本节考察Agent 化UI 框架（agentic UI frameworks）的图景：使丰富、透明、可信赖的人机协作成为可能的设计范式、组件库与实现模式。

## 27.1 动机：超越聊天框

![图 460-1](assets/ch27_p460_fig1.png)

执行20 步研究任务、浏览网页、撰写并运行代码、综合成一份报告时，用户需要解答的问题是简单的文本响应所无法提供的：

- Agent 此刻在做什么？长时间运行的任务需要进度反馈；沉默会滋生不信任。

- Agent 为何做出这个决策？推理过程的透明性使用户能够尽早发现错误。

- 使用了哪些工具，输入了什么？工具来源（tool provenance）对核实事实性论断与审计行为至关重要。

- 我应该在哪里介入？Agent必须呈现那些需要人类判断的决策点，同时避免用每一项微决策淹没用户。

- 我能撤销吗？不可逆操作（发送邮件、修改文件、执行代码）需要明确的确认与回滚路径。

自动化偏差风险

人机自动化交互的研究一致表明，用户会过度信任自动化系统，尤其是当这些系统自信地呈现输出而不附带不确定性信号时[398]。Agent 化UI 必须通过呈现不确定性、展示推理过程，以及让质疑或覆盖Agent 决策变得容易，来主动对抗自动化偏差（automation bias）。

因此，Agent 化UI 的设计位于人机交互（Human-Computer Interaction, HCI）、可解释AI（Explainable AI, XAI）与软件工程的交汇处。核心设计目标包括：

1. 透明性：让Agent 的内部状态对用户清晰可读。

2. 可控性：提供有意义的介入节点，而不要求持续监督。

3. 信任校准：帮助用户建立关于Agent 能力与局限的准确心智模型。

4. 效率：最小化认知负担；在合适的时间呈现合适的信息。

5. 可恢复性：让错误的发现与撤销成本低廉。

## 27.2 Agent 的UI 范式

没有任何单一的UI 范式能适应所有Agent 化用例。合适的界面取决于任务时长、所需的人类介入程度、输出类型以及用户的专业水平。其谱系从完全对话式的聊天界面，跨越到几乎无需人类交互的完全自主仪表盘。

### 27.2.1 基于聊天的界面

聊天范式——消息气泡、文本输入框和滚动历史——仍然是与LLM 交互最为熟悉的入口。其优势在于学习曲线平缓与自然语言的灵活性。在Agent 化场景下，聊天界面通常会扩展以下能力：

- 流式响应：Token在生成时即逐步出现，提供即时反馈并降低感知延迟。通过服务器推送事件（Server-Sent Events, SSE）或WebSockets 实现。

- 内联工具指示器：消息流中的小徽章或可展开区段，用于显示何时调用了工具（例如“[Searched the web for: climate change 2024]”）。

- 输入指示与状态消息：“Agent is thinking…”、“Running Python code…”、“Fetching results…”等提示在延迟间隙保持用户的知情感。

- 消息线程化：对于多轮Agent 任务，可折叠的子线程能容纳中间步骤，而不会让主对话变得杂乱。

![图 461-1](assets/ch27_p461_fig1.png)

### 27.2.2 画布与产物式界面

画布范式（canvas paradigm）由Claude Artifacts1 与ChatGPT Canvas2 推广，引入了分屏布局：左侧承载对话，右侧（“canvas” 或“artifact panel”）将生成的内容——代码、文档、图表、电子表格——以可实时编辑的产物（artifact）形式呈现。

关键特征：

- 持久化产物：生成的内容跨轮次持久存在，可以通过自然语言指令迭代优化（“把图表改成蓝色”、“给这个函数加上错误处理”）。

- 原位编辑：用户可以直接编辑产物，Agent也能观察并响应这些编辑。

- 版本历史：产物维护修订历史，支持回滚到任意先前状态。

- 多产物工作区：进阶实现支持多个并存的产物（例如一个代码文件、其测试套件以及一个文档页面）。

画布范式特别适合共创（co-creation）任务：写作、编程、数据分析与设计——这些场景的产出是文档或产物，而非对话式的回答。

### 27.2.3 工作流可视化

对于执行结构化计划（步骤序列或步骤图）的Agent，工作流可视化UI 使计划显式化并可追踪。该范式常见于：

- Agent 化流水线（LangGraph、AutoGen、CrewAI）：将Agent 的执行图渲染为有向无环图（Directed Acyclic Graph, DAG）或流程图，节点代表步骤，边代表数据流或控制流。

- 任务分解视图：将Agent 的高层计划呈现为清单或甘特图式的时间线，每个子任务可展开以显示其自身的步骤。

- 实时进度追踪：节点在执行时改变颜色或显示加载图标；完成的节点展示输出；失败的节点展示错误细节。

LangGraph Studio3是该范式的典范，它为LangGraph Agent 提供基于图的调试器与可视化工具。用户可以检查每个节点的状态、回放执行过程，并注入修改后的状态以测试备选路径。

### 27.2.4 仪表盘与监控界面

对于长时间运行或生产环境中的Agent，仪表盘UI 提供运维视角：

- 实时状态：哪些Agent 正在运行、空闲或失败；当前任务与步骤。

- 资源指标：Token消耗、API 调用计数、延迟直方图、成本估算。

- 队列管理：待处理任务、优先级排序、限流状态。

- 告警与异常检测：异常行为（过多的重试、成本激增、反复失败）以通知形式呈现。

- 历史分析：任务完成率、平均时长、错误频率随时间的变化。

仪表盘UI 通常基于Grafana4、自定义React 仪表盘或Streamlit 等工具构建，面向运维人员而非最终用户。

### 27.2.5 协作型界面

协作型UI 将Agent 视为共享工作区（文档、代码库或设计画布）中与人类协作者并列的同侪贡献者。关键特性包括：

- 在线指示：Agent以带名字的光标或头像出现在共享工作区中。

- 变更归属：Agent所做的编辑在视觉上与人类编辑相区分（例如用颜色编码的diff）。

- 内联建议：Agent以修订标记或评论的形式提出变更，人类可以接受、拒绝或修改。

- 冲突解决：当Agent 与人类同时编辑同一区域时，UI 会呈现冲突并辅助解决。

该范式正在Cursor5（带AI 的协作式代码编辑）、Notion AI6 以及集成Gemini 的Google Docs7等工具中兴起。

### 27.2.6 带检查点的自主模式

在自主性谱系的另一端，一些Agent 几乎完全独立运行——浏览网页、编写代码、执行命令——仅在预定义的、需要人类批准的检查点（checkpoint）处现身。该范式被用于：

- 计算机使用Agent（Anthropic Computer Use8、OpenAI Operator9）：Agent 控制浏览器或桌面；UI 显示实时屏幕画面，并在不可逆操作前暂停等待批准。

- 带门控的自动化流水线：CI/CD风格的工作流——Agent 完成一个阶段后等待人类“合并” 方可继续。

- 定时Agent：按计划运行并异步报告结果的Agent，通过基于通知的UI 来审阅输出并批准后续操作。

![图 462-1](assets/ch27_p462_fig1.png)

## 27.3 Agent 的关键UI 组件

无论采取何种总体范式，Agent 化UI 都共享一组反复出现的组件。本节梳理其中最重要的部分，并给出各自的设计指导。

### 27.3.1 思考与推理展示

现代LLM——尤其是采用思维链（Chain-of-Thought，CoT）或延展思考训练的模型（例如OpenAI o1/o3、带扩展思考的Anthropic Claude）——在产出最终响应之前会生成大量内部推理。呈现这些推理是一把双刃剑：它提升了透明度，但冗长的内部独白也可能淹没用户。

最佳实践：

- 可折叠的推理区块：显示摘要（“思考了12 秒”），并为想看细节的用户提供展开按钮。

- 渐进式披露：默认仅展示最终结论；推理可按需查看。

- 结构化推理：如果模型产出结构化思考（假设、证据、结论），应以视觉层次来呈现，而非堆成大段文字墙。

- 推理与响应的区分：在视觉上清晰区分内部推理（可能包含错误或失败尝试）与最终响应。

### 27.3.2 工具使用可视化

工具调用是Agent 与外部世界交互的主要机制。将其可视化对建立信任与进行调试至关重要。

![图 463-1](assets/ch27_p463_fig1.png)

- 内联工具卡片：消息流中的紧凑卡片，显示工具名、一行输入摘要与状态（运行中/成功/错误）。可展开查看完整细节。

- 工具时间线：横向时间线显示一轮中的全部工具调用及其耗时，便于识别瓶颈。

- 输入/输出diff：对于修改状态的工具（例如文件编辑），展示前后对比diff。

- 工具图标与品牌：常用工具（网页搜索、代码执行、文件系统、API）使用可辨识的图标，便于快速扫视。

- 错误高亮：失败的工具调用以红色显示，附带错误消息与任何重试尝试。

### 27.3.3 进度指示器

多步Agent 任务需要丰富的进度反馈：

- 步骤级进度：编号的计划步骤列表，每完成一项即打钩。对于动态计划，步骤可随Agent 调整而增删。

- Token 流指示器：生成期间的闪烁光标或动画省略号；为高级用户提供每秒Token 数计数器。

- 预计完成：在可行时，依据任务复杂度与历史表现给出预计到达时间（ETA），并以合适的不确定性表达（“大约2–5 分钟”）。

- 子任务嵌套：对于层级化任务，提供带可展开子任务的树状进度视图。

- 取消：清晰可见的“停止” 按钮，能优雅地终止Agent 并总结迄今为止已完成的工作。

### 27.3.4 审批门控

审批门控（approval gates）是人机协同（Human-in-the-Loop）控制的主要机制。它必须设计得既富含信息（给用户足够的上下文以做出良好决策），又不会令人疲劳（无需对每个琐碎操作都请求批准）。

![图 464-1](assets/ch27_p464_fig1.png)

- 操作摘要：用通俗语言描述Agent 想要做什么（“向john@example.com 发送附带报告的邮件”）。

- 风险指示器：操作可逆性的视觉信号（绿色= 易于撤销，黄色= 难以撤销，红色= 不可逆）。

- 批准/ 拒绝/ 修改：三选项界面；“修改” 会在批准前打开操作参数的编辑器。

- 上下文面板：可展开区域，展示Agent 想执行该操作的缘由（相关推理、先前步骤）。

- 超时行为：清晰说明用户不响应时会发生什么（Agent暂停而非继续）。

### 27.3.5 上下文展示

Agent 会维护影响其行为的内部状态——记忆、激活的工具、检索到的文档、对话历史。让这些状态可见有助于用户理解并预测Agent 的行为。

- 记忆面板：显示Agent 当前对用户、任务以及先前交互“记住” 了什么。用户可编辑。

- 激活工具列表：Agent当前可用的工具，配有启用/禁用开关。

- 检索到的上下文：当前位于Agent 上下文窗口中的文档或数据块，附带来源引用。

- Token 预算指示器：上下文窗口已消耗多少，帮助用户判断何时开启新会话。

### 27.3.6 错误与恢复UI

Agent 会失败——工具返回错误、模型出现幻觉、计划变得不可行。UI 必须优雅地处理失败：

- 错误卡片：内联显示失败信息，包括错误类型、消息以及Agent 的解读。

- 重试控件：手动重试按钮，可选地调整参数。

- 备选方案：当主方案失败时，Agent提出备选；UI 将其作为可选项呈现。

- 部分结果：若多步任务中途失败，UI显示已完成的步骤及其输出，保留部分价值。

- 升级路径：当Agent 无法继续时，提供通往人工支持或手动完成的清晰路径。

### 27.3.7 置信度指示器

LLM 是带有（已校准或未校准的）不确定性的概率系统。呈现置信度有助于用户判断何时该信任、何时该核实：

- 措辞犹豫的展示：高亮诸如“我不确定” 或“你或许需要核实” 之类的措辞，以引起对低置信度论断的注意。

- 来源质量指示：对检索到的信息显示来源新近度、权威性与相关性评分。

- 显式不确定性请求：“你有多自信？” 按钮促使Agent 自评并解释其不确定性。

- 核实建议：对于高风险输出，Agent主动建议核实步骤（“我建议独立核对这个计算”）。

## 27.4 框架与库

日益增长的框架生态加速了Agent 化UI 的开发。我们按主要语言与使用场景，考察其中被最广泛采用的框架。

### 27.4.1 Vercel AI SDK

Vercel AI SDK [399] 是一个TypeScript/JavaScript 库，用于在React、Next.js、Svelte 与Vue 中构建流式AI 界面。它是当前生产级Web Agent UI 中应用最广泛的框架。

核心抽象：

- useChat：管理聊天对话的React Hook，支持流式响应、消息历史与加载状态。

- useCompletion：用于带流式的单轮文本补全的Hook。

- useObject：流式传输结构化JSON 对象，使复杂输出能够渐进式渲染。

- streamText / streamObject：通过HTTP 流式传输LLM 响应的服务端函数。

生成式UI（AI SDK RSC）：Vercel AI SDK 最具特色的能力，是通过React Server Components（RSC）支持生成式UI（generative UI）。LLM 不再仅返回文本，而是可以调用工具，其结果被渲染为任意React 组件——天气小部件、股票图表、预订表单——并直接流式注入UI。进一步讨论参见

## 27.5 节。

### 27.4.2 Chainlit

Chainlit [400] 是一个Python 框架，以极少的样板代码构建生产可用的Agent UI。它在LangChain与LlamaIndex 生态中尤其流行。

关键特性：

- 步骤可视化：Chainlit原生地将LangChain 与LlamaIndex 的执行步骤渲染为可折叠的树，显示每次链调用、检索与工具调用。

- 多模态支持：开箱支持文件上传、图像展示、音频播放与PDF 渲染。

- 认证与会话：内置用户认证、持久化对话历史与多用户支持。

- 自定义元素：可从Python 注册并渲染React 组件，实现丰富的自定义可视化。

- 反馈采集：内置点赞/踩反馈，附带可选评论并存入数据库。

```
import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
@tool
def search(query: str) -> str:
# Search for information.
"""Search for information."""
return f"Results for: {query}"
agent = create_react_agent(
ChatOpenAI(model="gpt-4o"), tools=[search]
)
@cl.on_message
async def on_message(message: cl.Message):
# Chainlit 在使用回调处理器时，会自动将每一步渲染为可折叠的UI 元素
# Chainlit automatically renders each step as a collapsible UI element
```

```
# when using the callback handler
async with cl.Step(name="Agent", type="run") as step:
step.input = message.content
result = await agent.ainvoke(
{"messages": [{"role": "user", "content": message.content}]},
config={"callbacks": [cl.LangchainCallbackHandler()]}
)
output = result["messages"][-1].content
step.output = output
await cl.Message(content=output).send()
```

#### Listing 27.1: 最小化的Chainlit Agent 及其步骤可视化

### 27.4.3 Gradio

Gradio [401] 是一个Python 库，用于快速搭建ML demo 与Agent 界面。其gr.ChatInterface 与gr.Blocks API 使得用极少代码即可快速原型化对话式Agent。

#### 在Agent 化UI 方面的优势：

- 零配置部署：通过Hugging Face Spaces 一行代码完成分享。

- 自定义组件：Gradio自定义组件系统允许构建能与Python Backend 无缝集成的React 组件。

- 多模态输入：以极少配置支持文件上传、图像、音频、视频及网络摄像头输入。

- 流式：原生支持基于生成器（generator）的流式响应。

局限：Gradio的布局系统不如完整React 框架灵活，其状态管理以会话为作用域，使复杂的多Agent 协调具有挑战性。

### 27.4.4 Streamlit

Streamlit [402] 是一个用于数据应用的Python 框架，被广泛用于Agent 仪表盘与监控UI。其响应式执行模型——每次交互整个脚本重跑——简单直观，但对复杂的Agent 化工作流可能受限。

#### Agent 化用例：

- Agent 仪表盘：使用st.metric、st.dataframe 与st.status 提供实时指标、任务队列与状态展示。

- 会话状态：st.session_state 在重跑之间持久化Agent 状态，从而支持多轮对话。

- 流式：st.write_stream 渐进式渲染生成器输出。

- Fragments：@st.fragment 装饰器允许局部重跑，提升实时更新仪表盘的性能。

### 27.4.5 OpenAI Assistants Playground

OpenAI Assistants Playground 可视为Agent 化UI 设计的参考实现。它演示了：

- 基于线程（thread）、带持久化历史的对话管理。

- 文件附件与检索过程的可视化。

- 代码解释器执行及其输出展示（stdout、图像、文件）。

- 函数调用展示，附带输入/输出检查。

- 运行步骤可视化，展示模型调用与工具调用的序列。

尽管Playground 并非用于构建自定义UI 的框架，但其设计模式被广泛仿效。

### 27.4.6 LangGraph Studio

LangGraph Studio [403] 是一个桌面应用，为LangGraph Agent 提供可视化IDE。它是目前最成熟的工具使用与工作流可视化环境。

特性：

- 图可视化：交互式渲染Agent 的状态机，节点代表Agent 步骤，边代表状态转移。

- 状态检查：在执行的任意点都可以以结构化JSON 形式检视完整的Agent 状态（全部变量、记忆、工具结果）。

- 时间旅行调试：回放任意先前执行步骤、修改状态，并从该点重新运行。

- 人机协同集成：可在任意节点设置断点；执行会暂停并等待人类输入后再继续。

- 多Agent 支持：可视化supervisor 与子Agent 的层级以及Agent 间消息传递。

### 27.4.7 框架对比

**表27.1 总结了上文所讨论框架的关键特征。**

| 框架 | Table 27.1: Agent 化UI 框架对比。 语言 | 流式 工具可视化 多Agent 生成式UI 生产可用 |
| --- | --- | --- |
| Vercel AI SDK | TypeScript | ✓ 部分 部分 ✓ ✓ |
| Chainlit | Python | ✓ ✓ 部分 部分 ✓ |
| Gradio | Python | ✓ ◦ × ◦ ✓ |
| Streamlit | Python N/A（托管） | ✓ ◦ × × ✓ |
| OAI Playground |  | ✓ ✓ × × × |
| LangGraph Studio | Python/TS | ✓ ✓ ✓ × 部分 |

## 27.5 生成式UI

![图 467-1](assets/ch27_p467_fig1.png)

由模型动态选择并参数化适合当前上下文的UI 组件。

### 27.5.1 用React Server Components 构建动态界面

Vercel AI SDK 的RSC（React Server Components10）集成是生成式UI 最成熟的实现。其架构如下：

1. 用户向Next.js11 的server action 发送消息。

2. 服务端以一组工具调用LLM，每个工具关联一个React 组件。

3. 当LLM 调用某个工具（例如show_weather）时，服务端以工具输出作为props 渲染相应的React 组件。

4. 渲染后的组件作为React Server Component 流式发送给客户端，在聊天中内联出现。

```
// app/actions.tsx (Server Action)
import { streamUI } from 'ai/rsc';
import { openai } from '@ai-sdk/openai';
import { WeatherCard } from '@/components/WeatherCard';
import { StockChart } from '@/components/StockChart';
export async function chat(userMessage: string) {
const result = await streamUI({
model: openai('gpt-4o'),
messages: [{ role: 'user', content: userMessage }],
tools: {
show_weather: {
description: 'Display current weather for a location',
parameters: z.object({
location: z.string(),
unit: z.enum(['celsius', 'fahrenheit']),
}),
// 工具结果作为React 组件渲染
// Tool result rendered as a React component
generate: async ({ location , unit }) => {
const data = await fetchWeather(location , unit);
return <WeatherCard data={data} />;
},
},
show_stock: {
description: 'Display stock price chart',
parameters: z.object({ ticker: z.string() }),
generate: async ({ ticker }) => {
const data = await fetchStockData(ticker);
return <StockChart ticker={ticker} data={data} />;
},
},
},
});
return result.value;
}
```

Listing 27.2: 基于Vercel AI SDK RSC 的生成式UI（TypeScript）

### 27.5.2 基于内容类型的自适应界面

生成式UI 使界面能够根据所呈现内容的性质自适应：

- 表格数据→可排序、可过滤、带导出选项的数据表。

- 地理数据→带标记与图层的交互式地图。

- 时间序列→可缩放、带注释的折线图。

- 代码→带语法高亮与运行按钮的编辑器。

- 文档→带注释工具的格式化文档查看器。

- 表单/结构化输入→动态生成的表单字段。

模型在此扮演UI 编排者（UI orchestrator）的角色，为每条信息选择最合适的呈现方式。这降低了开发者预判所有可能输出类型并预先构建相应组件的需求。

![图 468-2](assets/ch27_p468_fig1.png)

## 27.6 流式与实时模式

流式（Stream）是Agent 化UI 的基础：它把体验从“等待一个结果” 转变为“观察Agent 在工作”。

本节涵盖关键的流式模式及其实现考量。

### 27.6.1 Token 流

Token 流在Token 被生成时增量传输LLM 输出，而不等待完整响应。常用的两种传输机制：

- 服务器推送事件（Server-Sent Events, SSE）12：从服务器到客户端的单向HTTP 流。每个事件携带一块Token。SSE 简单，运行于标准HTTP/1.1 之上，并由浏览器自动重连。它是LLM 流式API 的主导机制（OpenAI、Anthropic、Google 都使用SSE）。

- WebSockets：双向持久连接。实现更复杂，但在客户端需要在流中途发送数据的交互式流场景下是必要的（例如打断Agent、在生成中途提供反馈）。

```
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
import json
app = FastAPI()
client = AsyncOpenAI()
async def token_stream(prompt: str):
# 产出SSE 格式Token 块的生成器。
"""Generator that yields SSE-formatted token chunks."""
stream = await client.chat.completions.create(
model="gpt-4o",
messages=[{"role": "user", "content": prompt}],
stream=True,
)
async for chunk in stream:
delta = chunk.choices[0].delta
if delta.content:
# SSE 格式："data: <json >\n\n"
# SSE format: "data: <json >\n\n"
yield f"data: {json.dumps({'token': delta.content})}\n\n"
elif chunk.choices[0].finish_reason:
yield f"data: {json.dumps({'done': True})}\n\n"
@app.get("/stream")
async def stream_endpoint(prompt: str):
return StreamingResponse(
token_stream(prompt),
media_type="text/event-stream",
headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
)
```

#### Listing 27.3: 基于FastAPI 的SSE Token 流

### 27.6.2 工具调用流

现代LLM API 支持流式工具调用：工具名与参数被增量地流式传输，使UI 能够在工具尚未真正被调用之前就显示“Agent 正在调用search_web，查询：‘climate change 2024’…”。这需要解析不完整的JSON，可通过流式JSON 解析器实现。

工具调用流的模式：

- 渐进式参数展示：在参数流入时即展示，即便调用尚未完成。

- 并行工具调用指示器：当模型同时调用多个工具时，将它们全部标记为待处理，并在结果到达时分别更新。

- 工具结果流：有些工具（例如代码执行、网页抓取）本身就能流式输出结果；应将其渐进式管道传输至UI。

### 27.6.3 多Agent 流

在多Agent 系统中，可能有多个Agent 同时产出输出。UI 必须处理并行流：

- 带Agent 标签的流：每条流都标记Agent 身份；UI 将其渲染到不同的泳道或面板中。

- 流合并：对于supervisor-子Agent 模式，supervisor 的流可能与子Agent 流交错；UI 必须维护一致的顺序。

- 背压（Backpressure）：如果UI 渲染速度跟不上流的到达速度（例如多个Agent 同时生成），背压机制必须防止缓冲区溢出。策略包括：丢弃中间Token（仅展示最新）、批量更新或暂停较慢的流。

### 27.6.4 乐观UI 更新

乐观UI 更新（optimistic UI updates）通过在服务器确认前立即将用户操作反映到UI 上，提升感知响应性：

- 当用户发送消息时，它会（乐观地）立即出现在聊天历史中，而请求仍在传输中。

- 当审批门控被接受时，UI立即将该操作显示为“已批准”，并开始展示Agent 的后续步骤，甚至先于服务器处理该批准。

- 如果服务器返回错误，乐观更新将被回滚并显示错误状态。

### 27.6.5 背压处理

在高吞吐量的Agent 化场景下，数据到达速率可能超出UI 的渲染能力。管理背压的策略：

- Token 批处理：将Token 缓冲50–100ms 后批量渲染而非逐个渲染，降低DOM 更新频率。

- 虚拟滚动：对于长输出，仅渲染可视区域的内容，丢弃屏幕外的DOM 节点。

- 节流更新：对于指标与状态展示，无论入流速率如何，都以固定速率（例如10 Hz）更新。

- 渐进式细节：在高吞吐期间显示摘要视图；完整细节按需提供。

## 27.7 人机协同UI 设计

人机协同（Human-in-the-Loop, HITL）交互是Agent 化UI 中最关键的设计挑战之一。目标是在维持有效人类监督的同时，避免形成抹杀自动化效率收益的瓶颈。

### 27.7.1 何时打断Agent

并非所有Agent 操作都需要人类审查。一项有原则的打断策略应考虑：

- 可逆性：不可逆操作（删除文件、发送邮件、进行采购）始终需要批准。可逆操作（读取文件、网页搜索）一般不需要。

- 影响范围：影响外部系统或他人的操作比纯本地操作更需谨慎审查。

- 置信度：当Agent 对其对用户意图的解读置信度较低时，应请求澄清而非径直执行。

- 成本：高成本操作（大规模API 调用、昂贵计算）需要批准。

- 新颖性：Agent在此上下文中此前未执行过的操作，比例行操作更需审查。

### 27.7.2 分层审批工作流

分层审批策略可在监督与效率之间取得平衡：

三层审批模型

Tier 1（自动批准）：低风险、可逆、例行操作。例如：网页搜索、读取文件、调用只读API。

Agent 无须打断即可执行；操作被记录以便审计。

Tier 2（通知）：中风险操作。UI 显示非阻塞通知（“Agent 已向您的草稿箱发送一封草稿邮件”），用户可异步审阅。在短窗口期（例如30 秒）内允许在操作最终生效前取消。

Tier 3（要求批准）：高风险、不可逆或高成本操作。Agent 暂停并呈现阻塞式审批门控。用户必须显式批准、拒绝或修改，Agent 才会继续。

各层之间的阈值可由用户配置（“发送邮件前总是询问”），也可从用户行为中学得（如果用户总是批准网页搜索，则将来自动批准它们）。

### 27.7.3 反馈机制

除了审批门控，Agent化UI 还应提供丰富的反馈机制，帮助Agent 随时间改进：

- 点赞/踩：对响应的简单二元反馈，被存储并用于基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）微调或偏好学习。

- 内联纠正：用户可直接编辑Agent 的输出；原始输出与纠正输出之间的差异即是一个训练信号。

- 偏好选择：当Agent 提供多个选项时，用户的选择即偏好信号。

- 显式指令：“下次别这么做”、“X之前总是询问”、“偏好方法Y 而非Z”——更新Agent 行为策略的自然语言指令。

- 带理由的评分：评分附带可选的自由文本解释，提供比二元反馈更丰富的信号。

### 27.7.4 通过UI 交互教会Agent

最成熟的HITL UI 将每次交互视为教学机会：

- 示范：用户手动执行任务；Agent观察并学习首选方法。

- 带泛化的纠正：当用户纠正某个Agent 操作时，UI 询问“我以后是否总是这样改？”，以将纠正泛化。

- 偏好引出：周期性提示用户对比两种Agent 行为并指出更偏好哪种。

- 行为配置文件：UI维护一份可见的“偏好” 档案，用户可审阅与编辑，使Agent 习得的行为透明可控。

## 27.8 无障碍与信任

信任不是一项功能——它是一个系统在持续按预期行为、清晰自我解释并能从失败中优雅恢复时所涌现出的属性。Agent 化UI 必须将信任作为一等关切来设计。

### 27.8.1 解释Agent 的决策

Agent 化UI 中的可解释性超越了展示思维链。它要求：

- 决策理由：对于重要决策，Agent不仅应解释决定了什么，还应解释为什么——考虑了哪些因素、拒绝了哪些备选、做出了哪些假设。

- 来源归属：论断应链接到其来源；检索到的文档应可引用。

- 反事实解释：“如果你说的是X 而不是Y，我会做Z”——帮助用户理解Agent 的决策边界。

- 不确定性量化：明确说明置信度，以及驱动不确定性的因素。

### 27.8.2 展示置信度水平

置信度指示器必须经过校准且有意义：

- 语言化置信度：对大多数用户而言，自然语言表达（“我相当有信心”、“这点我不太确定”）比数值概率更易解读。

- 视觉化置信度：颜色编码（绿/黄/红）、图标变体或字重可在不增加文字的情况下编码置信度。

- 按论断的置信度：对于含多项论断的响应，逐项的置信度指示（例如内联脚注）比单一的响应级评分更具信息量。

### 27.8.3 撤销与回滚能力

每一项重要的Agent 操作在技术可行时都应可撤销：

- 带撤销的操作日志：所有Agent 操作的时序日志，每个可逆操作旁附“撤销” 按钮。

- 基于快照的回滚：对于有状态任务（例如代码编辑、文档写作），定期快照支持回滚到任意先前状态。

- Dry-run 模式：在执行计划之前，Agent 可模拟运行并展示预测的状态变化，允许用户在任何真实操作发生前批准或修改。

- 优雅降级：当撤销不可行时（例如邮件已发出），UI应清楚说明，并提供最佳可行的替代方案（例如发送跟进邮件）。

### 27.8.4 UI 中的审计轨迹

对于企业与受监管的使用场景，审计轨迹（audit trails）必不可少：

- 不可变操作日志：每一次Agent 操作、工具调用与人类批准都附带时间戳、用户身份与完整参数被记录。

- 可导出的历史：审计轨迹可导出为JSON、CSV 或PDF，用于合规报告。

- Diff 视图：对于文档或代码修改，审计轨迹包含前后对比diff。

- 会话回放：可逐步回放整个Agent 会话，用于调试或合规审查。

### 27.8.5 管理用户期望

失准的期望是用户不信任的主要来源。Agent 化UI 应主动管理期望：

- 能力披露：清晰、易获取的文档，说明Agent 能做什么、不能做什么。

- 局限承认：当Agent 遇到超出其能力的任务时，应明确说明，而不是默默尝试并失败。

- 不确定性沟通：主动沟通不确定性，而非等待用户去发现错误。

- 一致的人格：一致的Agent 身份与沟通风格能建立熟悉感与可预测性。

![图 472-1](assets/ch27_p472_fig1.png)

![图 473-1](assets/ch27_p473_fig1.png)

## 27.9 实现示例：全栈Agent 化UI

现在给出一个具体的实现示例，将流式、工具可视化与审批门控组合于一个Python/React 技术栈中。Backend 使用FastAPI 配合LangGraph；Frontend 使用React，借鉴Vercel AI SDK 的模式并适配自定义Backend。

### 27.9.1 Backend：FastAPI + LangGraph 实现流式与审批门控

```
# backend/main.py
import asyncio
import json
from typing import AsyncGenerator
from fastapi import FastAPI , HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
app = FastAPI()
# -- 工具定义----------------------------------------------------------
# -- Tool definitions ----------------------------------------------------------
@tool
def web_search(query: str) -> str:
# 在网上搜索信息。
"""Search the web for information."""
return f"Search results for '{query}': [simulated results]"
@tool
def send_email(to: str, subject: str, body: str) -> str:
# 发送一封邮件。需要人工批准。
"""Send an email. REQUIRES HUMAN APPROVAL."""
return f"Email sent to {to} with subject '{subject}'"
@tool
def read_file(path: str) -> str:
# 从文件系统读取文件。
"""Read a file from the filesystem."""
try:
with open(path) as f:
return f.read()
except FileNotFoundError:
return f"Error: File not found: {path}"
# 需要批准的工具（Tier 3）
# Tools requiring approval (Tier 3)
APPROVAL_REQUIRED_TOOLS = {"send_email"}
# -- 审批门控存储（内存版；生产环境请使用Redis ）------------------
# -- Approval gate store (in-memory; use Redis in production) ------------------
approval_store: dict[str, asyncio.Event] = {}
approval_results: dict[str, dict] = {}
# -- LLM 设置-----------------------------------------------------------------
# -- LLM setup -----------------------------------------------------------------
```

```
llm = ChatOpenAI(model="gpt-4o", streaming=True)
tools = [web_search , send_email , read_file]
llm_with_tools = llm.bind_tools(tools)
def should_request_approval(tool_name: str) -> bool:
return tool_name in APPROVAL_REQUIRED_TOOLS
# -- 流式端点--------------------------------------------------------
# -- Streaming endpoint --------------------------------------------------------
async def agent_stream(
session_id: str,
user_message: str,
) -> AsyncGenerator[str, None]:
# 以SSE 形式流式发送Agent 事件。
"""Stream agent events as SSE."""
def sse(event_type: str, data: dict) -> str:
return f"data: {json.dumps({'type': event_type , **data})}\n\n"
yield sse("status", {"message": "Agent starting..."})
# 模拟多步Agent 执行
# Simulate multi-step agent execution
steps = [
("thinking", {"content": "Analyzing the request..."}),
("tool_call", {
"tool": "web_search",
"input": {"query": user_message},
"tier": 1, # 自动批准/ Auto-approve
}),
("tool_result", {
"tool": "web_search",
"output": f"Results for: {user_message}",
"duration_ms": 342,
}),
]
for event_type , data in steps:
await asyncio.sleep(0.5) # 模拟处理时间/ Simulate processing time
yield sse(event_type , data)
# 模拟一项需要批准的Tier 3 操作
# Simulate a Tier 3 action requiring approval
approval_id = f"{session_id}_email_001"
approval_event = asyncio.Event()
approval_store[approval_id] = approval_event
yield sse("approval_required", {
"approval_id": approval_id ,
"tool": "send_email",
"tier": 3,
"risk": "irreversible",
"action_summary": "Send summary email to user@example.com",
"parameters": {
"to": "user@example.com",
"subject": f"Research results: {user_message}",
"body": "Here are the findings...",
},
})
# 等待人工批准（5 分钟后超时）
# Wait for human approval (timeout after 5 minutes)
try:
await asyncio.wait_for(approval_event.wait(), timeout=300)
```

```
result = approval_results.get(approval_id , {})
if result.get("approved"):
yield sse("tool_call", {
"tool": "send_email",
"input": result.get("parameters", {}),
"tier": 3,
"approved_by": "human",
})
await asyncio.sleep(0.3)
yield sse("tool_result", {
"tool": "send_email",
"output": "Email sent successfully",
"duration_ms": 128,
})
else:
yield sse("action_rejected", {
"tool": "send_email",
"reason": result.get("reason", "User rejected"),
})
except asyncio.TimeoutError:
yield sse("approval_timeout", {
"approval_id": approval_id ,
"message": "Approval timed out; action skipped",
})
# 最终响应
# Final response
yield sse("token", {"content": "I've completed the research. "})
yield sse("token", {"content": "Here's a summary of what I found..."})
yield sse("done", {"total_tokens": 847, "duration_ms": 2341})
@app.get("/chat/stream")
async def chat_stream(session_id: str, message: str):
return StreamingResponse(
agent_stream(session_id , message),
media_type="text/event-stream",
headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
)
class ApprovalRequest(BaseModel):
approval_id: str
approved: bool
parameters: dict | None = None
reason: str | None = None
@app.post("/chat/approve")
async def handle_approval(req: ApprovalRequest):
if req.approval_id not in approval_store:
raise HTTPException(status_code=404, detail="Approval not found")
approval_results[req.approval_id] = {
"approved": req.approved ,
"parameters": req.parameters ,
"reason": req.reason,
}
approval_store[req.approval_id].set()
return {"status": "ok"}
```

#### Listing 27.4: 带流式与审批门控的FastAPI 后端

### 27.9.2 Frontend：React 实现流式与工具可视化

```
// frontend/AgentChat.tsx
import { useState , useEffect , useRef } from 'react';
```

```
// -- 类型定义---------------------------------------------------------------------
// -- Types ---------------------------------------------------------------------
type AgentEvent =
| { type: 'status'; message: string }
| { type: 'thinking'; content: string }
| { type: 'token'; content: string }
| { type: 'tool_call'; tool: string; input: object; tier: number }
| { type: 'tool_result'; tool: string; output: string; duration_ms: number }
| { type: 'approval_required'; approval_id: string; tool: string;

tier: number; risk: string; action_summary: string; parameters: object }
| { type: 'action_rejected'; tool: string; reason: string }
| { type: 'done'; total_tokens: number; duration_ms: number };

// -- 工具卡片组件-------------------------------------------------------
// -- Tool Card Component -------------------------------------------------------
function ToolCard({ event }: { event: AgentEvent & { type: 'tool_call' } }) {
const [expanded , setExpanded] = useState(false);
const tierColors = { 1: '#22c55e', 2: '#f59e0b', 3: '#ef4444' };
const color = tierColors[event.tier as keyof typeof tierColors] || '#6b7280';
return (
<div style={{ border: `1px solid ${color}`, borderRadius: 8, padding: 8,
margin: '4px 0', fontSize: 13 }}>
<div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
<span style={{ color, fontWeight: 600 }}>[gear] {event.tool}</span>
<span style={{ color: '#6b7280', fontSize: 11 }}>
Tier {event.tier} . {event.tier === 1 ? 'Auto' : 'Approved'}
</span>
<button onClick={() => setExpanded(!expanded)}
style={{ marginLeft: 'auto', fontSize: 11 }}>
{expanded ? 'Hide' : 'Details'}
</button>
</div>
{expanded && (
<pre style={{ marginTop: 8, fontSize: 11, background: '#f3f4f6',
padding: 8, borderRadius: 4, overflow: 'auto' }}>
{JSON.stringify(event.input, null, 2)}
</pre>
)}
</div>
);
}
// -- 审批门控组件---------------------------------------------------
// -- Approval Gate Component ---------------------------------------------------
function ApprovalGate({
event,
onDecision ,
}: {
event: AgentEvent & { type: 'approval_required' };
onDecision: (approved: boolean , params?: object) => void;
}) {
const riskColors = { reversible: '#22c55e', 'hard-to-undo': '#f59e0b',
irreversible: '#ef4444' };
const riskColor = riskColors[event.risk as keyof typeof riskColors] || '#6b7280';
return (
<div style={{ border: `2px solid ${riskColor}`, borderRadius: 8,
padding: 16, margin: '8px 0', background: '#fef9f0' }}>
<div style={{ fontWeight: 700, color: riskColor , marginBottom: 8 }}>
[!] Approval Required: {event.tool}
</div>
<div style={{ marginBottom: 8 }}>{event.action_summary}</div>
```

```
<div style={{ fontSize: 12, color: '#6b7280', marginBottom: 12 }}>
Risk level: <span style={{ color: riskColor }}>{event.risk}</span>
</div>
<div style={{ display: 'flex', gap: 8 }}>
<button
onClick={() => onDecision(true, event.parameters)}
style={{ background: '#22c55e', color: 'white', border: 'none',
borderRadius: 6, padding: '8px 16px', cursor: 'pointer' }}>
[OK] Approve
</button>
<button
onClick={() => onDecision(false)}
style={{ background: '#ef4444', color: 'white', border: 'none',
borderRadius: 6, padding: '8px 16px', cursor: 'pointer' }}>
[X] Reject
</button>
</div>
</div>
);
}
// -- 主聊天组件-------------------------------------------------------
// -- Main Chat Component -------------------------------------------------------
export function AgentChat() {
const [events, setEvents] = useState <AgentEvent[]>([]);
const [response , setResponse] = useState('');
const [isStreaming , setIsStreaming] = useState(false);
const [input, setInput] = useState('');
const sessionId = useRef(`session_${Date.now()}`);
const sendMessage = async () => {
if (!input.trim() || isStreaming) return;
setEvents([]);
setResponse('');
setIsStreaming(true);
const url = `/chat/stream?session_id=${sessionId.current}`
+ `&message=${encodeURIComponent(input)}`;
const es = new EventSource(url);
es.onmessage = (e) => {
const event: AgentEvent = JSON.parse(e.data);
if (event.type === 'token') {
setResponse(prev => prev + event.content);
} else if (event.type === 'done') {
setIsStreaming(false);
es.close();
} else {
setEvents(prev => [...prev, event]);
}
};
es.onerror = () => { setIsStreaming(false); es.close(); };
setInput('');
};
const handleApproval = async (
approvalId: string,
approved: boolean ,
parameters?: object ,
) => {
await fetch('/chat/approve', {
method: 'POST',
headers: { 'Content -Type': 'application/json' },
body: JSON.stringify({ approval_id: approvalId , approved , parameters }),
```

```
});
};
return (
<div style={{ maxWidth: 800, margin: '0 auto', padding: 16 }}>
<div style={{ minHeight: 400, border: '1px solid #e5e7eb',
borderRadius: 8, padding: 16, marginBottom: 16 }}>
{events.map((event, i) => {
if (event.type === 'tool_call')
return <ToolCard key={i} event={event} />;
if (event.type === 'approval_required')
return (
<ApprovalGate key={i} event={event}
onDecision={(approved , params) =>
handleApproval(event.approval_id , approved , params)} />
);
if (event.type === 'status' || event.type === 'thinking')
return (
<div key={i} style={{ color: '#6b7280', fontSize: 12,
fontStyle: 'italic', margin: '4px 0' }}>
{event.type === 'thinking' ? event.content : event.message}
</div>
);
return null;
})}
{response && (
<div style={{ marginTop: 8, lineHeight: 1.6 }}>
{response}
{isStreaming && <span className="cursor -blink">|</span >}
</div>
)}
</div>
<div style={{ display: 'flex', gap: 8 }}>
<input
value={input}
onChange={e => setInput(e.target.value)}
onKeyDown={e => e.key === 'Enter' && sendMessage()}
placeholder="Ask the agent..."
style={{ flex: 1, padding: '8px 12px', borderRadius: 6,
border: '1px solid #d1d5db', fontSize: 14 }}
/>
<button onClick={sendMessage} disabled={isStreaming}
style={{ padding: '8px 16px', background: '#3b82f6',
color: 'white', border: 'none', borderRadius: 6,
cursor: isStreaming ? 'not-allowed' : 'pointer' }}>
{isStreaming ? 'Running...' : 'Send'}
</button>
</div>
</div>
);
}
```

#### Listing 27.5: 带流式工具可视化与审批门控的React 前端

![图 478-2](assets/ch27_p478_fig1.png)

![图 479-1](assets/ch27_p479_fig1.png)

## 27.10 小结

Agent化UI 框架代表了人机交互的新前沿，要求我们从第一性原理出发重新思考界面设计。本节的关键洞见包括：

1. 范式选择至关重要：合适的UI 范式（聊天、画布、工作流、仪表盘、协作、自主）取决于任务结构、所需的人类介入与输出类型。大多数生产系统会组合多种范式。

2. 透明性不可妥协：用户无法信任他们看不见的东西。思考展示、工具可视化与上下文面板并非可选功能——它们是可信赖Agent 化系统的基础。

3. 流式是基线：用户期望实时看到Agent 在工作。Token 流、工具调用流与多Agent 流是入门级必备能力。

4. 审批门控必须分层：平坦的审批策略（全批准或全不批准）在实践中会失败。对安全操作自动批准、对危险操作门控的分层策略，能在维持监督的同时避免瓶颈。

5. 生成式UI 是前沿：LLM 不仅能生成文本，还能生成UI 组件——图表、表单、地图、小部件

- —这使得界面可以适配内容，而不必把内容硬塞进固定模板。

6. 信任靠一致性与可恢复性赢得：在建立用户信任方面，撤销能力、审计轨迹与已校准的置信度指示器与原始能力同等重要。

![图 479-2](assets/ch27_p479_fig2.png)

- —提供了构建模块。实践者面临的挑战是依据自身用户的具体需求与所在领域的具体风险，深思熟虑地将它们组合起来。
