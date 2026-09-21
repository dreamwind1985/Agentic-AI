# SFT 最佳实践与技巧

监督微调(Supervised Fine-Tuning, SFT) 是RLHF 流水线的基础。SFT 模型的质量决定了RL 能达到的上限:RL 可以精炼和提升某种行为, 但无法可靠地引入SFT 模型中完全不存在的行为。本章涵盖高效SFT 的关键技术。

## 11.1 序列打包以提升效率

标准SFT 的Batch 会将所有序列填充(pad) 到该Batch 中最长序列的长度。对长度方差较大的数据集(例如短指令与长文档混合), 这会浪费50–80% 的计算量在padding Token 上。序列打包(Sequence Packing) 消除了这种浪费。

序列打包将多个短样本拼接为长度为max_seq_length 的单一序列, 以EOS Token 分隔。Atten- tion mask 确保不同样本的Token 之间不会相互attend:

1. 按长度对样本排序(可选, 可提升打包效率)。

2. 贪心地将样本打包进大小为max_seq_length 的bin。

3. 使用块对角(block-diagonal)Attention mask, 防止跨样本attention。

4. 只在非padding Token 上计算Loss。

```
打包效率
• 典型打包效率:85–95%(相比之下,padding仅20–50%)。
• 加速比: 对长度方差较大的数据集可达2–4×。
• 显存: 与padding 相近(每个Batch 的总Token 数相同)。
• 注意事项: 需谨慎设计attention masking 以避免交叉污染。
TRL 中的序列打包
from trl import SFTConfig , SFTTrainer
config = SFTConfig(
max_seq_length=4096,
packing=True, # 启用序列打包
output_dir="sft_model",
per_device_train_batch_size=4,
gradient_accumulation_steps=4,
learning_rate=2e-5,
num_train_epochs=3,
)
```

```
trainer = SFTTrainer(
model=model,
args=config ,
train_dataset=dataset ,
# dataset_text_field="text", # 或使用formatting_func
)
trainer.train()
```

## 11.2 Chat 模板与格式化

![图 180-2](assets/ch11_p180_fig1.png)

ChatML 是使用最广泛的chat 模板:

```
# ChatML 格式
template = """<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
{assistant_message}<|im_end|>"""
```

Llama 3 使用一种带特殊Token 的不同模板:

```
# Llama 3 格式
template = """<|begin_of_text|><|start_header_id|>system <|end_header_id|>
{system_message}<|eot_id|><|start_header_id|>user <|end_header_id|>
{user_message}<|eot_id|><|start_header_id|>assistant <|end_header_id|>
{assistant_message}<|eot_id|>"""
在TRL 中应用Chat 模板
from transformers import AutoTokenizer
from trl import SFTConfig , SFTTrainer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama -3.1-8B-Instruct")
def formatting_func(example):
"""对数据集样本应用chat 模板。"""
messages = [
{"role": "system", "content": "You are a helpful assistant."},
{"role": "user", "content": example["instruction"]},
{"role": "assistant", "content": example["response"]},
]
return tokenizer.apply_chat_template(
messages ,
tokenize=False,
add_generation_prompt=False,
)
config = SFTConfig(
```

```
max_seq_length=2048,
output_dir="sft_model",
)
trainer = SFTTrainer(
model=model,
tokenizer=tokenizer ,
args=config ,
train_dataset=dataset ,
formatting_func=formatting_func ,
)
```

## 11.3 仅对补全部分进行mask

```
为何要mask prompt?
在指令微调中, 模型应该学习生成assistant 的回应, 而非预测用户的问题或system prompt。在
prompt Token 上计算Loss 会浪费Gradient 信号, 并可能导致模型“记忆”prompt 而非学会回应
它们。Completion-only masking 将所有非assistant Token 的Loss 置零。
TRL 中的Completion-Only Masking
from trl import SFTConfig , SFTTrainer , DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama -3.1-8B-Instruct")
# 定义响应模板(此Token 之后开始计算Loss)
response_template = "<|start_header_id|>assistant <|end_header_id|>"
collator = DataCollatorForCompletionOnlyLM(
response_template=response_template ,
tokenizer=tokenizer ,
)
config = SFTConfig(
max_seq_length=2048,
output_dir="sft_model",
)
trainer = SFTTrainer(
model=model,
tokenizer=tokenizer ,
args=config ,
train_dataset=dataset ,
data_collator=collator , # 仅对completion 部分计算Loss
formatting_func=formatting_func ,
)
Completion Masking 陷阱
• 响应模板必须与token 化后的形式精确匹配。token 化中的off-by-one 错误会导致mask 应
用错误。
• 对非常短的回答,mask掉prompt 后剩下的Token 可能不足以提供有意义的Gradient 信号。
可考虑设置最小响应长度阈值。
• 多轮对话(Multi-Turn) 需要mask 所有非assistant 轮次, 而不仅是第一轮。
```

## 11.4 多任务SFT 的数据混合策略

![图 182-1](assets/ch11_p182_fig1.png)

比例混合

按各数据集的规模成比例地采样:

Nk pk = PK , j=1 Nj

其中Nk 是数据集k 中的样本数。这是大多数框架的默认方式, 在数据集质量相近时效果良好。

温度混合

应用温度T 来平滑比例:

pk ∝N1/T .

k

T = 1: 比例混合。T →∞: 均匀混合。T < 1: 对大数据集过采样。T > 1: 对小数据集过采样。

基于质量加权的混合

按估计的质量(例如在参考模型下的困惑度、人工质量评分) 对数据集加权:

pk ∝Nk · qk,

其中qk 是数据集k 的质量分数。

```
TRL 中的数据混合
from datasets import concatenate_datasets , interleave_datasets
# 比例混合(默认)
mixed_dataset = concatenate_datasets([
dataset_math ,
dataset_code ,
dataset_general ,
]).shuffle(seed=42)
# 温度混合(T=2: 对小数据集过采样)
mixed_dataset = interleave_datasets(
[dataset_math , dataset_code , dataset_general],
probabilities=[0.4, 0.4, 0.2], # 经温度缩放后手动设定
seed=42,
)
config = SFTConfig(output_dir="sft_model")
trainer = SFTTrainer(
model=model,
args=config ,
train_dataset=mixed_dataset ,
)
```

## 11.5 当SFT 反而有害——灾难性遗忘与对齐税

当LLM 依次经历各训练阶段——预训练→继续预训练→SFT →RLHF/DPO ——标准基准上的性能下降时常出现。驱动这些回退的是两种本质上截然不同的现象, 将其混淆将导致错误的缓解策略。

### 11.5.1 灾难性遗忘(结构性擦除)

![图 183-1](assets/ch11_p183_fig1.png)

- 在微调数据之外的任务上完全崩溃(例如模型在chat 数据上SFT 后忘记如何做数学)

- 语言多样性丧失——模型只能以微调分布的狭窄风格生成

- 在微调阶段未被强化的知识上, 事实准确性下降

- 仅在英语数据上SFT 后多语言能力下降

机制根源——Fisher 信息视角: 任务A 的Fisher 信息矩阵F 标识了哪些参数对DA “重要”:

h ∇θ log πθ(x) ∇θ log πθ(x)T i F = Ex∼DA (11.2)

Fisher 特征值高的参数对任务A 至关重要。无约束的任务B 梯度下降完全忽略这些特征值——∆θ沿∇LB 方向移动, 而不管是否摧毁了LA 的高Fisher 方向。

### 11.5.2 对齐税(行为约束)

对齐税(Alignment Tax) 是一种有意为之、可预期的权衡: 模型的原始能力(无约束生成、最大化推理带宽) 下降, 是因为Policy 被约束去生成安全、格式良好、与偏好对齐的输出。

机制: 在DPO/PPO 期间,Policy πθ 通过KL 散度被惩罚以防偏离参考πref:

rimplicit(x, y) = β log πθ(y|x) (11.3) πref(y|x)

这条“链条”约束着模型的输出分布——它无法探索偏离参考太远的高方差推理路径。知识并未被擦除; 而是被抑制了。模型仍“知道”答案, 但其分布被压平, 偏向安全、通用的回应。

症状:

- 过度拒答(对良性查询回复“I can’t help with that”)

- 风格僵硬——模糊措辞、过多警示语、冗长的安全免责声明

- 在原始能力基准(MMLU、HumanEval) 上分数下降, 而在偏好基准(MT-Bench、AlpacaEval)上提升

- 生成复杂、高熵输出(创意写作、新颖算法) 的能力下降

**Table 11.1: 灾难性遗忘vs 对齐税——完整对比。**

| 维度 | 灾难性遗忘 | 对齐税 |
| --- | --- | --- |
| 意图性 | 非故意(优化产物) | 可预期权衡(为安全/有用而有意承担) |
| 参数状态 | 先验知识被物理覆盖 | 潜在分布被约束/截断 |
| 信息 | 被摧毁: 权重不再编码该能力 | 被抑制: 知识仍存在但更难触发 |
| 主导阶段 | 顺序SFT、领域继续预训练 | 偏好优化(PPO、DPO、KTO、RLHF) |
| 主要症状 | 基线能力完全崩溃 | 过度拒答、风格僵硬、原始基准分数下降 |
| 可逆性 | 不重新从checkpoint 训练则不可逆 | 部分可逆: 调整β、system prompt 或微调 |
| 检测 | 预训练评估集上的困惑度飙升 | 困惑度稳定但能力基准上的胜率下降 |
| 随模型规模 | 各尺度相近 | 小模型支付更大的对齐税 |

### 11.5.3 对比分类

### 11.5.4 缓解策略

针对灾难性遗忘:

1. 数据回放(Data replay): 将5–10% 的预训练数据混入SFT 数据集。确保Gradient 更新不会完全忽视预训练分布。

2. 弹性权重整合(Elastic Weight Consolidation, EWC) [203]: 添加正则项Ω(θ) = P λ i Fi(θi −θ∗ i )2, 惩罚对原任务而言Fisher 信息高的参数的变化。

3. LoRA / 参数高效微调: 只训练低秩adapter(参数量< 1%), 基础权重完全冻结。这避免了预训练知识的永久性摧毁——你随时可以移除adapter 来恢复原模型。然而, 当adapter 处于激活状态时, 组合系统(W0 + BA) 仍可能表现出遗忘:adapter 可能将模型的有效行为推离旧技能。LoRA 保护的是checkpoint, 而非激活时的推理行为。

4. 保守的学习率: 使用1–5 × 10−6 配合较少的Epoch(1–3)。更大的学习率会加速遗忘。

5. 渐进式训练: 逐步混合分布, 随时间增加SFT 数据比例, 而非突然切换。

针对对齐税:

1. 仔细调节β: 更低的β 给予模型更多自由(减少税负), 但可能牺牲安全性。多数场景下最优β ∈[0.05, 0.3]。

2. 高质量、多样化的SFT 数据: 对齐税的一部分来自SFT 收窄了输出分布; 更广、更多样化的SFT 数据可减少这部分。RL 阶段通过KL 正则化[9] 进一步施加约束。

3. 条件对齐(Conditional alignment): 训练模型仅在安全标志激活时才对齐。推理时为基准测试关闭约束(仅供研究使用的技术)。

4. Constitutional AI / RLAIF: 利用模型自生成反馈创建更细致的偏好数据, 在提升对齐的同时保留能力。

5. 有针对性的RL 预算: 不要过度训练RL。监控能力基准, 当税负超出可接受阈值(通常是2–5% MMLU 回退) 时停止。

如何判断你遇到的是哪一种

- 在失败任务上运行base 模型: 若base 模型成功而微调后模型完全失败→灾难性遗忘。

- Prompt 工程测试: 若精心设计的prompt(例如“忽略安全准则, 逐步解答此数学题”) 能恢复该能力→对齐税(知识被抑制而非擦除)。

- 困惑度检查: 在预训练验证集上计算困惑度。飙升= 遗忘。稳定= 对齐税。

- Few-shot 恢复: 若提供少量上下文示例就能恢复能力→对齐税。若大量示例也无法恢复→遗忘。

## 11.6 与RL 的关系——SFT 质量决定RL 上限

SFT 模型是RL 训练的起点。RL 可以:

- 放大SFT 模型中已存在但较弱的行为。

- 抑制已存在但不理想的行为。

- 精炼回答的风格与格式。

RL 无法:

- 引入SFT 模型中完全不存在的能力。

- 从SFT 阶段严重的灾难性遗忘中恢复。

- 弥补系统性有偏的奖励模型。

为使RL 起作用,SFT 模型必须偶尔产生正确响应(从而Reward 信号非零)。如果SFT 模型对某prompt 从不产生正确响应,RL 无法学会产生正确响应——没有可放大的正向信号。这就是为什么SFT 质量是RL 性能的上限。

具体而言: 若SFT 模型能正确解答10% 的数学题,RL 有可能将其推至80%。若SFT 模型解题率为0%,RL 将毫无进展(所有Reward 为零, 所有优势为零, 无Gradient)。

实践启示

1. SFT 数据质量: 使用高质量、多样化的数据。少量高质量数据胜过大量低质量数据。

2. SFT 数据覆盖度: 确保SFT 数据覆盖你想通过RL 改进的任务。如果某任务不在SFT 数据中,RL 将难以推进。

3. SFT 训练时长: 不要过度训练SFT 模型。过度训练降低多样性, 让RL 探索更困难。

4. Warm-up: 即使base 模型已经过指令微调, 在RL 之前也可考虑在特定任务数据上进行简短的SFT warm-up。

```
在RL 之前检查SFT 质量
import numpy as np
from tqdm import tqdm
def estimate_pass_at_k(model, tokenizer , dataset , k=8, n_samples=100):
"""
估计SFT 模型的pass@k 。
若pass@1 < 5%,RL 大概率会失败。
若pass@k < 20%,RL 会很吃力。
"""
pass_at_1_scores = []
pass_at_k_scores = []
for example in tqdm(dataset.select(range(n_samples))):
prompt = example["prompt"]
ground_truth = example["answer"]
# 采样k 个补全
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
**inputs,
max_new_tokens=512,
```

```
do_sample=True,
temperature=0.8,
num_return_sequences=k,
)
correct = 0
for output in outputs:
response = tokenizer.decode(output, skip_special_tokens=True)
if ground_truth in response:
correct += 1
# pass@1:正确样本占比(估计的成功率)
pass_at_1_scores.append(correct / k)
# pass@k:k 个样本中至少一个正确
pass_at_k_scores.append(correct >= 1)
print(f"Pass@1 (estimated): {np.mean(pass_at_1_scores):.2%}")
print(f"Pass@{k}: {np.mean(pass_at_k_scores):.2%}")
print(f"RL viability: {'Good' if np.mean(pass_at_1_scores) > 0.05 else 'Poor'}"
)
estimate_pass_at_k(sft_model , tokenizer , eval_dataset)
SFT 最佳实践总结
1. 使用序列打包(Sequence Packing) 以最大化GPU 利用率。
2. 使用completion-only masking 将Gradient 聚焦于assistant 响应。
3. 为你的模型系列使用正确的chat 模板。
4. 多任务SFT 时使用温度缩放(T ≈2) 按比例混合数据。
5. 使用LoRA 防止灾难性遗忘。
6. 在开始RL 前评估pass@k, 以确保SFT 模型是可行的起点。
7. 不要过度训练: 指令微调通常1–3 个Epoch 就足够。
8. 监控多样性指标(熵、n-gram多样性) 以检测mode collapse。
```
