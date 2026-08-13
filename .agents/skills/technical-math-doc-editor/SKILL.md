---
name: technical-math-doc-editor
description: >
  Use when reviewing, rewriting, or restructuring technical documents
  involving mathematical models, machine learning systems, performance
  modeling, memory modeling, throughput modeling, regression, statistics,
  or optimization. Improve mathematical rigor, clarity, terminology,
  derivations, assumptions, units, and engineering interpretation without
  changing source-of-truth formulas, parameters, experimental results,
  algorithms, or business rules.
---

# Technical Math Document Editor

## Purpose


将已有的机器学习、深度学习、系统优化、性能建模类技术文档，重构为：

* 数学上严谨
* 工程上准确
* 表述专业
* 结构清晰
* 对非数学专业的工程师仍然容易理解

目标不是把文档写得“更复杂”，而是让读者能够理解：

1. 要解决什么问题；
2. 为什么采用这个数学模型；
3. 每个公式从哪里来；
4. 每个变量代表什么；
5. 模型做了哪些假设；
6. 哪些部分是物理解析模型；
7. 哪些部分是数据拟合；
8. 最终公式如何对应真实工程系统。

---

# 1. Core Principle

始终遵循：

> 直觉 → 数学定义 → 推导 → 工程含义 → 数值例子

不要直接从概念跳到最终公式。

不要为了“专业”而增加不必要的术语。

专业性来自：

* 定义准确
* 推导完整
* 假设明确
* 量纲正确
* 结论有依据

而不是来自复杂措辞。

---

# 2. Preserve Source Truth

重写已有技术文档时：

## MUST

* 保留原有算法逻辑。
* 保留原有实验结果。
* 保留原有参数值。
* 保留原有模型版本。
* 保留已有公式真实含义。
* 保留业务决策逻辑。

## MUST NOT

不要为了让推导“看起来漂亮”而修改：

* 数学模型；
* 系数；
* 特征定义；
* 实验结果；
* 模型结构；
* 业务规则。

如果发现公式、文字解释和代码实现之间可能存在矛盾：

不要自行修正。

使用：

> **一致性检查：**
> 当前文档中的公式为 ……，但根据 …… 可能需要进一步确认实现是否完全一致。

如果代码仓库中存在对应实现，应优先检查代码实现。

---

# 3. Separate Four Kinds of Statements

任何技术解释都必须区分以下四类信息。

### ① Mathematical Fact

数学上可以严格推出的结论。

例如：

[
y = \log\frac{M_{\mathrm{obs}}}{M_{\mathrm{ref}}}
]

因此：

[
M_{\mathrm{obs}}
================

M_{\mathrm{ref}}e^y
]

这是严格数学关系。

---

### ② Modeling Assumption

人为建立模型时采用的假设。

例如：

> 假设解析模型未捕获的误差主要表现为乘性误差，因此在 log-ratio 空间进行回归。

必须明确写成“模型假设”，不能描述成数学定理。

---

### ③ Empirical Observation

数据中观察到的事实。

例如：

> 在当前训练集上，高 logits share 的样本整体呈现负 residual。

如果没有数据支持，不得写成 observation。

---

### ④ Mechanistic Interpretation

根据模型系数或实验结果提出的解释。

例如：

> 一种可能解释是 logits tensor 生命周期较短，因此解析模型按照完整 workspace 计入时产生系统性高估。

必须使用：

* “可能”
* “一种解释是”
* “这与……一致”

而不能仅凭回归系数宣称因果关系。

---

# 4. Standard Structure for Every Mathematical Model

介绍任何数学模型时，优先按照以下结构组织。

## 4.1 Problem Definition

首先明确：

* 输入是什么；
* 输出是什么；
* 优化目标是什么；
* 为什么需要这个模型。

例如：

> 给定训练配置 (x)，目标是预测单卡 CUDA reserved memory 峰值 (M(x))，并构造具有安全裕量的上界 (U(x))。

---

## 4.2 Modeling Idea

在公式之前先用自然语言解释模型设计。

推荐结构：

> 整个模型分成三层：
>
> 1. 物理模型负责给出具有正确量纲和趋势的 baseline；
> 2. 数据模型负责修正物理模型无法描述的系统误差；
> 3. 风险模型负责覆盖预测误差和 OOM 尾部风险。

让读者先理解架构，再进入数学。

---

# 5. Mathematical Notation Rules

每个符号第一次出现必须定义。

例如：

[
B = \text{micro batch size}
]

[
S = \text{sequence length}
]

[
H = \text{hidden dimension}
]

[
L = \text{number of Transformer layers}
]

如果是 tensor，尽可能给出 shape。

例如：

[
X\in\mathbb{R}^{B\times S\times H}
]

不要只写：

> H 是 hidden。

---

# 6. Always Explain Units

涉及显存、FLOPs、带宽、吞吐等物理量时必须注明量纲。

例如：

[
M_{\mathrm{param}}
==================

P_{\mathrm{load}}\times 2\ \text{bytes}
]

其中：

* (P_{\mathrm{load}})：参数个数；
* BF16 每个参数占 2 bytes；
* 因此结果单位为 bytes。

最终转换为：

[
1\ \mathrm{GiB}=2^{30}\ \mathrm{bytes}
]

不要只给数字结果。

---

# 7. Derive Formulas Instead of Dropping Formulas

例如介绍显存参数项时，不要只写：

[
M_{\mathrm{param}}=2P
]

应该解释：

模型包含 (P) 个参数。

BF16 中每个参数：

[
16\text{ bits}=2\text{ bytes}
]

因此：

[
M_{\mathrm{param}}
==================

P\times2
]

如果 ZeRO-3 将参数平均 shard 到 (N) 张 GPU：

[
M_{\mathrm{param,perGPU}}
\approx
\frac{2P}{N}
]

随后说明：

> 这里描述的是稳定状态下的参数 shard，不包含 ZeRO-3 在计算某个 module 时临时 all-gather 产生的 live parameter 峰值，因此后者需要单独建模。

---

# 8. Statistical Model Explanation

统计模型不能只展示最终闭式解。

必须按照：

> 建模目标 → loss function → regularization → matrix form → solution

解释。

例如 Weighted Ridge：

首先定义：

[
y_i
===

\log
\frac{M_{\mathrm{obs},i}}
{M_{\mathrm{ref},i}}
]

模型：

[
y_i
===

\beta_0+
\tilde z_i^\top\beta+
\epsilon_i
]

然后定义优化问题：

[
\min_{\beta_0,\beta}
\sum_{i=1}^{n}
w_i
\left(
y_i-\beta_0-\tilde z_i^\top\beta
\right)^2
+
\alpha|\beta|_2^2
]

解释：

* 第一项：拟合训练数据；
* (w_i)：样本权重；
* 第二项：L2 regularization；
* (\alpha)：正则强度；
* intercept (\beta_0) 不进行惩罚。

之后才进入矩阵形式：

[
\hat\theta
==========

(D^\top WD+\alpha R)^{-1}D^\top Wy
]

并解释：

[
\theta=
\begin{bmatrix}
\beta_0\
\beta
\end{bmatrix}
]

---

# 9. Explain Why Log Space Is Used

如果模型预测：

[
\log
\frac{M_{\mathrm{obs}}}{M_{\mathrm{ref}}}
]

必须回答：

> 为什么不用 (M_{\mathrm{obs}}-M_{\mathrm{ref}})？

推荐解释：

使用 ratio 意味着模型学习的是**相对误差而不是绝对误差**。

例如：

解析模型：

[
M_{\mathrm{ref}}=20\text{ GiB}
]

实际：

[
M_{\mathrm{obs}}=22\text{ GiB}
]

以及：

[
M_{\mathrm{ref}}=100\text{ GiB},
\quad
M_{\mathrm{obs}}=110\text{ GiB}
]

虽然绝对误差分别为 2 GiB 和 10 GiB，但本质上都是：

[
\frac{M_{\mathrm{obs}}}{M_{\mathrm{ref}}}=1.1
]

log transform 后：

[
\log(1.1)
]

两者具有相同 residual。

然后解释：

[
M_{\mathrm{center}}
===================

M_{\mathrm{ref}}
\exp(\hat y)
]

因此 Ridge 学到的是对物理模型的**乘性校正**。

---

# 10. Explain Feature Standardization

如果使用：

[
\tilde z_j
==========

\frac{z_j-\mu_j}{\sigma_j}
]

必须说明：

* (\mu_j)：训练集该特征均值；
* (\sigma_j)：训练集标准差；
* 标准化不会改变特征包含的信息；
* 它主要改善 Ridge 中不同量纲特征的可比性和数值稳定性。

不要直接使用“std z”而不解释来源。

---

# 11. Explain Feature Interactions

出现：

[
\text{LoRA}\times\text{GC}
]

或者：

[
\log_2(B)\times\log_2(S)
]

时必须解释：

线性模型默认假设每个变量的影响可以相互独立相加。

加入 interaction term 是为了允许：

> 一个变量的影响随着另一个变量改变。

例如：

[
\log_2(B)\log_2(S)
]

允许模型描述：

> 增大 MBS 对显存的影响可能随着 sequence length 增长而进一步增强。

---

# 12. Explain Physical Share Features

如果：

[
s_k
===

\frac{M_k}{M_{\mathrm{ref}}}
]

首先说明：

[
\sum_k s_k=1
]

然后解释设计动机：

两个配置即使：

[
M_{\mathrm{ref}}=40\text{ GiB}
]

完全相同，它们的组成仍可能不同。

Configuration A：

[
70%\ \text{parameters}
]

Configuration B：

[
70%\ \text{activations}
]

两种显存结构可能具有不同：

* 生命周期；
* allocator 行为；
* fragmentation；
* workspace 峰值。

因此 share features 描述的是：

> **显存组成结构，而不仅是显存总量。**

---

# 13. Explain Uncertainty Separately from Mean Prediction

中心预测和安全上界必须明确区分。

例如：

[
M_{\mathrm{center}}
]

回答：

> 最可能需要多少显存？

而：

[
M_{\mathrm{upper}}
]

回答：

> 为了控制 OOM 风险，我们应该按照多少显存进行容量规划？

不能把 P95 描述成普通 prediction。

---

# 14. Explain Conformal Quantile Carefully

第一次出现 conformal prediction 时，不要直接写：

> conformal Q95。

先解释目的：

即使中心模型平均误差很小，容量规划真正关心的是：

> 模型低估显存的概率有多大？

然后定义 residual：

[
e_i
===

\log
\frac{M_{\mathrm{obs},i}}
{M_{\mathrm{center},i}}
]

如果：

[
e_i>0
]

表示中心模型低估。

选择 residual 的 95% 上分位：

[
q_{0.95}
]

最终：

[
M_{\mathrm{upper}}
==================

M_{\mathrm{center}}
e^{q_{0.95}}
]

最后再介绍 finite-sample rank：

[
k=
\left\lceil
(n+1)(1-\alpha)
\right\rceil
]

避免统计术语先于直觉出现。

---

# 15. Explain Right Censoring

OOM 数据必须单独解释。

OOM 时我们并不知道：

[
M_{\mathrm{true}}
]

具体是多少。

只知道：

[
M_{\mathrm{true}}

>

M_{\mathrm{available}}
]

因此这是：

> right-censored observation。

不能把显存容量直接当作真实 peak memory。

应该把它作为下界约束或 safety guard，而不是普通 regression label。

---

# 16. Throughput Model Explanation

吞吐模型必须首先定义：

[
T
=

\frac{W_{\mathrm{effective}}}
{t_{\mathrm{step}}}
]

其中：

* (W_{\mathrm{effective}})：每 step 有效 token；
* (t_{\mathrm{step}})：step time；
* (T)：effective tokens/s。

之后才能解释 absolute head 为什么预测 step time。

---

# 17. Explain Pairwise Ranking Mathematically

不要只说：

> pairwise 更适合排序。

设同一场景：

[
\log T_i
========

c_s+f(x_i)+\epsilon_i
]

其中：

* (c_s)：场景共同偏置；
* (f(x_i))：配置本身造成的性能差异。

两个候选作差：

[
\log T_i-\log T_j
]

得到：

[
f(x_i)-f(x_j)
+
(\epsilon_i-\epsilon_j)
]

共同偏置：

[
c_s-c_s=0
]

因此 pairwise learning 天然弱化了：

* 模型架构差异；
* 软件版本差异；
* 数据集共同开销；

造成的 scenario-level scale shift。

这才是“排序比绝对值更容易迁移”的数学原因。

---

# 18. Case Study Format

真实 Case 统一按照以下模板。

## Configuration

清晰列出输入。

## Step 1 — Physical Components

逐项计算，并解释单位。

## Step 2 — Feature Construction

展示关键 feature，而不是机械列出所有 feature。

## Step 3 — Statistical Correction

展示：

[
z
\rightarrow
\tilde z
\rightarrow
\tilde z^\top\beta
\rightarrow
e^{\beta_0+\tilde z^\top\beta}
]

## Step 4 — Final Prediction

得到：

[
M_{\mathrm{center}}
]

## Step 5 — Safety Bound

得到：

[
M_{\mathrm{upper}}
]

## Step 6 — Engineering Interpretation

解释：

为什么会得到这个结果？

哪些解释是：

* 数学结论；
* 数据观察；
* 工程假设。

---

# 19. Writing Style

整体风格：

> ML systems paper / engineering whitepaper + 教材式解释

避免：

* “其实”
* “显然”
* “很简单”
* “无脑”
* “玄学”
* “一把梭”
* 大量感叹号
* 营销式语言

减少：

> 核心、关键、非常重要

等词语的重复使用。

优先使用：

> 该设计的目的在于……

> 从数学上看……

> 从工程实现上看……

> 需要注意的是……

> 这里隐含了一个假设……

> 更严格地说……

---

# 20. Terminology

第一次出现术语：

> 中文名称（English term, abbreviation）

例如：

> 岭回归（Ridge Regression）

> 留一场景交叉验证（Leave-One-Scenario-Out, LOSO）

> 折外预测（Out-of-Fold, OOF）

> 右删失（Right Censoring）

后续统一使用缩写。

不要同一个概念在：

* residual
* correction
* bias
* error

之间随意切换。

---

# 21. Layered Explanation

重要概念同时提供三层解释。

### Level 1 — Intuition

一句话解释。

### Level 2 — Mathematics

严格数学定义。

### Level 3 — Engineering Meaning

它在训练系统里对应什么。

例如 Ridge residual：

**直觉**

> 解析模型负责预测“大概多少”，Ridge 负责学习“通常需要乘多少”。

**数学**

[
y=
\log\frac{M_{\mathrm{obs}}}{M_{\mathrm{ref}}}
]

**工程**

> 如果预测 (y=0.1)，意味着实际 reserved memory 大约是物理解析值的 (e^{0.1}\approx1.105) 倍。

---

# 22. Document Hierarchy

推荐整篇技术文档采用：

1. 摘要
2. 问题定义
3. 系统总体架构
4. 数学符号与基本假设
5. 显存模型

   * 物理模型
   * 统计校准
   * 不确定性建模
   * OOM 数据处理
6. 吞吐模型

   * 物理代理
   * absolute regression
   * pairwise ranking
   * set-aware fusion
7. 端到端推荐算法
8. 数值案例
9. 实验设计与验证
10. 局限性
11. 结论
12. 附录

* 完整特征表
* 完整系数
* 代码路径
* 术语表

不要把完整 70 维 feature list 放在主体叙事中打断数学逻辑。

---

# 23. Separate Main Text and Appendix

主体负责：

> 为什么。

附录负责：

> 全部是什么。

例如 70 维特征：

主体只介绍：

* mechanism features
* model geometry
* workload
* physical proxies
* interactions

完整 70 维名称放 Appendix。

同样：

28 个 Ridge coefficient 不需要全部进入正文。

正文只展示：

* 数学结构；
* 最有解释力的几个 feature；
* Case 中主要贡献项。

---

# 24. Review Before Rewriting

面对已有文档时，先完成内部 review：

检查：

* 数学定义是否完整；
* 推导是否跳步；
* terminology 是否统一；
* dimensional analysis 是否正确；
* statistical claim 是否严谨；
* correlation 是否被误写为 causation；
* empirical result 是否被误写成理论结论；
* engineering assumption 是否明确；
* 公式是否与代码实现一致；
* 数字是否前后一致。

然后再重写。

不要简单进行语言润色。

---

# 25. Final Quality Bar

完成的文档应该让一个具有以下背景的工程师：

* 熟悉 Transformer；
* 知道 SFT；
* 知道 GPU 显存；
* 知道基本线性代数；
* 不熟悉统计建模；

在阅读之后能够自己解释：

1. 为什么显存模型需要 physical reference；
2. 为什么 residual 用 log ratio；
3. Ridge 到底优化什么；
4. share feature 为什么存在；
5. conformal bound 在解决什么问题；
6. OOM 为什么是 right-censored；
7. 为什么吞吐模型需要 absolute + rank 两个 head；
8. pairwise ranking 为什么能够抵消 scenario bias；
9. 最终推荐器为什么先做 memory admission 再做 throughput ranking。

如果读者只能记住公式，却无法回答上述问题，则文档仍然不合格。
