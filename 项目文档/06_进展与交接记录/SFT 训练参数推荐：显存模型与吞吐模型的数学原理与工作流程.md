# SFT 训练参数推荐：显存 V3 与结构化吞吐 V5 的数学原理和工作流程

> **2026-08-11 当前运行时更新：** 默认端到端入口已经切换为“统一有界显存
> V3 → 结构化吞吐 V5”。入口为 `h800_resource_predictor.py`，输出 schema 为
> `sft_h800_unified_v3_throughput_v5_prediction/v1`，发布模式为
> `active_recommendation`，但自动执行仍为 `false`。旧显存门禁只保留为显式
> 整链路 rollback。本文显存章节仍保留 2026-08-05 的 M1 双中心/双准入头 V5，
> 它是历史影子候选，不等同于当前显存 V3；吞吐 V5 章节则同时描述当前运行时
> 使用的结构化吞吐模型。
>
> 当前显存 V3 是另一条版本序列：一个共享 reserved 中心头加一个独立共享风险
> 头，不按 LoRA/FULL 路由拆成多个业务模型。其准入公式为
>
> $$
> C=M_{\mathrm{ref}}e^{f_c(z)},\qquad
> R=M_{\mathrm{ref}}e^{f_r(z)},\qquad
> U=\max(C,1.0170738699R),
> $$
>
> $$
> A_{\mathrm{memory}}=\mathbb I[U\le132.8392700195\ \mathrm{GiB}].
> $$
>
> $C$ 是中心预测，$R$ 是使用精确 success 与右删失 OOM 约束拟合的风险头，
> $U$ 是校准准入上界；$U$ 不是 P95。当前 531 行同集回放中，V3 的来源等权
> 中心 MAPE 为 8.77%，安全配置放行率为 381/394=96.70%，OOM 放行率为
> 0/106=0%。独立业务盲测当前只完成 35/60 个有效观测，对应 14.96%、
> 21/30=70% 和 0/1=0%，因此仍需继续监控泛化，不应把同集指标解释成全域保证。

> 本文说明 H800 SFT 参数推荐器的当前链路：**统一有界显存 V3**与**结构化吞吐
> V5**；同时保留显存 V5 双准入头的历史研究章节。显存 V5 公式、阈值、样本数
> 和验证结果以 2026-08-05 的版本化产物为准，不能代替当前 V3 接口。
>
> - 显存 V5 候选：`offline_experiments/diagnostics/h800_m1_lora_full_admission_v5_20260805/candidate_model_m1_lora_full_admission_v5.json`
> - 显存 V5 验证：`offline_experiments/diagnostics/h800_m1_lora_full_admission_v5_20260805/refit_and_all_unused_validation_report.json`
> - 吞吐 V5 产物：`offline_experiments/artifacts/structured_throughput_modeling.json`
>
> **状态边界：**显存 V5 仍是未发布的历史影子候选；结构化吞吐 V5 已通过
> `h800_unified_v3_throughput_v5_predictor.py` 接入当前统一入口。吞吐产物本身仍
> 保留 `analysis_only=true`、`publishable=false` 的冻结来源状态；本次显式授权
> 只启用推荐输出，不开放自动训练执行。

---

## 摘要

SFT 配置推荐需要回答两个不同问题：

1. **显存准入**：候选是否可以在单卡安全显存线内运行？
2. **吞吐排序**：在已经通过显存准入的候选中，哪个候选的有效 token 吞吐最高？

两者采用串联而不是混合决策：

| 组件 | 数学结构 | 主要输出 | 决策职责 |
|---|---|---|---|
| 当前显存 V3 | 解析显存参考 + 共享 reserved 中心 + 独立共享风险头 | reserved 中心、准入上界、准入结果 | 过滤可能 OOM 或超出支持域的候选 |
| 吞吐 V5 | 静态工作量 + 五分量正耗时主干 + 35 维有界校正 + 卡型 adapter + 联合目标 | step time、effective tokens/s | 在已放行候选中排序 |

最终业务策略为：

> 先找到仍有安全候选的最小 GPU 数，再在该 GPU 数内选择吞吐 V5 预测值最高的候选。

### 版本命名必须加组件前缀

本文中存在两个不同的版本序列：

- **显存 V5**：`S5_lora_full_separate_admission_heads`，即 LoRA/FULL 双准入头候选；
- **吞吐 V5**：五分量结构化 step-time 模型；
- **V4b**：历史吞吐 absolute/rank 双头模型，不是显存模型版本。

因此，后文不单独使用“V5”指代整个系统，而写成“显存 V5”或“吞吐 V5”。

---

## 1. 问题定义

### 1.1 候选配置

用 $x$ 表示一个候选配置。它至少包含：

- 模型结构和参数规模；
- Full 或 LoRA 训练方式；
- GPU 数 $N$、micro batch size（MBS）$B$、gradient accumulation（GA）和目标 global batch size（GBS）；
- ZeRO stage、梯度检查点（Gradient Checkpointing, GC）和 packing；
- attention、cross entropy、optimizer、compile 等 kernel 路径；
- 训练前可获得的静态数据画像。

同一用户场景的候选集合记为

$$
\mathcal C=\{x_1,x_2,\ldots,x_m\}.
$$

### 1.2 历史显存 V5 输出

本节至显存 V5 验证章节描述的是历史影子候选，不是当前 V3 运行时。显存 V5
不再用一个 $M_{\mathrm{center}}$ 同时代表所有显存口径，而是分别输出：

$$
C_a(x),\qquad C_r(x),\qquad U_{\mathrm{memory}}(x),
$$

$$
p_{\mathrm{unsafe}}(x),\qquad A(x)\in\{0,1\}.
$$

其中：

- $C_a$：单卡 CUDA peak allocated memory 的中心预测，单位为 bytes/GPU；
- $C_r$：单卡 CUDA peak reserved memory 的中心预测，单位为 bytes/GPU；
- $U_{\mathrm{memory}}$：用于容量报告的 reserved memory 运行安全上界，单位为 bytes/GPU；
- $p_{\mathrm{unsafe}}$：对应准入头预测的不安全风险，无量纲；
- $A(x)$：最终是否准入。

这里必须区分两个问题：

- $U_{\mathrm{memory}}$ 回答“保守容量规划应按多少显存准备”；
- $A(x)$ 回答“在当前已校准机制内是否允许运行”。

显存 V5 的目标路径不会再简单使用 $U_{\mathrm{memory}}\le L_{\mathrm{safe}}$ 作为唯一准入规则。

### 1.3 吞吐模型输出与业务目标

吞吐 V5 只处理 $A(x)=1$ 的候选，输出

$$
\widehat t_{\mathrm{step}}(x),
\qquad
\widehat T(x).
$$

设仍有准入候选的 GPU 数集合为

$$
\mathcal N_{\mathrm{admit}}
=
\left\{
N:\exists x\in\mathcal C, A(x)=1, \operatorname{gpu}(x)=N
\right\}.
$$

若该集合非空，先取

$$
N^*=\min\mathcal N_{\mathrm{admit}},
$$

再取

$$
x^*
=
\arg\max_{x\in\mathcal C}
\widehat T(x)
\quad
\text{s.t.}\quad
A(x)=1, \operatorname{gpu}(x)=N^*.
$$

若 $\mathcal N_{\mathrm{admit}}=\varnothing$，系统应返回 `no_admitted_candidate`，而不是强行推荐显存风险未知的候选。

---

## 2. 系统总体架构

```text
模型配置 + 静态数据画像 + 候选集合
                   │
                   ▼
         输入规范化与支持域路由
                   │
                   ▼
┌──────────────────────────────────────────────────┐
│ 显存 V5                                          │
│ 1. profile max → 有效序列长度 S_eff              │
│ 2. 九个解析分量 → M_ref                          │
│ 3. 两个 11 维 Ridge → C_a 与 C_r                 │
│ 4. reserved OOF Q95 + exact OOM guard → U_memory │
│ 5. LoRA/FULL 独立风险头 → A(x)                   │
└──────────────────────────────────────────────────┘
                   │ 仅保留 A(x)=1
                   ▼
┌──────────────────────────────────────────────────┐
│ 吞吐 V5                                          │
│ 6. 静态画像 → 每 step 工作量                     │
│ 7. 五个正耗时分量 → physical step time           │
│ 8. 35 维有界校正 + 卡型 adapter                  │
│ 9. 唯一 predicted effective tokens/s → 排序      │
└──────────────────────────────────────────────────┘
                   │
                   ▼
      最小准入 GPU 数内选择吞吐 rank=1
```

显存链路中的三个数值层和一个决策层各自回答不同问题：

| 层 | 问题 |
|---|---|
| 解析锚点 $M_{\mathrm{ref}}$ | 按显式物理代理，主要显存尺度是多少？ |
| 中心 $C_a,C_r$ | 在历史成功样本中，allocated/reserved 通常是多少？ |
| 容量上界 $U_{\mathrm{memory}}$ | 如何覆盖中心低估尾部与已知 OOM 下界？ |
| 准入 $A(x)$ | 在当前机制证据与风险阈值下，是否允许运行？ |

把这四层混成一个“P95 显存预测”会掩盖显存 V5 相比旧模型最重要的结构变化。

---

## 3. 数学符号、单位和建模边界

### 3.1 主要符号

| 符号 | 含义 | 单位或取值 |
|---|---|---|
| $P_{\mathrm{base}}$ | 基座模型参数个数 | elements |
| $P_{\mathrm{load}}$ | 实际加载参数个数；LoRA 时包含 adapter | elements |
| $P_{\mathrm{train}}$ | 可训练参数个数 | elements |
| $H$ | hidden dimension | elements/token |
| $I$ | MLP intermediate dimension | elements/token |
| $L$ | Transformer layer 数 | 无量纲 |
| $K$ | KV width，即 KV heads × head dimension | elements/token |
| $h$ | attention head 数 | 无量纲 |
| $V$ | vocabulary size | classes |
| $B$ | 单卡 MBS | samples/GPU/micro-step |
| $S_{\mathrm{cutoff}}$ | 配置 cutoff | tokens/sample |
| $S_{\mathrm{eff}}$ | 由静态 profile 重建的有效解析序列长度 | tokens/sample |
| $N$ | GPU 数 | cards |
| $M_{\mathrm{ref}}$ | 九个解析显存分量之和 | bytes/GPU |
| $C_a,C_r$ | allocated/reserved 中心预测 | bytes/GPU |
| $L_{\mathrm{safe}}$ | 单卡显存安全线 | bytes/GPU |
| $U_{\mathrm{memory}}$ | reserved 容量安全上界 | bytes/GPU |
| $p_{\mathrm{unsafe}}$ | 不安全风险头输出 | $[0,1]$ |
| $W_{\mathrm{eff}}$ | 每 step 有效 token 数 | tokens/step |
| $t_{\mathrm{step}}$ | step time | seconds/step |
| $T$ | 有效 token 吞吐 | tokens/second |

显存先以 bytes 计算，再按

$$
1\ \mathrm{GiB}=2^{30}\ \mathrm{bytes}
$$

转换为 GiB。参数个数是 elements，只有乘以 bytes/element 后才得到显存字节数。

### 3.2 四类陈述的证据边界

本文用以下口径区分事实与解释：

- **数学事实**：可以从公式严格推出，例如 $C_r=M_{\mathrm{ref}}e^{\hat y_r}$；
- **模型假设**：为构造代理而引入，例如解析层保守计入完整 logits workspace；
- **经验观察**：冻结验证数据上实际测得，例如严格未见 FULL 的安全配置放行率为 9/9；
- **机制解释**：对系数或误差方向的可能解释，不作为因果结论。

### 3.3 解析显存模型的主要假设

以下均是当前实现假设，不是任意训练栈都成立的定理：

- 参数和梯度按 BF16 计，每个 element 为 2 bytes；
- AdamW optimizer state 按 12 bytes/可训练参数计；
- logits workspace 按 FP32 计，每个 element 为 4 bytes；
- attention kernel 采用 `fa3_orig` 代理；
- fused cross entropy 的 chunk shape 未绑定，因此保守计入完整 logits；
- ZeRO bucket 使用 DeepSpeed 0.19.2 的保守默认上限；
- ZeRO 通信不假设 overlap；
- ZeRO-3 live parameter 使用历史 Qwen 结构代理，不是精确 tensor-liveness 仿真。

---

## 4. 显存 V5：双中心、单层容量尾部与双准入头

### 4.1 显存 V5 解决了什么问题

旧的 stacked 准入路径会把 allocated 中心误差尾部与 reserved/allocated 膨胀尾部连续相乘。两层尾部可能重复表达 allocator 风险，导致容量上界过度保守。显存 V5 将“容量报告”和“准入决策”拆开：

1. 解析锚点提供显存量纲和结构趋势；
2. 两个 Ridge 分别预测 allocated 与 reserved 中心；
3. reserved 中心只叠加一层 success 上尾和 exact-mechanism OOM guard；
4. LoRA 与 FULL 使用两个独立风险头做最终准入；
5. 证据不足的 FULL 机制 fail closed。

目标路径为

$$
x
\longrightarrow
S_{\mathrm{eff}}
\longrightarrow
M_{\mathrm{ref}}
\longrightarrow
(C_a,C_r)
\longrightarrow
U_{\mathrm{memory}}
\longrightarrow
p_{\mathrm{unsafe}}
\longrightarrow
A(x).
$$

### 4.2 有效序列长度：先修正物理锚点

M1 中心变体不直接假设所有数据都填满 cutoff。静态数据画像先给出按当前 cutoff 截断后的最大 token 数 $S_{\mathrm{profile,max}}$，再按 8 token 对齐：

$$
S_{\mathrm{eff}}
=
\operatorname{round\_up}
\left(
\min(S_{\mathrm{cutoff}},S_{\mathrm{profile,max}}),
8
\right).
$$

然后使用 $S_{\mathrm{eff}}$ 重新计算激活、attention 和 logits 等解析分量。

这里的设计顺序很重要：数据集中不存在的更长序列不应先进入物理锚点，再完全依赖统计残差把高估拉回。M1 的名称 `effective_sequence_no_ratio` 也表示：有效长度已经进入解析锚点，最终 11 维中心特征不再额外加入 `log_effective_fraction`。

### 4.3 九个解析显存分量

定义每层、每 token 的激活代理 elements 为

$$
A_{\mathrm{token,layer}}=6H+2K+3I.
$$

九个分量都以单卡 bytes 计算。

#### 4.3.1 参数、梯度和优化器状态

| 分量 | ZeRO-0 | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---:|---:|---:|---:|
| parameters | $2P_{\mathrm{load}}$ | $2P_{\mathrm{load}}$ | $2P_{\mathrm{load}}$ | $2P_{\mathrm{load}}/N$ |
| gradients | $2P_{\mathrm{train}}$ | $2P_{\mathrm{train}}$ | $2P_{\mathrm{train}}/N$ | $2P_{\mathrm{train}}/N$ |
| optimizer | $12P_{\mathrm{train}}$ | $12P_{\mathrm{train}}/N$ | $12P_{\mathrm{train}}/N$ | $12P_{\mathrm{train}}/N$ |

Full 训练中 $P_{\mathrm{train}}=P_{\mathrm{base}}$。LoRA `target=all` 时，当前几何代理为

$$
P_{\mathrm{adapter}}
=
rL(9H+2K+3I),
$$

并使用

$$
P_{\mathrm{load}}=P_{\mathrm{base}}+P_{\mathrm{adapter}},
\qquad
P_{\mathrm{train}}=P_{\mathrm{adapter}}.
$$

因此 LoRA 会明显缩小梯度和 optimizer state，但不会等比例缩小基座参数、激活或 logits。

#### 4.3.2 保存激活与重算工作区

GC 关闭时：

$$
M_{\mathrm{saved}}
=
BS_{\mathrm{eff}}L(6H+2K+3I)\times2.
$$

GC 开启时：

$$
M_{\mathrm{saved}}
=
BS_{\mathrm{eff}}LH\times2,
$$

$$
M_{\mathrm{recompute}}
=
BS_{\mathrm{eff}}(6H+2K+3I)\times2.
$$

重算项不乘 $L$，因为代码把它建模为当前 layer 的峰值 workspace。

#### 4.3.3 Attention、logits 与 ZeRO workspace

$$
M_{\mathrm{attention}}
=
BS_{\mathrm{eff}}(2H+2K)\times2
+4BhS_{\mathrm{eff}},
$$

$$
M_{\mathrm{logits}}
=
BS_{\mathrm{eff}}V\times4.
$$

令 reduce bucket 和 all-gather bucket 上限均为 $5\times10^8$ elements：

$$
M_{\mathrm{reduce}}
=2\min(P_{\mathrm{train}},5\times10^8),
$$

$$
M_{\mathrm{allgather}}
=2\min(P_{\mathrm{train}},5\times10^8).
$$

则

$$
M_{\mathrm{zero\_workspace}}
=
\begin{cases}
0, & \text{ZeRO-0},\\
M_{\mathrm{reduce}}, & \text{ZeRO-1},\\
\max(M_{\mathrm{reduce}},M_{\mathrm{allgather}}), & \text{ZeRO-2},\\
M_{\mathrm{reduce}}, & \text{ZeRO-3}.
\end{cases}
$$

#### 4.3.4 ZeRO-3 live parameters

令 $d_{\mathrm{head}}$ 为 head dimension：

$$
P_{\mathrm{max\ layer}}
=2H^2+2HK+3HI+2H,
$$

$$
P_{\mathrm{max\ module}}
=\max(P_{\mathrm{max\ layer}},VH),
$$

$$
P_{\mathrm{persistent}}
=
\min\left(
P_{\mathrm{load}},
L(2H+2d_{\mathrm{head}})+H
\right).
$$

进一步定义

$$
P_{\mathrm{retained}}=\min(P_{\mathrm{load}},2\times10^9),
$$

$$
P_{\mathrm{live}}
=
\min\left(
P_{\mathrm{load}},
P_{\mathrm{retained}}+P_{\mathrm{max\ module}}+P_{\mathrm{persistent}}
\right).
$$

当 $N>1$ 时：

$$
M_{\mathrm{stage3\_live}}
=2P_{\mathrm{live}}\left(1-\frac1N\right).
$$

#### 4.3.5 解析参考值与 activation share

九项依次为 parameters、gradients、optimizer、saved activations、recompute workspace、attention workspace、logits workspace、ZeRO collective workspace 和 stage-3 live parameters。记为 $M_1,\ldots,M_9$：

$$
M_{\mathrm{ref}}=\sum_{k=1}^{9}M_k.
$$

显存 V5 不再把九个 physical share 全部送入中心模型，而只保留结构激活占比：

$$
M_{\mathrm{activation}}
=
M_{\mathrm{saved}}
+M_{\mathrm{recompute}}
+M_{\mathrm{attention}},
$$

$$
s_{\mathrm{activation}}
=
\frac{M_{\mathrm{activation}}}{M_{\mathrm{ref}}}.
$$

它描述解析参考中由结构激活代理占据的比例，是无量纲特征。

### 4.4 中心层的 11 维特征

中心 Ridge 使用特征向量 $z\in\mathbb R^{11}$：

$$
z=
\begin{bmatrix}
\mathbb 1_{\mathrm{LoRA}},
\mathbb 1_{\mathrm{GC}},
\mathbb 1_{\mathrm{ZeRO2}},
\mathbb 1_{\mathrm{ZeRO3}},
\log_2N,
\log_2B,
\log_2(P_{\mathrm{base}}/P_0),
s_{\mathrm{activation}},
\mathbb 1_{\mathrm{LoRA}}s_{\mathrm{activation}},
\mathbb 1_{\mathrm{LoRA}}\log_2(P_{\mathrm{base}}/P_0),
s_{\mathrm{activation}}\log_2B
\end{bmatrix},
$$

其中

$$
P_0=4{,}022{,}468{,}096.
$$

三个 interaction term 允许 LoRA 的影响随显存组成或模型规模改变，也允许 MBS 的影响随 activation share 改变。线性模型若没有 interaction，只能假设这些影响彼此独立相加。

### 4.5 allocated/reserved 双中心 Ridge

#### 4.5.1 两个 log-ratio 标签

对成功样本 $i$，分别定义：

$$
y_i^{(a)}
=
\log\frac{M_{\mathrm{allocated},i}^{\mathrm{obs}}}
{M_{\mathrm{ref},i}},
$$

$$
y_i^{(r)}
=
\log\frac{M_{\mathrm{reserved},i}^{\mathrm{obs}}}
{M_{\mathrm{ref},i}}.
$$

使用 log-ratio 表示模型学习的是相对误差。20 GiB 到 22 GiB 与 100 GiB 到 110 GiB 虽然绝对误差不同，但比例同为 1.1，因而拥有相同的 $\log(1.1)$ 标签。

这里隐含一个**模型假设**：解析锚点未覆盖的主要误差更适合表示为乘性误差，而不是固定 GiB 的加性误差。

#### 4.5.2 按独立数据源平衡

设样本 $i$ 属于独立源 $s(i)$，该源有 $n_{s(i)}$ 个成功配置，则中心拟合权重为

$$
w_i=\frac{1}{n_{s(i)}}.
$$

因此每个 `split_unit_id` 的总权重为 1，候选更多的数据源不会仅因行数更多而主导拟合。

对每个特征使用同一组加权均值和标准差：

$$
\tilde z_{ij}
=
\frac{z_{ij}-\mu_j}{\sigma_j}.
$$

若 $\sigma_j<10^{-9}$，代码将 scale 置为 1。预测时必须复用冻结的 $\mu_j$ 与 $\sigma_j$。

#### 4.5.3 Ridge 优化与解

对 $t\in\{a,r\}$，模型为

$$
y_i^{(t)}
=
\beta_0^{(t)}
+\tilde z_i^\top\beta^{(t)}
+\epsilon_i^{(t)}.
$$

优化问题为

$$
\min_{\beta_0^{(t)},\beta^{(t)}}
\sum_i w_i
\left(
y_i^{(t)}-\beta_0^{(t)}-\tilde z_i^\top\beta^{(t)}
\right)^2
+\alpha_t\lVert\beta^{(t)}\rVert_2^2.
$$

截距不受惩罚。令 $D=[\mathbf 1,\tilde Z]$、$W=\operatorname{diag}(w_i)$、$R=\operatorname{diag}(0,1,\ldots,1)$，则代码使用 Moore–Penrose 伪逆：

$$
\hat\theta_t
=
\left(D^\top W D+\alpha_tR\right)^\dagger
D^\top Wy^{(t)}.
$$

$\alpha$ 从 $\{0.01,0.1,1,10,100\}$ 中按 leave-one-source-out 的 source-equal MAPE 选择。冻结候选中两个中心都选择

$$
\alpha_a=\alpha_r=0.01.
$$

两个中心都使用 230 个成功配置和 40 个独立源。OOM 不进入中心回归。

#### 4.5.4 从 log-ratio 还原中心

$$
C_a(x)
=
M_{\mathrm{ref}}(x)
\exp\left(
\hat\beta_0^{(a)}+\tilde z(x)^\top\hat\beta^{(a)}
\right),
$$

$$
C_r(x)
=
M_{\mathrm{ref}}(x)
\exp\left(
\hat\beta_0^{(r)}+\tilde z(x)^\top\hat\beta^{(r)}
\right).
$$

这两个公式是严格的代数还原关系。Ridge 的职责可以概括为：解析锚点预测“大约多少”，中心模型学习 allocated 和 reserved 分别“通常还需要乘多少”。

### 4.6 单层 reserved 容量上界

#### 4.6.1 success 折外 residual

对独立数据源做 leave-one-source-out。对被留出的成功样本定义 reserved 中心 residual：

$$
e_i^{(r)}
=
\log
\frac{M_{\mathrm{reserved},i}^{\mathrm{obs}}}
{C_{r,\mathrm{OOF},i}}.
$$

- $e_i^{(r)}>0$ 表示 reserved 中心低估；
- $e_i^{(r)}<0$ 表示 reserved 中心高估。

同一独立源在同一桶中可能有多个配置。代码先取该源最坏 residual，再对独立源做分位：

$$
e_s^{\max}=\max_{i:s(i)=s}e_i^{(r)}.
$$

这样可以避免同一数据源仅因候选更多而重复增加尾部样本数。

#### 4.6.2 有限样本单侧 95% 分位

将一个桶内 $n$ 个独立源最坏 residual 排序为

$$
e_{(1)}\le\cdots\le e_{(n)}.
$$

单侧 95% finite-sample rank 为

$$
k=\left\lceil(n+1)\times0.95\right\rceil.
$$

当 $k\le n$ 时：

$$
q_r=e_{(k)}.
$$

运行时按以下层级选择第一个正式可用的桶：

1. exact mechanism：`(training_mode, zero_stage, GC, gpu_count, packing)`；
2. `training_mode × GC`；
3. pooled。

这里的 exact mechanism 不包含模型、数据集、MBS、dtype 或 kernel，不能解释成完整运行环境完全相同。

#### 4.6.3 OOM 右删失 guard

OOM 时并不知道真实峰值，只知道它超过一个下界：

$$
M_{\mathrm{true},i}\ge d_i,
$$

$$
d_i
=
\max\left(
M_{\mathrm{observed\_lower},i},
L_{\mathrm{safe},i}+1\ \mathrm{byte}
\right).
$$

因此 OOM 是右删失（Right Censoring）观测，不能把卡容量伪造成精确 regression label。其折外 log 下界为

$$
g_i^{\mathrm{OOM}}
=
\log\frac{d_i}{C_{r,\mathrm{OOF},i}}.
$$

同一 exact mechanism 内取最严格值：

$$
g_{\mathrm{OOM}}
=
\max_i g_i^{\mathrm{OOM}}.
$$

OOM guard 不向 `mode × GC` 或 pooled 传播。

#### 4.6.4 最终容量上界

success 路径与 OOM 路径分别为

$$
U_{\mathrm{success}}
=
C_r\exp(\max(0,q_r)),
$$

$$
U_{\mathrm{OOM}}
=
C_r\exp(\max(0,g_{\mathrm{OOM}})).
$$

若 exact OOM guard 存在，则

$$
U_{\mathrm{memory}}
=
\max(U_{\mathrm{success}},U_{\mathrm{OOM}});
$$

否则只使用 $U_{\mathrm{success}}$，并输出 `no_exact_oom_guard` 审计项。

这个公式只保留一层 reserved 中心尾部，不再计算

$$
C_a
\times\text{allocated residual tail}
\times\text{reserved/allocated expansion tail}.
$$

因此，V5 目标路径中的容量上界不存在旧 stacked 规则的重复尾部乘积。

### 4.7 LoRA/FULL 双准入头

#### 4.7.1 不安全标签

准入头是二分类模型。标签定义为

$$
y_i^{\mathrm{unsafe}}
=
\begin{cases}
1, & \text{OOM},\\
1, & \text{success 且 }M_{\mathrm{reserved},i}^{\mathrm{obs}}>L_{\mathrm{safe},i},\\
0, & \text{success 且 }M_{\mathrm{reserved},i}^{\mathrm{obs}}\le L_{\mathrm{safe},i}.
\end{cases}
$$

OOM 在这里作为“不安全”分类约束出现，但仍没有被当成精确峰值。

#### 4.7.2 两个无量纲压力特征

每个准入头只使用：

$$
z_r=\log\frac{C_r}{L_{\mathrm{safe}}},
\qquad
z_a=\log\frac{C_a}{L_{\mathrm{safe}}}.
$$

代码中的向量顺序为 $[z_r,z_a]$。当 $z_r=0$ 时，reserved 中心正好等于安全线；当 $z_r<0$ 时，reserved 中心低于安全线。

#### 4.7.3 Logistic Ridge

对头 $h\in\{\mathrm{LoRA},\mathrm{FULL}\}$，分别使用该头自己的均值 $\mu_h$、scale $\sigma_h$、截距 $b_h$ 和系数 $\gamma_h$：

$$
\tilde z_h
=
\frac{[z_r,z_a]^\top-\mu_h}{\sigma_h},
$$

$$
\eta_h=b_h+\gamma_h^\top\tilde z_h,
$$

$$
p_{\mathrm{unsafe},h}
=
\sigma(\eta_h)
=
\frac{1}{1+e^{-\eta_h}}.
$$

拟合使用 source-balanced 权重，再使 safe/unsafe 两类的总权重各占一半。对应优化目标可写为

$$
\min_{b_h,\gamma_h}
-\sum_i w_i
\left[
y_i\log p_i+(1-y_i)\log(1-p_i)
\right]
+\frac{\alpha_{\mathrm{logit}}}{2}\lVert\gamma_h\rVert_2^2,
$$

其中

$$
\alpha_{\mathrm{logit}}=0.1,
$$

截距不受惩罚。实现通过 Newton/IRLS 迭代求解，并把线性项截断在 $[-30,30]$ 后计算 sigmoid，以避免指数溢出。

LoRA 与 FULL 只共享特征含义，不共享任何拟合参数或阈值。

#### 4.7.4 阈值与准入规则

冻结阈值为：

| 头 | 阈值 $\tau_h$ | 拟合行 | 独立源 | safe/unsafe |
|---|---:|---:|---:|---:|
| Critical LoRA | 0.454102 | 117 | 40 | 90/27 |
| 支持的 FULL | 0.596547 | 29 | 12 | 26/3 |

Critical LoRA 的分类阈值取 leave-one-source-out 分类器预测中 unsafe 风险的最小值。支持的 FULL 因 unsafe 证据更少，阈值取 classifier-fit calibration scores 中 unsafe 风险的最小值；它不是分类器级 OOF 阈值，因此证据强度弱于 LoRA。

Critical LoRA 的最终规则为

$$
A_{\mathrm{LoRA}}(x)=1
\iff
p_{\mathrm{unsafe,LoRA}}(x)<0.454102.
$$

支持的 FULL 额外加入不参与调参的中心物理 guard：

$$
A_{\mathrm{FULL}}(x)=1
\iff
p_{\mathrm{unsafe,FULL}}(x)<0.596547
\land C_a(x)\le L_{\mathrm{safe}}
\land C_r(x)\le L_{\mathrm{safe}}.
$$

### 4.8 支持域与路由

显存 V5 候选的路由不是“所有 H800 配置都使用同一个新头”：

| 路由 | 条件 | 行为 |
|---|---|---|
| Critical LoRA | LoRA、ZeRO-2、GC 关、2×H800、packing=false | 使用 LoRA 独立头 |
| 支持的 FULL | Full、ZeRO-3、GC 开、2×H800、packing=false | 使用 FULL 独立头与中心硬 guard |
| 其他 FULL | GPU 数、ZeRO、GC 或 packing 不满足上行 | fail closed，不沿用已知有问题的 stacked 准入 |
| 其他非 FULL 路由 | 当前主要是其他 LoRA 机制 | 保留当前 S0 旧路径，不属于 V5 双头目标域 |

因此，“显存 V5 泛化更好”只能在已验证目标路径内讨论，不能外推成任意 GPU 数、ZeRO、GC、packing 或训练模式都已获得 V5 证据。

### 4.9 为什么容量上界可以拒绝，但风险头仍可准入

旧规则把“上界是否低于安全线”同时当成容量规划和准入判据。显存 V5 将两者分开：

- $U_{\mathrm{memory}}$ 要覆盖 success residual 的高尾，因此天然偏保守；
- 风险头直接利用 OOM 与超安全线 success 的二分类证据，目标是在零已知 unsafe admission 约束下提高安全配置 recall；
- FULL 通过 $C_a,C_r\le L_{\mathrm{safe}}$ 保留明确的物理硬约束；
- LoRA 当前没有启用该中心硬 guard，以冻结代码为准。

所以，出现

$$
U_{\mathrm{memory}}>L_{\mathrm{safe}}
\quad\text{但}\quad
A(x)=1
$$

并不表示公式矛盾，而表示容量报告与业务准入采用了不同风险目标。第 7.1 节给出真实案例。

---

## 5. 吞吐 V5：结构化物理主干与联合目标

### 5.1 吞吐的数学定义

每 step 的有效 token 吞吐为

$$
T
=
\frac{W_{\mathrm{eff}}}{t_{\mathrm{step}}},
$$

其中 $W_{\mathrm{eff}}$ 的单位为 tokens/step，$t_{\mathrm{step}}$ 的单位为 seconds/step，因此 $T$ 的单位为 tokens/s。取对数后：

$$
\log T
=
\log W_{\mathrm{eff}}
-\log t_{\mathrm{step}}.
$$

吞吐 V5 先预测 step time，再除以运行前静态画像重建的有效工作量。绝对预测与排序分数使用同一个输出：

$$
\text{absolute prediction}
=
\text{ranking score}
=
\widehat T.
$$

### 5.2 运行前静态工作量

吞吐 V5 不把实测 step time、实测 token counter 或运行结果作为输入。静态 profile 对给定 dataset、cutoff、MBS 和 packing 估计：

- 每个物理序列的有效 token；
- 实际计算 token；
- attention token pair；
- 逻辑样本数。

对 `packing=false`：

$$
\mathrm{GA}
=
\frac{\mathrm{GBS}}{NB}
\in\mathbb N.
$$

每 step 物理序列数为

$$
NB\mathrm{GA}=\mathrm{GBS}.
$$

若静态 profile 给出每个物理序列的有效 token 期望 $\bar w_{\mathrm{eff}}$：

$$
W_{\mathrm{eff}}
=
\bar w_{\mathrm{eff}}\times\mathrm{GBS}.
$$

Packing 的工作量公式已实现，但冻结吞吐训练集没有 `packing=true` 证据，因此该路径不是已验证支持域。

### 5.3 五个正耗时物理分量

理想耗时代理为

$$
\tau_{\mathrm{compute}}
=
\frac{F_{\mathrm{total}}}{NC_{\mathrm{peak}}},
$$

$$
\tau_{\mathrm{hbm}}
=
\frac{B_{\mathrm{kernel}}}{B_{\mathrm{HBM}}},
\qquad
\tau_{\mathrm{optimizer}}
=
\frac{B_{\mathrm{optimizer}}}{B_{\mathrm{HBM}}},
$$

$$
\tau_{\mathrm{comm}}
=
\frac{B_{\mathrm{collective}}}{B_{\mathrm{link}}}
+N_{\mathrm{collective}}t_{\mathrm{latency}},
$$

$$
\tau_{\mathrm{launch}}
=
N_{\mathrm{launch}}t_{\mathrm{launch}}.
$$

$F_{\mathrm{total}}$ 为 FLOPs/step，吞吐率为 FLOPs/s；各 $B$ 为 bytes/step，带宽为 bytes/s。因此五个 $\tau$ 都以 seconds/step 计。

主要线性层 FLOPs 代理为

$$
F_{\mathrm{linear,full}}
=6P_{\mathrm{linear}}W_{\mathrm{comp}},
$$

$$
F_{\mathrm{linear,lora}}
=4P_{\mathrm{linear}}W_{\mathrm{comp}}
+6P_{\mathrm{adapter}}W_{\mathrm{comp}}.
$$

Attention 项为 $6LHA$，其中 $A$ 是 attention token pair 数；GC 开启时再加入前向重算 FLOPs 和 traffic。这些是物理代理，不是逐 kernel 精确计数。

### 5.4 可学习逆效率与物理主干

对每个分量和卡型 $c$：

$$
m_{k,c}
=
1+\operatorname{softplus}(a_k+u_{c,k}).
$$

因此 $m_{k,c}>1$，不会把理想下界学成负耗时或超越峰值效率。

Compute 与 kernel HBM 用 $p=4$ 的平滑 roofline 合并：

$$
\tau_{\mathrm{roof},c}
=
\left[
(m_{\mathrm{compute},c}\tau_{\mathrm{compute}})^4
+(m_{\mathrm{hbm},c}\tau_{\mathrm{hbm}})^4
\right]^{1/4}.
$$

物理主干为

$$
\tau_{\mathrm{physical},c}
=
m_{\mathrm{launch},c}\tau_{\mathrm{launch}}
+\tau_{\mathrm{roof},c}
+m_{\mathrm{optimizer},c}\tau_{\mathrm{optimizer}}
+m_{\mathrm{comm},c}\tau_{\mathrm{comm}}.
$$

冻结全量模型的共享逆效率倍数约为 launch 5.43、compute 2.85、kernel HBM 2.00、optimizer HBM 2.01、communication 2.08。它们是联合拟合参数，不能解释为单个 kernel 的独立实测利用率。

### 5.5 35 维有界校正与卡型 adapter

吞吐 V5 使用 35 维结构特征，分组为：

| 特征组 | 数量 |
|---|---:|
| 训练机制与 kernel | 11 |
| 资源几何 | 4 |
| 结构化工作量与数据分布 | 14 |
| 连续硬件比 | 2 |
| 机制交互 | 4 |
| **合计** | **35** |

特征变换不是普通 z-score，而是

$$
\tilde z_j
=
2\tanh\left(
\frac{z_j-\mu_j}{2\sigma_j}
\right),
$$

所以 $\tilde z_j\in(-2,2)$。它限制数值外推，但不会把 OOD 请求变成已验证请求。

共享校正与卡型 adapter 为

$$
r_c
=
b+\beta^\top\tilde z+\gamma_c^\top\tilde z_S,
$$

其中 $\tilde z_S$ 只取 8 个卡型敏感特征，$\gamma_c$ 使用更强 L2 正则。未知卡型使用 $u_{c,k}=0$ 和 $\gamma_c=0$，并把置信度降为 `low`。

### 5.6 最终 step time 与吞吐

对数校正被限制在 $(-1.25,1.25)$：

$$
\delta_c
=
1.25\tanh\left(\frac{r_c}{1.25}\right).
$$

于是

$$
\widehat t_{\mathrm{step},c}
=
\tau_{\mathrm{physical},c}e^{\delta_c},
$$

$$
\widehat T_c
=
\frac{W_{\mathrm{eff}}}{\widehat t_{\mathrm{step},c}}.
$$

结构化 residual 对物理主干的乘性修正在

$$
e^{-1.25}<e^{\delta_c}<e^{1.25},
$$

约 $[0.287,3.49]$。这是模型结构约束，不是预测区间。

### 5.7 绝对、成对与列表联合训练

令 $v_i=\log T_i$、$\hat v_i=\log\widehat T_i$：

$$
\mathcal L
=
\lambda_{\mathrm{abs}}\mathcal L_{\mathrm{Huber,abs}}
+\lambda_{\mathrm{mag}}\mathcal L_{\mathrm{Huber,pair}}
+\lambda_{\mathrm{order}}\mathcal L_{\mathrm{logistic,pair}}
+\lambda_{\mathrm{list}}\mathcal L_{\mathrm{KL,list}}
+\mathcal R.
$$

冻结权重为：

| 项 | 权重 |
|---|---:|
| absolute Huber | 1.00 |
| pairwise magnitude Huber | 0.50 |
| pairwise logistic order | 0.02 |
| listwise KL | 0.05 |
| 共享 residual L2 | 0.02 |
| 卡型 adapter L2 | 0.10 |
| 全局效率先验 | 0.02 |

Absolute Huber 阈值为 0.25，pairwise magnitude Huber 阈值为 0.20。实现使用带解析梯度的 L-BFGS-B，而不是吞吐 V4b 的 IRLS。

Pairwise loss 的迁移直觉来自场景偏置抵消。若

$$
\log T_i=c_s+f(x_i)+\epsilon_i,
$$

同场景两个候选作差：

$$
\log T_i-\log T_j
=
f(x_i)-f(x_j)+\epsilon_i-\epsilon_j,
$$

公共偏置 $c_s$ 消失。吞吐 V5 保留这个优点，但所有 loss 共同作用于同一个 $\widehat T$，不再产生独立 rank head。

### 5.8 参数规模与 Predictor 契约

两种卡型共同拟合时共有 67 个标量参数：5 个共享物理效率参数、10 个卡型分量偏移、16 个卡型 residual 系数、35 个共享系数和 1 个截距。冻结拟合使用 391 个成功候选、148 个场景和 1268 个同场景 pair，覆盖 H800 与 RTX 4090。

`ThroughputPredictor` 输出 `predicted_effective_tokens_per_second`、`predicted_step_seconds` 和 `confidence`。它同时返回：

- `requires_memory_safety_filter=true`；
- `memory_safety_checked=false`。

因此吞吐 Predictor 必须位于显存准入之后。

---

## 6. 端到端推荐流程

### 6.1 显存 V5 目标路径

对每个候选：

1. 校验模型清单、H800 容量、数据 profile 和配置字段；
2. 根据 `profile max` 与 cutoff 计算 $S_{\mathrm{eff}}$；
3. 使用 $S_{\mathrm{eff}}$ 重建九个解析分量和 $M_{\mathrm{ref}}$；
4. 构造 11 维特征并计算 $C_a$ 与 $C_r$；
5. 选择 reserved residual 层级和 exact OOM guard，计算 $U_{\mathrm{memory}}$；
6. 根据训练模式、ZeRO、GC、GPU 数和 packing 路由；
7. Critical LoRA 使用 LoRA 风险头；支持的 FULL 使用 FULL 风险头和中心硬 guard；
8. 其他 FULL fail closed；其他非 FULL 路由保留当前 S0 行为；
9. 输出中心、容量上界、风险、阈值、支持域和最终准入审计字段。

### 6.2 吞吐与业务选择

对 $A(x)=1$ 的候选：

1. `packing=false` 时校验

   $$
   \mathrm{GBS}\bmod(NB)=0,
   $$

   并计算

   $$
   \mathrm{GA}=\frac{\mathrm{GBS}}{NB}.
   $$

2. 从静态画像重建 effective/computed tokens 和 attention pairs；
3. 计算五个物理耗时分量；
4. 构造 35 维特征、共享校正和卡型 adapter；
5. 得到唯一的 `predicted_effective_tokens_per_second`；
6. 先选择最小准入 GPU 数，再在该卡数内取吞吐 rank=1；
7. 输出被选候选、跨卡吞吐诊断、每个拒绝原因、模型版本与审计哈希。

### 6.3 当前实现边界

显存 V5 候选尚未成为统一 Predictor 的默认显存实现。当前链路由外层显存 V3
先准入，再把 `admitted=true` 的候选交给吞吐 V5；吞吐 V5 自身不承担显存安全
判断。开放自动训练前仍需要：

- 继续对显存 V3 的支持域与 fail-closed 路由做业务回放；
- 确保吞吐 V5 始终只接收 `admitted=true` 的候选；
- 保留 memory/throughput 两套独立置信度和拒绝原因。

---

## 7. 数值案例

### 7.1 Case A：Critical LoRA 容量上界超线，但 V5 风险头准入

该案例来自未参与拟合的严格数据集 `fresh_s3_rare_tail_short_v1`：

| 配置 | 取值 |
|---|---|
| 模型 | Qwen3-8B |
| 训练方式 | LoRA |
| GPU | 2×H800 |
| ZeRO / GC / packing | ZeRO-2 / 关 / false |
| MBS | 2 |
| $S_{\mathrm{eff}}$ | 4096 |

安全线为

$$
L_{\mathrm{safe}}=132.8393\ \mathrm{GiB}.
$$

V5 中心与容量上界为：

| 量 | 结果 |
|---|---:|
| $C_a$ | 57.0363 GiB |
| $C_r$ | 84.8715 GiB |
| $q_r$ | 0.463001 |
| $U_{\mathrm{memory}}=C_r e^{q_r}$ | 134.8467 GiB |

因此容量上界略高于安全线：

$$
\frac{U_{\mathrm{memory}}}{L_{\mathrm{safe}}}
=1.0151.
$$

如果仍使用“上界小于安全线”的旧准入规则，该候选会被拒绝。显存 V5 改为计算压力特征：

$$
z_r=\log\frac{84.8715}{132.8393}=-0.448002,
$$

$$
z_a=\log\frac{57.0363}{132.8393}=-0.845452.
$$

LoRA 头标准化后：

$$
\tilde z_r=-0.123966,
\qquad
\tilde z_a=-0.186845.
$$

对应 logit 贡献为

$$
\eta
=
0.001743
+0.928921(-0.123966)
+0.945520(-0.186845)
=-0.290078.
$$

所以

$$
p_{\mathrm{unsafe}}
=\sigma(-0.290078)
=0.427985.
$$

由于

$$
0.427985<0.454102,
$$

最终 `admitted=true`。

该实验实际成功，reserved 峰值为 99.4277 GiB，低于安全线。这个 outcome 只用于事后验证，不是预测输入。该案例展示了显存 V5 的设计目的：保留保守容量上界用于报告，同时用独立风险头减少安全候选的过度拒绝。

### 7.2 Case B：支持的 FULL 被风险头与中心硬 guard 同时拒绝

该案例来自严格未见数据集 `fresh_s3_broad_nontruncated_longtail_v1`：

| 配置 | 取值 |
|---|---|
| 模型 | Qwen3-14B |
| 训练方式 | Full |
| GPU | 2×H800 |
| ZeRO / GC / packing | ZeRO-3 / 开 / false |
| MBS | 2 |
| $S_{\mathrm{eff}}$ | 12520 |

预测结果为：

| 量 | 结果 |
|---|---:|
| $C_a$ | 134.3123 GiB |
| $C_r$ | 141.8457 GiB |
| $U_{\mathrm{memory}}$ | 228.4109 GiB |
| $L_{\mathrm{safe}}$ | 132.8393 GiB |

压力特征为

$$
z_r=0.065600,
\qquad
z_a=0.011028.
$$

FULL 头标准化后：

$$
\tilde z_r=0.566655,
\qquad
\tilde z_a=0.527778.
$$

于是

$$
\eta
=
-0.239032
+0.737241(0.566655)
+0.724561(0.527778)
=0.561137,
$$

$$
p_{\mathrm{unsafe}}
=\sigma(0.561137)
=0.636716.
$$

因为

$$
0.636716>0.596547,
$$

风险头拒绝；同时 $C_a$ 和 $C_r$ 都超过安全线，中心硬 guard 也拒绝。最终 `admitted=false`。

该实验虽然训练成功，但 reserved 峰值为 133.1953 GiB，已经超过 132.8393 GiB 安全线，因此按显存准入口径属于 unsafe success。显存 V5 的拒绝方向与实测一致。

### 7.3 吞吐 V5 新数据集案例

2026-08-05 的时间留出场景 `lora_s2_src01_live_pk_script` 使用 2×H800、Qwen3-14B LoRA、ZeRO-2、GC 关、GBS=64、packing=false，只比较 $B\in\{1,2,4\}$：

| MBS | GA | V5 预测 tokens/s | 实测 tokens/s | APE |
|---:|---:|---:|---:|---:|
| 1 | 32 | 1482.97 | 1251.52 | 18.49% |
| 2 | 16 | 2691.37 | 2513.01 | 7.10% |
| 4 | 8 | **4586.57** | **4874.00** | 5.90% |

吞吐 V5 选择 MBS=4，oracle 也为 MBS=4：

$$
\mathrm{Top1Regret}
=
\max\left(
0,
1-\frac{4874.00}{4874.00}
\right)
=0.
$$

该静态 profile 未参与吞吐 V5 拟合，且部分长度特征超出冻结 min/max，所以 Predictor 仍将置信度标为 `low`。一次经验上选对不能把 OOD 请求追溯性改成域内预测。

### 7.4 端到端解释

Case A 说明显存 V5 可以在容量上界保守的同时放行校准风险较低的 Critical LoRA；Case B 说明 FULL 风险头与中心硬 guard 会共同阻止已经逼近或超过安全线的候选。只有通过显存路由与准入的候选才进入第 7.3 节所示的吞吐排序。

---

## 8. 校准、验证与指标口径

### 8.1 防泄漏原则

显存与吞吐需要按各自的独立性单元切分：

- 显存中心、尾部和准入头按 `split_unit_id` 整源留出，同一源的候选不能跨训练/验证两侧；
- 重复执行结果先按配置折叠，成功峰值取重复值中位数，不能把重复 run 当成独立证据；
- 显存 success 用于中心回归；OOM 只用于右删失 guard 和 unsafe 分类；
- 吞吐按完整场景留出，OOM 和失败候选不伪造吞吐标签；
- 数据集级未见与同数据集配置级未拟合必须分开报告。

### 8.2 显存 V5 拟合与验证数据

拟合集为：

| 数据 | 原始记录 | 去重配置 | 独立源 | 用途 |
|---|---:|---:|---:|---|
| 历史标定 + 第一阶段 | 263 | 213 | 20 | 拟合 |
| 第二阶段补充 | 60 | 60 | 20 | 拟合 |
| **合计** | **323** | **273** | **40** | 中心、尾部和双头 |

273 个配置中有 230 个 success 和 43 个 OOM。

验证数据为：

| 验证组 | 配置 | 计分源 | 是否参与拟合 | 证据含义 |
|---|---:|---:|---|---|
| 5 个严格未见数据集 | 31 | 5 | 否 | 数据集级独立诊断 |
| 历史未拟合配置 | 106 | 1 个内容连通源组 | 否 | 配置级留出 |
| **全部未拟合** | **137** | **6** | 否 | 两类证据合并统计 |

### 8.3 显存 V5 严格未见结果

| 范围 | 配置 | 安全配置放行 | allocated MAPE | reserved MAPE | 上界覆盖率 | unsafe success 错放 | OOM 错放 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 全部机制 | 31 | 25/25 | 3.98% | 15.43% | 100% | 0 | 0 |
| Critical LoRA | 16 | 12/12 | 4.53% | 25.12% | 100% | 0 | 0 |
| 支持的 FULL | 10 | 9/9 | 0.62% | 3.54% | 100% | 0 | 0 |

这些是**经验观察**，不是公式必然保证。Critical LoRA 的 16 个配置包含 15 个 success 和 1 个 OOM；15 个 success 中有 3 个 reserved 超过安全线。支持的 FULL 有 10 个 success，其中 1 个超过安全线。上述 unsafe/OOM 都未被放行。

### 8.4 全部未拟合结果

| 范围 | 配置 | 安全配置放行 | 放行率 | allocated MAPE | reserved MAPE | 上界覆盖率 | unsafe/OOM 错放 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 全部机制 | 137 | 62/96 | 64.58% | 6.61% | 16.71% | 100% | 0/0 |
| Critical LoRA | 20 | 13/14 | 92.86% | 5.06% | 21.37% | 100% | 0/0 |
| 所有 FULL | 58 | 13/37 | 35.14% | 2.78% | 5.64% | 100% | 0/0 |
| 支持的 FULL | 18 | 13/15 | 86.67% | 1.41% | 4.99% | 100% | 0/0 |

“所有 FULL”的 35.14% 不能解读为支持的 FULL 头只有 35.14% recall。58 个 FULL 中只有 18 个属于 V5 支持路径，其余 40 个因机制证据不足而 fail closed。

### 8.5 与旧准入规则的同集对比

以下比较来自完全相同的验证记录。这里的“旧模型”指显存旧准入规则，不是吞吐 V4b：

| 验证范围 | 旧规则放行率 | 显存 V5 放行率 | 变化 | V5 unsafe/OOM 错放 |
|---|---:|---:|---:|---:|
| 严格未见 Critical LoRA | 75.00% | 100% | +25.00 个百分点 | 0/0 |
| 严格未见支持的 FULL | 0% | 100% | +100.00 个百分点 | 0/0 |
| 全部未拟合 Critical LoRA | 71.43% | 92.86% | +21.43 个百分点 | 0/0 |
| 全部未拟合支持的 FULL | 0% | 86.67% | +86.67 个百分点 | 0/0 |
| 全部未拟合、所有 FULL | 0% | 35.14% | +35.14 个百分点 | 0/0 |

严格未见支持 FULL 的旧 allocated/reserved 中心误差已经只有 0.49%/1.64%，但 9 个安全配置一个都未放行；显存 V5 中心误差变为 0.62%/3.54%，安全配置却从 0/9 提升到 9/9。这是**经验观察**：主要改进来自准入结构，而不是中心回归误差继续下降。

### 8.6 拟合侧嵌套折外检查

每个外层 fold 都重新拟合中心、尾部和准入头：

| 路径 | 配置 | 安全配置放行 | 放行率 | 上界覆盖率 | unsafe success 错放 | OOM 错放 |
|---|---:|---:|---:|---:|---:|---:|
| Critical LoRA | 117 | 80/90 | 88.89% | 97.08% | 0 | 0 |
| 支持的 FULL | 29 | 21/26 | 80.77% | 95.83% | 0 | 0 |

FULL 拟合集只有 2 个独立源包含 unsafe 或 OOM，阈值稳定性证据弱于 LoRA。特别是 FULL 阈值没有使用分类器级 OOF calibration，这一限制不能被严格未见集上的 9/9 放行结果抹去。

### 8.7 吞吐 V5 验证

冻结产物的阻塞式留出为：

| 协议 | 场景等权 MAPE | 场景等权 pairwise | Top-1 regret |
|---|---:|---:|---:|
| H800 固定留出 | 14.98% | 90.39% | 1.75% |
| RTX 4090 完整场景留出 | 13.36% | 94.12% | 0.58% |
| RTX 4090 完整模型留出 | 28.86% | 94.12% | 0.58% |
| 完整数据集留出 | 19.49% | 85.40% | 1.24% |

2026-08-05 的 H800 新数据集时间留出回放为：

| 范围 | 数据集 | 权威 success | MAPE | P90 APE | Pairwise | Top-1 regret | Hit@10% |
|---|---:|---:|---:|---:|---:|---:|---:|
| 严格域内主结果 | 6 | 18 | 10.46% | 18.62% | 100% | 0% | 100% |
| 全部新数据集诊断 | 20 | 33 | 13.92% | 19.23% | 100% | 0% | 100% |

吞吐 V5 参数生成于 2026-07-28，新结果没有用于重拟合；但逐候选预测没有在 outcome 前预注册，因此属于参数冻结后的时间留出诊断，不是预注册 prospective acceptance。

### 8.8 发布证据边界

显存 V5 严格验证数据在本轮候选冻结前已经被查看过；吞吐新数据集结果也没有预注册逐候选预测。因此，两者都不能仅凭当前表格自动发布。

若在预先冻结的独立场景中观察到零推荐失败，并希望以约 95% 置信度支持“失败率不高于 5%”，需要约 59 个独立场景。该数值来自零失败二项分布上界，不表示当前已经完成 59 个 prospective 场景。

---

## 9. 模型演进与当前绑定

### 9.1 显存模型演进

| 阶段 | 设计 | 结论 |
|---|---|---|
| 旧 11 维资源回归 | 缺少可审计物理锚点 | 分布外失控，已退出安全主线 |
| physical-shares | 九分量解析参考 + 28 维中心 Ridge + success/OOM 上界 | 改善历史 holdout，但后续新数据暴露泛化与过度保守问题 |
| M1 + S0 stacked | 有效序列 11 维双中心 + allocated 尾部 × allocator expansion 尾部 | 安全但重复叠加导致 Critical LoRA/FULL 过度拒绝 |
| 显存 V4 | 保留直接 reserved 上界，为 Critical LoRA 增加独立准入头 | 修复 LoRA 重复叠加，未覆盖 FULL |
| **显存 V5** | 保留 V4 LoRA 头，新增独立 FULL 头；FULL 域外 fail closed | 本文主讲影子候选；严格未见目标路径 0 unsafe/OOM 错放，尚未发布 |

### 9.2 吞吐模型演进

| 版本 | 设计 | 结论 |
|---|---|---|
| V1 | 纯物理 Roofline | MAPE 223%、pairwise 43%，错排明显 |
| V4 | 全因素单头 | 历史 absolute MAPE 约 20.9% |
| V4b | 70 维 absolute/rank 双头 + set-aware 融合 | 历史排序较好，但两个输出可能不一致，集合依赖不利于迁移 |
| **吞吐 V5** | 静态画像 + 五分量主干 + 35 维有界校正 + 卡型 adapter + 联合目标 | 本文吞吐主线，同一输出同时服务绝对值与排序 |

### 9.3 当前绑定事实

- 当前稳定入口为 `h800_resource_predictor.py::H800ResourcePredictor`，默认
  `memory_gate=unified_v3`；
- 当前适配层为 `h800_unified_v3_throughput_v5_predictor.py`，显存绑定
  `h800_unified_bounded_memory_candidate_v3.json`；
- 显存 V3 使用一个共享中心和一个独立共享风险头，历史锚点覆盖固定关闭；
- 吞吐绑定 `structured_throughput_modeling.json` 的 V5，同一个
  `predicted_effective_tokens_per_second` 同时服务绝对预测和排序；
- V5 不把 `dataset_id` 作为拟合身份特征；`dataset_id` 只负责定位运行前静态画像，
  模型消费长度分布、有效 token、attention pair、模型结构和并行机制；
- 当前默认路径不加载 `joint_throughput_modeling.json`，也不加载它绑定的旧
  `h800_challenger_modeling.json`；
- `h800_physical_v4b_predictor.py` 保持不变，作为历史实现和回滚依据；显式传入
  `--memory-gate legacy_physical_v1` 才会回滚旧显存 + V4b 整条链路；
- `release.mode=active_recommendation` 只表示推荐门禁已经切换，
  `automatic_execution_allowed=false` 仍禁止自动启动训练；
- 显存 V5 公式仍是历史研究参考；吞吐 V5 公式和冻结系数是当前默认吞吐运行时。

---

## 10. 局限性与一致性检查

### 10.1 显存 V5 局限性

- 九分量锚点是物理代理，不是 tensor 生命周期仿真；完整 logits 和 ZeRO-3 live set 带有保守假设。
- $S_{\mathrm{eff}}$ 只使用 profile 最大长度，不能完整表达 P95/P99、长样本比例、动态 batch max 或 allocator 碎片。
- 冻结验证中存在 M1 输入完全相同、reserved 峰值却明显不同的配置碰撞；这说明当前 11 维信息不足以精确识别所有数据分布差异。
- reserved 中心在严格未见 Critical LoRA 上 MAPE 为 25.12%，显著高于 allocated 的 4.53%；容量上界与风险头不能替代对 reserved 生成机制的进一步建模。
- FULL 头只支持 2×H800、ZeRO-3、GC 开、packing=false；其他 FULL fail closed 不是“模型预测它们一定 OOM”，而是证据不足时的业务安全策略。
- FULL 阈值只有 3 个 unsafe 拟合行，且不是分类器级 OOF 阈值，需要前瞻数据继续校准。
- Logistic 系数描述条件相关性，不能解释为 allocated 或 reserved 对 OOM 的因果效应。

### 10.2 吞吐 V5 局限性

- 绝对误差依赖验证协议：H800 固定留出 MAPE 为 14.98%，完整数据集留出为 19.49%，未见模型时可能更高。
- 温度、功耗、同机干扰等运行状态没有进入特征，可能形成不可约噪声。
- 冻结证据主要是 dense Qwen3、GPU 数 1/2/4、packing=false 和有限参数规模；MoE、8 卡/跨机、packing 或未知卡型需要单独验证。
- 未知卡型退化为共享物理主干与零 adapter，只保证公式可执行，不保证误差已校准。

### 10.3 文档—实现一致性检查结果

本次重构按代码和产物统一了以下口径：

1. **版本身份**：当前链路是统一有界显存 V3 + 结构化吞吐 V5。M1 双准入头
   显存 V5 属于另一条历史候选序列，显存与吞吐版本号不能横向比较。
2. **有效序列**：M1 使用 `round_up(min(cutoff, profile_max), 8)` 重建物理锚点。
3. **中心数量**：历史显存 V5 有 allocated/reserved 两个 11 维中心；当前统一
   有界 V3 运行时只输出一个 reserved 中心，并另设独立共享风险头。
4. **Ridge 权重**：每个 `split_unit_id` 总权重为 1；两个中心的冻结 $\alpha$ 均为 0.01。
5. **安全尾部**：目标路径使用 reserved success 尾部与 exact OOM guard，不再乘 allocated tail 与 allocator expansion tail。
6. **尾部桶键**：exact mechanism 包含 GPU 数，顺序为 `(mode, zero, GC, gpu_count, packing)`。
7. **最终准入**：容量上界不是 LoRA/FULL 目标路径的唯一判据；两个头分别使用独立参数和阈值。
8. **FULL 物理 guard**：支持的 FULL 同时要求 $C_a,C_r\le L_{\mathrm{safe}}$；LoRA 当前代码不启用该 hard guard。
9. **域外策略**：其他 FULL fail closed；其他非 FULL 路由仍保留旧 S0，不能被写成全面迁移。
10. **吞吐输出**：吞吐 V5 的绝对值和排序使用同一个 `predicted_effective_tokens_per_second`，不存在 V4b 式独立 rank head。
11. **发布状态**：统一 V3 源 artifact 仍为 immutable、`publishable=false`；
    2026-08-11 的用户授权记录在版本化适配层，未篡改源 artifact。
12. **上界术语**：V3 的 $U=\max(C,1.0170738699R)$ 是校准准入上界，输出中的
    `operational_p95_reserved_bytes` 只是兼容别名，不能继续解释成 P95。

---

## 11. 结论

2026-08-11 的实际替换结果是：

- 显存侧由统一 V3 负责一个端到端门禁：共享中心估计 reserved memory，独立共享
  风险头吸收 success 尾部与右删失 OOM 约束，二者共同形成单一准入上界；
- 当前不再按“Critical LoRA / 支持的 FULL”选择不同准入头，也不允许历史锚点
  覆盖，因此业务只面对一个显存模型和一条准入规则；
- 吞吐侧使用结构化 V5：静态画像和物理耗时主干给出唯一
  `predicted_effective_tokens_per_second`，同一个输出同时用于绝对预测和排序；
- 旧显存模型与 artifact 原样保留，回滚必须显式指定，不会与 V3 隐式混用；
- 当前替换只授权推荐输出，未授权自动训练执行。

从当前回放证据看，V3 比旧门禁同时改善中心 MAPE、安全放行率和 OOM 放行率；
但独立业务盲测尚未完成，尤其 OOM 只有 1 个有效分母。工程上可以先使用 V3
继续迭代，统计上仍需持续积累独立业务 case，并同时检查最终推荐吞吐是否达到
可运行最优配置的 90%。

本文显存 V5 章节用于说明历史双中心、独立准入头方法，不应被当作当前显存接口；
吞吐 V5 章节描述当前吞吐模型。当前端到端接口以第 9.3 节和
`H800统一资源配置预测器使用说明_2026-07-30.md` 的 2026-08-11 更新为准。

---

## 附录 A：显存 V5 中心特征

按冻结顺序共 11 维：

1. `is_lora`
2. `gradient_checkpointing`
3. `zero2`
4. `zero3`
5. `log2_gpu_count`
6. `log2_mbs`
7. `log2_parameters_over_4b`，分母为 4,022,468,096，分子为 base parameters
8. `activation_share`
9. `is_lora_x_activation_share`
10. `is_lora_x_log2_parameters`
11. `activation_share_x_log2_mbs`

---

## 附录 B：显存 V5 冻结系数

### B.1 中心模型共同 mean 与 scale

| 特征 | mean | scale |
|---|---:|---:|
| `is_lora` | 0.860052 | 0.346933 |
| `gradient_checkpointing` | 0.146400 | 0.353507 |
| `zero2` | 0.829729 | 0.375870 |
| `zero3` | 0.140217 | 0.347212 |
| `log2_gpu_count` | 1.012694 | 0.269519 |
| `log2_mbs` | 0.859197 | 0.794234 |
| `log2_parameters_over_4b` | 1.006136 | 0.764877 |
| `activation_share` | 0.418121 | 0.254358 |
| `is_lora_x_activation_share` | 0.411988 | 0.262823 |
| `is_lora_x_log2_parameters` | 0.835505 | 0.770746 |
| `activation_share_x_log2_mbs` | 0.408825 | 0.488377 |

### B.2 中心 Ridge 截距与标准化系数

| 特征 | allocated | reserved |
|---|---:|---:|
| intercept | 0.024905 | 0.293864 |
| `is_lora` | -0.022917 | -0.099861 |
| `gradient_checkpointing` | -0.067039 | -0.058134 |
| `zero2` | 0.005983 | 0.077499 |
| `zero3` | 0.023345 | 0.071036 |
| `log2_gpu_count` | 0.005110 | 0.014923 |
| `log2_mbs` | 0.003944 | 0.051811 |
| `log2_parameters_over_4b` | 0.017496 | -0.046487 |
| `activation_share` | -0.293990 | -0.284415 |
| `is_lora_x_activation_share` | 0.294375 | 0.464130 |
| `is_lora_x_log2_parameters` | -0.043604 | 0.022895 |
| `activation_share_x_log2_mbs` | -0.010405 | -0.054745 |

两个中心的 $\alpha$ 都是 0.01。表中系数作用于标准化特征，不能直接与原始量纲特征相乘。

### B.3 双准入头参数

代码中的特征顺序均为 `[log_reserved_center_over_safe_limit, log_allocated_center_over_safe_limit]`。

| 参数 | Critical LoRA | 支持的 FULL |
|---|---:|---:|
| mean 1 | -0.343815 | -0.167697 |
| mean 2 | -0.709941 | -0.229731 |
| scale 1 | 0.840447 | 0.411709 |
| scale 2 | 0.725258 | 0.456175 |
| intercept | 0.001743 | -0.239032 |
| coefficient 1 | 0.928921 | 0.737241 |
| coefficient 2 | 0.945520 | 0.724561 |
| threshold | 0.454102 | 0.596547 |

---

## 附录 C：吞吐 V5 的 35 维结构特征

### C.1 训练机制与 kernel（11）

1. `is_lora`
2. `gradient_checkpointing`
3. `zero2`
4. `zero3`
5. `packing`
6. `kernel_fa2`
7. `kernel_fa3`
8. `kernel_liger`
9. `kernel_fused_ce`
10. `kernel_fused_optimizer`
11. `kernel_compile`

### C.2 资源几何（4）

1. `log2_gpu_count`
2. `log2_mbs`
3. `log2_gradient_accumulation`
4. `log2_cutoff_over_512`

### C.3 结构化工作量与数据分布（14）

1. `log_compute_to_hbm_ratio`
2. `log1p_comm_to_roof_ratio`
3. `log1p_optimizer_to_roof_ratio`
4. `log1p_launch_to_roof_ratio`
5. `effective_to_computed_token_ratio`
6. `attention_flop_share`
7. `recompute_flop_share`
8. `log1p_model_state_to_hbm_capacity`
9. `mean_length_to_cutoff`
10. `length_cv`
11. `p90_to_mean_length`
12. `p99_to_mean_length`
13. `label_token_ratio`
14. `mean_turns_over_10`

### C.4 连续硬件比（2）

1. `log1p_machine_balance_flops_per_hbm_byte`
2. `log1p_link_to_hbm_bandwidth_ratio`

### C.5 机制交互（4）

1. `lora_x_gc`
2. `gc_x_zero3`
3. `zero3_x_log2_gpu_count`
4. `log2_mbs_x_length_cv`

---

## 附录 D：术语表

| 术语 | 定义 |
|---|---|
| effective sequence | `round_up(min(cutoff, profile_max), 8)` 得到的解析长度 |
| analytic reference | 九个显存解析分量之和 $M_{\mathrm{ref}}$ |
| allocated center | allocated log-ratio Ridge 还原得到的 $C_a$ |
| reserved center | reserved log-ratio Ridge 还原得到的 $C_r$ |
| capacity upper | reserved success 上尾与 exact OOM guard 的最大值 $U_{\mathrm{memory}}$ |
| right censoring | OOM 只给出真实需求下界，不给出精确 peak |
| admission head | 用 $[\log(C_r/L),\log(C_a/L)]$ 预测 unsafe 风险的 Logistic Ridge |
| fail closed | 证据不足时拒绝，而不是输出未经验证的安全结论 |
| source-balanced | 每个独立 `split_unit_id` 的总拟合权重相同 |
| OOF | Out-of-Fold，折外预测 |
| throughput V4b | 历史 absolute/rank 双头与 set-aware 融合模型 |
| throughput V5 | 五分量物理主干、35 维有界校正、卡型 adapter 与联合目标 |
| effective tokens | 排除 padding 后，每 step 真正有效的 token 数 |
| Top-1 regret | $\max(0,1-T_{\mathrm{selected}}/T_{\mathrm{best}})$ |
| shadow candidate | 只用于离线评估或影子预测，尚未发布 |

---

## 附录 E：实现与产物索引

### E.0 当前运行时：统一显存 V3 + 结构化吞吐 V5

- 稳定入口：`offline_experiments/scripts/h800_resource_predictor.py`
- V3 + V5 适配层：`offline_experiments/scripts/h800_unified_v3_throughput_v5_predictor.py`
- 统一有界 V3 推理公式：`offline_experiments/scripts/h800_unified_bounded_memory_model.py`
- V3 冻结 artifact：`offline_experiments/artifacts/h800_unified_bounded_memory_candidate_v3.json`
- V3 特征构造：`offline_experiments/scripts/fit_h800_unified_resource_partial_v1.py`
- V3 531 行回放：`offline_experiments/diagnostics/h800_unified_bounded_memory_shadow_v3_20260810/report.json`
- 独立业务盲测：`offline_experiments/artifacts/h800_final_memory_business_blind_results_v2.json`
- 吞吐 V5：`offline_experiments/artifacts/structured_throughput_modeling.json`
- 吞吐 V5 推理：`offline_experiments/scripts/throughput_predictor.py`
- 当前入口测试：`offline_experiments/tests/test_h800_unified_v3_throughput_v5_predictor.py`

### E.1 显存 V5

- 九分量解析参考：`offline_experiments/scripts/h800_theory_basis.py`
- 有效序列、11 维特征、双中心 Ridge 与尾部：`offline_experiments/scripts/fit_h800_lora_source_disjoint_recalibration_v1.py`
- 单层 reserved 上界：`offline_experiments/scripts/calibrate_h800_m1_safety_upper_v2.py`
- Critical LoRA 独立准入头：`offline_experiments/scripts/fit_h800_m1_separate_admission_v4.py`
- FULL 扩展与 V5 路由：`offline_experiments/scripts/fit_h800_m1_full_admission_v5.py`
- 重拟合入口：`offline_experiments/scripts/refit_h800_m1_all_unused_validation_v3.py --full-admission-head-v5`
- V5 影子候选：`offline_experiments/diagnostics/h800_m1_lora_full_admission_v5_20260805/candidate_model_m1_lora_full_admission_v5.json`
- V5 验证报告：`offline_experiments/diagnostics/h800_m1_lora_full_admission_v5_20260805/refit_and_all_unused_validation_report.json`
- 未拟合逐条预测：`offline_experiments/diagnostics/h800_m1_lora_full_admission_v5_20260805/all_unused_validation_predictions.jsonl`
- V5 单元测试：`offline_experiments/tests/test_h800_m1_full_admission_v5.py`

### E.2 吞吐 V5

> 本节列出当前 `H800ResourcePredictor` 已绑定的结构化吞吐 V5 来源。

- 模型训练与公式：`offline_experiments/scripts/structured_throughput_modeling.py`
- 冻结产物：`offline_experiments/artifacts/structured_throughput_modeling.json`
- Python API 与 CLI：`offline_experiments/scripts/throughput_predictor.py`
- 自动化测试：`offline_experiments/tests/test_throughput_predictor.py`
- 2026-08-05 新数据集验证：`offline_experiments/artifacts/v5_dataset_generalization_h800_lora_stage2_20260805.json`

### E.3 方法与验证文档

- `项目文档/02_显存模型/显存模型LoRA与FULL双准入头V5验证报告_2026-08-05.md`
- `项目文档/02_显存模型/LoRA显存准入模型去叠加改造与验证_2026-08-05.md`
- `项目文档/02_显存模型/LoRA显存模型40源重拟合与全量未拟合数据验证_2026-08-05.md`
- `项目文档/03_吞吐模型/结构化吞吐模型重建说明_2026-07-28.md`
- `项目文档/03_吞吐模型/统一吞吐预测器使用说明_2026-07-28.md`
- `项目文档/03_吞吐模型/吞吐模型各版本泛化对比_2026-07-28.md`

---

*原文生成于 2026-08-04；2026-08-05 按 `technical-math-doc-editor` 规范重构为
LoRA/FULL 双准入头显存 V5；2026-08-11 补充当前统一显存 V3 + 结构化吞吐 V5 运行时绑定。*
