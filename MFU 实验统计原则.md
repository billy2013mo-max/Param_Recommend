# MFU 实验统计原则

## 1. 目的与正式指标

本文档固定本项目的 MFU 统计口径。实验报告只使用两类正式指标：

1. **Per-step MFU**：衡量每个完整训练 optimizer step 的模型计算效率，并对稳态 step 做统一汇总。
2. **E2E MFU**：衡量从训练命令启动到训练任务完全结束的端到端效率。

两类指标必须使用同一套模型 FLOPs 定义、GPU 理论峰值和单位。MFU 与包含冗余重计算的 HFU/Profiler FLOPS 不得混用。

`Useful-token MFU` 不作为第三个正式指标。non-padding token、padding fraction 和各 rank 序列长度只用于数据效率与多卡负载均衡诊断，不能替代模型架构与真实执行 shape 对应的 MFU 分子。

## 2. 统一符号

<table><tr><td>符号</td><td>含义</td></tr><tr><td>$$
(F_{\text{train},i})

$$</td><td>第 (i) 个 optimizer step 的有效模型 FLOPs</td></tr><tr><td>$$
(F_{\text{eval},j})

$$</td><td>第 (j) 个 eval batch 的有效 forward FLOPs</td></tr><tr><td>$$
(T_i)

$$</td><td>第 (i) 个训练 optimizer step 的 wall-clock 时间</td></tr><tr><td>$(T_{\text{start}})$</td><td>外层 launcher 启动训练命令的时间</td></tr><tr><td>$(T_{\text{end}})$</td><td>所有训练进程退出且最终 checkpoint 完全持久化的时间</td></tr><tr><td>$$
(N_{\text{GPU}})

$$</td><td>实验占用的 GPU 数量</td></tr><tr><td>$(P_{\text{peak}})$</td><td>单张 GPU 在实验精度和计算模式下的理论 dense 峰值 FLOP/s</td></tr></table>

固定卡数时，总理论算力为：

$P_{\text{capacity}}=N_{\text{GPU}}P_{\text{peak}}$

当前 H800 BF16/FP16 dense Tensor Core 实验默认使用：

$P_{\text{peak}}=989.4\times 10^{12}\ \text{FLOP/s/GPU}$

不得在没有启用结构化稀疏时使用 sparse 峰值。不同 GPU SKU、精度或执行模式必须重新填写对应峰值，不能沿用 H800 的数值。

## 3. MFU 分子的统一定义

### 3.1 训练 step

每个训练 step 的分子是完成一次逻辑模型训练所需的有效模型 FLOPs：



$$
F_{\text{train},i}
=
\sum_{m\in\text{microbatches of step }i}
\left(F_{\text{forward},m}+F_{\text{backward},m}\right)
$$

计算要求：

- 根据模型架构、当步真实输入 shape、序列布局和算子 FLOPs 公式计算。
- gradient accumulation 中的全部 microbatch 都必须计入同一个 optimizer step。
- Full SFT 可以在适用算子上使用“backward 约为 forward 的两倍”，但 LoRA/冻结参数场景必须根据 `requires\_grad` 和实际需要的 input-gradient、weight-gradient分别计算，不能无条件使用 `3 × forward`。
- 多模态、GQA、SwiGLU、MoE、变长 attention 等结构必须有对应公式；不支持的算子必须显式记录，不能静默按 0 处理。
- `label\_tokens` 不能单独作为分子，因为未参与 loss 的输入位置仍可能参与 forward 和 hidden-state backward。

以下工作不进入训练 MFU 分子：

- activation recomputation/gradient checkpointing 产生的额外 forward；
- 通信、optimizer、scheduler、日志、数据加载；
- checkpoint 序列化和存储；
- 编译、allocator、Python 和框架调度开销。

这些工作造成的耗时仍进入相应时间分母，因此减少重计算、通信等待或存储阻塞会提高 MFU。

### 3.2 LoRA 训练分子的特殊性

LoRA 与 Full SFT 必须使用各自训练任务实际需要的逻辑 FLOPs，不能统一套用

`backward = 2 × forward` 或 `training FLOPs = 3 × forward`。

以一个不含 bias 的 Linear 为例：

$Y=XW$

其中：

- $X$ 的 shape 为 $[M,K]$； 
- $W$ 的 shape 为 $[K,N]$；
- $M$ 是参与该 Linear 的 token/row 数；
- $K$、$N$ 分别是输入和输出维度。

Linear forward 的 FLOPs 为：

$F_{\text{forward}}=2MKN$

Full SFT 中，输入梯度和权重梯度都需要计算：

$F_{dX}=2MKN$

$F_{dW}=2MKN$

因此一个全参 Linear 的逻辑训练 FLOPs 为：



$$
F_{\text{Full Linear}}
=
2MKN+2MKN+2MKN
=
6MKN
$$

LoRA 训练中，原始权重 $W$ 冻结，但梯度通常仍需穿过 $W$ 传给前层。此时保留

forward 和 input-gradient，不计算原始权重的 weight-gradient：



$$
F_{\text{LoRA base Linear}}
=
2MKN+2MKN
=
4MKN
$$

LoRA adapter 的计算形式为：

$Y=XW+(XA)B$

其中 $A$ 的 shape 为 $[K,r]$，$B$ 的 shape 为 $[r,N]$，$r$ 是 LoRA rank。

当 $A$、$B$ 都可训练且需要向 $X$ 传播梯度时，adapter 的 forward 和 backward

逻辑 FLOPs 为：



$$
F_{\text{LoRA adapter}}
=
6Mr(K+N)
$$

所以该 LoRA Linear 的完整逻辑训练 FLOPs 为：



$$
F_{\text{LoRA Linear}}
=
4MKN+6Mr(K+N)
$$

通常 $r\ll K,N$，因此可以在解释上近似写为：

$F_{\text{LoRA Linear}}\approx4MKN$

但正式统计必须保留 adapter FLOPs，不能直接用 $4MKN$ 代替完整公式。

LoRA MFU 还必须遵守以下规则：

- 对每个模块根据 `weight.requires\_grad`、`input.requires\_grad` 和 adapter 的实际梯度路径决定是否计算 $dW$、$dX$、$dA$、$dB$。
- 如果某个冻结模块的输入不需要梯度，则不能机械加入其 $dX$；必须按实际 autograd 路径计算。
- attention core、激活函数等非权重梯度计算在 LoRA 和 Full SFT 中仍可能相同，因而完整模型的 LoRA/Full FLOPs 比例不一定等于 $4/6$。
- 常见的 $6N$ 训练近似隐含“主要权重参与 weight-gradient”的 Full SFT 假设，不得直接作为 LoRA MFU 分子。
- Full SFT MFU 与 LoRA MFU 代表不同逻辑训练 workload；比较时必须同时报告训练方式和分子定义，不能只根据百分比判断哪种训练的硬件效率更高。

### 3.3 eval

E2E workload 包含 eval，因此每个 eval batch 的逻辑 forward FLOPs进入 E2E 分子：

$F_{\text{eval},j}=F_{\text{forward},j}$

eval 不包含 backward。eval 的数据加载、同步、指标汇总和日志没有模型 FLOPs，只贡献 E2E 时间。

### 3.4 优化前后的分子一致性

在模型、数据和逻辑训练任务不变时，FA3、CCE、LoRA Fusion、通信 overlap、activation recomputation 策略以及同步/异步 checkpoint 不得通过改变 MFU 分子的定义获得更高结果。

- FA3、LoRA Fusion、通信 overlap 和 checkpoint 优化通常不改变逻辑模型 FLOPs。
- CCE 等改变物理执行路径但保持逻辑 loss 任务等价的优化，使用同一个 dense-reference 逻辑 FLOPs，并在报告中标记 `numerator\_mode: logical\_reference`。
- 如果 batch、sequence shape、packing、图像 token 数或模型结构发生变化，必须按每步真实 workload 重新计算分子。

## 4. Per-step MFU

### 4.1 单步定义

第 (i) 个 optimizer step 的 MFU：



$$
MFU_i
=
\frac{F_{\text{train},i}}
{T_iN_{\text{GPU}}P_{\text{peak}}}
$$

一个 optimizer step 的计时范围固定为：

1. 在请求该 step 的第一个训练 microbatch 之前开始计时；
2. 包含数据等待、全部 gradient-accumulation microbatch、forward、backward、训练通信、optimizer 和 scheduler update；
3. 在 optimizer step 完成后执行 CUDA synchronize；
4. 在 eval、日志重任务和 checkpoint callback 开始之前结束计时。

计时开始和结束都必须使用同一 monotonic clock。需要 CUDA synchronize 的实验必须所有对照组使用相同同步协议。

为避免 shape profiler 改变 step 延迟，正式实验优先采用“shape/FLOPs pass + clean timing pass”两轮确定性 replay，并逐 step 校验样本或 shape fingerprint 对齐。

### 4.2 warmup 与稳态汇总

每个 step 都保存原始结果，但配置中指定的前 (W) 个 warmup step 不进入稳态主结果。例如运行 20 step、`warmup\_steps: 5` 时，正式窗口为 step 6–20。

稳态主结果不能直接对各 step MFU 做算术平均，必须先累计 FLOPs 和时间：



$$
MFU_{\text{per-step summary}}
=
\frac{\sum_{i\in S_{\text{steady}}}F_{\text{train},i}}
{\left(\sum_{i\in S_{\text{steady}}}T_i\right)
N_{\text{GPU}}P_{\text{peak}}}
$$

它等价于按 step 时间加权的 per-step MFU：



$$
MFU_{\text{per-step summary}}
=
\frac{\sum_iT_iMFU_i}{\sum_iT_i}
$$

实验报告同时给出 per-step MFU 的 `median`、`min`、`max`、`p25`、`p75`，用于观察抖动；这些描述统计不能替代上述主结果。

## 5. E2E MFU

### 5.1 时间边界

E2E 使用完整训练任务 wall-clock：

```plain text
launcher 启动训练命令
  → Python/torchrun/DeepSpeed 初始化
  → 模型、checkpoint 和数据加载
  → warmup、CUDA graph/JIT/Triton 编译
  → 全部训练 step
  → 全部 eval
  → 中间 checkpoint
  → 最终 checkpoint 与异步 drain
  → 所有训练进程正常退出

```

定义：

$T_{\text{E2E}}=T_{\text{end}}-T_{\text{start}}$

- `T\_start` 必须由训练进程外层 launcher 在启动命令前记录，不能从第一个训练 step 开始。
- `T\_end` 必须等待所有 rank 退出、异步 checkpoint writer/drain 完成，并确认最终要求的 checkpoint 已持久化。
- 如果后台保存进程晚于主训练进程退出，则以两者中更晚的完成时间作为 `T\_end`。
- E2E 包含 warmup、eval、checkpoint、数据等待、日志、初始化和收尾，不允许从总时间中扣除。

### 5.2 E2E 分子

E2E 分子包含完整 workload 中的逻辑模型工作：



$$
F_{\text{E2E}}
=
\sum_{i\in S_{\text{all train}}}F_{\text{train},i}
+
\sum_{j\in S_{\text{all eval}}}F_{\text{eval},j}
$$

规则：

- 被 per-step 稳态汇总排除的前几个真实训练 step，仍然进入 E2E 分子和时间。
- 如果 warmup 是正常训练的前几个 optimizer step，它们按训练 FLOPs计入分子。
- 如果 warmup 是不产生有效训练更新、随后会丢弃状态的纯编译/虚拟预热，其时间进入 E2E，但其冗余 FLOPs不进入 MFU 分子。
- checkpoint、初始化、数据加载、日志、通信和 optimizer 没有逻辑模型 FLOPs，只进入时间分母。
- eval 的逻辑 forward FLOPs计入分子，eval 其余开销只进入时间分母。
- activation recomputation 仍不进入 MFU 分子。

最终公式：



$$
MFU_{\text{E2E}}
=
\frac{F_{\text{E2E}}}
{T_{\text{E2E}}N_{\text{GPU}}P_{\text{peak}}}
$$

E2E MFU 是本项目定义的完整 workload 指标；论文常见的 iteration MFU 对应本文的 per-step/稳态汇总口径，而不是 E2E MFU。

## 6. 实验报告格式

所有 FLOPs 保存为整数，时间保存为秒，MFU 原始值保存为 `[0, 1]` ratio，同时输出百分比。推荐 YAML 结构如下：

```yaml
mfu_protocol:
  version: 1
  numerator_mode: logical_model_flops
  recomputation_in_numerator: false

hardware:
  gpu_model: NVIDIA H800
  gpu_count: 1
  precision: bf16
  peak_mode: dense_tensor_core
  peak_tflops_per_gpu: 989.4

per_step_mfu:
  warmup_steps_excluded_from_summary: [1, 2, 3, 4, 5]
  steady_steps: [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
  records:
    - step: 1
      model_flops: 0
      step_time_s: 0.0
      mfu: 0.0
      mfu_percent: 0.0
  summary:
    total_model_flops: 0
    summed_step_time_s: 0.0
    time_weighted_mfu: 0.0
    time_weighted_mfu_percent: 0.0
    median: 0.0
    min: 0.0
    max: 0.0
    p25: 0.0
    p75: 0.0

e2e_mfu:
  timer_start: launcher_command_start
  timer_end: all_processes_exited_and_final_checkpoint_durable
  wall_time_s: 0.0
  completed_training_steps: 0
  training_model_flops: 0
  evaluation_batches: 0
  evaluation_forward_flops: 0
  total_logical_model_flops: 0
  final_checkpoint_verified: false
  mfu: 0.0
  mfu_percent: 0.0

data_balance_diagnostics:
  nonpad_tokens_per_rank: []
  padded_tokens_per_rank: []
  padding_fraction_per_rank: []
  rank_step_time_s: []
  straggler_gap_s: 0.0

```

## 7. 对照实验有效性要求

对照实验至少固定并记录：

- 模型版本、训练方式（Full SFT/LoRA）、数据集和 seed；
- global batch、microbatch、gradient accumulation、sequence/packing 策略；
- 训练 step 数、eval 频率和 eval workload；
- checkpoint 频率、保存 payload、持久化位置和恢复有效性；
- GPU 型号、数量、精度、理论峰值和软件环境；
- warmup 排除范围以及 per-step/E2E 计时边界；
- FLOPs 公式版本、shape 记录和不支持算子清单。

异步 checkpoint 只有在同步组和异步组保存 payload 等价、最终文件持久化且能够正常恢复时，才允许比较 E2E MFU。任何配置变化导致 workload 不等价时，必须在报告中显式说明，不能只比较 MFU 百分比。

### 不采用 Useful-token MFU

本项目不再将“把所有 Linear 的 padded token 替换为 non-padding token 后得到的反事实 FLOPs”命名为 MFU。该值主要反映 padding 浪费，不能解释多卡通信、通信计算 overlap、节点间网络瓶颈或 rank straggler，也不代表模型实际执行的逻辑 FLOPs。

如需诊断 packing、padding 或多卡负载不均衡，只记录以下辅助数据：

- 每个 rank 的 padded/non-padding token 数；
- 每个 rank 的 sequence-length 分布；
- padding fraction 和 global non-padding token/s；
- 每个 rank 的 step 时间与 straggler gap。

这些字段统一归入 `data\_balance\_diagnostics`，不得使用 `useful\_mfu` 名称。