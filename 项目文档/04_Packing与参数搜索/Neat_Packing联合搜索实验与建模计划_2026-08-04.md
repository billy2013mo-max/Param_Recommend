# Neat-Packing 联合搜索实验与建模计划

版本：`v4.1`（对齐 `Packing决策逻辑V2_修订版_2026-08-04.md` 与平台化 DataProfile 合同）  
日期：`2026-08-05`  
状态：`phase_b_complete_phase_c_safe_interaction_matrix_materialized`  
工作目录：`/wanqing-develop/luowenjing/Param_Recommend`

> 本文仍是实验与建模的总设计，不自动授权整个计划或模型发布。Packing 语义 canary、第一批 Packing×GC/ZeRO、W4 GBS 修复和 W3/W5/W7/W8 Phase B 均已完成；Phase B 新增 36/36 success、0 OOM，并通过语义、ledger 与重复稳定性门禁。DataProfile estimator 的 utilization/mean 中心和 mean/P99 safety coverage 通过，但 P99 guard 仍过宽，只允许 cache-miss shadow inference。后续执行计划中的 Phase C 已形成独立的安全 24-job 交互矩阵：长 cutoff GC-off 因冻结 operational P95 超过 H800 90% 容量线而禁止入队，GC 轴改用通过同一门禁的 W3@4096 与 W8@10240。该矩阵仍是 fit-only，不允许自动发布。

## 1. 目标与核心结论

目标是在文本 SFT 场景中，把以下变量纳入同一个可解释、可验收的推荐流程：

```text
DataPlan:
  cutoff_len × packing_branch

ExecutionPlan:
  No-Packing: card_type × gpu_count × ZeRO × GC × physical_MBS × derived_GA
  Neat-Packing: card_type × gpu_count × ZeRO × GC × physical_MBS(=1) × derived_GA
```

其中 No-Packing 分支继续搜索 MBS，并由 `target_GBS/(DP×MBS)` 精确派生 GA；Neat-Packing 分支固定 `physical_MBS=1`，cutoff 直接决定每个 pack/每卡 microstep 的 token 容量，是 Packing 吞吐的核心旋钮。Packing 下每个 pack 的逻辑样本数会变化，因此样本级 GBS 采用期望值管理：推荐阶段用缓存 DataProfile 预测 utilization 和平均 `n_pack`，再由目标 GBS、GPU 数和 `n_pack` 推导 GA。稳定和可控的主要训练负载是 token/step，而不是要求每个 optimizer step 恰好包含固定条数的逻辑样本。

本计划采用以下建模结论：

1. `neat_packing=false/true` 不能只共用一套系数再加一个二值偏置。
2. 两条路径仍应共享模型状态、理论 FLOPs、HBM、通信等物理基础，不应完全拆成两个无关黑盒。
3. 显存模型共享模型状态、activation/attention 等物理主干，以实际 active tokens、attention pairs 和 kernel workspace 表达主要差异；Packing 只增加专属 workspace/residual 与安全 upper guard，不重建一套无关黑盒。
4. Packing 分支必须把显存可行、GBS 可控、总优化器步数达标三个硬门槛放在同一个 cutoff 循环中，不能先“尽量拉长 cutoff”再独立修补。
5. 数据上传阶段异步 tokenize，并按 packer fingerprint 缓存候选 cutoff 的紧凑 pack-count 分布；训练推荐阶段不得读取或重新 tokenize 原始数据，只能由画像/缓存估计 `pack_utilization/n_pack/packs/expected_GBS/opt_steps` 并做硬门槛。
6. 严格配对的 effect head 为分支切换提供置信区间；预测接近或超出支持域时由双分支短跑兜底。
7. 扩卡收益在固定 cutoff、Packing 状态、目标 GBS、数据快照和运行时合同下单独建 ratio head。
8. 数据集名称不能成为自由系数；数据差异必须通过 Base DataProfile/CandidateProfile 连续特征表达。

推荐模型拓扑：

```text
Shared physical basis
├── Shared memory state/activation/attention model
│   ├── Unpacked workload adapter + upper safety envelope
│   └── Neat-Packing workspace/residual + upper safety envelope
├── Unpacked throughput residual
├── Neat-Packing throughput residual
├── Paired Packing effect head
└── Packing-aware cross-card ratio head
```

产品决策遵循三层结构：上传期 DataProfile 与 utilization estimator 计算每个 cutoff 的期望 GBS 和总优化器步数，并完成硬门槛；双分支模型比较各自最优安全配置；只有收益置信区间重叠、画像信息不足或数据超出标定支持域时，才对两边 Top 候选执行短跑验证。

### 1.1 Neat-Packing 的 batch/GBS 合同

对 cutoff `C`、由缓存画像预测的 pack utilization `U_hat(C)`、平均样本长度 `L_mean`、数据并行度 `DP` 和目标样本级 GBS `G`，定义：

```text
n_pack_mean_center = C × U_center(C, DataProfile, packer_fingerprint) / L_mean
n_pack_mean_upper  = epoch平均samples/pack的置信上界
n_pack_step_p99    = 单pack逻辑样本数分布的P99上界
GA = max(1, round(G / (n_pack_mean_center × DP)))
expected_epoch_sample_GBS = n_pack_mean_center × DP × GA
tokens_per_microstep_per_rank ≈ C
tokens_per_optimizer_step ≈ C × DP × GA
```

`n_pack_mean_center` 的长度分母必须使用离线上传画像中的均值，不能使用中位数。如果上传期已经缓存同 packer fingerprint 的 utilization/pack-count curve，优先使用实测 `{mean,p50,p90,p95,p99,max}`；否则由长度直方图、CV、分位数和长尾特征分别预测 epoch mean 置信界与 step 分布上尾。`U_upper` 只能形成 mean 的不确定性上界，不能替代 `n_pack_step_p99`。实际每个 pack 和 optimizer step 的样本数允许波动，GA 只基于中心期望 GBS 派生，不作为 Packing 分支的独立自由搜索变量。

由于 `GA≥1`，如果一个 microstep 全局已经装入超过目标 GBS 允许范围的样本，就无法再通过 GA 降低。因此 Packing 候选必须满足：

```text
n_pack_step_p99 × DP ≤ G × (1 + epsilon_GBS)
```

对画像记录数 `N`、`n_pack` 和 epoch 数 `E`，预计总 pack/优化器步数为：

```text
packs_center = N / n_pack_mean_center
opt_steps_center = packs_center × E / (DP × GA)

safety gate uses epoch-mean upper because it produces fewer packs/steps
opt_steps_lower = (N / n_pack_mean_upper) × E / (DP × GA)
```

并要求 `opt_steps_lower ≥ K_min_steps`。`epsilon_GBS` 和 `K_min_steps` 必须作为产品/训练合同预注册：前者控制样本级 GBS 的可接受偏差，后者覆盖 warmup、decay 和基本收敛所需步数。二者不能针对某个候选临时调宽；具体默认值在阶段 B/C 的收敛与波动实验后冻结。

benchmark 和生产都必须按真实 pack、optimizer boundary 和有效 label/token denominator 记录 consumed ledger；dataloader prefetch 不得计入已消费样本。短 probe 的窗口均值只允许报告为 `observed_probe_window_GBS`，不得代替完整画像推导的 `expected_epoch_GBS`。正确的块对角 attention mask、position 和 loss mask 仍是发布前提。

## 2. 首版支持域与非目标

### 2.1 首版支持域

- 纯文本 causal SFT；
- Dense Qwen3/Qwen3.5；
- BF16；
- H800；
- Full 与 LoRA；
- 1/2/4 GPU；
- 单机已冻结互联拓扑；
- 无 offload；
- `packing=true` 必须等价于 `neat_packing=true`；
- neat-packing 固定物理 `MBS=1`；
- 目标样本级 GBS 优先为 32/64，Packing 侧按平均 `n_pack` 派生 GA 并报告期望值与波动；
- cutoff 从不截断真实样本的 `data_max_len` 起，采用粗搜、事件边界与局部 512 精搜到模型、显存和 GBS 三重上界；
- tokenizer、template、DataProfile schema/version、sampler、packer、shuffle、loss-normalization 与 runtime 必须进入 contract fingerprint；
- 训练推荐请求只消费已缓存 DataProfile，不依赖原始数据可读性。

### 2.2 首版明确不覆盖

- 普通 `packing=true, neat_packing=false`；
- VL、多模态、DPO/ORPO、分类专用 collator；
- MoE、量化训练、CPU/NVMe offload；
- 多机拓扑；
- H800 系数直接迁移到 RTX 4090；
- 未绑定 tokenizer、chat template、processor 和 runtime fingerprint 的数据。
- 未记录样本级 GBS 波动、token/step、实际 optimizer steps 和 consumed ledger 的 Packed 自动推荐。
- 只有 mean、没有 records/max/长度分布且不能给出 utilization 不确定区间的数据画像自动推荐。

RTX 4090 作为 H800 发布之后的独立硬件 track，复用 schema 和物理 basis，但重新标定硬件 adapter、Packing residual 与扩卡比例。

## 3. 当前证据基线

### 3.1 已完成的真实业务交互实验

`h800_real_business_packing_cutoff_mbs_20260804_v1` 共 40/40 成功、0 OOM。它覆盖：

- Qwen3-8B LoRA；
- 1×H800；
- ZeRO-0；
- GC on；
- 4 个真实业务长度画像；
- 每个画像 5 个 arm、2 次重复。

关键结果：

| 数据画像 | `P-kC-1 / N-C-k` | `P-C-1 / N-C-1` | `P-kC-1 / P-C-1` |
|---|---:|---:|---:|
| 极短集中 | 3.188× | 4.386× | 2.860× |
| 短文本长尾 | 1.526× | 4.515× | 1.161× |
| 教育会话中长 | 1.214× | 1.267× | 1.041× |
| 内容评测宽长尾 | 1.672× | 3.692× | 0.702× |

这组数据足以证明：Packing 效应取决于长度分布和 cutoff，扩大 cutoff 不是单调增益；但它不能标定 GC、ZeRO、Full/LoRA、多卡和卡型交互。只要 tokenizer/template/packer/runtime fingerprint 与目标发布机制一致，这些 Packed 点可以进入当前期望 GBS 路线的 calibration，而不再需要迁移到另一套 exact-GBS BatchSampler。

同 cutoff、MBS=1 的四个业务画像中，No-Packing→Packing 峰值显存约变化 `+0.4%/+1.2%/+4.1%/+10.8%`；而增大 Packed cutoff 可带来约 `65%～67%` 的显存增长。它支持“配置 cutoff 与实际 active tokens 是主效应，Packing workspace/fragmentation 是次级 residual”的建模方向，但现有点仍不足以发布 80 GiB 边界 upper guard。

### 3.2 已完成的 24-job Packing calibration

现有 calibration 有 24/24 成功、12 个完整配对，覆盖 8B/14B、Full/LoRA、1/2 GPU、ZeRO-0/2/3 与 GC 的部分组合。四个 family 的 Packing 路线收益约为：

| Family | 收益 | 当前形式状态 |
|---|---:|---|
| C1 | +131% | 可用于 fit-only |
| C2 | +42% | 可用于 fit-only |
| C3 | +170% | 可用于 fit-only |
| C4 | -68% | 3 个 packed repeat 的语义标志为 false |

因此当前 evaluator 给出 `calibration_complete_for_joint_fit=false`。首先应审计 C4 的 consumed ledger、prefetch 计数、loss mask 和跨样本 attention 证据；若不能从冻结证据无歧义恢复，应完整重跑 C4 的 3 组 U/P 配对，不能只重跑 packed 一侧。

### 3.3 当前代码能力与缺口

已有能力：

- `candidate_generator.py`：固定 cutoff 下生成 GPU 数、ZeRO、GC、MBS 候选；
- `packing_aware_candidates.py`：生成 unpacked rankable 与 packed shadow 两分支；
- `structured_throughput_modeling.py`：已有物理组成、结构化特征和卡型 adapter；
- `cross_card_scaling.py`：已有相邻翻倍、显存先验和保守 1.8× 判定合同。

主要缺口：

- cutoff 外层候选生成；
- Packing 专属 Base/CandidateProfile 特征；
- Packing 显存中心与上界；
- Packed 吞吐 residual 和 paired effect head；
- Packing-aware 扩卡 ratio 置信界；
- source-disjoint prospective holdout；
- expected-GBS、token/step、总 optimizer steps 与 consumed ledger 的统一合同；
- 平台 DataProfile schema、utilization/n_pack 估计器及不确定区间；
- 不读取原始数据的 Packed/Unpacked 双分支最终决策器。

## 4. 数据画像定义

### 4.1 平台化两层画像合同

数据上传/注册阶段运行一次异步 tokenizer/template pipeline，以 streaming aggregation 生成 cutoff 无关的冻结 `DataProfile`。推荐服务不保存或要求全量 token IDs，也不在训练请求路径重新读取原始数据。DataProfile 至少包含：

#### 上传期 Base DataProfile

- dataset revision、tokenizer revision、chat template/processor revision；
- records `N`、统计覆盖行数、采样/全量标志和 profile confidence；
- mean/std/CV；
- P50/P90/P95/P99/max；
- mean/P50/P90/P99 label tokens 与 label token ratio；
- turns、assistant turns、segments 的统计；
- 预注册长度直方图、长尾比例和双峰特征；
- profile schema/version、生成时间、输入/输出 hash。

全量上传 tokenize 可以流式执行，只持有当前 batch 和聚合器；不要求为了推荐永久保存完整 token 序列或逐样本长度。大数据集如果只能抽样，必须记录 sampling rule、sample size 和统计置信界。缺失 `N/mean/max` 时不能生成自动 Packed 推荐；只有 mean 而缺少直方图/分位数时，只能使用宽 utilization 区间并返回 shadow/provisional。

训练推荐阶段基于 Base DataProfile 和候选 cutoff 派生 `CandidateProfile`，不接触原始数据：

#### cutoff 派生 CandidateProfile

- `mean/Pxx/max ÷ cutoff`；
- 截断风险和 token 保留率下界；
- No-Packing 各 MBS 的 padding/active-token/attention-pair 估计；
- `U_lower/U_center/U_upper(cutoff)`；
- `n_pack_mean_lower/center/upper`、packs interval；
- `n_pack_step_p50/p90/p95/p99/max` 及其校准上界；
- derived GA、expected GBS/error interval；
- tokens/optimizer-step、`opt_steps_lower/center`；
- GBS controllable、`K_min_steps`、memory 和 support gates。

dataset ID 不进入模型自由系数；数据差异只通过上述连续画像表达。

### 4.2 Utilization/n_pack 估计器

推荐阶段不再运行完整静态 packer。utilization 来源按优先级选择：

1. 上传期已经针对同一 packer fingerprint 缓存的 `U(cutoff)` 与 samples-per-pack 分布曲线；
2. 由 DataProfile 预测的 `U_hat(cutoff)`；
3. 只有极简画像时使用保守宽区间，并保持 Packed shadow。

utilization 模型最低输入：

- `mean/cutoff`、CV；
- P50/P90/P99/max 相对 cutoff；
- 长度直方图与短/中/近-cutoff 占比；
- 双峰/长尾特征；
- turns/segments；
- packer、shuffle/sharding fingerprint。

中心估计用于 GA 和性能排序：

```text
n_pack_center = cutoff × U_center / mean_length
GA = max(1, round(target_GBS / (n_pack_center × DP)))
```

这里的 `n_pack_center` 等价于 `n_pack_mean_center`。两个不同风险必须使用两个不同上界：

```text
n_pack_mean_upper = epoch平均samples/pack的校准置信上界
n_pack_step_p99   = 单pack逻辑样本数P99的校准上界
GBS gate: n_pack_step_p99 × DP ≤ target_GBS × (1 + epsilon_GBS)
steps lower: (N / n_pack_mean_upper) × epochs / (DP × GA) ≥ K_min_steps
```

同一均值可能对应集中、长尾或双峰等完全不同的 pack-count 分布，因此 mean 是期望公式核心，但不能作为 GBS 单步安全代理。utilization/pack-count estimator 必须在 W1～W9 和 source-disjoint holdout 上分别验收 epoch-mean 中心误差、mean 区间 coverage 和 step-P99 coverage；超出画像支持域时不允许给出无置信界的点估计。

上传期 utilization 曲线和推荐期派生画像的缓存键必须包含 `dataset/profile/tokenizer/template/packer/shuffle/schema revision`。缓存命中时推荐路径只做数值计算和模型推理，目标 P95≤2 秒。

### 4.3 目标画像格

| 画像 ID | 定义目标 | 主要辨识内容 |
|---|---|---|
| W1 极短集中 | `P50/cutoff ≤0.25`，低 CV，高 samples/pack | Packing 高收益上界 |
| W2 短文本稀有长尾 | P50 很短但 P99 接近 cutoff | 尾部和截断敏感性 |
| W3 自然多轮中等长度 | turns 较多，`P50/cutoff≈0.3～0.7` | segment/turn 与 kernel 开销 |
| W4 宽长尾 | `P99/mean` 与 CV 高 | cutoff×Packing 交互 |
| W5 近 cutoff 饱和 | P50≥0.75 cutoff，samples/pack 接近 1 | Packing 低收益/负收益边界 |
| W6 高截断压力 | cutoff 下截断率 2%～10% | DataPlan 质量门控；只作诊断 |
| W7 双峰混合 | 短样本与近 cutoff 长样本各占稳定比例 | greedy packing 的组合效应 |
| W8 结构化/代码 | 长标点、JSON、代码块、label ratio 变化大 | tokenizer 与 kernel workload 迁移 |
| W9 标签占比高的推理 | assistant label 较长 | label/optimizer 路径迁移 |

W6 默认不参与自动推荐胜负；只有 token 保留率和截断率满足产品门槛的 cutoff 才能进入最终候选。

## 5. 现有本地数据盘点

### 5.1 已冻结的公开源池

| 源 | 当前冻结 revision | 本地用途 |
|---|---|---|
| `yahma/alpaca-cleaned` | `12567cabf869d7c92e573c7c783905fc160e9639` | 短单轮、短集中 |
| `HuggingFaceH4/ultrachat_200k` | `8049631c405ae6576f93f445c6b8166f76f5505a` | 自然多轮、2K/4K |
| `Yukang/LongAlpaca-12k` | `46dce924ed8786979556018e191c0f557d8f4aa2` | 8K～32K、近 cutoff 和高截断构造 |

### 5.2 现有代表画像

| 数据 | cutoff | mean / P50 / P99 tokens | pack fill | samples/pack |
|---|---:|---:|---:|---:|
| `short_512` | 512 | 160 / 130 / 431 | 0.978 | 3.12 |
| `multiturn_2048` | 2,048 | 1,135 / 1,094 / 1,984 | 0.947 | 1.71 |
| `multiturn_4096` | 4,096 | 1,176 / 1,094 / 3,155 | 0.977 | 3.40 |
| `longtail_8192` | 8,192 | 819 / 245 / 8,128 | 0.952 | 9.52 |
| `longcontext_16384` | 16,384 | 8,283 / 7,959 / 16,320 | 0.936 | 1.85 |
| `longcontext_32768` | 32,768 | 9,502 / 7,959 / 26,673 | 0.873 | 3.01 |
| 真实业务极短集中 | 1,024 / 4,096 | 219 / 209 / 376 | 0.942 / 0.979 | 4.44 / 18.33 |
| 真实业务短尾 | 4,096 / 16,384 | 589 / 565 / 917 | 0.955 / 0.978 | 6.67 / 27.22 |
| 真实业务教育中长 | 8,192 / 32,768 | 2,638 / 2,525 / 4,739 | 0.915 / 0.966 | 2.84 / 12.00 |
| 真实业务宽长尾 | 16,384 / 32,768 | 1,041 / 661 / 6,282 | 0.974 / 0.973 | 15.36 / 30.61 |

现有数据已覆盖 W1～W5 的大部分区域，但以下缺口仍明显：

- source-disjoint 的自然多轮 holdout；
- source-disjoint 的 8K～64K 长上下文 holdout；
- 结构化代码/JSON token 形态；
- 标签占比高的推理输出；
- 预注册的双峰画像；
- 没有参与当前阈值选择的新业务数据。

## 6. 新数据源检索结果与使用决策

所有数据都必须在下载时冻结 repo revision、原文件列表和 SHA256。下表中的 revision 是 2026-08-04 查询到的候选版本；正式 acquisition 必须再次核验并写入 manifest。

| 数据源 | 当前 revision | 许可证/访问 | 目标画像 | 决策 |
|---|---|---|---|---|
| [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) | `179dd21fc55192153d94adb0e0ce8f69e222bf75` | Apache-2.0，非 gated | W3 自然多轮、多语言 | 采用；优先保留为 source-disjoint holdout |
| [microsoft/orca-math-word-problems-200k](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) | `29255d1770cc4eac66e5e7fa378cba542c026350` | MIT，非 gated | W9 标签占比高的推理 | 采用；holdout/迁移诊断 |
| [nvidia/OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) | `8f3ba5bafe4d6e8db46082cf7ae6741bc370604d` | CC-BY-4.0，非 gated | W8 代码、结构化输出 | 采用；streaming 采样进入 calibration |
| [ise-uiuc/Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) | `5f839b1f368a76b161028bb9edff055db34022b2` | MIT metadata；数据卡提示额外关注生成模型使用政策 | W8 代码 source-disjoint 验证 | 条件采用；政策审核通过后作 holdout |
| [LongAlign-10k](https://github.com/THUDM/LongAlign) / HF redirect `zai-org/LongAlign-10k` | `12f17c4baff1001f0d44c4f8feab09ee2ee8c6dc` | 代码仓 Apache-2.0；HF 数据仓缺 license tag | W5/W6，8K～64K | 条件采用；必须单独确认数据许可 |
| [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | `26d14ebfe18b1f7b524bd39b404b50af5dc97866` | CC-BY-4.0，gated | JSON/tool-call | V1 不作为硬依赖；避免 gated 数据阻塞计划 |

### 6.1 数据角色分配

建议分配如下：

```text
Calibration:
  现有 Alpaca / UltraChat / LongAlpaca / 真实业务与 S3 画像
  + NVIDIA OpenCodeInstruct 的冻结小样本

Prospective holdout:
  OASST2（自然多轮）
  Orca Math（推理/长 label）
  Magicoder（代码；政策通过后）
  LongAlign（长上下文；许可通过后）
  至少 1 份全新业务 S3/Blobstore 快照
```

若 Magicoder 或 LongAlign 未通过许可/访问门禁，不得临时换数据后沿用原 holdout 名称；必须生成新设计版本并重新冻结。代码 holdout 可退回 OpenCodeInstruct 的 publisher-disjoint 子源，但只能标为 `row/source-family holdout`，不能冒充 dataset-source-disjoint。

### 6.2 双峰和边界数据的构造

双峰画像不依赖寻找一个现成“双峰数据集”。从互不重叠的短、长源按确定性 hash 抽样：

```text
W7-25/75: 25% short + 75% near-cutoff long
W7-50/50: 50% short + 50% near-cutoff long
W7-75/25: 75% short + 25% near-cutoff long
```

每个 mixture 固定 source row ID、比例、shuffle seed 和最终行序。Calibration 与 holdout 不得共享 source row、conversation tree、近重复文本或派生样本。

LongAlpaca/LongAlign 按 tokenizer 后长度构造：

- near-cutoff：`0.75C ≤ L ≤ 1.05C`；
- high-truncation：`L ≥ 1.20C`；
- broad tail：同时包含 `<0.25C`、`0.25C～0.75C`、`>0.75C` 三层。

## 7. 数据准备与冻结协议

### 7.1 Acquisition

每个源生成：

- repo ID、revision SHA、split、config；
- 原始文件列表、etag/SHA256、下载时间；
- license tag、数据卡 URL、访问/gated 状态；
- 原始 row ID 或可稳定重建的复合 ID；
- 本地 snapshot SHA256；
- 转换脚本 SHA256；
- 采样 seed 和 hash 规则。

大数据集采用 streaming 或固定 parquet row-group 采样，不需要完整下载。禁止依赖“当前 main”而不固定 revision。

### 7.2 规范化

统一成项目支持的 ShareGPT 或 prompt/response schema，并记录：

- 原始字段到目标字段映射；
- system/user/assistant role；
- tool/JSON/code fence 是否原样保留；
- 丢弃规则与丢弃数量；
- tokenizer/template ID；
- 多轮 tree 展平路径。

OASST2 必须按 `message_tree_id/parent_id/message_id` 重建从 root 到 leaf 的合法对话路径，不能把 depth-first flat rows 当成独立对话。

### 7.3 去重与 split

- exact normalized-content SHA256 去重；
- MinHash/5-gram 近重复检查；
- conversation tree、source document、code seed 必须作为 group split 单元；
- calibration/holdout 先按 source 数据集分隔，再做 group-level hash split；
- holdout 在预测、阈值、family 和 evaluator 冻结前不可查看 GPU 结果；
- 已用于本轮方案选择的四份真实业务数据只能进入 calibration，不能再称 prospective holdout。

### 7.4 样本量

- 每个抽样 Base DataProfile 至少 4,096 条；极短样本优先 8,192 条；全量 streaming profile 不受此下限替代统计约束；
- 无法达到 4,096 条的自然长上下文画像至少 1,000 条，并对 P99 不确定性单列报告；
- GPU probe slice 至少覆盖 `warmup + measure` 所需逻辑样本的 2 倍；
- source-disjoint holdout 每个 family 至少 1,000 条，优先 2,048～4,096 条。

### 7.5 平台数据生命周期

```text
dataset_uploaded
  → offline_profiling（异步 tokenizer/template + streaming aggregation）
  → profile_ready（冻结 DataProfile + revision/hash）
  → recommendation/training requests（只读 DataProfile）
```

推荐接口不得隐式触发 tokenizer、下载数据或完整 pack 模拟。`profile_ready` 之前可以返回 `profiling_pending`，但不能为了降低响应时间用无来源默认均值自动开启 Packing。数据、tokenizer 或 template 任一 revision 改变时生成新 profile，不原地覆盖旧画像；已物化训练任务必须绑定具体 profile hash。

## 8. cutoff 候选生成

### 8.1 候选规则

Packing 分支不再全区间固定按 512 密扫，而是采用“粗搜＋事件候选＋局部 512 精搜”：

```text
C_data_min = ceil_to_512(data_max_len_after_template)
C_upper = min(C_model_max, C_memory_feasible_max, C_GBS_controllable_max)
```

`C_data_min` 保证不截断真实样本；如果原始样本超过模型上下文上限，应先由数据质量合同明确截断/过滤，不能由推荐器静默处理。对每个 `(card_type,gpu_count,ZeRO,GC)` 分别确定：

- `C_model_max`：模型上下文、position encoding、runtime 和 kernel 的已验证上限；
- `C_memory_feasible_max`：显存规划器预测峰值不超过 `memory_margin×capacity` 的最大 cutoff；
- `C_GBS_controllable_max`：仍满足 `n_pack_step_p99×DP≤G×(1+epsilon_GBS)` 的最大 cutoff。

初始粗网格：

```text
cutoff ≤ 4K:        step=512
4K < cutoff ≤ 16K: step=1024
cutoff > 16K:      step=2048
```

强制加入事件候选：`C_data_min/C_upper`、每个合理 GA 对应的目标 GBS cutoff、显存边界、GBS 可控边界、`opt_steps_lower≈K_min_steps` 边界。对 coarse Top 3～5 和所有区间跨门槛的点，再加入 `C-512/C/C+512`。近似 GA 目标点可由下式给出，并用 `U_hat(C)` 迭代一到两次：

```text
C_GA ≈ target_GBS × mean_length / (U_hat(C) × DP × GA)
```

LoRA + 已验证 dense 模型初始 `memory_margin=0.95`；Full fine-tuning 使用单独标定且更小的 margin。在显存模型未通过对应模式的 upper-coverage 验收前，只能返回 shadow。GPU 实验和短跑只选信息量最高、预测最优及边界点，不跑完整候选网格。

吞吐对 cutoff 不作单调假设：cutoff 增大通常提高每卡 token 负载，但同时增加激活显存、每 pack 样本数并减少总 pack/optimizer steps；最终必须在同一循环中裁决。

### 8.2 Packing 候选的联合硬门槛

No-Packing 分支只保留满足下式的整数候选：

```text
GA = target_GBS / (DP × physical_MBS)
GA ∈ positive_integer
```

Neat-Packing 分支固定 `physical_MBS=1`。对每个 cutoff，只读取 DataProfile 并调用 utilization estimator：

```text
n_pack_mean_center = cutoff × U_center / mean_sample_length
n_pack_mean_upper  = epoch平均samples/pack的校准置信上界
n_pack_step_p99    = 单pack逻辑样本数P99的校准上界
GA = max(1, round(target_GBS / (n_pack_mean_center × DP)))
expected_epoch_GBS = n_pack_mean_center × DP × GA
expected_GBS_error = (expected_epoch_GBS - target_GBS) / target_GBS
packs_center = records / n_pack_mean_center
opt_steps_center = packs_center × epochs / (DP × GA)
opt_steps_lower = (records / n_pack_mean_upper) × epochs / (DP × GA)
```

然后在同一次循环中施加硬门槛：

```text
不截断:   cutoff ≥ C_data_min
显存可行: predicted_peak_upper ≤ memory_margin × GPU_capacity
GBS可控:  n_pack_step_p99 × DP ≤ target_GBS × (1 + epsilon_GBS)
步数达标: opt_steps_lower ≥ K_min_steps
运行支持: cutoff/packer/kernel/runtime fingerprint in supported_domain
```

`GBS可控` 判断的是 GA 已经降到 1 后单步上尾是否仍过大，是 cutoff/DP/目标 GBS 的联合硬约束；GBS gate 使用 `n_pack_step_p99`，步数 gate 使用 `n_pack_mean_upper`，排序使用中心值。`expected_epoch_GBS`、`observed_probe_window_GBS` 和逐 step 样本数预测区间必须分开报告，不能把短窗口波动伪装成 epoch mean 或 exact GBS。`K_min_steps` 按总训练 optimizer steps 判断，包含 epochs 对步数的杠杆。

在所有存活候选中，用 Packing memory/throughput head 预测有效吞吐、epoch time 和置信区间，选择吞吐最高的 cutoff/执行配置。若候选为空，则该硬件/卡数下 Packing 不可用，回到 No-Packing 分支或其他卡型卡数。

### 8.3 cutoff、MBS 与 Packing GA 的 arm

为分离“Packing 机制效应”和“替代最佳 MBS 的路线收益”，每个关键 `(data, cutoff)` 至少保留：

```text
N-C-1-gN      No-Packing, physical MBS=1, gN=G/DP
P-C-1-gP      Neat-Packing, physical MBS=1, gP按平均n_pack派生
N-C-m*-gN     No-Packing, 当前卡数下最快安全 MBS, GA精确派生
```

若研究 cutoff 交互，再增加：

```text
N-kC-1-gN
P-kC-1-gP
```

估计量：

```text
MBS=1配对效应  = log(T(P-C-1-gP)) - log(T(N-C-1-gN))
路线决策效应   = log(T(P-C-1-gP)) - log(T(N-C-m*-gN))
cutoff交互效应 = [log(T(P-kC-1-gP))-log(T(N-kC-1-gN))]
                 - [log(T(P-C-1-gP))-log(T(N-C-1-gN))]
```

Packing 会改变物理 pack 数、GA 和样本级 GBS 分布，因此这里的“MBS=1配对效应”是完整 Packing 路线的机制效果，不声称是在 GA 和逐步样本数完全相同下的单一 kernel 因果效应。只比较 `P-C-1-gP` 与 `N-C-m*-gN` 会把 Packing、MBS 和 GA 路线差异合在一起，适合产品最终决策，但不足以单独辨识各执行系数。

## 9. 分阶段实验设计

### 阶段 A：CPU-only 画像与设计矩阵

GPU 数：0。

1. 下载/冻结新数据源；
2. 用平台上传 pipeline 生成 cutoff 无关 Base DataProfile，验证 streaming aggregation 与 hash/revision；
3. 只在 calibration/上传画像环境中运行真实 packer，生成监督标签 `U(cutoff)/n_pack_mean/n_pack_step_distribution/packs`；用户推荐路径不运行该步骤；
4. 拟合并冻结 utilization、epoch-mean n_pack 与 step-P99 模型，计算 `expected_epoch_GBS/opt_steps` 与三项硬门槛；
5. 生成候选 family universe；
6. 用预注册 D-optimal/分数因子选择器挑选 family。

D-optimal 选择器的目标不是最大化数据量，而是保证以下预声明列满秩、每个二元水平至少有 3 个 family 支撑，并控制设计矩阵条件数：

```text
packing × cutoff/workload features
packing × GC
packing × zero3
packing × log2(gpu_count)
packing × train_mode
packing × model_geometry
packing × target_GBS/derived_GA
packing × n_pack/expected_GBS_error
packing × opt_steps/GBS_controllable_margin
```

候选选择必须在 GPU 结果前冻结。

### 阶段 B：Packing 语义与 instrumentation closeout

1. 审计旧 C4 的 3 个 semantic false；能从权威 consumed ledger 无歧义重建则只重算报告，否则重跑完整 C4 U/P×3；
2. 对最高 samples/pack、宽长尾和最长上下文画像分别做最短 No-Packing/Packing canary，至少覆盖两个 DP；
3. 逐字节验证块对角 attention mask，确保 pack 内不同样本互不 attend，并检查 position reset、loss mask 和边界 token；
4. 验证 optimizer boundary、实际 label/token loss denominator、真实逐 step 样本数和 dataloader prefetch 排除；
5. 对比 DataProfile 估计与完整静态 pack 标签的 `pack_utilization/n_pack_mean/n_pack_step_p99/packs/expected_epoch_GBS/opt_steps`；短 probe 只检查窗口观测和 consumed ledger，中心误差、step-P99 coverage 或区间 coverage 未过门槛时阻断发布；
6. 记录多 rank 的 pack/sample/token 负载不均衡和 step-time tail。

本阶段任一语义项失败时停止后续 Packed GPU 实验。尤其在块对角 attention 未被逐字节证明前，不能使用“attention 代价随 cutoff 近似线性”的建模假设。

### 阶段 C：DataPlan 曲面补测

原 40-job 结果与本次 18-job W4 修复结果按各自 GBS 合同拆分后，合格子集作为当前 runtime 的 fit-only calibration；不合格点继续只作 shadow。新增重点放在尚未覆盖的 W3/W5/W7/W8，并让每个 Packed arm 使用 DataProfile 的 `n_pack_mean_center` 派生 GA：

| Profile | 建议数据 | cutoff 档 | arms | 初始重复 |
|---|---|---:|---|---:|
| 自然多轮 | UltraChat calibration slice | `C_data_min/中间/近上界` | 5-arm | 2 |
| 近 cutoff 饱和 | LongAlpaca stratum | `C_data_min/中间/近上界` | 5-arm | 2 |
| 双峰 50/50 | Alpaca + LongAlpaca | `C_data_min/中间/近上界` | 5-arm | 2 |
| 代码/结构化 | OpenCodeInstruct | `C_data_min/中间/近上界` | 5-arm | 2 |

每个画像使用粗网格、GA/显存/GBS/K 事件点和局部 512 精搜；GPU 点优先选择预测最优点、显存上界邻点、GBS 可控上界邻点和 `opt_steps_lower≈K_min_steps` 邻点。优先包络为 30～40 个新增运行；与现有 58 次 execution fingerprint 和 GBS 合同均完全相同的合格点可以复用，任何删点都不能参考新 GPU 结果。

W1/W3/W4 至少各保留 `C_data_min/中间/近上界` 三个 Packed 点，用来检验“长 cutoff 下 pack utilization 近似稳定”的假设。若画像预测或实跑 utilization 随 cutoff 系统漂移，不得使用常数 utilization 外推，必须把 `cutoff×profile` 交互带入 `n_pack` 和吞吐预测。

另设收敛质量子实验，不与短吞吐 probe 混为一类：选 W1/W3/W4 至少 3 个画像，在固定模型、数据、token budget、optimizer 和学习率口径下，对比 No-Packing、步数充足的 Packed cutoff、接近 `K_min_steps` 的长-cutoff Packed 候选；记录逐 step 样本 GBS 波动、训练/验证 loss、最终任务指标和 warmup/decay 覆盖。该实验用于冻结 `epsilon_GBS` 与 `K_min_steps`，预计 12～24 个较长训练 jobs；在它完成前，相关门槛只能处于 shadow。

重复规则：

- treatment CV>5%：补第 3 次；
- 路线差异绝对值<5%：补第 3 次；
- 保守下界已领先≥20%且语义/显存健康：不补第 4 次；
- 并发实验记录 GPU assignment、CPU/I/O 压力和运行时 cohort。

### 阶段 D：Packing 与执行参数交互

从候选 universe 选择 8～12 个 base family，每个 family 固定模型、数据、cutoff、卡型、卡数、ZeRO、GC、目标 GBS、dtype、kernel 和 optimizer，对 `N-C-1-gN/P-C-1-gP` 做 counterbalanced 配对；`N-C-m*-gN` 作为路线 anchor。Packing GA 一律由该 cutoff 的 `n_pack_mean_center` 派生，不为追求更好结果手动改 GA。

必须获得以下对比支持：

| 交互 | 最低对照要求 |
|---|---|
| Packing×GC | 同模型/数据/cutoff/卡数/ZeRO 下 GC on/off |
| Packing×ZeRO | 同一 2-GPU family 的 ZeRO-2/3 |
| Packing×卡数 | 同一语义 family 的 1/2 或 2/4 GPU |
| Packing×Full/LoRA | 同模型规模、数据、cutoff 与卡数 |
| Packing×模型规模 | 同训练方式和执行配置的 8B/14B |
| Packing×GBS/GA | 至少一个高 samples/pack 和一个近饱和画像；覆盖 GA=1 的 GBS 上界及 GA>1 |
| Packing×optimizer steps | 至少覆盖步数宽裕和 `opt_steps_lower≈K_min_steps` 的 cutoff |

每个 U/P treatment 默认 3 次，顺序采用 counterbalanced `U-P-P-U-U-P` 或镜像 `P-U-U-P-P-U`。预计 36～60 个新增运行；能与阶段 C 或扩卡端点完全同配置的运行按 execution fingerprint 复用。

#### 阶段 D 第一批 closeout（2026-08-04）

第一批 Packing×GC/ZeRO 的 24/24 个正式 jobs 全部成功，0 OOM，语义与 consumed ledger 均通过。配对结果如下：

| 画像/交互 | Packing 吞吐比（samples/s） | No-Packing→Packing reserved | 完整画像 expected epoch GBS | 短 probe window GBS | GBS 合同 |
|---|---:|---:|---:|---:|---|
| W1, GC off | 13.347× | 26.29→61.49 GiB | 63.751 | 63.75 | 通过；step P99×DP=41 |
| W1, GC on | 13.509× | 17.76→22.02 GiB | 63.751 | 63.75 | 通过；step P99×DP=41 |
| W4, ZeRO-2 | 3.095× | 25.04→41.51 GiB | 61.224 | 50.25 | 不通过；step P99×DP=120 |
| W4, ZeRO-3 | 8.631× | 21.86→39.39 GiB | 61.224 | 50.25 | 不通过；step P99×DP=120 |

Packing×GC 的 log 交互约 `-0.012`，当前近似可忽略；Packing×ZeRO 的 log 交互约 `1.026`，很大但 W4 Packed arms 未通过修正后的目标 GBS=64 合同。因此 W1 可进入严格目标 GBS 的 fit-only calibration；W4 这批只能作为带明确分布标签的吞吐/显存 shadow 证据，不能拟合或验收严格 target-GBS effect。自动下一批与发布 gate 均保持 false。

#### 阶段 D W4 GBS 合同修复补测 closeout（2026-08-04）

修复补测固定 `W4, cutoff=32768, MBS=1, GC=on`，覆盖 `DP1/GBS64/none` 与 `DP2/GBS128/ZeRO-2/3`，每个 U/P treatment 3 次，共 18/18 jobs 成功、0 OOM。Packing 语义、全 rank authoritative ledger 和测量步一致性全部通过。

| 配置 | Packing GA | expected epoch GBS(P) | 8-step probe GBS(P) | Packing 吞吐比（samples/s） | No-Packing→Packing reserved |
|---|---:|---:|---:|---:|---:|
| W4, DP1, GBS64, none | 2 | 61.224 | 50.250 | 2.574× | 26.86→44.13 GiB |
| W4, DP2, GBS128, ZeRO-2 | 2 | 122.449 | 138.625 | 4.273× | 33.58→41.51 GiB |
| W4, DP2, GBS128, ZeRO-3 | 2 | 122.449 | 138.625 | 12.246× | 30.98→39.42 GiB |

三组配对重复的吞吐倍数范围分别为 `2.568～2.580×`、`4.245～4.296×` 和 `12.179～12.299×`，重复性良好。这里的 target-GBS 门禁依据完整上传画像的 epoch 期望与 step-P99 上界；8 个 measured steps 的 probe 均值只用于观测短窗口波动，不能替代 epoch 合同。Packing 在三组里都提高 reserved 显存约 `7.93～17.27 GiB`，因此吞吐收益不能被解释成“Packing 不增加显存”。

修复批可进入严格 target-GBS 的 fit-only calibration；原先 `DP2,target_GBS64` 的 W4 第一批仍只能保留为 shadow 证据。ZeRO-3 的 Packing 相对收益显著大于 ZeRO-2，当前 log-ratio interaction 为约 `1.053`，后续建模必须保留 Packing×ZeRO 交互或分支 residual，不能用一个全局 Packing 常数覆盖。自动发布与自动下一批 gate 仍为 false。

#### DataProfile estimator v1 closeout（2026-08-04）

本轮没有重复 tokenize，而是复用 DataProfile v2 已冻结的 exact static pack 曲线，整理出 W1～W9 共 225 个 cutoff 标签。模型只使用 non-truncating 点：W1/W2/W3/W4/W5/W7/W8 共 102 点作为 calibration，W6 保持 diagnostic-only，W9 的 20 点作为一次性 source-disjoint prospective evaluation。W3/W5/W7/W8 本轮新增整理的 non-truncating exact 标签分别为 18/11/11/21 点，共 61 点。

utilization 使用 aggregate profile 连续特征的 ridge head；`n_pack_mean` 不另建黑盒，而由 `capacity×predicted_utilization/mean_length` 物理恒等式派生；step-P99 使用独立 ridge head并由 family-LOO 残差形成 safety upper。

| 指标 | family-LOO calibration | W9 prospective | 状态 |
|---|---:|---:|---|
| utilization center relative MAE | 4.27% | 1.72% | 通过 ≤5% |
| n_pack_mean center relative MAE | 4.27% | 1.72% | 通过 ≤5% |
| n_pack_mean safety interval coverage | 100% | 100% | 通过 ≥95% |
| n_pack_step_p99 center relative MAE | 42.69% | 34.32% | 不满足自动剪枝锐度 ≤25% |
| n_pack_step_p99 upper coverage | 100% | 100%，0 次低估 | 安全 coverage 通过 |

P99 coverage 依赖 `2.570×` upper multiplier，虽然 false-safe 为 0，但会造成明显的 false-reject。因此 exact cached curve 仍是正式推荐的首选；estimator 只允许缓存未命中时做 shadow inference，`automatic_gbs_candidate_pruning_allowed=false`、`automatic_packing_recommendation_allowed=false`。仅增加非线性模型不能解决这个问题：探索性 tree/boosting head 虽改善 center，family-LOO 所需 guard 反而扩大，故未采纳。

GPU evidence membership 同时冻结为 42 个 jobs：W1 与 W4 修复批合计 30 个严格 target-GBS fit-only jobs；原 W4 `DP2,target_GBS64` 的 12 个 jobs 保持 distribution-labeled shadow，禁止进入严格 GBS 系数拟合。

#### W3/W5/W7/W8 信息增益批次完成（2026-08-05）

基于 exact cached curve 和严格 `DP=2,target_GBS=128,ZeRO-2,GC=on` 合同，从 7 个端点候选中用受约束 ridge-logdet 选择 6 个 family：`W3@4096/40960`、`W5@16384`、`W7@20480`、`W8@2048/10240`。每个 family 做 U/P×3 counterbalanced 配对，共 36 jobs、72 GPU-job equivalents。

修订后的正式队列已完成 `36/36`，成功 `36/36`、OOM `0`；Packing 语义、authoritative ledger、expected/probe GBS 分字段和所有 treatment CV≤5% 门禁均通过，不需要第4次重复。六个 family 的 logical samples/s P/U 为 `2.721×/2.979×/1.228×/1.644×/4.392×/7.856×`；effective tokens/s P/U 为 `2.737×/3.193×/1.397×/1.863×/4.418×/7.709×`。本批仍只估计 MBS=1 matched 机制效应，不能单独解释为最终 route-effect。

fit-only 模型选型将既有严格证据与 Phase B 合并，49 个 repeat pairs 折叠为 19 个配置；GC-off 与 ZeRO-3 因各只有一个独立画像，保留为诊断而不拟合通用交互。17 个主配置、10 个画像组的嵌套画像留一结果显示：低自由度 spline-GAM 将 effective/logical effect 的 group-equal log-MAE 从 Ridge 的 `0.516/0.473` 降至 `0.230/0.236`，但 effective-token 目标出现额外 Top-1 regret。因此当前只能把 GAM 保留为 effect-center challenger，生产吞吐模型仍需 absolute＋pairwise 两头选型，不能把本轮 center 胜负直接发布为 ranker。

#### 安全 Phase C 交互矩阵（2026-08-05）

后续执行计划中的 Phase C 使用 Phase B 的 `ZeRO-2,GC-on,U/P×3` 作为同画像基线，并新增 24 个 U/P jobs。交互目标是 `log((P/U)_new/(P/U)_PhaseB)`，而不是直接比较不同画像的绝对吞吐。

| 交互轴 | 新执行臂 | 新 jobs | 冻结显存结论 |
|---|---|---:|---|
| Packing×ZeRO | W7@20480、W3@40960 的 `ZeRO-3,GC-on` | 12 | 最大 guarded upper 56.69 GiB，通过 |
| Packing×GC | W3@4096、W8@10240 的 `ZeRO-2,GC-off` | 12 | 最大 guarded upper 110.72 GiB，通过 |

原拟用于 GC 轴的 W7@20480 与 W3@40960 在 GC-off 下 operational P95 分别达到约 188～204 GiB 和 362～392 GiB，超过 H800 140 GiB 实测容量的 90% 线，因此明确排除。该结果来自冻结显存模型的执行预检，是模型预测与工程门禁，不是 OOM 实测。由此产生的支持域限制是：本批可以估计当前低/中 cutoff 的 GC 交互，但不得外推长 cutoff GC-off；若产品必须覆盖该域，需要先构造新的安全执行策略并单独标定。

### 阶段 E：Packing 显存边界

原 40-job 交互实验、18-job 修复补测与 Phase B 36-job 合计 94/94 成功，且该证据集合的峰值最高仍约 44.13 GiB；这些点没有逼近 80 GiB 容量边界，不能据此标定 80 GiB 卡的 false-safe OOM 边界。

先用阶段 C/D 的同 cutoff、MBS=1 Packing off/on 配对拟合共享物理主干。每个点必须记录 configured cutoff、observed active tokens、max/mean segment length、`Σsegment_len²`、kernel workspace、`max_memory_allocated/reserved`。按顺序比较：

1. 只用 configured cutoff；
2. 改用 active tokens、attention pairs、GC 和模型几何；
3. 再增加 segments/kernel workspace 和 Packing residual。

只有第 3 步在 family holdout 上仍显著降低残差时才保留 Packing 中心 residual；无论中心 residual 是否保留，安全 upper guard 都单独验收。该设计用于区分“Packing 本身耗显存”和“Packing 把 cutoff 填满导致 active tokens 增加”。

本阶段不是全笛卡尔积，只在以下触发条件出现时执行：

- Packed memory center/upper 残差随 cutoff 或 profile 系统偏移；
- prospective P95 coverage<95%；
- false-safe OOM>0；
- Packing workspace 无法从已有点辨识。

触发后围绕预测占用 70%～98% 容量选择 8～12 个 boundary family：

- 8B/14B；
- Full/LoRA；
- cutoff 4K/8K/16K/32K；
- 1/2/4 GPU；
- ZeRO-2/3；
- GC on/off；
- Packing on/off。

每个 family 先取明显安全点和接近安全线点；CUDA OOM 作为右删失证据，不伪造峰值。预计 24～48 jobs，仅在门槛触发后审批。

### 阶段 F：Packing-aware 扩卡比例

至少覆盖四条 scale chain：

| Chain | 画像 | 模型/训练 | 卡数 |
|---|---|---|---|
| S1 | 极短/高 pack 密度 | 8B LoRA | 1→2→4 |
| S2 | 自然多轮 | 14B LoRA | 1→2→4，按可行性 |
| S3 | 宽长尾 | 14B Full | 2→4 |
| S4 | 近 cutoff 饱和 | 8B Full | 2→4 |

每条 chain 在 Packing off/on 下分别计算：

```text
strict_matched_ratio:
  数据顺序、cutoff、Packing状态、目标GBS、GC、ZeRO、runtime一致
  No-Packing 固定MBS，GA随DP精确派生
  Packing 固定n_pack画像；N端GA必须可被2整除，2N端GA减半
  两端expected_GBS和tokens/optimizer_step保持一致

best_tuned_ratio:
  数据、Packing状态、目标GBS和质量合同一致
  每个卡数内独立选择最佳安全cutoff/MBS/derived_GA、GC和ZeRO
```

strict matched 的 Packing 比较要求 N 端派生 GA 可减半；若 N 端已经 `GA=1`，扩到 2N 后无法保持相同 token/optimizer-step 和期望样本 GBS，该端点不构成 strict chain。best-tuned ratio 衡量扩卡后重新选 cutoff/执行配置的实际产品收益，必须与 strict ratio 分开报告，不能把 cutoff 或预期 GBS 变化称为线性度。

自动扩卡只使用：

```text
lower_throughput(2N) / upper_throughput(N) >= 1.8
```

即相邻翻倍保守并行效率至少 90%。如果点估计≥1.8但区间跨阈值，输出 `uncertain`；如果 N 卡 OOM 而 2N 卡可行，只能声明最低资源为 2N，不能计算 N→2N 线性度。

预计新增 12～24 jobs；优先复用阶段 D 的端点。

### 阶段 G：冻结模型后的 prospective holdout

模型、特征、阈值、support gate、family、数据 revision、执行顺序和 evaluator 全部冻结后，至少运行 4 个 source-disjoint family：

| Holdout | 来源 | 主要检查 |
|---|---|---|
| H1 | OASST2 | 新自然多轮/多语言迁移 |
| H2 | Orca Math | 长 label、推理输出迁移 |
| H3 | Magicoder 或等价 source-disjoint code | 结构化/代码迁移 |
| H4 | LongAlign 或许可通过的长上下文源 | 8K～64K 迁移 |
| H5 可选 | 全新业务 S3/Blobstore | 真实业务 source holdout |

每个 family 至少做 `best unpacked / best neat-packed` 各 3 次；若要验收 MBS=1 paired head，再加 `N-C-1-gN` 三次。每个 Packed holdout 只向推荐器提供冻结 DataProfile，先通过 utilization interval 和三项硬门槛；不得在推荐时偷看 holdout 原始 lengths。预计 24～36 jobs。Holdout 不得回写本版系数或阈值。

### 阶段 H：RTX 4090 独立 track

H800 通过后再启动：

- 新 card/runtime fingerprint；
- 6～8 个 Packing calibration family；
- 至少 3 个 source-disjoint holdout family；
- 1→2、2→4 独立 scale chain；
- 卡间 PCIe/拓扑作为 ratio 输入。

4090 可以共享物理公式和数据画像，但必须拥有独立的 card adapter、Packing residual、memory upper guard 与 scale ratio 验收。预计 36～60 个 calibration/holdout jobs，扩卡端点另行复用。

## 10. 建模方案

### 10.1 共享物理基础

继续复用现有结构化物理 basis：

```text
model/optimizer state
linear FLOPs
attention FLOPs based on ΣL²
activation/HBM traffic
optimizer HBM traffic
launch overhead
communication volume
hardware FLOPs/HBM/link ratios
```

模型名、数据集名、具体 GPU 商品名不作为任意倍率键。模型通过结构参数进入，硬件通过容量、带宽、峰值算力和互联比率进入；卡型仅允许有受正则约束的 adapter。

### 10.2 显存模型

建议形式：

```text
M_off = M_shared_state
        + M_activation(T_active_off, model_geometry, GC)
        + M_attention(attention_pairs_off, kernel)
        + residual_off(selector, profile)

M_neat = M_shared_state
         + M_activation(T_active_pack, model_geometry, GC)
         + M_attention(attention_pairs_pack, kernel)
         + M_varlen_workspace(segments, packer, kernel)
         + residual_pack(selector, profile)
```

`M_shared_state`、activation 和 attention 的主物理系数优先共用；路径差异通过实际 workload 表达。No-Packing 的配置 cutoff 通常只是上限，`T_active_off` 取动态 batch 的实际/predicted padded tokens；Packing 会把 physical MBS=1 的序列填到接近 cutoff，`T_active_pack` 由 runtime 合同取 `cutoff` 或 `cutoff×fill`。因此 Packing 显存的一阶变化主要来自更长、更满的实际序列，不是一个任意的 `packing=1` 倍率。

Packing 专属部分只拟合 varlen/block-diagonal kernel workspace、segments metadata、allocator fragmentation residual 和 safety upper guard。若加入 `T_active/attention_pairs/segments/kernel` 后 residual 在 prospective 数据上与 0 不可区分，允许把中心 residual 收缩为 0，但 upper guard 在 OOM/coverage 验收前不能删除。

中心模型使用带正则的稳健回归或受限 GAM；安全上界使用 scenario-level conformal/quantile envelope，并按 selector 保留只增不减的 residual guard。CUDA OOM 作为右删失边界；software failure 不进入显存拟合。

最低 Packing memory 特征：

- cutoff 与模型几何；
- predicted/observed active tokens per microstep；
- predicted/observed attention pairs（块对角为 `Σ segment_len²`）；
- pack fill/waste；
- samples/segments per pack；
- varlen/block-diagonal kernel workspace 与 metadata；
- 每步 pack token/sample 分布和 rank token imbalance；
- length CV 与 tail ratios；
- GC、ZeRO、GPU 数；
- sampler、packer、runtime/kernel fingerprint。

GA、总 packs 和总 optimizer steps 不是单 microstep 峰值显存的主特征，只用于训练/吞吐合同；除非实测证明存在持久 buffer 交互，不进入 Packing memory 主公式。

### 10.3 吞吐模型

```text
log T_off  = log T_physical + r_off(x)
log T_neat = log T_physical + r_neat(x, pack_profile)
```

`r_off` 和 `r_neat` 使用独立系数，但通过 hierarchical ridge 向共享物理先验收缩。若数据量不足，不允许增加任意高阶交互；只拟合预注册交互。

Phase B 后的选型规则进一步明确为：Ridge 是必须保留的线性 baseline；受限 spline-GAM 只在连续 cutoff/profile residual 上作为 challenger；生产排序必须另有同场景 pairwise head。平均 center 误差降低不能替代 Top-1 regret、最坏画像误差和 source/profile-disjoint 验收，因此当前 `production_throughput_model_selected=false`。

现有 `FEATURE_NAMES` 已有 `packing` 主效应，但缺少足够的 Packing 交互。新增或显式派生：

- `packing_x_log2_cutoff`；
- `packing_x_mean_length_to_cutoff`；
- `packing_x_length_cv`；
- `packing_x_fill_ratio_center` 与 utilization interval width；
- `packing_x_log_samples_per_pack_mean_center`、`n_pack_mean_upper/center` 与 `n_pack_step_p99`；
- `packing_x_log2_derived_GA`；
- `packing_x_expected_GBS_error`；
- `packing_x_log_opt_steps_center/lower` 与 `packing_x_GBS_controllable_margin`；
- `packing_x_rank_token_imbalance` 与 `packing_x_step_tail`；
- `packing_x_attention_share`；
- `packing_x_gc`；
- `packing_x_zero3`；
- `packing_x_log2_gpu_count`；
- `packing_x_is_lora`。

但生产实现优先采用 block-separated residual head，而不是让一个无约束大回归同时学习所有二值乘积。不同 sampler/packer/runtime fingerprint 不能只靠同一个 Packing 二值项无条件混合；只有经过 bridge 验证后才允许共享受正则约束的 adapter。

### 10.4 Packing paired-effect head

核心目标：

```text
delta_mbs1  = log T(P-C-1-gP) - log T(N-C-1-gN)
delta_route = log T(P-C-1-gP) - log T(N-C-m*-gN)
```

effect head 直接在 ABBA 配对差值上拟合，可抵消机器状态、模型绝对尺度和部分运行时漂移。

输入包括：

- Base DataProfile/CandidateProfile 连续特征与 profile confidence；
- cutoff；
- derived GA、n_pack、expected GBS/error、opt_steps、GBS controllable margin；
- rank imbalance 与 step-tail；
- unpacked best MBS 或 `replaces_no_packing_mbs`；
- GC、ZeRO、GPU 数；
- Full/LoRA 与模型结构；
- hardware/runtime ratios。

输出包括中心、95% 区间和 decision support status。产品流程先在两个分支内分别选出最佳质量合格、memory-safe 候选，再用 paired route effect interval 检查分支切换是否稳健。自动开启门槛：

- 原最佳 unpacked MBS=1：收益下界≥10%；
- 原最佳 unpacked MBS>1：收益下界≥20%；
- `n_pack_step_p99×DP≤target_GBS×(1+epsilon_GBS)`；
- `opt_steps_lower≥K_min_steps`，且期望/实际样本级 GBS 波动位于已验证支持域；
- 语义检查100%；
- Packed false-safe OOM=0。

若 Packed 与 Unpacked 的预测区间重叠、收益下界未过线或 DataProfile/运行时超出支持域，不根据画像 fill ratio 点估计强行开启 Packing；对两分支 Top 1～3 候选各执行同合同短跑。短跑仍无法区分时默认 No-Packing，并报告 `uncertain_no_pack_fallback`。

### 10.5 扩卡 ratio head

按 Packing 状态分别拟合：

```text
log_ratio = log T(2N) - log T(N)
```

输入必须绑定：

- 相同模型、数据 snapshot、cutoff、Packing 状态、目标 GBS；
- from/to GPU count；
- topology/collective/runtime fingerprint；
- 两端选择的 ZeRO、GC、MBS、derived GA；
- 两端 expected GBS、tokens/optimizer-step 和 opt_steps；
- compute/communication 与 pack workload ratios。

输出中心、lower/upper bound。只有保守下界和 fresh measured lower bound 均≥1.8，才允许自动扩卡。

### 10.6 cutoff 与最终排序

Packing cutoff 候选都必须不截断真实样本；在块对角 attention、position 和 loss mask 语义通过后，不同 cutoff 才能视为同一训练数据的不同物理装箱方式。最终排序顺序：

1. `cutoff≥C_data_min`、模型上下文与 runtime 支持门控；
2. 块对角 attention/loss/position、GBS 可控和总 optimizer steps 门控；
3. memory P95/upper 门控；
4. No-Packing 与 Neat-Packing 分支内分别按有效训练吞吐排序；
5. 用 paired route effect interval 比较两个分支的最佳安全候选，必要时触发短跑；
6. 通过收敛质量支持域后按预计 epoch time 排序；
7. 输出 epoch time、GPU-hours、费用、显存余量、expected GBS 和 opt_steps Pareto；
8. 用户未给定质量/费用权重时，不压缩为任意单一分数。

“有效训练吞吐”优先使用有效 label tokens/s，并同时报告 logical samples/s、computed tokens/s、tokens/optimizer-step 和 step time。吞吐最高但 `opt_steps_lower<K_min_steps` 或 GA=1 时 `n_pack_step_p99×DP` 已不可控的长 cutoff 必须淘汰，不能靠性能排序救回。

## 11. 训练、验证与防泄漏

### 11.1 数据划分

- calibration 与 holdout 按 upstream source 隔离；
- 同一 dataset 的 row split 只能叫 row holdout，不能叫 source-disjoint；
- 同一 model/data/cutoff/GBS 的 repeats 必须全部落在同一 fold；
- 同一 packer/sampler/runtime version 的 repeats 必须全部落在同一 fold，不同机制不得互相充当 repeats；
- ABBA pair 不得跨 fold；
- scale chain 的相邻端点不得跨 fold；
- 公布前执行 leave-one-source-profile-out 和 leave-one-model-scale-out。

### 11.2 拟合顺序

1. 冻结 shared physical basis；
2. 拟合 Unpacked heads，确认既有能力不退化；
3. 拟合 Neat-Packing memory/throughput heads；
4. 拟合 paired-effect head；
5. 拟合 scale-ratio head；
6. 冻结全部 artifact；
7. 运行 prospective holdout；
8. holdout 只验收，不回写。

### 11.3 模型复杂度门槛

- 每个自由 interaction 至少有 3 个独立 family 支撑；
- 设计矩阵必须满秩，条件数超过预注册门槛时先减少交互；
- 优先 ridge/hierarchical shrinkage，不用 dataset dummy；
- 若 residual 明显非线性，只允许在 cutoff/profile 连续特征上增加受限 spline；
- 不因一个 holdout 失败给某个 dataset ID 增加专属倍率。

## 12. 验收门槛

| 模块 | 指标 | 门槛 |
|---|---|---:|
| 语义 | block-diagonal attention/loss mask/position 检查 | 100%通过 |
| 语义 | optimizer boundary、consumed/preload ledger | 100%一致 |
| DataProfile | `N/mean/max/histogram/revision` 完整性 | 自动推荐100%满足 |
| Utilization | `U/n_pack_mean/packs` 中心误差 | 预注册门槛内 |
| Utilization | epoch-mean `n_pack` safety interval coverage | ≥95% |
| Pack-count 上尾 | `n_pack_step_p99` safety coverage | ≥95%，且 prospective 不低估 |
| GBS 可控 | `n_pack_step_p99×DP≤target_GBS×(1+epsilon_GBS)` | 100%候选满足 |
| GBS 可观测 | `expected_epoch_GBS/observed_probe_window_GBS/step distribution` | 100%分开报告 |
| 优化器步数 | `opt_steps_lower≥K_min_steps` | 100%候选满足 |
| 收敛质量 | GBS 波动及近 K cutoff 的质量非劣性 | 通过预注册 recipe-specific 门槛 |
| 显存 | prospective successful P95 upper coverage | ≥95% |
| 显存 | Packed false-safe OOM | 0 |
| 显存 | memory-gated safety failure | 0 |
| 显存 | center MAPE（次要） | ≤20% |
| 吞吐 | 同卡数 Top-1 regret | ≤10% |
| 吞吐 | Hit@90% | 100% |
| Packing | 原 MBS=1 的 paired gain 95%下界 | ≥10% |
| Packing | 原 MBS>1 的 paired gain 95%下界 | ≥20% |
| Packing | profile/predicted false-enable | 0 |
| Packing 决策 | 预测区间重叠/OOD 时的未验证自动开启 | 0 |
| 推荐响应 | 已缓存画像、≤100候选的 profile-based 搜索 P95 | ≤2秒 |
| 冷数据 | tokenize 未完成时阻塞在线请求 | 0；返回 `profiling_pending/provisional` |
| 扩卡 | `lower(2N)/upper(N)` | ≥1.8 |
| 扩卡 | fresh measured lower bound | ≥1.8 |
| 扩卡 | false-positive claim | 0 |
| 支持域 | DataProfile/tokenizer/template/packer/runtime fingerprint | 完整绑定 |

若只通过 calibration 而未通过 source-disjoint holdout，Packed 候选继续为 `shadow_only`。若 only H800/某一模型训练方式通过，只发布窄支持域，不扩大文字描述。

## 13. 停止、追加与失败规则

### 13.1 停止规则

- Base DataProfile 不完整/置信区间过宽，或 cutoff 候选未通过显存、GBS 可控、`K_min_steps` 任一门槛时，不启动对应 Packed GPU 候选；
- 块对角 attention、loss/position 或 consumed ledger canary 失败时停止全部 Packed 后继；
- `epsilon_GBS/K_min_steps` 尚未完成收敛质量冻结时，Packed 结果保持 shadow，不自动发布；
- 某条 scale doubling 的 ratio 下界<1.8 时停止继续自动翻倍；
- 当前 cutoff 已被质量门控拒绝时，不参与胜负统计；
- 某 family 已有安全点、近边界点且区间收敛时停止显存加点；
- 差异小于噪声时增加重复，不增加模型自由度；
- 新残差集中在单一机制时补机制对照，不重铺全矩阵。

### 13.2 失败分类

| 类型 | 处理 |
|---|---|
| CUDA allocator OOM | 有效右删失边界 |
| software/runtime failure | 修复后按原 job 合同重跑，不进入模型 |
| infrastructure failure | 不进入模型，不推进后继 |
| 数据/template 错误 | 原 attempt 作废，重新冻结 profile 和 job |
| fingerprint/hash 不一致 | incomplete，不可用于 calibration |
| Packing 语义错误 | 阻断该 packer/runtime fingerprint 的发布 |

## 14. 预计实验规模

原 40 次真实业务实验、本次 18 次 W4 GBS 修复补测和 C1～C3 的有效 ABBA 证据，在 execution fingerprint 与各自 GBS 合同一致时作为当前 Packing 路线的 fit-only calibration；原 W4 `DP2,target_GBS64` 不合格点仍仅作 shadow，C4 完成语义 closeout 前也不可进入联合拟合。新增运行按信息增益触发：

| 阶段 | 新增 H800 jobs | 说明 |
|---|---:|---|
| A CPU画像/DOE | 0 | 上传期 DataProfile、calibration pack 标签和 utilization estimator |
| B 语义/C4/instrumentation | 4～10 | C4 可重建时较少；否则含完整6-job重跑 |
| C DataPlan曲面 | 30～40 | 新画像；与 D 可复用 |
| C-Q 收敛质量标定 | 12～24 长训练 | 冻结 `epsilon_GBS/K_min_steps`，不与短 probe 等价计数 |
| D 执行参数交互 | 36～60 | D-optimal 选择，不跑全笛卡尔积 |
| E 显存边界 | 条件性24～48 | 只有残差/coverage触发 |
| F 扩卡比例 | 增量12～24 | 尽量复用 C/D 端点 |
| G prospective holdout | 24～36 | 不回写模型 |

通过现有 Packed/No-Packing 点以及 C/D/F 端点复用，首轮吞吐、显存和扩卡性能包络预计为新增约60～90个 H800 jobs；另有 12～24 个不能用短 probe 替代的收敛质量 jobs，用于冻结 `epsilon_GBS/K_min_steps`。只有显存安全残差触发时，再增加24～48个 boundary jobs。最终物化前必须由 DOE 选择器给出精确唯一配置数、训练时长和 GPU-job-equivalent，不能直接把上表上限相加。

RTX 4090 track 不计入 H800 首轮包络，单独预计36～60个 calibration/holdout jobs，并另行复用扩卡端点。

## 15. 需要修改的代码与产物

### 15.1 代码

| 组件 | 计划动作 |
|---|---|
| `candidate_generator.py` | 支持分段粗网格、GA/显存/GBS/K 事件点与局部512精搜；No-Packing 精确派生 GA，Packing 按 `n_pack_mean_center` 派生 GA |
| `packing_aware_candidates.py` | 在 Packing 验收后把 in-domain packed 从 shadow 升级为 rankable |
| cutoff/GBS/steps gate | 联合计算显存上界、`n_pack_step_p99×DP` GBS 可控性、`n_pack_mean_upper` 派生的 packs/optimizer-steps 区间和三项淘汰理由 |
| consumed/loss contract | 逐 step 记录逻辑样本、物理 pack、token、有效 label denominator、optimizer boundary；排除 prefetch |
| `structured_throughput_modeling.py` | 增加分头 residual、Packing profile 特征和预注册交互 |
| `throughput_predictor.py` | 消费 Packing heads、effect interval 和 support gate |
| H800 memory modeling | 共享 active-token/attention 主干，增加 Packing workspace/residual 与独立 upper guard |
| `cross_card_scaling.py` | 复用现有 policy，接入 Packing-aware conservative bounds |
| 上传期 DataProfile pipeline | streaming tokenizer aggregation；冻结 N/mean/max/quantiles/histogram/labels/turns 与 revision/hash |
| Utilization/pack-count estimator | 由 DataProfile/上传期缓存预测 `U/n_pack_mean/packs` 中心区间及 `n_pack_step_p99` 上界，不读取原始数据 |
| 联合决策器 | 分支内 Top-K 排序，再用 paired route interval 比较；区间重叠/OOD 时生成双分支短跑或 No-Packing fallback |
| 新 DOE selector | 从合法候选 universe 选择满秩、平衡、信息量最大的 family |
| evaluator | 同时报告机制效应、路线效应、cutoff交互、显存和扩卡验收 |

### 15.2 冻结产物

- `packing_data_source_manifest_v2.json`；
- `packing_data_profile_schema_v1.json`；
- `packing_base_profiles_v4.jsonl`；
- `packing_candidate_profiles_v4.jsonl`；
- `packing_utilization_model_v1.json`；
- `packing_utilization_acceptance_v1.json`；
- `packing_decision_contract_v2.json`；
- `packing_experiment_design_v4.json`；
- `packing_candidate_universe_v4.jsonl`；
- `packing_doptimal_selection_v4.json`；
- `packing_memory_head_v4.json`；
- `packing_throughput_head_v4.json`；
- `packing_effect_head_v4.json`；
- `packing_scaling_ratio_head_v4.json`；
- `packing_prospective_acceptance_v4.json`；
- source/calibration/holdout membership ledger；
- rollback 指针与上一版 artifact SHA。

## 16. 最终产品决策流程

```text
用户请求
  ↓
读取已缓存的 Base DataProfile；不读取原始数据
  ↓
读取 model/card候选/目标GBS/epochs 与训练合同
  ↓
并行生成两个条件分支
  ├─ No-Packing：cutoff × MBS；精确派生GA；显存门控
  └─ Neat-Packing：
       计算C_data_min；粗网格+事件点+局部512精搜
       → DataProfile/cache得到U、n_pack mean区间与step-P99上界
       → 派生GA/expected_epoch_GBS/tokens-per-step/opt_steps区间
       → memory upper + n_pack step-P99 + opt_steps lower三项硬门槛
  ↓
两个分支都展开 card type/count × ZeRO × GC × MBS/derived_GA
  ↓
两个分支内分别做同卡数 Top-K 吞吐排序
  ↓
Packing paired route-effect interval gate
  ↓
预测重叠或 OOD → 双分支短跑；仍不确定则 No-Packing fallback
  ↓
固定 cutoff/Packing/目标GBS的相邻扩卡 ratio gate
  ↓
epoch time / GPU-hours / cost / retention Pareto
  ↓
推荐 + support_status + confidence + fallback reason
```

### 16.1 语言化决策说明

数据集上传后，平台异步执行一次 tokenizer/template 和 streaming aggregation，冻结包含 N、mean、max、分位数、长度直方图、label/turn 画像及 revision/hash 的 Base DataProfile。它不要求永久保存全量 token IDs。训练推荐请求只读取这份画像；画像尚未 ready、缺失关键字段或版本不匹配时返回 `profiling_pending/shadow`，不得在请求路径重新 tokenize 原始数据。

收到推荐请求后，系统固定模型、DataProfile、训练模式、目标样本级 GBS、epochs，以及可选卡型卡数。这些信息缺一不可：画像决定平均长度和装箱率区间，卡数决定全局一次 microstep 会消费多少 pack，epochs 决定最终还有多少 optimizer steps。

对 Packing 分支，先从 DataProfile 的真实/保守 max 得到 `C_data_min`，向上取整到 512。然后针对每一种卡型、卡数、ZeRO 和 GC，先走分段粗网格，并强制加入 GA 跳变、显存上界、GBS 可控上界和 `K_min_steps` 边界；只在 Top 候选和跨门槛位置用 512 局部精搜。cutoff 不能直接拉到模型最大上下文，因为每增加一档都会同时改变显存、每 pack 样本数、GA 和总 optimizer steps。

每个 cutoff 都优先读取上传期按相同 packer fingerprint 缓存的 utilization 与 samples-per-pack 分布，而不是重新 pack 数据。缓存未覆盖该 cutoff 时，系统才用 mean、长度直方图、CV、P90/P99、长尾/双峰、turns/segments 和 packer fingerprint 预测 `U_lower/center/upper`、`n_pack_mean` 区间及 `n_pack_step_p99` 上界。mean 中心派生 GA 和性能预测，mean 上界用于 optimizer-steps 下界，step-P99 专门用于 GBS 安全门控。

接下来对同一个 cutoff 同时做三项硬判断。第一，共享显存物理模型给出的 upper 必须低于该卡容量的安全 margin；第二，即使 GA 已经是 1，`n_pack_step_p99×DP` 也不能超过目标 GBS 的允许范围；第三，由 `n_pack_mean_upper` 得到的保守 optimizer-steps 下界必须不少于 `K_min_steps`。任意一项失败，这个 cutoff 都直接淘汰。

所有硬门槛通过后，Packing 吞吐模型对存活的 cutoff 和执行配置预测有效吞吐、epoch time、显存余量与置信区间；在 Packing 分支内部选择预测吞吐最高的安全候选。与此同时，No-Packing 分支按原逻辑独立搜索 cutoff、MBS、GA、卡型卡数、ZeRO 和 GC，得到自己的最优安全候选。

最后比较两个分支，而不是仅凭 Base DataProfile 决定是否打开 Packing。只有 Packing 最优候选的收益下界超过发布门槛，且块对角 attention、GBS 波动、optimizer steps、显存和支持域全部通过，系统才自动开启 Packing。两边预测区间重叠或遇到新数据分布时，对各自 Top 候选做短跑；短跑仍不能可靠区分时默认关闭 Packing，并返回不确定原因。

卡数变化时要对每个卡数重新执行这套流程，因为 DP 会改变派生 GA 和 GBS 可控上界。只有 cutoff、数据、Packing 和 token/optimizer-step 都匹配，且 N 卡 GA 能在 2N 卡减半时，才把 N→2N 称为严格扩卡线性度；重新选择 cutoff/GA 后得到的提升只能称为 best-tuned 扩卡收益。

输出必须解释：

- 为什么选择这个 cutoff；
- Packing 替代了哪个 unpacked MBS；
- 绑定的 DataProfile revision、完整度和 utilization support status；
- 目标 GBS、派生 GA、`expected_epoch_GBS`、`observed_probe_window_GBS` 及逐 step 区间；
- cutoff、utilization、`n_pack_mean` 中心/置信界、`n_pack_step_p99/max`、tokens/optimizer-step；
- packs/opt_steps 中心与安全下界、K_min_steps 余量和三项 gate 结果；
- 显存中心、上界和余量；
- epoch time 与 GPU-hours；
- 扩卡 ratio 与并行效率；
- 是否位于已验证数据/模型/硬件支持域。

## 17. 执行顺序与下一步

后续分阶段补实验的精确目标、最小矩阵、预算和门禁见 [`Packing后续补充实验执行计划_2026-08-05.md`](./Packing后续补充实验执行计划_2026-08-05.md)。该补充计划同样不构成 GPU 执行或自动发布授权。

截至 2026-08-04，本节第 1～2 项和 W4 最小修复补测均已完成：W1～W9 共生成 9 份 schema-v2 画像和 2,025 个 CPU 静态候选；W4 的 `DP=2,target_GBS=64` 合法候选为 0，`DP=1,target_GBS=64` 与 `DP=2,target_GBS=128` 各有 2 个。对应 18-job GPU 补测在不抢占外部任务的 H800 0/1/5/6/7 上完成，18/18 success、0 OOM、全部语义门禁通过。

1. 冻结 `DataProfile/pack-count schema v2`：上传期缓存每个 packer fingerprint、候选/事件 cutoff 的 `{U, samples_per_pack mean,p50,p90,p95,p99,max,packs}`，推荐请求禁止读取原始 lengths；
2. 对当前 W1/W4 和 W1～W9 画像做 CPU-only cutoff×DP×target-GBS 复筛，生成修正后的候选 universe；任何 `n_pack_step_p99×DP>G×(1+epsilon_GBS)` 的 Packed arm 在入队前淘汰。W4 已完成首轮复筛：在不截断区间 `29184～32768` 内，`DP=2,target_GBS=64` 无 Packed 候选通过；
3. 将已完成阶段 D 证据按合同拆分入模：W1 进入严格 target-GBS fit-only；W4 原 `DP2,target_GBS64` 第一批仍为 shadow；W4 修复批的 `DP1,target_GBS64` 与 `DP2,target_GBS128` 进入严格 target-GBS fit-only。必须保留 Packing×ZeRO 交互，并把 probe-window GBS 与 epoch expected GBS 分字段建模；
4. 已完成最小 GPU 修复设计与执行：`W4, cutoff=32768, DP=1, target_GBS=64, GA=2` 及 `W4, cutoff=32768, DP=2, target_GBS=128, GA=2, ZeRO-2/3` 共 18 jobs；二者完整画像的 expected epoch GBS 为约 `61.224/122.449`，step-P99 全局上尾为 `60/120`，在 10% 合同下可行。如果产品合同固定 `GBS=64, DP=2`，则该分支仍直接 No-Packing fallback，不为保留 Packing 放宽 gate；
5. 已补齐并冻结 W3/W5/W7/W8 的 61 个 non-truncating exact pack 标签，完成 utilization、`n_pack_mean` 与 `n_pack_step_p99` estimator/coverage；utilization/mean 和 P99 safety coverage 通过，但 P99 guard 锐度未过，自动 GBS 剪枝保持关闭；
6. 已完成 6-family、36-job Phase B，并完成 Ridge/GAM fit-only 诊断；生产吞吐模型和 Packing 显存 center/upper 尚未达到发布门槛；
7. 已物化安全 Phase C 24-job queue 和显存预检：W7@20480/W3@40960 测 ZeRO，W3@4096/W8@10240 测 GC；下一步完成 provenance、独立审批与 scheduler dry-run 后，仅在真实空闲卡启动；
8. Phase C 完成后重拟合交互 residual，再执行显存边界、扩卡和收敛质量子实验，冻结 `epsilon_GBS/K_min_steps`、多头模型和双分支阈值；
9. 最后冻结并执行 prospective holdout；只有全部发布门槛通过，才解除 `packing_not_validated` 和 `scale_out_enabled=false`。

## 18. 参考与现有证据

### 本地证据

- `Packing决策逻辑V2_修订版_2026-08-04.md`（本版 Packing cutoff/GBS/optimizer-steps 决策基准）
- `真实业务Packing_Cutoff_MBS交互实验计划_2026-08-04.md`
- `offline_experiments/artifacts/h800_real_business_packing_cutoff_mbs_results_v1.md`
- `offline_experiments/artifacts/h800_packing_calibration_evaluation_v1.json`
- `offline_experiments/artifacts/h800_packing_platform_v4_interactions_batch1_continuation_combined_results_v1.md`
- `offline_experiments/artifacts/h800_packing_platform_v4_interactions_batch1_gbs_contract_correction_v2.md`
- `offline_experiments/artifacts/packing_data_profile_schema_v2.json`
- `offline_experiments/artifacts/packing_cutoff_dp_gbs_screen_w1_w9_v2.md`
- `offline_experiments/artifacts/h800_packing_gbs_repair_batch_results_v1.md`
- `offline_experiments/artifacts/packing_fit_membership_v1.json`
- `offline_experiments/artifacts/packing_profile_estimators_acceptance_v1.md`
- `offline_experiments/artifacts/packing_information_gain_design_v1.md`
- `offline_experiments/artifacts/h800_text_neat_packing_strict_acceptance_design_v1.md`
- `../01_项目总览与使用指南/通用参数配置模型实验与建模计划_2026-07-31.md`
- `offline_experiments/artifacts/dataset_analysis.json`
- `offline_experiments/artifacts/static_workload_profiles.json`

### 外部数据卡

- [OpenAssistant OASST2](https://huggingface.co/datasets/OpenAssistant/oasst2)
- [Microsoft Orca Math Word Problems 200K](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k)
- [NVIDIA OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct)
- [Magicoder OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K)
- [THUDM LongAlign](https://github.com/THUDM/LongAlign)
- [Salesforce xLAM Function Calling 60K](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
