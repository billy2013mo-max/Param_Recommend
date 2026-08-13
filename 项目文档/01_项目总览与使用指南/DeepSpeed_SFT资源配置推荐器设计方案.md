# DeepSpeed SFT 资源配置推荐器设计方案

> 目标:为普通 SFT(走 DeepSpeed 纯 ZeRO 数据并行)提供"推荐训练资源 + 分布式配置"的建模引擎,
> 复刻现有 Megatron/CPT `strategy` 模块的"建模 + 实验校准"方法论,但面向 ZeRO 而非 TP/PP。
>
> 状态:理论优先版设计已定稿(2026-07-21 经逐项 grilling 确认)。第一阶段只使用已完成且
> provenance 可追溯的 H800 实验校准有物理意义的实现参数和预测区间,不得替代解析模型。
> 4090 流水线尚未完成,当前只定义静态硬件 profile 与数据接口；不得吸收部分结果或标记为 calibrated，
> 待实验收口后按同一套独立 holdout 门槛单独完成。
>
> 实现进展(2026-07-22,离线 CPU-only 审计,`offline_experiments/scripts/h800_theory_calibration.py`
> + `h800_evidence_gap.py`,固定 `theory_only`/`bootstrap`/`nonpublishable`,绝不生成生产 profile):
> 全局场景 LOOCV + **内层场景折外(OOF)**安全区间(显存严格 LOSO、吞吐有界 K 折);显存尾部
> 成功残差=精确事件、OOM=右删失 KM,统计 P95 与运行护栏 `max(censor_lower, safe_limit+1)−center`
> 分开;吞吐排序/1.8× 扩卡先经 `预测 P95 ≤ 0.95×容量` 联合门控;拟合前独立校验 basis 摘要+schema、
> 原子写、绑定实现/折哈希、拒绝 NaN/Inf。**历史 bootstrap 明确未达发布门槛**:显存 P95 覆盖
> 0.9289/0.9063(<0.95)、假安全 OOM 2 例均为 `LoRA+ZeRO-3`(中心估低)、吞吐 top-1 regret 0.203
> (>0.10)、存活的 1.8× 声明 0、无 native-v2 已验证锚点。缺口报告按机制点名并 `creates_gpu_queue=false`。
> readiness 因被绑定的 `EXPERIMENT_DESIGN.md` 在证据冻结后被改过而 **fail-closed 保持 v5**;
> 刷新 v6 属需单独授权的成套操作(定稿文档→重生成 recovery→重跑 readiness),不通过绕过校验来做。

---

## 1. 背景:现有两条链路

| 链路 | 代码 | 方法 | 产出 |
|---|---|---|---|
| **Megatron / CPT** | `pkg/jobmanager/strategy/` | 建模 + 37 次 H800 实测校准;网格搜索 TP/PP/DP/CP/EP/mbs,建显存+计算+通信模型,按 step time 排序 | 完整分布式方案 |
| **DeepSpeed / SFT(现状)** | `recommender/` + `estimator/` | 粗糙经验估算:模型状态固定 16P,激活按模型规模分桶;`gpuCards=ceil(state/(mem-act))`;ZeRO stage 硬编码 zero3 | 仅卡数,无分布式配置 |

本方案 = 用与 Megatron 引擎同等严谨的建模,替换 DeepSpeed 链路的粗糙估算器。

---

## 2. 已确认的核心决策(决策树)

| # | 决策点 | 结论 |
|---|---|---|
| A | 方法论 | **新建 DeepSpeed 专用建模模块**,复用 `strategy` 原语(config.json 参数量、维度激活、ring 通信时间、cluster 拓扑),替换并行网格为 ZeRO 搜索 |
| X | batch 口径 | **GlobalBatchSize 为不变量**,引擎接管 micro-batch;`gradAccum = globalBS/(mbs×N)` 反推。与 CPT 口径统一。需新增 SFT 前端收 GlobalBatchSize 的契约 |
| 路线2 | 是否建吞吐 | **建吞吐 + step-time 排序**(不止显存可行性)。**不含 offload**(砍掉 host/PCIe/CPU-Adam profile 与校准) |
| P | ZeRO stage | **进搜索网格** {1,2,3},由 step time 排序自然选出 |
| 候选3 | 优化目标 | **最小安全卡数优先 + 性价比爬坡**:先推荐最小可行卡数;翻倍卡数的保守吞吐提升达到 **1.8×** 才推荐扩卡 |
| — | 搜索维度 | 基础四维网格:**卡数 N × stage{1,2,3} × mbs × GC{true,false}**;packing 在有数据画像时作为条件分支 |
| C | 校准策略 | **解析模型 + 受约束离线校准**。线上只读经审核、版本化的 kconf profile;不在线自动拟合 |
| — | 校准数据 | 我方定义统一 **数据 schema**。第一阶段 H800 离线校准 → 写 kconf；4090 完成后独立校准、独立验收，不混用系数 |
| D1 | 输出结构 | **新建 `DeepSpeedStrategyResult` / `DeepSpeedSearchRequest`**,字段对齐 `HyperParams`/`ResourceConfig` 落库口径 |
| W1 | 接线 | **并列新分支 + kconf 灰度 + 失败 fallback 老 recommender**(与 CPT 引擎已验证的模式一致) |
| N2 | 多机 | **第一版单机 only(≤8 卡)**,多机二期。单机放不下的大模型 → 触发兜底护栏 |
| — | 覆盖范围 | **full + LoRA;SFT;dense + VL**。MoE 暂不、DPO/RFT/KTO 二期 |
| — | 卡数档位 | {1,2,4,8} |
| — | mbs 档位 | {1,2,4,8,16}(砍 32/64) |
| — | 显存门槛 | `P95(真实峰值总需求) ≤ 0.95 × 实际 GPU 显存`;95% 容量线固定,碎片风险进入 P95 上界 |

---

## 3. 搜索空间与优化目标

### 3.1 网格
```
for N in {1,2,4,8}:                    # 单机卡数 = DP 度
  for stage in ({0} if N==1 and launcher disables DS else {1,2,3}):
                                         # stage 0 = 单卡 non-DeepSpeed
    for mbs in {1,2,4,8,16}:           # per-device micro-batch
      for gc in {true,false}:          # gradient checkpointing
        gradAccum = globalBS / (mbs × N)
        skip if globalBS % (mbs × N) != 0      # 整除约束
        skip if gradAccum < 1
        eval → (peakMemP50, peakMemP95, stepTimeP50, stepTimeInterval)
        keep if peakMemP95 ≤ 0.95 × gpuMemGB
```
dtype∈{fp16,bf16} 是固定输入,字节数相同但 kernel profile 可以不同。
`N=1` 若实际 launcher 启用 DeepSpeed，才评估 ZeRO 1/2/3；当前生产契约单卡禁用 DeepSpeed，因此规划器
用内部 `stage=0` 精确表示 non-DeepSpeed 路径，并要求它拥有独立的 stage-0 calibration 才能正式推荐。
它不能借用任一 ZeRO profile，也不能把单卡理论候选误标为 calibrated。

packing 是依赖数据画像的条件搜索分支,不是一个无条件布尔倍率:

- 用户输入的 GBS 始终表示每次参数更新的**原始逻辑样本数**。
- 由真实长度分布做确定性装箱模拟,得到 `pack_utilization`、`mean_samples_per_pack`、
  token 数和 attention pair 数,再反推物理 batch 与 gradient accumulation。
- 当前 neat-packing + FlashAttention 路径物理 batch size 固定为 1;逻辑 GBS 误差不得超过 5%。
- 缺少数据画像、装箱门槛不通过或样本间 attention 隔离不受支持时只搜索 packing=off。

### 3.2 auto-search(性价比爬坡)
1. 从最小卡数起(1→2→4→8),每档只保留通过 P95/95% 安全线的组合,再按稳健吞吐排序。
2. 最小安全卡数是主推荐。从 N 升到 2N,只有
   `throughput_lower(2N)/throughput_upper(N) ≥ 1.8` 才把 2N 标为推荐扩卡;否则只展示收益估计并停止爬坡。
3. 同时展示最小卡方案、每次翻倍的预计收益和绝对最快方案,避免把“最快”误写成“最划算”。
4. **兜底护栏**:爬到 8 卡仍无可行解(或 step time 超合理上限),不静默失败 →
   返回显式信号,建议**改 LoRA / 降 seq / 降 GBS**(N2 下不建议多机)。

### 3.3 fixed 模式
用户显式指定卡数时,只在该卡数下搜 stage×mbs×gc×packing(若满足条件),返回安全 top-K。
吞吐区间重叠时优先吞吐置信下界更高者;仍无法区分则优先显存余量与配置简单度,并生成短吞吐初筛需求。

---

## 4. 建模:解析主干 vs 受约束校准

> 最高原则:凡能由 dtype、张量形状、可训练性、ZeRO 语义、算子计算图和 collective 算法推导的量,
> 必须解析计算,不得回归。实验只校准硬件/软件实现造成的效率、重叠、工作区、碎片和预测区间。
> 若受约束参数无法解释系统性残差,结论是“理论结构缺项”,应补机制与定向实验,不能添加黑盒残差模型。

### 4.0 不可校准量与校准白名单

| 类别 | 解析量(禁止拟合) | 可校准量(必须有物理含义和边界) |
|---|---|---|
| 模型状态 | tensor 数量、shape、dtype bytes、trainable/frozen、ZeRO 分片比例 | allocator 对齐/碎片的非负残差分布 |
| 激活 | 算子图、token/attention pair 数、各保存 tensor shape、GC 重计算集合 | kernel 路径导致的 activation liveness/工作区,受图上界约束 |
| 计算 | 各 matmul/attention/elementwise FLOPs 与读写 bytes | `(0,1]` 的计算/带宽效率 |
| 通信 | collective 种类、payload bytes、ring 因子 | 不超过链路上限的有效带宽、非负延迟、`[0,1]` overlap |
| packing | 长度分布、装箱结果、逻辑样本/token/attention pair 数 | 非负的数据整理与 kernel 开销 |
| 风险 | GPU 实际容量、95% 容量线 | 成功/OOM 约束得到的 P95 安全残差 |

禁止以 `model_id`、`dataset_id` 为校准键,也禁止任意多项式、树模型、神经网络或无物理边界的自由截距
直接预测显存/吞吐。校准键只允许包含会改变执行机制的 GPU/profile、runtime fingerprint、dtype/kernel、
training mode、ZeRO stage、GC 和 packing 路径。

### 4.1 模型结构与兼容层级

优先读取 HF config + runtime-unique tensor 清单及 module-group 清单,转成通用 `ModelStructureManifest`:
Embedding、GQA/MHA、MLP/SwiGLU、Norm、LM Head、Vision Encoder、Projector、参数组和可训练性。
参数量使用真实、去重后的 runtime tensor shape 之和,模型名中的“8B/14B”只可作一致性检查。仅有总参数量
而没有逐 tensor/module 清单时不能标为 exact；LoRA exact 还必须包含真实 trainable tensor 清单，以覆盖
`modules_to_save`、trainable embedding/bias 和非标准 target，不能只依赖 rank×矩阵维度推导。

新模型无需先跑实验,按三层降级:

1. **exact**:架构完整识别,使用精确算子公式与已校准 profile。
2. **generic_structural**:主要 transformer 维度可识别,模型状态精确,激活/效率使用跨架构保守上包络。
3. **conservative_fallback**:只有总参数量和少量维度,强制搜索 GC/小 MBS 并扩大区间;允许过度配置但优先不 OOM。

只有遇到不可归类且占比不可忽略的新算子、保守上界导致无候选或线上证据击穿上界时才要求补实验。

### 4.2 显存模型(每卡峰值)

```text
peak_total = persistent_model_state
           + saved_activations + recompute_workspace
           + logits_or_fused_ce_workspace + zero_collective_workspace
           + cuda_nccl_framework + allocator_fragmentation
```

输出必须同时给出解析基线、校准中心估计和 P95 上界。可行性只看 P95 上界。

**(a) 模型状态(ZeRO 解析公式)**

令 `Pf` 为冻结参数量、`Pt` 为可训练参数量、`Pall=Pf+Pt`;参数/梯度/Adam 状态字节分别为
`bp/bg/bo`。bf16/fp16 + AdamW 的默认值是 `bp=2`、`bg=2`、`bo=12`(fp32 master + m/v),
但实际 dtype/optimizer 配置必须作为输入,不能用回归修改。

| stage | 每卡持久模型状态 |
|---|---|
| 1 | `bp*Pall + bg*Pt + bo*Pt/N` |
| 2 | `bp*Pall + (bg+bo)*Pt/N` |
| 3 | `(bp*Pall + (bg+bo)*Pt)/N` |

Full 模式 `Pf=0, Pt=Pall`;LoRA 模式基座进入 `Pf`,adapter 进入 `Pt`。因此 stage3 也会分片冻结基座参数,
但冻结参数不产生 gradient/optimizer state。

stage3 all-gather 峰值不是自由倍率。只有受信 operator manifest 可按 DeepSpeed 的**逐 tensor** threshold 与
`stage3_model_persistence_threshold` 精确计算 persistence；当前 state-only inventory 或缺少受信 module trace 时，
persistence 与最大当前 module 均以完整 loaded parameter set 作安全上界。受信 manifest 的最大当前 module 仍
必须包含 embedding/lm_head，而不只是 decoder block。`stage3_max_reuse_distance` 与
`stage3_max_live_parameters` 是相互独立且必须进入 runtime contract 的 retained 上界；它们可与不受 prefetch
cap 限制的当前 module 共存，因此至少使用
`live_parameter_elements = min(Pall, retained_envelope + max_current_module + persistent_upper)`，并计算
`bp*(1-1/N)*live_parameter_elements`。默认配置缺失时使用绑定 DeepSpeed 版本的真实默认值，不能把
persistence threshold 当成 0。实验只能校准 allocator 对齐/残差，不能把这个结构上界拟合小。

**(b) 激活与 GC**

按每个算子实际 tensor shape 累加 autograd 需要保存的张量。decoder 主项由
`B×S×L×H`、GQA 的 `KVHeads×HeadDim`、MLP intermediate 和 attention pair 数构成。
FlashAttention 与普通 attention 使用不同的解析工作区;长序列不能靠经验分桶。

GC 根据 checkpoint 覆盖的模块删除其 saved tensors,并把对应 forward FLOPs 加入重计算项。
不得把 `gc=true` 简化成固定 `0.35` 显存倍率或任意 step-time 倍率。

**(c) logits / fused CE / VL**

非 fused CE 的保守 logits 峰值是 `B×S×V×logits_dtype_bytes`;fused/chunked CE 根据已知 chunk shape
推导工作区,实现模式未知时使用全 logits 上界。VL 由上游提供 vision token,再按视觉塔/projector 结构计算;
`seq_len` 是总序列长度的单一真相,不重复叠加 vision token。

**(d) framework 与碎片**

CUDA context、NCCL bucket、DeepSpeed contiguous gradient/bucket 和 allocator reserved-but-unallocated 分开建项。
不复用现有 `denseMemoryOverheadGB` 的模型规模分桶。OOM 日志中的 allocated、reserved-unallocated 和下一次
申请块共同构成显存需求下界;例如“allocated 127.35 GiB”发生 OOM 不能被当成 127.35 GiB 峰值。

### 4.3 通信模型(每 step,DP=N 单机组)

令 `K=gradAccum`、`G=communication_dtype_bytes*Pt` 为可训练梯度 payload、`W=bp*Pt` 为更新参数
payload、`A=bp*Pall` 为 stage3 需要按层物化的全部参数 payload。payload 与 ring 因子禁止拟合。
在当前 DeepSpeed 0.19.2 默认路径中，stage1 只在 accumulation boundary 做梯度 all-reduce；stage2
每个 micro-step 进入 `allreduce_and_scatter`，但在默认 `use_multi_rank_bucket_allreduce=true` 下其实际
collective 是 all-reduce，不得按函数名误算为 reduce-scatter；stage1/2 optimizer 后都要 all-gather
更新后的权重分片。stage3 梯度使用真正的 reduce-scatter，参数 materialization 随 micro-step 重复：

| stage | collective 结构 | 每 rank 结构化传输量 |
|---|---|---|
| 1 | accumulation boundary gradient all-reduce + updated-param all-gather | `(N-1)/N * (2*G+W)` |
| 2 | 每 micro-step gradient all-reduce + 每 optimizer step updated-param all-gather | `(N-1)/N * (2*K*G+W)` |
| 3 | 每 micro-step param all-gather + gradient reduce-scatter | `(N-1)/N * K*(M*A+G)`，普通路径 `M=2`；GC 重算未有 trace 证明复用时 `M=3` |

时间使用 `latency*collective_count + bytes/effective_bandwidth`;有效带宽不得超过 GPU profile 的链路上限,
overlap 限制在 `[0,1]`。bucket 大小、collective 次数和 accumulation boundary 行为来自绑定的
DeepSpeed runtime fingerprint；若显式关闭 `use_multi_rank_bucket_allreduce` 或关闭 stage3 reduce-scatter，
必须选择对应的结构分支，不能让带宽效率吸收 payload 差异。运行时变化必须重新验证结构假设。
第一版仅单机,不假装估计跨机拓扑。

### 4.4 计算与 step time

使用算子级 roofline。以 Linear 为例,forward、input-gradient、weight-gradient 各自约 `2*B*S*I*O` FLOPs;
冻结基座仍有 forward 和 input-gradient,但无 weight-gradient。LoRA adapter 的 matmul 另行按 rank 计算。
attention quadratic、embedding、LM head、VL projector 和 GC 重计算均按结构加入。

每类 kernel 时间为 `max(FLOPs/(peak*compute_eff), bytes/(mem_bw*memory_eff))`;step time 为 kernel critical path、
未被 overlap 的通信、optimizer 与非负 launch/framework 开销之和。排序使用吞吐置信下界;MFU 只做诊断,
不得作为推荐目标。

---

## 5. 校准闭环(策略 C)

### 5.0 分阶段数据准入

- **Phase H800（当前）**：只读入最终状态已完成、attempt 时间窗明确、运行配置和结果来源可追溯的 H800
  observation。旧记录缺少完整 execution fingerprint 时仍必须标为 `legacy_incomplete`，不能作为正式
  calibration/publication 或 prospective holdout 证据，也不能提升置信度；但若来源绑定的 retrospective
  recovery sidecar 能从原始 status/render/config/event/summary/telemetry 重建并逐项验证 attempt、运行配置、
  硬件范围、终态和 measurement route，则允许它进入 `historical_bounded` 的理论参数 bootstrap 与历史交叉
  验证。该例外只用于减少重复实验，输出固定为 `theory_only`/`publishable=false`，并强制加入 runtime-cohort
  nuisance effect、legacy 不确定性膨胀和后续 prospective acceptance blocker。
- **Phase 4090（延后）**：流水线未收口期间，即使已有部分 success/OOM，也不得计算或发布 4090 校准系数。
  当前服务只允许加载 4090 的版本化静态物理 profile；推荐结果最多为 `theory_only`。
- 两种 GPU 的 observation、系数、覆盖范围、holdout 和准入结论完全隔离。H800 的结论不能自动迁移为
  4090 的效率/残差；只能共享下文明确列出的解析公式。
- `h800_first` 是 fail-closed 的发布阶段，而不是“优先使用 H800”的软提示：H800 exporter、校准器和
  holdout 审计必须拒绝任何声明或运行时证明为 4090 的记录；尚未完成的 4090 记录不能以默认值、混合总体、
  fallback 样本或先验校正等形式间接影响 H800 profile。只有 4090 流水线完整结束、使用独立 observation
  集完成独立 holdout 并发布独立 profile 后，才能切换到 `per_gpu_validated`。

### 5.1 数据 schema(同事网格回填,我方定义)
每行 = 一个 run:
- **自变量**:`gpu_type, model_name, num_gpus, zero_stage, micro_batch, seq_len, vision_token, grad_checkpoint, freeze_vision_tower, packing, dtype`
- **结构输入**:`model_structure_sha256, runtime-unique tensor/module/trainable inventory, total_params, trainable_params, tensor/communication dtype, zero bucket/prefetch/persistence/reuse config, logical_gbs, physical_mbs, grad_accum, token/attention-pair totals`
- **因变量**:`step_time_s`、有效/计算 token throughput、`max_allocated_bytes`、`max_reserved_bytes`、OOM 时 requested allocation bytes
- **元信息**:`framework_version, runtime_fingerprint, run_date, outcome_class, failure_signature`

execution fingerprint 使用两段式生成：进程启动前的 `sft_execution_inputs/v2` 绑定静态输入、审批与实际硬件；模型初始化后，
每个 rank 输出按 Parameter 身份去重、保留 tied-weight aliases、优先使用 `ds_shape/ds_numel` 的运行时模型清单。
只有 job/attempt/rank/world-size 均匹配且所有 rank 的逻辑 inventory hash 一致时，才生成
`sft_execution_fingerprint/v2`。这个 v2 SHA 只用于逐 run 的完整性证明；校准键中的 `runtime_fingerprint` 使用
`sft_runtime_mechanism/v2`，它由 runtime identity、launcher/config-template source、patch-set、构建身份和清洗后的机制环境组成，不能包含
模型、数据集、batch 或卡数。stage/GC/training mode/kernel/packing 仍是独立显式 selector。OOM 若发生在运行时
清单产生前，仍保留为诊断证据，但不进入正式校准。

运行证据必须同时满足以下约束，缺少任意一项都只能标为 `incomplete`：

- 每次运行使用独立的 attempt 根目录；status、render、配置、任务元信息、日志、事件、summary、遥测和所有
  fingerprint 文件都绑定同一个 `job_id + execution_attempt_id`，不得依赖时间窗从可追加的跨 attempt 文件中猜测。
- 启动 worker 前实时记录所分配物理 GPU 的 index、UUID、驱动名称、总显存、PCI bus、compute capability、
  driver、power/MIG 状态及 live topology。声明的 H800 profile 与实际每张卡必须同构且精确匹配；GPU family
  由该 attestation 推导，不能由 exporter 硬编码。
- execution inputs 必须绑定审批模式、design/approval/queue 证据和 canonical authorized job payload。未批准的
  smoke/diagnostic 可具有完整 provenance，但永远是 `calibration_eligible=false`；calibration/holdout role 与
  预先声明的 split unit 也必须属于被审批的 job payload。
- runtime tensor inventory 必须做结构校验，不只校验 JSON hash：非空 tensor、唯一 tensor/alias、shape×dims 与
  numel、dtype/trainability、全局与 module-group 汇总、largest group 均须自洽；LoRA 还必须有非零的真实
  trainable inventory。各 rank 的逻辑 inventory 必须一致，并分别绑定实际设备 attestation。
- 上述 execution inventory 的自洽性只能证明“实际运行了哪些 tensor”，不能证明调用方声明的 tensor→算子/
  module 语义可信。当前 `dsplanner.tensor_inventory.v1` 因此只用于精确的 model-state 总量，并固定降为
  `generic_structural`：任意识别出的 `model_type`、自哈希或 module grouping 都不能升级 `exact`、不能消费
  calibrated profile。其 ZeRO-3 module/persistence 上界按完整 `LoadedParameters` 处理。只有后续由平台内部
  CPU checkpoint scanner 生成、逐算子验证 shape/覆盖且具有受信来源边界的新版本 manifest，才能开放
  `exact`；这不妨碍未知新架构先获得偏保守、以不 OOM 为首要目标的 theory-only 建议。
- runtime mechanism 只哈希真正改变执行机制的软件 build、训练代码/配置模板/patch 及清洗后的机制环境；完整
  run provenance 仍可绑定模型、数据和硬件，但其总 SHA 不得作为可复用 calibration key。CUDA/cuDNN/NCCL、
  driver、Torch/DeepSpeed/LLaMA-Factory、FlashAttention/Liger/CCE/Triton、optimizer、compile、GC variant、
  ZeRO bucket/prefetch/persistence/reuse 和通信/dtype 路径必须由 mechanism hash 或显式 selector 精确覆盖。

canonical exporter 必须重新验证上述原始证据后才设置 `evidence_verified=true`；readiness 审计不能只信导出
JSONL 中的 `quality="complete"` 字符串，还要重算嵌入证据、拒绝 observation/attempt/fingerprint 重复，并要求
calibration 与 holdout 的预先绑定 split unit 不相交。

结果必须区分 `success`、`oom`、`software_failure` 和 `infrastructure_failure`。只有
`success` 与真实 `oom` 可以参与可行性边界和性能回归；软件或基础设施失败必须先修复并重跑，
不得据此排除某个 GPU、训练模式或 ZeRO stage。特别地，H800 的 LoRA + ZeRO-3 始终保留在
搜索空间中；旧运行时出现的混合 dtype all-gather 错误属于运行时缺陷，不是策略不可行证据。

成功样本给出峰值/时间观测;OOM 是“真实需求大于当时可用容量”的**删失不等式**,不得伪造一个峰值数值参与最小二乘。
`software_failure`/`infrastructure_failure` 不进入可行性或性能校准。

采样哲学是**缺什么证据补什么**:固定大部分维度做最小单维扫描。显存不确定时在预测边界做 success/OOM
夹逼;activation 扫 mbs×seq×GC;通信扫 stage×卡数;计算效率选择远离 OOM 边界的短任务。每轮更新区间后
若已可辨识就停止,不重铺完整网格,并始终保留按模型规模或 seq 区间隔离的独立 holdout。

“边界完成”只针对支持域 `MBS∈{1,2,4,8,16}`：每个预注册 family 得到 success/OOM 夹逼即可停止；
若 MBS=16 已有完整、获批且绑定 H800 的 success，则证明本支持域全部可行，同样停止，禁止为了制造 OOM
继续试 32/64。packing 固定物理 MBS=1，以 ABBA 配对、零新增 OOM 和对应 unpacked 边界为依据，不要求
packing 自身人为跑出 OOM。

H800 补实验的静态 JSONL 只是最大审批包络，不能被普通并发队列直接逐行执行。获批执行队列必须把每个
source row 的 canonical hash、条件依赖和顺序绑定进 approval design；scheduler 先验证同族、先序、无环的
condition DAG，再执行 anchor→单向边界和 ABBA 前置关系。条件不满足的 row 写
`conditional_skipped` scheduler terminal，不启动训练、也不生成 calibration observation；软件/基础设施失败
保持非终态并阻塞后继。每次运行硬超时 2700 秒，必须终止整个 launcher/torchrun process group；campaign
墙钟硬上限 72 小时，到点终止在途进程且不再发射新任务。候选冻结、审批 promotion 与实际执行仍是三个
分离事务，离线冻结绝不能修改 live approval/queue 或启动 GPU。

### 5.2 回归 → 落库
静态实验 JSONL → 解析理论 basis → 受约束校准白名单参数与 conformal/分位数安全残差 → 独立 holdout 验证 →
写入 kconf `GetDeepSpeedPlannerConfig()`。引擎运行时**只读批准后的 profile,不读原始网格**。

校准配置按 `gpu_type + runtime_fingerprint + dtype/kernel path + stage/GC/training mode + packing path` 分层；
选择发生在**每个候选配置求值时**，不能在一次搜索开始时只按 GPU/runtime 取一套参数后复用于全部 stage/GC/mode；
模型名和数据集名不进入系数键。线上数据可自动收集,单次 OOM 可立即把相应区域降级并单调扩大安全上界,
但不得自动重写中心系数;新版本必须离线验证、review、版本化发布并可回滚。

历史 resource 交叉验证的 split unit 固定为
`(model_id, train_type, dataset_id, target_gbs)`，一次 outer fold 必须从所有 runtime cohort 中整体移除该
场景的全部 stage/GC/MBS/卡数/重复记录。runtime cohort 只能作为训练折内 nuisance/fixed effect；不得先按
cohort 分开再各自留出场景，因为同一场景会经由另一 cohort 泄漏进训练集。当前恢复集对应 84 个全局
resource folds（memory boundary 48、正式 throughput 78）；按 cohort 分开的 214 个局部 folds 只保留为诊断。

实现中另以 `runtimeContractSha256` 绑定规范化后的 dtype 字节语义、固定 AdamW 状态、attention/CE kernel、
CE chunk、ZeRO bucket/prefetch/persistence/max-reuse、collective 与 overlap；它与软件/模板的 `runtime_fingerprint` 必须
同时精确匹配。profile 选择必须发生在显存/吞吐计算**之前**：只有 `exact` 且处于显式 coverage 内的候选
可以消费 calibrated profile，generic、fallback、coverage miss 或 contract miss 必须真正切回版本化保守
default，不能只在计算后把 confidence 改低。当前 kconf schema 为 `dsplanner.config.v2`。

严禁把现有形如 `log(observed/analytic)=intercept+is_lora+gc` 的自由回归直接当成规划器主模型。
这类结果只能用于发现理论缺项或形成有边界的候选先验,必须重新映射到上述物理参数并通过 holdout。

### 5.3 显存反馈缺口
线上 Kafka 指标(`kafka.go`)已回收 `throughput`/`total_tokens` → step time 可反推(在线校吞吐)。
峰值显存暂无线上回收 → **第一批校准依赖同事网格的 `peak_mem_gb`**;线上以 OOM 事件做单边兜底(OOM→调高显存系数)。

### 5.4 置信度、准入与补实验

- `calibrated`:请求处于已验证覆盖范围,使用 profile 中心值和 P95 区间。
- `theory_only`:只有解析/家族保守 profile,可输出但默认影子验证;生产路径回退旧推荐器。
- `out_of_distribution`:超出模型规模、seq、GPU/runtime 覆盖,使用保守 fallback 并生成补实验需求。

进入灰度的最低验收:

1. 独立验证中真实 OOM 被判为安全候选的数量为 0。
2. 至少 95% 成功任务的真实峰值不超过输出 P95 上界。
3. 推荐配置相对实测最优的吞吐损失不超过 10%。
4. 声称扩卡收益达到 1.8× 的场景,独立实测也必须达到 1.8×;否则只展示、不推荐。
5. 不同 GPU、不同 runtime fingerprint 分开验收；当前只执行 H800 验收，4090 等其流水线完成后再验收。

### 5.5 CPU-only 服务与新增 GPU

推荐代码运行环境没有 GPU,不得在请求路径执行 NVML、CUDA、GEMM 或 NCCL 探测。新增
`DeepSpeedGPUProfile` kconf 注册表,与现有仅含名称/显存的 `MachinePackage` 解耦:

```go
type DeepSpeedGPUProfile struct {
    GPUType              string
    ArchitectureFamily   string
    MemoryBytes          uint64
    DenseBF16PeakFLOPS   float64
    MemoryBandwidthBps   float64
    IntraNodeLink        string
    IntraNodeBandwidthBps float64
    MaxGPUsPerNode       int
    Source, Version      string
    Confidence           string
}
```

profile 来源可以是硬件公开规格、资源平台登记或独立 GPU 实验流水线;线上服务只消费版本化结果。
没有训练实验的新 GPU 使用“精确 SKU → 架构/互联家族 → 全局保守下界”三级 fallback。
若只有名称与显存,仍可给出低置信度的保守显存方案,但不得正式宣称 2N/N 达到 1.8×。

---

## 6. 输出结构(D1)

```go
type DeepSpeedSearchRequest struct {
    ModelConfig          map[string]interface{} // HF config.json
    GpuType              string
    GpuMemGB             float64
    Nodes, GpusPerNode   int    // 0/0 = auto-search(单机)
    GlobalBatchSize      int    // 不变量(口径 X)
    SeqLen               int    // 已含 vision token(上游给)
    VisionToken          int    // 上游 token 模块提供(VL)
    TrainingMode         string // full / lora
    GradientCheckpointing bool  // 默认 true
    FreezeVisionTower    bool   // 默认 true(VL)
    Dtype                string // bf16 / fp16
    Packing              bool   // 默认 false
    Lora                 LoraCfg
    TopK                 int
    // 校准系数由 kconf 注入,不在 request
}

type DeepSpeedStrategyResult struct {
    NumGpus, NodeCount     int
    ZeroStage              int
    MicroBatch, GradAccum  int
    GlobalBatchSize        int
    GradientCheckpointing  bool
    Dtype                  string
    TrainableParams        int64
    StepTimeMs             float64
    ThroughputSamplesPerS  float64
    ThroughputLower        float64
    PeakMemTheoryGB        float64
    PeakMemP50GB           float64
    PeakMemP95GB           float64
    MemBreakdown           map[string]float64
    Packing                bool
    LogicalGBSError        float64
    Confidence             string
    Evidence               []string
    Assumptions            map[string]interface{} // profile/version/覆盖范围/公式依据
}
```
映射:`NumGpus→ResourceConfig.GpuCount`、`MicroBatch→HyperParams.BatchSizePerDevice`、
`GradAccum→HyperParams.GradientAccumulationSteps`。

---

## 7. 接线(W1)

```
非 CPT 任务:
  if kconf.DeepSpeedPlannerEnabled(project/model):   # 灰度开关
      result = deepspeedPlanner.Search(req)
      if err: fallback → RecommendLLMWithResourcePoolPriority(老)   # 安全网
  else:
      RecommendLLMWithResourcePoolPriority(老)
```
与 CPT 现有的"新引擎 + `cptErr` fallback 老 recommend"完全同构。

当前接入默认 `off`；`shadow` 使用有界、零等待队列的异步观测器，饱和时只丢弃观测，绝不阻塞旧推荐。
H800-first 会将资源平台的 H800/140GiB 身份与 registry 的 canonical SKU、Hopper family、容量和单机卡数
交叉校验；已知 SKU 的 alias 必须保留同一 SKU 标记，H800 profile 不能反向声明 4090 alias（该限制在后续
`per_gpu_validated` 阶段也继续生效）。正式 profile/holdout 尚未发布前，域内 `enforce` 一律 fail-closed，
shadow 结果不写回任务配置。

---

## 8. 代码落点

```
pkg/jobmanager/strategy/                 # 复用:ring 时间函数、cluster、参数量/激活/维度 helpers
pkg/jobmanager/dsplanner/  (新增)
    request.go        # DeepSpeedSearchRequest/Result
    manifest.go       # HF config/tensor inventory → 通用结构描述,含三级 fallback
    memory.go         # ZeRO state/activation/workspace 的解析 basis + P95 上界
    performance.go    # 算子 roofline + ring collective + overlap critical path
    packing.go        # 数据画像装箱分支与逻辑 GBS 守恒
    searcher.go       # 网格 + 最小安全卡数 + 1.8× 保守爬坡 + 兜底护栏
    calib.go          # 受约束 profile、覆盖范围、置信度与补实验需求
    profiles.go       # CPU-only GPU profile 解析与分层 fallback
pkg/utils/kconf/       # +GetDeepSpeedCalibConfig()
pkg/jobmanager/job_manager.go  # W1 接线 + 灰度
pkg/engine/types/train.go      # +DeepSpeed request/result 类型;SFT 收 GlobalBatchSize
```

---

## 9. 范围与二期

**第一版**:full + LoRA;SFT;dense + VL;单机 ≤8 卡;解析模型 + 已批准离线校准;packing 条件推荐。
**二期**:多机(ZeRO-3 跨机)、DPO(+reference model 显存)、MoE、RFT/KTO。

## 10. 已知风险
1. **第一版 profile 覆盖不足** → theory_only/OOD 分级、W1 灰度 + fallback;不得伪装成精确结果。
2. **显存无线上反馈** → 依赖同事网格 + OOM 单边兜底;后续可补 `peak_mem_gb` 上报。
3. **大模型单机放不下**(N2)→ 兜底护栏建议 LoRA/降 seq,不静默失败。
4. **CPT 引擎 `seqTotal = TrainSeqLen + visionTokens` 疑似重复计 vision token**(纯文本 CPT 未触发)。
   本引擎口径为 `seq_len` 单一真相(已含 vision token),不叠加。
5. **框架兼容性故障污染校准标签** → 校准按 `runtime_fingerprint` 绑定；非 OOM 失败进入修复/重跑队列，
   不进入策略排除规则。生产镜像发布前执行 LoRA + ZeRO-3 短 canary。
6. **新模型/新 GPU 无专属实验** → 使用结构/家族保守 profile,优先不 OOM;不以模型 ID 建专属倍率。
