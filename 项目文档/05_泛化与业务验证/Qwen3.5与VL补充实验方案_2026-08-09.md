# Qwen3.5 与 VL 补充实验方案（2026-08-09）

## 1. 结论

当前证据不足以把 Qwen3.5 系列或真实 VL 场景纳入统一资源推荐器的发布域。

- Qwen3.5 只有 4B 文本路径形成了少量 H800 真实观测；0.8B、9B、27B 缺少跨尺寸标定，4B 的中心误差在不同批次间也不稳定。
- Qwen3.5 本身是统一视觉语言架构。历史 Qwen3.5 作业虽然 checkpoint 含视觉塔和投影层，但输入是纯文本，视觉塔/投影层被冻结，因此不能代表真实图像训练。
- Qwen2.5-VL-7B 和 Qwen3-VL-8B 已通过真实图片 media canary，但这只证明图片、像素和视觉网格实际进入模型，不等于显存或吞吐模型已经标定。
- 历史 Qwen3-VL-8B 纯文本代理回放与真实图片域不一致，不能用来校准图像 token、视觉激活或多模态 workspace。

本次冻结为两个阶段：15 个真实图像语义 canary 和 86 个正式标定作业，共 101 个作业、188 个 GPU-job equivalents。只有 canary 全部通过，正式阶段才会自动开始。

## 2. 当前证据

### 2.1 Qwen3.5-4B 文本路径

以下均为已有实验观测，不是本次新实验的结果。

- 一批 25 个 follow-up 终态包含 21 个 success、4 个确认 OOM、0 个软件失败。
- 边界子集为 5 个 success、4 个 OOM。
- 该批中心显存 MAPE 为 13.84%，安全上界覆盖率为 60%，4 个 OOM 中 false-safe 为 0，5 个可运行点中安全准入 4 个。
- 吞吐排序验证包含 16 个候选、12 对比较，排序正确率 100%、top-1 regret 为 0，但绝对吞吐 MAPE 为 127.24%。这说明排序信号可用，不代表绝对吞吐已经准确。
- 后续来源隔离的 3 个 Qwen3.5-4B LoRA 点均为 process-success。实测峰值 reserved 分别为 55.1875、32.2559、137.7441 GiB；安全线为 132.8393 GiB。两个安全点被准入，超过安全线的点被拒绝。
- 这 3 个点的中心 APE 分别为 35.64%、40.42%、20.95%，平均约 32.34%；安全上界覆盖 3/3，false-safe 为 0。

工程解释：安全上界暂时比中心预测稳定，但 Qwen3.5-4B 的中心项仍需要重新标定。不能用安全上界的零 false-safe 代替中心误差评估。

### 2.2 真实 VL 路径

- 历史 Qwen3-VL-8B 纯文本代理回放中，LoRA 中心 MAPE 为 66.80%、安全上界覆盖为 0、false-safe OOM 为 2/4；Full 中心 MAPE 为 21.68%、安全上界覆盖为 50%。这些值只能作为代理域失败证据。
- 结构拆解估计 Qwen3-VL-8B 的视觉参数约为 5.7639 亿，BF16 权重约 1.0736 GiB，只能解释历史约 21.2 GiB 观测差值中的约 5%。把视觉权重再次加到 checkpoint 总参数上还会造成双计数。
- Qwen2.5-VL-7B 与 Qwen3-VL-8B 的真实图片 canary 已在所有 rank 通过：真实图片路径、pixel values、image grid 均存在；视觉塔与投影层冻结；语言侧 LoRA 生效；没有视觉 LoRA 命中。
- 上述 canary 的 `fit_allowed=false`、`recommendation_release_allowed=false`。原有 16 个真实图片标定设计尚未执行，本次将它等价纳入新的正式队列，避免另跑一套不一致的基准。

### 2.3 本次本地 checkpoint 端点

参数量来自 safetensors 的逐 tensor 逻辑元素计数，不使用模型名称中的标称规模。

| 模型 ID | 实际总参数 | 视觉塔与投影层估计参数 | 本次角色 |
|---|---:|---:|---|
| `qwen3p5_0p8b` | 0.8734B | 0.1006B | Qwen3.5 小尺寸 |
| `qwen3p5_4b` | 4.6599B | 0.3335B | 已有文本锚点、真实图片主锚点 |
| `qwen3p5_9b` | 9.6531B | 0.4560B | Qwen3.5 中尺寸迁移 |
| `qwen3p5_27b` | 27.7814B | 0.4607B | Qwen3.5 大尺寸迁移 |
| `qwen2p5_vl_3b` | 3.7546B | 0.6687B | Qwen2.5-VL 尺寸迁移 |
| `qwen2p5_vl_7b` | 8.2922B | 0.6766B | Qwen2.5-VL 机制锚点 |
| `qwen3_vl_2b` | 2.1275B | 0.4070B | Qwen3-VL 小尺寸 |
| `qwen3_vl_4b` | 4.4378B | 0.4153B | Qwen3-VL 训练范围消融 |
| `qwen3_vl_8b` | 8.7671B | 0.5764B | Qwen3-VL 机制锚点 |
| `qwen3_vl_32b` | 33.3574B | 0.5953B | Qwen3-VL 大尺寸迁移 |
| `qwen3_vl_30b_a3b` | 31.0708B | 0.5386B | Qwen3-VL MoE 诊断 |

## 3. 本次实验矩阵

### 3.1 语义 canary：15 个作业，27 GPU-job equivalents

| 轨道 | 作业数 | 目的 | 通过条件 |
|---|---:|---|---|
| 11 个模型端点的 language-only LoRA | 11 | 验证 checkpoint、Qwen3.5 TileLang/FLA overlay、真实图片数据链路和多卡 ZeRO 路径 | 每个作业 success；每个 rank 均观测到真实图片、pixel values、image grid；冻结声明与实际参数可训练状态一致 |
| Qwen3.5-4B 与 Qwen3-VL-4B 的训练范围 canary | 4 | 分别验证“投影层+语言”和“视觉塔+投影层+语言” | 对应视觉组件确实出现可训练参数；未声明训练的组件保持冻结 |

canary 为语义证据，永不进入显存或吞吐拟合。canary 出现 OOM、软件失败、结构声明不一致或视觉数据未进入模型时，正式队列保持锁定。

### 3.2 正式标定：86 个作业，161 GPU-job equivalents

| 轨道 | 作业数 | 覆盖内容 |
|---|---:|---|
| Qwen3.5 文本跨尺寸 | 28 | 0.8/4/9/27B；3 个来源画像；每个尺寸 2 个 LoRA MBS 边界点；另加 4 个 Full 短序列锚点 |
| Qwen2.5-VL-7B / Qwen3-VL-8B 机制锚点 | 16 | 低/高分辨率；1 卡无 ZeRO、2 卡 ZeRO-3 MBS 1/2、2 卡 ZeRO-2 无 GC 四种机制 |
| Qwen3.5 真实图片跨尺寸 | 16 | 0.8/4/9/27B；低/高分辨率；每个尺寸一个安全点和一个压力点 |
| VL Dense 尺寸迁移 | 14 | Qwen2.5-VL-3B、Qwen3-VL-2B/4B 的低/高分辨率安全/压力点；Qwen3-VL-32B 的低/高安全点 |
| VL MoE 迁移 | 2 | Qwen3-VL-30B-A3B 的低/高分辨率安全点 |
| 视觉训练范围消融 | 4 | Qwen3.5-4B 与 Qwen3-VL-4B；投影层+语言、全视觉+语言 |
| 重复测量 | 6 | Qwen3.5-4B、Qwen2.5-VL-7B、Qwen3-VL-8B 高分辨率安全点各 2 次额外重复 |

正式文本作业采用 3 个 warmup + 10 个 measurement optimizer steps；正式图片作业采用 2 个 warmup + 8 个 measurement optimizer steps。全部作业固定 BF16、GBS 64、packing 关闭、offload 关闭。

## 4. 数据和处理器约束

- 真实图片数据固定为 1000 条、每条 2 张图片的 `vl_pzfj38_calibration_v1`。
- 低分辨率上限为 $448^2=200{,}704$ pixels；高分辨率上限为 $768^2=589{,}824$ pixels。
- 画像按 Qwen2.5-VL、Qwen3-VL、Qwen3.5 三个 processor 家族分别用真实图片验证。
- 每个代表模型至少抽 16 张真实图片，要求预计算的 `image_grid_thw` 与实际 processor 输出完全一致。
- 同家族跨尺寸共享画像前，固定 LlamaFactory 模板实际使用的 `tokenizer.json` 以及图像/视频 processor 文件必须逐字节一致。
- Qwen3-VL-32B 缺少冗余 `merges.txt`，Qwen3.5-0.8B 自带 chat template 与其它尺寸存在差异；这两项不参与固定 LlamaFactory 模板的实际编码。差异已写入 processor 等价性记录，未将目录文件不一致误报为完全相同。

## 5. 指标和统计语义

### 5.1 显存

每个 success 作业记录所有 rank 的峰值 allocated 和 reserved bytes。主要拟合目标为峰值 reserved：

$$
M_i^{\mathrm{obs}}=\max_r M_{i,r}^{\mathrm{reserved}}.
$$

单位同时保留 bytes 与 GiB，其中 $1\ \mathrm{GiB}=2^{30}$ bytes。

确认的 CUDA OOM 不是一个精确显存点，而是右删失观测：

$$
M_i^{\mathrm{required}}>M_i^{\mathrm{available}}.
$$

软件错误、数据错误、审批错误和外部占卡均不得改标为 OOM。

对 success 点，中心误差使用：

$$
\mathrm{APE}_i=\frac{|\hat M_i-M_i^{\mathrm{obs}}|}{M_i^{\mathrm{obs}}},
\qquad
\mathrm{MAPE}=\frac{1}{n}\sum_i\mathrm{APE}_i.
$$

安全评价必须同时报告 success 上界覆盖、OOM false-safe 数量和安全点误拒绝，不能只报告中心 MAPE。

### 5.2 吞吐

正式结果使用 consumed-token ledger，报告有效 token 与计算 token 两个口径。有效吞吐为：

$$
T_i^{\mathrm{effective}}
=\frac{\sum_r N_{i,r}^{\mathrm{effective\ tokens}}}
{\max_r t_{i,r}^{\mathrm{measured}}}
\quad(\mathrm{tokens/s}).
$$

OOM 不具有吞吐点值。绝对吞吐误差、候选排序正确率和 top-1 regret 必须分别报告。

### 5.3 重复和数据划分

- 同一物理 arm 的重复测量用于估计运行方差，拟合显存中心前先折叠，不能当作独立样本扩大样本量。
- 训练/验证划分按来源或 workload profile 分组，不按重复行随机切分。
- 本批 86 个正式点是 calibration/fit 数据，不是 prospective acceptance。

## 6. 自动接力门禁

自动接力器依次执行以下状态转换：

1. 当前 `h800_unified_resource_evidence` 队列必须恰好为 220 个作业，且 220/220 都得到 `calibration_eligible=true` 的 success 或 OOM。
2. 上游 launcher、scheduler、run_job 和 train_entry 进程全部退出。
3. GPU 0–7 必须是精确的 8 张 NVIDIA H800，并连续 3 次、每次间隔 60 秒没有任何 compute process。
4. 安装本批冻结配置，重新捕获 provenance，执行静态验证。
5. 冻结并提升 canary 的精确 approval，执行 15 个 canary。
6. canary 验收 15/15 通过后，才冻结并提升正式 approval，执行 86 个正式作业。
7. 正式结果要求 86/86 均为可标定 success/OOM，且不存在软件或基础设施失败。

如果上游控制器退出时仍不足 220 个可标定终态，或任一门禁失败，接力器写入 `blocked` 状态并停止，不会抢占 GPU 或跳过失败点。

执行补充记录（2026-08-09）：上游在 186/220 个可标定终态时由用户手动暂时中止，尚余 34 个作业；用户随后明确要求先运行本批 Qwen3.5/VL 实验。接力器因此仅对本次执行启用显式 `--allow-incomplete-upstream-after-user-stop` 接管开关。该开关只豁免上游 220/220 终态条件；上游进程必须全部退出、GPU 0–7 必须是精确 8 张 H800 且仍需通过连续 3 次、间隔 60 秒的空闲门禁。Canary 与正式阶段之间的验收门禁不变，上游剩余 34 个作业也不被改写为成功或 OOM。

## 7. 正式阶段可拟合门槛

除 86/86 为 success/OOM 外，还必须满足：

- Qwen3.5 的 0.8/4/9/27B 文本轨道每个尺寸至少有一个 success；
- 11 个模型的每个低/高分辨率 language-only 组合至少有一个 success；
- 4 个视觉训练范围消融全部 success；
- 6 个重复测量全部 success。

压力点允许 OOM，并作为右删失边界进入安全模型。达到以上条件仅允许开始拟合，不允许直接发布。

## 8. 本批未覆盖的实验

以下不是遗漏，而是因为输入或语义门禁尚未满足而明确延期：

- 真实视频：当前没有冻结的真实视频训练集、frame/fps 分层画像和 runtime ledger 字段。需要先建立低/高帧数、低/高分辨率、时长分层的数据与 canary。
- 多模态 packing：当前真实图片路径没有通过 packing 语义验收。需要先验证 pack 边界不会跨样本错配图片，再做 U/P 配对实验。
- Qwen3.5 MoE：本地没有 Qwen3.5-35B-A3B 或更大 MoE checkpoint。Qwen3-VL-30B-A3B 只能验证 VL MoE 机制，不能代替 Qwen3.5 MoE。
- Qwen3.5-27B-FP8：本项目当前建模域是 BF16 SFT；FP8 checkpoint 不与 BF16 训练点混合。
- 发布验收：本批数据完成后，需要拟合 media-aware 显存/吞吐头，再冻结来源隔离的前瞻 holdout。

## 9. 记录与可追溯文件

- 静态方案：`artifacts/h800_qwen35_vl_supplement_{canary,formal}_design_v1.json`
- 精确队列：`matrix/h800_qwen35_vl_supplement_{canary,formal}_v1.jsonl`
- 模型清单：`artifacts/h800_qwen35_vl_supplement_model_inventory_v1.json`
- processor 画像：`artifacts/h800_qwen35_vl_supplement_processor_profiles_manifest_v1.json`
- 冻结选点依据：`artifacts/h800_qwen35_vl_supplement_frozen_selection_v1.json`
- 静态验证：`artifacts/h800_qwen35_vl_supplement_static_validation_v1.json`
- 自动接力状态：`artifacts/h800_qwen35_vl_autostart_state_v1.json`
- 自动接力日志：`qwen35_vl_staging/h800_qwen35_vl_autostart_v1.log`
- canary 验收：`artifacts/h800_qwen35_vl_supplement_canary_acceptance_v1.json`
- 正式结果：`artifacts/h800_qwen35_vl_supplement_results_v1.json` 与 `.md`

每个作业还保留 approval、队列 payload hash、模型与数据 hash、runtime identity、GPU 硬件声明、逐 rank summary、模型结构清单和 terminal classification。
