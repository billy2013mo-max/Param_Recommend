# Qwen3.5 与 VL 统一建模实施记录（2026-08-11）

## 1. 当前结论

本轮只覆盖“冻结视觉塔、冻结多模态 projector、语言侧 LoRA、不开 packing”的产品路径。Qwen2.5-VL、Qwen3-VL 与 Qwen3.5 的语言主干和视觉塔结构不同，不能只用模型参数量和最终 token 数互相外推；但它们可以共享同一套两阶段资源模型接口：

1. 语言主干继续使用已有纯文本显存/吞吐模型；
2. 视觉侧使用处理器绑定的图片/视频工作量特征估计增量；
3. 通过逐样本总 token 完全相等的文本/真实媒体配对实验，拟合视觉残差；
4. 总峰值按阶段最大值建模，不能把视觉阶段峰值和语言反向阶段峰值直接相加。

截至 2026-08-12 03:09 UTC，修复后的合并 Canary 已 12/12 通过，`formal_stage_allowed=true`。Canary approval transaction 为 `approval-20260812T030607Z-a1e8bb6c0165`；Formal approval transaction 为 `approval-20260812T030918Z-0f50be5696e4`。87 个 Formal 作业已经进入 `running_formal`，七个 `run_job` 分别绑定物理 GPU 0–6，GPU 7 不在授权池。状态以 `h800_frozen_vl_combined_canary_autostart_state_v1.json` 为准。

2026-08-12 02:07 UTC 启动的上一轮 Canary 在 02:16 UTC fail-closed，未进入 Formal。12 个进程中 11 个满足正测量门禁；Qwen3.5 真实视频臂虽然进程返回 0，但 optimizer step、measured seconds 和 effective tokens 都为 0，因此被正确分类为 `incomplete_metrics`。此前把视频 Canary 增加到 16 samples 只保证名义 dataloader 容量，并没有解决该问题。

本次实际发现了两个独立问题。第一个经验观察是：业务视频校准视图中的 `response=""` 在 Qwen2.5-VL 与 Qwen3-VL 模板下仍会保留 EOS 类监督标签，但在 `qwen3_5_nothink` 下该样本的 `label_ids` 全部为 `-100`。空响应因此被替换为固定最小校准哨兵 `"0"`；它只用于触发可比较的前向/反向测量，不代表业务语义标注。所有视频 workload profile 与逐样本等长文本控制均据此重新生成，三族真实视频画像的最小 label token 分别为 3、7、3，且文本/视频总 token 继续逐样本严格相等。

修复监督标签后，Qwen3.5 真实视频仍在 step 0 停止，所以“空响应是 step 0 的完整根因”这一判断被实测否定；把训练视图从 Alpaca 格式统一为 ShareGPT 格式也没有解决 step 0。显式保留异常 traceback 后确认，真正阻断 batch 的第二个问题是 Transformers 5.3 Qwen3.5 的 `get_rope_index`：Qwen3-VL 数据插件按采样帧生成多个由时间戳分隔的视频 token 段，但 collator 只传入每个源视频一行 `[T,H,W]` 网格。Qwen3.5 按连续视频 token 段逐行消费网格，第二段开始触发 `StopIteration`；Trainer 又把该异常误当成 epoch iterator 耗尽，从而产生返回码为 0、测量步为 0 的假成功。

工程修复只改变 Qwen3.5 的 RoPE 输入副本：若一行视频网格为 `[T,H,W]`，且 token 中恰有 $T$ 个按帧分隔的视频段，则 RoPE 侧展开为 $T$ 行 `[1,H,W]`；送入视觉塔和模型 forward 的原始 `[T,H,W]`、pixel tensor、采样帧和训练数学均不修改。无法由 token 段数和 $T$ 精确解释的情况直接报错。修复后独立 GPU 诊断完成 1 个优化步，输入 token 为 31,752、测量阶段为 5.92 秒，并观察到 9 个真实视频 batch。正式 Canary 中同一 Qwen3.5 视频作业完成 1 个测量步、测量阶段 6.03 秒，`real_video_path_observed=true`。旧的 0-step 结果不进入拟合。

## 2. 推荐器的输入契约

### 2.1 离线处理器已经执行时的必要输入

| 输入组 | 字段 | 单位/语义 |
|---|---|---|
| 模型 | `model_id`、`model_family`、实际参数量 | 本地 checkpoint 身份与逻辑参数个数 |
| 语言结构 | hidden size、层数、attention/KV heads、head dim、FFN size | 模型配置中的整数几何量 |
| 视觉结构 | depth、hidden/FFN size、heads、patch size、spatial merge、temporal patch、projector 参数量 | 视觉塔/merger 的固定结构量 |
| 训练范围 | 视觉塔/projector/语言模型是否冻结，LoRA rank/target | 本轮固定为视觉塔和 projector 冻结、语言 LoRA |
| 训练机制 | MBS、GBS、GPU 数、GA、GC、ZeRO、offload、dtype、kernel path | 物理运行配置；packing 固定为 false |
| 语言工作量 | `text_tokens`、`label_tokens`、`visual_tokens_total`、`total_tokens` 的均值及尾部分位数 | token/样本；视觉 token 会进入语言主干 |
| 视觉工作量 | `raw_patch_units_total`、`pixel_values_elements_total`、图片/视频数 | patch unit/样本、tensor element/样本、个/样本 |
| 视频工作量 | `sampled_video_frames_total`、视频时长、采样 FPS | frame/样本、秒/视频、frame/s |
| 处理器绑定 | min/max pixels、video max frames、patch/merge/temporal patch、processor/version/hash | 防止画像与训练处理器配置漂移 |

因此，图片最大像素数不是唯一新增输入。即使最终 `visual_tokens_total` 相同，原始 patch 数、像素 tensor 元素数、媒体个数和视频采样帧数仍会改变视觉塔前向时间、临时 tensor 和 kernel 启动开销。

### 2.2 还没有执行离线处理器时

产品侧至少还要提供每个样本的图片数量与宽高，以及每段视频的宽高、时长、源 FPS；同时提供处理器的图片/视频最小和最大像素、采样 FPS、最大帧数、patch size、空间 merge 和 temporal patch。离线画像脚本据此生成上一节的处理器后特征。原始文本、图片 URL 和视频 URL不进入资源模型 artifact。

## 3. 数学分解与量纲

以下是待标定的工程模型，不是已经由 GPU 结果证明的物理定律。

记持久显存为 $M_{\mathrm{persist}}$，跨阶段公共显存为 $M_{\mathrm{common}}$，语言动态阶段峰值为 $M_{\mathrm{lang,dyn}}$，视觉动态阶段峰值为 $M_{\mathrm{vision,dyn}}$。统一峰值参考式为：

$$
M_{\mathrm{peak}}
= M_{\mathrm{persist}} + M_{\mathrm{common}}
+ \max\!\left(M_{\mathrm{lang,dyn}}, M_{\mathrm{vision,dyn}}\right).
$$

所有 $M$ 的单位都是 byte。这里使用 `max` 是因为冻结视觉塔的前向阶段与语言反向峰值是不同时间段的峰值；不能因为视觉 embedding 最终变成语言 token，就把两个阶段峰值相加。若运行证据显示某个实现保留了跨阶段视觉 tensor，则其保留部分应进入 $M_{\mathrm{common}}$，而不是改变上述阶段语义。

吞吐按 step time 分解：

$$
T_{\mathrm{step}}
= T_{\mathrm{lang}}(N_{\mathrm{text}}+N_{\mathrm{visual}},\ \text{mechanism})
+ T_{\mathrm{vision}}(P_{\mathrm{raw}},F,C_{\mathrm{media}},\ \text{vision geometry})
+ \varepsilon,
$$

其中 $T$ 的单位为 second/optimizer-step，$N$ 为 token/样本，$P_{\mathrm{raw}}$ 为 raw patch unit/样本，$F$ 为 sampled frame/样本，$C_{\mathrm{media}}$ 为媒体个数/样本。当前生成的 FLOP、HBM byte 与临时显存字段都是代理特征；在正式配对 GPU 结果产生前，不拟合或伪造任何 VL 系数。

## 4. 数据与画像分工

| 数据源 | 模态 | 规模 | 当前用途 |
|---|---:|---:|---|
| pzfj38 | 图片 | 1,000 样本，每样本 2 图 | 图片配对拟合与机制诊断 |
| zltbjg V2 | 视频 | 100 样本，每样本 1 视频 | 视频帧数/分辨率配对拟合与机制诊断 |
| qype19 V7 | 图片 | 2,011 样本、2,425 张图 | 保留为来源隔离的业务泛化/前瞻候选，不并入本轮拟合 |

视频源实际分辨率为 320×180。`upscaled_64` 是固定 640×360 处理器分辨率的机制实验，只能解释分辨率处理机制，不能宣称业务源包含高分辨率视频。

V2 workload profile 共 24 份：6 份 pzfj38 图片画像、6 份 qype19 图片画像、12 份 zltbjg 视频画像。画像明确区分进入语言主干的 `visual_tokens_total` 与视觉塔实际工作的 `raw_patch_units_total`。

zltbjg 的源数据没有可直接用于 SFT 的非空答案。本轮资源校准视图统一使用最小响应 `"0"`，目的是保证不同模板都有正的监督标签并执行真实反向。它不用于模型质量评估，也不能视为新增的业务标签。由于该响应会改变语言 token 数，真实视频画像和 matched-text 控制必须同时重建；只修改训练 JSON 而不重建画像会破坏配对实验的 token 等长条件。

GPU 启动前先用 Transformers 4.57.1 验证三套处理器，又用 Qwen3.5 实际训练覆盖层中的 Transformers 5.3.0 + PyAV 16.0.0 重跑完整抽帧和 pixel tensor 构造。两次均通过：Qwen2.5-VL 得到 grid `[8,12,22]`、2,483,712 个 pixel tensor elements；Qwen3-VL 和 Qwen3.5 均得到 grid `[8,12,20]`、2,949,120 个 elements。它们与离线 V2 profile 逐项一致。这是 CPU 媒体/处理器门禁，不是 GPU 显存或吞吐结果。

## 5. 实验设计及其目的

### 5.1 图片实验

- Canary：3 个模型 ×（逐样本 token 等长文本、真实图片）= 6 个作业。
- Formal：45 个作业，包含 SAFE、NOGC、PRESSURE、低/高像素 tier 与重复测量。
- 目的：在语言主干 token 完全一致时，识别图片解码、视觉塔前向和 family/grid 的显存与 step-time 残差；同时识别 GC 与 MBS 交互。

### 5.2 视频实验

- Canary：3 个模型 ×（逐样本 token 等长文本、真实视频）= 6 个作业。
- Formal：42 个作业，覆盖 `native_16`、`native_64`、`native_128`、`upscaled_64`，并包含 GC/MBS 机制对照与重复。
- 目的：把最终语言 token 数的影响与采样帧数、raw patch 数、视频像素 tensor 和媒体启动开销分开；验证分辨率与帧数不能被一个 `max_pixels` 字段替代。

### 5.3 合并执行顺序

1. 合并 Canary：12 个作业，先图片后视频，只使用物理 GPU 0–6；
2. Canary 验收：12/12 success、calibration eligible、媒体 tensor/grid、视觉阶段 hook、冻结范围、语言 LoRA 全部通过；
3. 合并 Formal：87 个作业，只有 Canary 验收通过才会重新冻结和提升 formal approval；
4. Formal 结果允许 success 和明确 CUDA OOM，OOM 只作为右删失下界；软件/基础设施失败不能改标为 OOM；
5. 正式结果只允许拟合与诊断。模型拟合后仍需使用 qype19 或新的来源隔离数据做前瞻验收。

## 6. 自动启动与真实性门禁

自动接续控制器不会终止或抢占任何进程，也不会使用 GPU 7。它按以下顺序执行：

1. 检查仓库内不存在正在执行的 scheduler/run_job/pipeline；
2. 连续 3 次按配置间隔确认 GPU 0–6 是 H800 且无计算进程；2026-08-12 03:05 UTC 的最终重启使用 5 秒间隔；
3. 备份当前 experiment/provenance，原子安装 Canary staging config；
4. 重新捕获 provenance、重跑图片/视频静态校验、提升精确 12-job approval；
5. 启动 Canary，写入真实 status、metrics、runtime evidence；
6. Canary 验收通过后，原子安装 Formal staging config，重新捕获 provenance 并提升精确 87-job approval；
7. 启动 Formal 并生成结果汇总；任何失败都将状态置为 `blocked`。

“队列已生成”“控制器已挂起”“训练已启动”是三个不同状态。只有出现 `running_canary`/`running_formal` 状态、对应 scheduler/run_job 进程和结果 attempt 目录，才可以对外报告训练已启动。

本轮还要求终态满足 `measured_steps > 0`、`measured_seconds > 0` 和 `effective_tokens > 0`。因此，进程返回码为 0 只是必要条件，不足以证明训练产生了可用于显存或吞吐建模的观测。

## 7. 主要 artifact

- Workload V2：`offline_experiments/artifacts/h800_vl_business_workload_profiles_manifest_v2.json`
- VL 资源特征：`offline_experiments/artifacts/h800_vl_resource_features_manifest_v1.json`
- 图片静态验收：`offline_experiments/artifacts/h800_frozen_vl_decomposition_static_validation_v1.json`
- 视频静态验收：`offline_experiments/artifacts/h800_frozen_video_decomposition_static_validation_v1.json`
- 视频完整解码验收：`offline_experiments/artifacts/h800_vl_video_full_decode_validation_v1.json`
- 合并 Canary 队列：`offline_experiments/matrix/h800_frozen_vl_combined_canary_v1.jsonl`
- 合并 Formal 队列：`offline_experiments/matrix/h800_frozen_vl_combined_formal_v1.jsonl`
- 自动接续状态：`offline_experiments/artifacts/h800_frozen_vl_combined_canary_autostart_state_v1.json`
- 自动接续日志：`offline_experiments/frozen_vl_staging/h800_frozen_vl_combined_canary_autostart_v1.log`
