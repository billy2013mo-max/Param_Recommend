# Offline SFT efficiency experiments

本目录是 H800 1/2/4 卡 SFT 训练效率实验的可复现工程。设计说明见 [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md)。

当前开发机阶段只运行 Qwen3-1.7B、4B、8B、14B，统一使用 Qwen3 tokenizer 和 `qwen3_nothink` 模板。0.6B 不进入本轮；0.6B/1.7B 后续补 4090 结果，32B 延后，72B 的 8 卡或多机实验暂不执行。

常用的无训练命令：

```bash
cd /wanqing-develop/luowenjing/Param_Recommend/offline_experiments

# 数据已准备好时离线重算统计
/fine-tuning-launcher/.venv/bin/python scripts/prepare_datasets.py --reuse-source-pools --offline

# 生成不覆盖冻结基线的模型指纹（含 VL 视觉塔扩展）与实验矩阵
/fine-tuning-launcher/.venv/bin/python scripts/inventory_models.py \
  --output artifacts/model_inventory_vl_v1.json
/fine-tuning-launcher/.venv/bin/python scripts/generate_matrix.py

# 从既有 runtime manifest 生成 VL/组件冻结 sidecar；不修改严格 runtime v2 manifest
/fine-tuning-launcher/.venv/bin/python scripts/model_structure_manifest.py \
  --runtime-manifest <runtime_model_manifest.json> \
  --job-metadata <job_metadata.json> \
  --model-inventory artifacts/model_inventory_vl_v1.json \
  --output artifacts/model_structure_manifest_v1.json

# 证明六个切片的 Qwen3 静态长度/packing 与真实 processor 一致
/fine-tuning-launcher/.venv/bin/python scripts/validate_preprocessing.py

# 冻结运行时、源码、数据与模型复现指纹
/fine-tuning-launcher/.venv/bin/python scripts/capture_provenance.py

# 只运行四个严格限界的 1.7B smoke case，不会启动正式矩阵
/fine-tuning-launcher/.venv/bin/python scripts/run_smoke_tests.py

# 只预览调度，不训练
/fine-tuning-launcher/.venv/bin/python scripts/scheduler.py

# 物化短测候选并写入可恢复队列，不启动训练
/fine-tuning-launcher/.venv/bin/python scripts/run_pipeline.py --prepare-throughput

# 完整静态校验并冻结待审批设计（最后一步执行）
/fine-tuning-launcher/.venv/bin/python scripts/validate_setup.py

# 生成 H800 fresh prospective holdout 设计清单（只写 artifact，不建队列、不启动 GPU）
/fine-tuning-launcher/.venv/bin/python \
  scripts/prepare_h800_prospective_holdout.py

# 生成给数据/运行时负责人的 fresh profile 交付契约；不伪造 profile/data，也不允许执行
/fine-tuning-launcher/.venv/bin/python \
  scripts/prepare_h800_profile_requirements.py

# 整理既有四卡排序未结账项（只读，不重跑旧脚本）
/fine-tuning-launcher/.venv/bin/python \
  scripts/prepare_h800_rank_closeout.py

# 在取得 H800、fresh profile 和新 approval 后做最终门禁审计
# 仍然只写 readiness report，不物化队列、不启动训练；--strict 在任一项失败时返回 2
# 默认同时读取 artifacts/h800_fresh_profile_requirements_v1.json
/fine-tuning-launcher/.venv/bin/python \
  scripts/check_h800_campaign_gate.py --strict

# GPU 结果到位后，先运行下面的结果回收 adapter 生成验收输入；再用统一 evaluator
# 计算显存/排序/扩卡验收（只读）。空模板不能作为验收证据。
/fine-tuning-launcher/.venv/bin/python \
  scripts/prospective_acceptance.py \
  --input artifacts/prospective_acceptance_input_v1.json \
  --output artifacts/prospective_acceptance_report_v1.json \
  --scale-output artifacts/scale_out_acceptance_report_v1.json

# GPU 结果回收（先运行）：将冻结预测、queue manifest、canonical observations 和显式
# cross-card lower-bound 证据拼成 evaluator 输入；缺证据时只生成 blocked 输入
/fine-tuning-launcher/.venv/bin/python \
  scripts/build_h800_prospective_acceptance_input.py \
  --queue artifacts/h800_prospective_queue_manifest_v1.json \
  --prediction artifacts/h800_fresh_prediction_report_v1.json \
  --observations artifacts/h800_fresh_observations.jsonl \
  --scale-evidence artifacts/h800_fresh_scale_lower_bounds_v1.json \
  --output artifacts/prospective_acceptance_input_v1.json

# 生成 fresh holdout 的非执行队列 manifest；当前 gate 未通过时不会写 queue
/fine-tuning-launcher/.venv/bin/python \
  scripts/materialize_h800_prospective_holdout.py

# 先把上一步 manifest 中的 queue_binding_sha256 写入新 approval，
# 再重新运行 campaign gate；旧 approval 或 queue hash 不匹配时会拒绝写队列

# 只有 exact approval、profile/data 注册和最终 gate 全通过后才允许写 JSONL，仍不会启动 scheduler
/fine-tuning-launcher/.venv/bin/python \
  scripts/materialize_h800_prospective_holdout.py --write-queue

# 1. 规范化已完成的 H800 终态 attempt（不运行 GPU、不触碰队列）
/fine-tuning-launcher/.venv/bin/python scripts/export_h800_observations.py

# 2. 恢复并逐源文件验证历史证据（不运行 GPU、不触碰队列）
/fine-tuning-launcher/.venv/bin/python \
  scripts/recover_h800_historical_evidence.py --verify-source-files

# 3. 审计恢复后 H800 证据的拟合就绪度（不拟合、不发布系数）
/fine-tuning-launcher/.venv/bin/python scripts/audit_h800_calibration_readiness.py \
  --historical-recovery artifacts/historical_h800_recovery.json \
  --verify-historical-sources \
  --output artifacts/h800_calibration_readiness.json
```

`export_h800_observations.py` 默认写出
`artifacts/canonical_h800_observations.jsonl`，每行对应一个终态 attempt。由于
新运行从 `results/<job>/attempts/<attempt-id>/` 读取隔离的事件与证据；历史 flat
记录才使用 `status.json` 的 `[started_unix, finished_unix]` 时间窗。时间/显存跨
rank 取最大，token、sample 和 attention-pair 等工作量计数跨 rank 求和。OOM
必须有结构化 CUDA allocator 证据，只写右删失约束；缺失 peak 保持 `null`，不会
用前一个 step 或前一次运行的显存峰值填充。软件失败可作为诊断记录导出，但不会
成为显存或吞吐校准标签。

该命令只接受 H800 根目录，遇到 4090 campaign 或带 4090 身份的 job 会直接
拒绝。canonical 导出仍将没有 run-bound 完整 execution fingerprint 的历史运行标为
`legacy_incomplete`；这个原始标签不会自行升级，只有随后经过来源绑定和一致性检查的
historical recovery sidecar 才能赋予独立的历史证据等级。4090 流水线完成前不通过
本导出器吸收任何 4090 success/OOM。

新运行采用两段式证据链：启动前的 `sft_execution_inputs/v2` 绑定审批与队列、
任务快照、源码、runtime、实际分配的 H800/UUID/显存/拓扑、数据、静态模型清单
和 attempt-local DeepSpeed 配置；`on_train_begin` 再由每个 rank 写入按 Parameter
身份去重的运行时 tensor/module/trainable inventory 与 CUDA device attestation。
只有所有 rank 的逻辑 inventory 一致且逐卡身份与启动前硬件证据吻合，执行结束后
才生成 `sft_execution_fingerprint/v2`。某个 rank 缺失、hash 不匹配或重跑 attempt
不一致时均标为 `incomplete`，不能用于校准。逐 run 的 v2 SHA 只用于完整性证明；
可跨模型复用的校准 selector 使用单独的 `sft_runtime_mechanism/v2` SHA，它排除
模型、数据集、batch、卡数和 GPU 分配，但绑定实际执行机制与非敏感环境 allowlist。

当前就绪度报告位于 `artifacts/h800_calibration_readiness.json`。对 1143 条历史
H800 记录（1036 success、106 OOM、1 software failure）的追溯恢复得到：
`legacy_verified=97`、`legacy_consistent=1014`、`diagnostic_only=31`、`rejected=1`。
证据恢复等级与具体测量用途是两个独立维度；软件失败和 smoke/thermal 等诊断记录
不会进入显存或吞吐校准标签。

当前决策是 `fit_decision=historical_component_scoped_fit_allowed`、
`ready_for_full_bounded_fit_bundle=true`、`calibration_publishable=false`。这里的
“fit ready”只表示相应历史输入足以开始受约束的离线建模，不表示系数已经拟合、
验证或获准发布。分用途审计结果为：

- feasibility：1062 条，6 个可用 runtime cohort；
- memory boundary：594 条，2 个可用 runtime cohort；
- 正式 throughput：203 条，3 个可用 runtime cohort；另有 241 条短测只作低保真初筛；
- Profiler：19 条可用观测，校准特征 rank=3，并保留 16 条实际 holdout；旧辅助拟合的
  5 个绑定 holdout MAPE 为 5.02%，但它不是新理论 planner 的验收结果；
- packing：10 个完整的单次成对比较，只允许低置信效应估计，不能据此自动开启 packing。

新的 resource 分区按 `(model_id, train_type, dataset_id, target_gbs)` 跨所有 runtime
cohort 整体留出，共 84 个全局 LOOCV fold；其中 memory boundary 有 48 个场景，正式
throughput 有 78 个场景。旧实现按 cohort 分开得到的 214 个 folds 会让同一场景从另一
cohort 泄漏进训练集，现只保留为诊断统计。旧 `stage_decisions.json` 中的 78-fold 指标
也没有保存与当前 observation 集一致的 membership digest，因此只能作历史参考，不能
复用成新模型的验证分数。下一步是在无 GPU 环境中进行“理论公式 + 历史结果校准”的
有界拟合；`legacy_consistent` 证据必须做不确定性膨胀，runtime cohort 只作为折内
nuisance/fixed effect。95% 显存上界、零 unsafe holdout OOM 和吞吐排序验收通过之前，
不发布系数。4090 未完成结果继续与 H800 隔离，稍后按同样证据规则单独接入。

`artifacts/candidates/h800_calibration_*` 下原有的 109-row source candidate、执行队列
和 approval-design candidate 均是保留的历史审计材料，不是待执行 backlog。以下两个
schema 已永久退役：

- `sft_h800_calibration_candidate/v1`
- `sft_h800_calibration_approval_freeze/v1`

不得重新 prepare、freeze、promote 或交给 `run_job.py`/`scheduler.py` 执行；代码闸门
也会按 schema 拒绝，而不是依赖会变化的文件 SHA。若离线拟合确实暴露不可由现有证据
覆盖的缺口，需另建新 schema，将 recovery/readiness SHA、缺口、停止规则和验收标准
绑定到一份最小的前瞻实验设计，再单独审批。旧 artifact 不删除、不覆盖。

## H800 理论校准审计（已实现，`theory_only` / 不可发布）

无 GPU 的"理论公式 + 历史结果校准"有界拟合与验证已实现,产物固定为
`theory_only` / `bootstrap` / `nonpublishable`,绝不生成生产 profile:

```bash
cd /wanqing-develop/luowenjing/Param_Recommend/offline_experiments

# 全局场景 LOOCV 校准审计（不拟合生产 profile、不触碰 GPU/队列）
/fine-tuning-launcher/.venv/bin/python scripts/h800_theory_calibration.py

# 由上一步的诚实结果派生机器可读的证据缺口（不建队列、不发起 campaign）
/fine-tuning-launcher/.venv/bin/python scripts/h800_evidence_gap.py
```

`h800_theory_calibration.py` 在每个外层场景折内使用**内层场景折外(OOF)残差**估计安全
区间以消除泄漏:显存尾部用严格 leave-one-scenario-out(安全关键、无自由分组参数、实测
假安全更少),吞吐尾部用有界的分组 K 折(非安全关键,避免近二次的拟合开销)。显存尾部把
成功残差当**精确事件**、OOM 下界不等式当**右删失**,用 Kaplan–Meier 给出统计 P95,并与
运行安全上界 `max(censor_lower, safe_limit+1) − center` 分开报告(不臆造 OOM 峰值)。吞吐
排序与 1.8× 扩卡先经**预测 P95 ≤ 0.95×容量**的联合显存门控准入,再按吞吐置信下界排序;
regret 以"实测且不越 95% 线的最优安全候选"为分母,预测安全却 OOM 计为安全失败而非 regret。
校准器还在拟合前独立重算 basis 摘要与 schema、原子写出并绑定实现/折成员哈希、拒绝 NaN/Inf。

当前历史 bootstrap **明确未达发布门槛**(符合预期):显存 P95 覆盖 0.9289 pooled /
0.9063 scenario-equal(< 0.95);假安全 held-out OOM 2/106,**均为 `LoRA + ZeRO-3`
(non-GC)**——其显存中心估计偏低,故中心相对的护栏无法把 P95 抬过安全线;联合门控还有
10 个安全失败;吞吐 top-1 regret 0.203(> 0.10);两端均通过显存准入且保守/实测都 ≥ 1.8
的扩卡声明存活数为 0。证据层全为 legacy(无 native-v2 已验证锚点),`verified_calibration
_anchor_missing` 保持为 blocker(不折算成倍率)。`h800_evidence_gap.py` 据此点名各机制缺口
与前瞻验收标准,并显式声明 `creates_gpu_queue=false`、`requires_separately_approved_design=true`。

## H800 native-v2 显存校准：第一阶段

最新 native-v2 结果已通过一条独立的显存校准入口接入；旧的 historical bootstrap
产物保留不覆盖：

```bash
cd /wanqing-develop/luowenjing/Param_Recommend/offline_experiments

# CPU-only；只用 calibration 拟合，冻结后才读取 holdout
/fine-tuning-launcher/.venv/bin/python \
  scripts/h800_native_memory_calibration.py
```

产物为 `artifacts/h800_native_memory_calibration.json`。核心 unpacked 域限定
`MBS ∈ {1,2,4,8,16}`，packing ABBA 的 packed 与 paired-only unpacked 两侧均不进入
本次核心显存拟合。数据分区如下：

- historical memory boundary：594 条；
- native calibration：158 条（112 success、46 OOM）；
- 冻结的 native holdout：167 条（120 success、47 OOM），拟合使用数为 0；
- augmented candidate 的训练集共 752 条。

在完全相同的 native holdout 上，新增数据后的 augmented candidate 并非所有指标都变好：
成功样本 operational P95 覆盖率从 95.00% 提高到 96.67%，但假安全 OOM 仍为 2/47
（同一 `Qwen3-14B + LoRA + ZeRO-3 + non-GC + 2 GPU + MBS=2 + cutoff=4096`
配置的两次重复）；allocated-center mean MAPE 从 12.99% 变为 16.14%，reserved-center
mean MAPE 从 10.66% 变为 12.29%，安全 success 的误拒绝从 13 增至 19。calibration
场景外 LOOCV 为 93.75% pooled P95 coverage、0/46 false-safe OOM。因而第一阶段的
诚实结论是：新数据改善了尾部覆盖，但没有修复中心精度和 LoRA+ZeRO-3 假安全缺口，
当前 candidate 未通过显存验收，仍不可发布。报告同时保留 native-only 诊断拟合，但
不允许根据 holdout 结果反向选择或调参。

## H800 显存/吞吐 alternative challenger（无新实验）

在不启动 GPU 实验的前提下，增加了一次纯离线换模比较：

```bash
cd /wanqing-develop/luowenjing/Param_Recommend/offline_experiments

/fine-tuning-launcher/.venv/bin/python \
  scripts/h800_challenger_modeling.py
```

产物为 `artifacts/h800_challenger_modeling.json`，明确记录
`gpu_experiments_launched=false`、`queues_mutated=false`。两个 challenger 的模型族和
超参数均只由 native calibration 的 leave-scenario-out 结果选择，冻结后才读取 holdout：

- 显存：以 analytic reference 为基线，对物理分量占比以及
  `mode × ZeRO × GC × GPU × MBS × cutoff` 交互拟合 log-residual ridge；安全尾部继续使用
  成功样本场景外 conformal Q95 与 exact-selector OOM 下界 guard 的最大值。
- 吞吐：保留旧物理模型作为 base score，但不再追求绝对 step time 精确拟合；对同一场景
  内候选配置的 log-throughput 差值做 scenario-equal pairwise ridge，直接优化排序。

calibration-only 选出的显存模型为 `physical_shares + historical_weight=0.25 +
ridge_alpha=1`。nested calibration LOOCV 的 reserved-center mean MAPE 为 6.67%，
P95 coverage 为 99.11%，false-safe OOM 为 0/46。相同 native holdout 上：

- reserved-center mean MAPE：12.29% → 5.55%；
- allocated-center mean MAPE：16.14% → 5.64%；
- P95 coverage：96.67% → 96.67%；
- false-safe OOM：2/47 → 0/47；
- 安全 success 误拒绝：19 → 15。

calibration-only 选出的吞吐模型为 `physical pairwise residual +
historical_weight=0.25 + ridge_alpha=0.01`。相同 native holdout 的 79 个去重候选、
541 个可比较配置对上：

- pooled pairwise accuracy：70.98% → 89.46%；
- scenario-equal pairwise accuracy：72.75% → 91.94%；
- 全候选 top-1 regret：15.56% → 2.93%；
- 固定 GPU 数的 top-1 regret：16.49% → 2.10%；
- 固定 GPU 数 hit@10%：50.00% → 92.86%。

吞吐的绝对值仍只作诊断（scenario-equal MAPE 71.27% → 31.27%），生产用途应采用排序
score。该 holdout 在前一阶段已经被查看过，因此这里只能证明“同一历史测试集上的明显
改进”，不能当作新的无偏发布验收；两种 challenger 仍保持 nonpublishable。

**readiness v6 目前 fail-closed 保持在 v5(有意为之)**:`audit_h800_calibration_readiness.py
--verify-historical-sources` 校验历史 recovery 绑定的源文件时,发现 `EXPERIMENT_DESIGN.md`
在证据冻结后被改过、SHA 失配,按 fail-closed 拒绝出报告(`context_source_3_missing_or_changed`)。
这是防篡改校验的正确行为;不通过重生成 recovery 证据来"绕过"校验(那会削弱校验本身)。
彻底刷新到 v6 需要一套**单独授权**的操作:先定稿并更新被绑定的设计文档,再重生成
`historical_h800_recovery.json` 重新绑定其 SHA,最后重跑 readiness 审计。


训练执行被 `config/APPROVED_TO_RUN.json` 锁定。仓库中若残留旧审批，也必须与当前
源码、provenance、设计和队列逐 SHA 匹配；本轮证据链变更会使旧审批失效，不能
用于启动新的 H800 校准任务。
正式执行还会重新校验冻结清单中的每个 SHA256，并在 scheduler 启动和每个 job 启动前检查任务实际分配的 GPU 上没有外部计算进程。卡池和物理 GPU ID 必须以当前 campaign 的冻结设计为准；H800 fresh holdout 设计默认要求物理 GPU 4–7，若实际卡名或数量不匹配则 fail-closed，不把其他型号结果并入 H800 证据。
共享机资源临时冲突会触发等待和重新排队；只有明确 CUDA OOM 才会形成显存边界标签。旧审批文件不会跨源码、配置、队列或硬件指纹复用。
恢复场景可给 scheduler 添加 `--join-busy-pool`，先使用 GPU 1–4 中的空闲卡并动态接纳随后释放的卡。审批或 launcher 控制面失败会熔断发射并保留待运行任务，不会清空队列。

运行阶段顺序：

1. `memory_boundary_families.jsonl`
2. `materialize_jobs.py throughput-screen`（每个场景保留各卡数代表项并补充策略对照，最多 4 个候选，短测 2+4 steps）
3. `stage_decisions.py`（每个场景保留最低资源项与全局最快项 Top-2）
4. `materialize_jobs.py throughput`（只物化入围配置，正式测 3+10 steps）
5. `materialize_jobs.py scaling`（每个卡数只测一个静态优选策略，2+8 steps）
6. `materialize_jobs.py packing-memory`
7. `materialize_jobs.py packing-formal`
8. `materialize_jobs.py profiler`
9. `collect_results.py`

调度器在当前 campaign 的冻结卡池内尽量填满不相交 mask：最多四个单卡任务、两个双卡任务，或一个双卡加两个单卡任务；只有四卡任务独占实验池。实验池外仍允许其他用户运行任务。

## 实时 Dashboard

Dashboard 是只读的独立进程。它增量读取调度器、Trainer callback 和现有
`nvidia_smi.csv`，不会启动、停止训练，也不会额外调用 `nvidia-smi`。

启动：

```bash
cd /wanqing-develop/luowenjing/Param_Recommend/offline_experiments
/fine-tuning-launcher/.venv/bin/python -m dashboard --host 127.0.0.1 --port 8501
```

如果从其他机器访问，使用 SSH 隧道：

```bash
ssh -L 8501:127.0.0.1:8501 <user>@<experiment-host>
```

浏览器打开 `http://127.0.0.1:8501`。主要页面包括：

- 实验总览：阶段进度、运行/成功/OOM/失败数量和当前 GPU 卡池状态；
- 实时任务：配置筛选、滚动 step time、tokens/s、samples/s、MFU、显存和日志；
- 显存边界：最大可行 MBS、首个失败 MBS 和边界显存；
- 吞吐与 MFU：每场景最多 4 个代表候选先短测，最低资源与全局最快 Top-2 再做单次正式测量；展示 epoch 时间与 GPU-hours，失败或测量窗口/多 rank 汇总异常时自动重跑；
- 多卡扩展：1/2/4 卡加速比、并行效率和 1.8× 保守吞吐比扩卡规则；只有 `lower(2N) / upper(N) ≥ 1.8` 且两端通过显存门控时才允许相邻翻倍，否则保留当前卡数；
- Packing：严格成对的 epoch 时间收益与开关结论；
- 推荐结果：分别展示短测和正式测进度，可展开比较全部 GPU/ZeRO/GC/MBS 候选；短测只决定入围，最小资源、最快和默认方案只使用正式测结果。

实时接口使用 SSE，浏览器断线后会自动重连。服务自己的可重建索引位于
`runtime/dashboard/dashboard.sqlite`；原始实验文件始终是权威数据源。

常用接口：

```text
GET /healthz
GET /api/v1/overview
GET /api/v1/runs
GET /api/v1/runs/{job_id}
GET /api/v1/analysis/memory
GET /api/v1/analysis/throughput
GET /api/v1/analysis/scaling
GET /api/v1/analysis/packing
GET /api/v1/recommendations
GET /api/v1/events
```

运行 Dashboard 测试：

```bash
/fine-tuning-launcher/.venv/bin/python -m unittest -v dashboard.tests.test_dashboard
```
