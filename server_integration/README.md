# server_integration —— Go 参数推荐器接入产物

**日期**：2026-08-26
**目的**：把 Python 侧冻结的 H800 参数推荐器产物打包到一个稳定目录，
Go 服务只需读这里就够，不需要跨越 `offline_experiments/` 的实验现场
和它上面的白名单/黑名单规则。

## 目录结构

```
server_integration/
├── README.md                       ← 本文件
├── build_test_vectors.py           ← 只读驱动脚本；重新生成 test_vectors.json 用
├── artifacts/                      ← 冻结产物（Go 侧要拷贝或读取的所有东西）
│   ├── memory_v3.json              ← V3 显存模型（center + risk 双系数）
│   ├── memory_feature_inventory.json  ← 显存原始特征清单（V2 inventory，V3 沿用）
│   ├── throughput_v5.json          ← V5 吞吐模型（structured throughput）
│   ├── throughput_model_inventory.json  ← V5 用的模型 catalog
│   ├── theory_basis.json           ← 理论基础常量（FLOPs、带宽、TFLOPS 等）
│   ├── static_workload_profiles.json    ← 数据集类别 → 静态 workload 属性
│   ├── hardware_h800.json          ← H800 硬件描述（容量、带宽、TFLOPs）
│   ├── static_packing_policy.json  ← neat-packing 静态策略（Go 侧当只读事实读）
│   ├── packing_production_release.json  ← packing 发布门控（Go 侧当只读事实读）
│   └── dataset_profiles/           ← 3 个采样后的数据集 profile（jsonl）
│       ├── short_512.qwen3_nothink.jsonl
│       ├── multiturn_2048.qwen3_nothink.jsonl
│       └── longtail_8192.qwen3_nothink.jsonl
└── testdata/
    └── test_vectors.json           ← 5 个场景 × 全候选 × Python 侧中间量（用于 Go 单测）
```

## 产物来源与拷贝映射

所有 JSON 产物是从 `offline_experiments/` 用 `cp` 直接复制的冻结副本，未做任何
数值修改。原路径 → 本目录路径：

| 本目录路径 | 上游路径 |
|---|---|
| `artifacts/memory_v3.json` | `offline_experiments/artifacts/h800_unified_bounded_memory_candidate_v3.json` |
| `artifacts/memory_feature_inventory.json` | `offline_experiments/artifacts/h800_bounded_memory_v2_model_inventory_v1.json` |
| `artifacts/throughput_v5.json` | `offline_experiments/artifacts/structured_throughput_modeling.json` |
| `artifacts/throughput_model_inventory.json` | `offline_experiments/artifacts/model_inventory.json` |
| `artifacts/theory_basis.json` | `offline_experiments/artifacts/h800_theory_basis.json` |
| `artifacts/static_workload_profiles.json` | `offline_experiments/artifacts/static_workload_profiles.json` |
| `artifacts/hardware_h800.json` | `offline_experiments/config/hardware.json` |
| `artifacts/static_packing_policy.json` | `offline_experiments/artifacts/static_packing_policy_v1.json` |
| `artifacts/packing_production_release.json` | `offline_experiments/config/h800_text_packing_production_release_v1.json` |
| `artifacts/dataset_profiles/*.jsonl` | `offline_experiments/artifacts/dataset_profiles/*.jsonl` |

上游路径下的 SHA / schema / 冻结绑定都保持不变，Go 侧的 SHA 校验可以直接沿用。

## 为什么没有 VL/hybrid 相关的产物

Python 侧的 `H800UnifiedV3ThroughputV5Predictor` 在初始化时会强制加载
`vl_overlay`、`vl_model_inventory`、`hybrid_memory_artifact`、
`hybrid_model_inventory`、`hybrid_vl_safety_upper` 这 5 个额外产物 —— 因为
predictor 目前是**统一入口**，不管当前请求是不是 VL 都会做绑定校验，避免"某天来
了 VL 请求但缺产物"的失败。

Go 侧现在只走**纯文本 Qwen3 dense** 的路径，请求进来时 `model_id` 都在
`{qwen3_1p7b, qwen3_4b, qwen3_8b, qwen3_14b, ...}`，永远不会触发 VL/hybrid
分支。所以本目录**没有拷贝那 5 个 VL/hybrid 产物**：Go 侧的推理路径完全用不上，
拷过来反而制造"以为要用其实用不上"的困惑。当 Go 侧后续开始接 VL 时，再单独扩包
即可。

## test_vectors.json 是什么，怎么用

`testdata/test_vectors.json` 是 Go 单元测试的对齐样本。每一行 Go 计算出的
中间量都能在这里找到对应字段做 assertEqual。

### 覆盖的 5 个场景

| 场景 ID | model | 训练方式 | 数据集 | GBS | cutoff | packing | 意图 |
|---|---|---|---|---:|---:|:---:|---|
| S1-qwen3_8b-lora-short | qwen3_8b | lora | short_512 | 64 | 512 | 关 | 短样本 8B LoRA 基线 |
| S2-qwen3_8b-full-multiturn | qwen3_8b | full | multiturn_2048 | 64 | 2048 | 关 | FULL admission head |
| S3-qwen3_8b-lora-longtail-packing | qwen3_8b | lora | longtail_8192 | 64 | 8192 | **开** | packing 分支（physical_mbs 强制为 1） |
| S4-qwen3_14b-lora-multiturn | qwen3_14b | lora | multiturn_2048 | 128 | 2048 | 关 | 14B 中等负载 |
| S5-qwen3_1p7b-lora-short | qwen3_1p7b | lora | short_512 | 32 | 512 | 关 | 小模型全绿 |

每个场景都跑了 `candidate_generator.generate_candidates()` 的完整候选枚举（10-50 个），
没有截断。

### 顶层结构

```jsonc
{
  "schema": "sft_server_integration_test_vectors/v1",
  "generated_at_utc": "2026-08-26T00:00:00+00:00",
  "python_version": "...",
  "predictor_source_sha256": "<sha256(h800_unified_v3_throughput_v5_predictor.py)>",
  "capacity_bytes": 150142189568,
  "scenarios": [
    {
      "scenario_id": "S1-qwen3_8b-lora-short",
      "request": { ... 用户请求原样 ... },
      "capacity_bytes": 150142189568,
      "candidates_count": 50,
      "generator_output": {
        "comparison_group": "...",
        "scenario": { ... },
        "generation_policy": { ... },
        "rejected": [ ... 静态剪枝掉的配置 ... ]
      },
      "results": [
        {
          "input_index": 0,
          "request_id": "...",
          "candidate": { ... 生成的候选原样 ... },
          "memory": {
            "final_row": { ... predictor 报告里的 memory 段 ... },
            "intermediates": {
              "center": { raw, expanded, standardized, correction, ... },
              "risk":   { raw, expanded, standardized, correction, ... }
            }
          },
          "throughput": {
            "final_row": { ... predictor 报告里的 throughput 段（未准入时 null） ... },
            "intermediates": {
              "features", "standardized",
              "physical_components", "multipliers",
              "component_launch/compute/hbm/optimizer/communication",
              "step_base", "raw_correction", "correction",
              "log_step", "static_log_work", "log_throughput"
            }
          },
          "final_report_row": { ... predictor 完整返回的这一行 ... }
        }
      ]
    }
  ]
}
```

### Go 侧可以对齐哪些量

**内存路径**（V3）：
- `intermediates.center.raw` —— Go 侧 raw feature 向量应该逐位一致
- `intermediates.center.expanded` —— 展开基向量（`basis_kind` 已在同一记录里）
- `intermediates.center.standardized` —— 标准化后（decimal 十进制 ≤ 1e-10 应该完全相同）
- `intermediates.center.correction` —— shrinkage 之后
- `final_row.center_bytes`, `risk_guard_bytes`, `admission_upper_bytes`,
  `safe_limit_bytes`, `admitted`

**吞吐路径**（V5）：
- `intermediates.features` / `standardized` —— structured feature 向量
- `intermediates.physical_components` / `multipliers` —— 5 个组件（launch,
  compute, hbm, optimizer, communication）拆分
- `intermediates.step_base` / `roof` —— smooth roofline 组合
- `intermediates.raw_correction` / `correction` —— tanh 钳位前后
- `intermediates.log_step` / `log_throughput` —— 最终 log 空间的
  `log_throughput = static_log_work - log_step`
- `final_row.predicted_effective_tokens_per_second`

### 数值一致性验证

抽样验证脚本：随便挑一个 admitted 记录，`math.exp(intermediates.log_throughput)`
必须等于 `final_row.predicted_effective_tokens_per_second`（相对误差 < 1e-12）。
本次生成后已抽验通过。

## 如何重新生成 test_vectors.json

```bash
cd Param_Recommend
python server_integration/build_test_vectors.py
```

`build_test_vectors.py` 的做法：

1. `sys.path.insert(0, offline_experiments/scripts)` 后**只读**导入
   `H800UnifiedV3ThroughputV5Predictor`
2. 用 monkey-patch 包住 `benchmark_h800_memory_center_models_v1._predict_correction`
   和 `structured_throughput_modeling._predict_log_throughput`，记录每次调用的
   中间量到侧信道，然后**转调原函数**保证数值 byte-identical
3. 用 `candidate_generator.generate_candidates()` 静态枚举每个场景的候选
4. 用 `predictor.predict(candidates)` 跑推理
5. 把中间量按 `input_index` / `request_id` 挂回到最终报告行

**不会**修改任何 `.py` 源文件；只在运行时 rebind 模块级函数指针。

## 冻结绑定

- 上游 predictor 源码 SHA（当前）：`9f5e0036eccaa89c871e55923063d586b8fa8f1185eb587dad91effe11766817`
- 本次生成时的 git HEAD：`b7b41c48`
- 上游 artifacts 的 SHA 见 `test_vectors.json.scenarios[*].final_report_row`
  里的 `dataset_profile_sha256` 等字段，以及各 JSON 内嵌的 `schema` 字段

如果重新生成时 `predictor_source_sha256` 和 artifact SHA 都没变，`test_vectors.json`
应该逐字段相等（除 `generated_at_utc` / `python_version`）。

## 边界

- **场景只覆盖 Qwen3 dense 纯文本 SFT**（1.7B / 8B / 14B）。VL、hybrid、Qwen2.5
  系列都不在本次覆盖里。
- **H800 与 RTX 4090 都已打包**（2026-08-29 起）。见下面「RTX 4090」一节。
  4090 的 VL 与 Qwen3.5 独立预测器（`rtx4090_vl_v11_predictor.json`、
  `rtx4090_qwen35_v11_predictor.json`）仍未打包，它们是另外两条线。
- **候选集是静态枚举结果**，不是 Go 侧要复现的推荐结果 —— Go 侧的排序、准入、
  MBS 选择这些决策全由 predictor 报告里的 `final_report_row` 给出，`test_vectors`
  的角色只是"给同一份候选和请求，Go 应该算出和 Python 一样的中间量与结论"。

## RTX 4090

**吞吐不用改模型。** `artifacts/throughput_v5.json` 本来就是双卡的
（`cards: ["h800", "rtx4090"]`），带 `card_component_raw_offsets`、
`card_residual_coefficients`、`card_inverse_efficiency_multipliers` 三处 4090 条目。
之前只是没有导出成向量、Go 侧没有按卡分派。Go 只需要把 `card_id` 传成
`"rtx4090"`，其余公式完全一样。

**显存要用联合模型。** 出厂的
`h800_unified_bounded_memory_candidate_v3.json` 是 H800 专属
（`gpu_family: H800`，全文 0 次提到 card）。双卡联合重拟合的产物打包在
`artifacts/joint_card_v3_memory.json`。

### ⚠ 准入乘子是「按卡两个值」，不是一个常量

```json
"per_card_upper_multiplier": {
  "h800":    0.9189479086462471,
  "rtx4090": 1.2535290332832412
}
```

出厂那个 H800 单卡标定的 `1.0171` 用在 4090 上会**放行 168 个真 OOM 里的 18 个**
—— 用户会被告知"可以跑"然后炸掉。Go 侧必须按 `card_id` 取乘子。

准入规则本身不变：

```
admit iff max(centre_bytes, risk_bytes × upper_multiplier[card_id])
          <= safe_limit_fraction × capacity_bytes
```

### 向量

`testdata/test_vectors_rtx4090.json`，52 条，来自 `rtx4090_20260717` 实测活动，
覆盖 26 种 `(训练模式, zero, 梯度检查点, packing, 卡数)` 组合、3 个模型规模
（0.6B / 1.7B / 4B）、37 条成功 + 15 条 OOM。

每条向量给出两个头的全部中间量（原始特征、基展开维度、raw_correction、
correction、predicted_bytes；以及吞吐的 features、standardized、五个分量、
multipliers、roof、step_base、correction、log_step、log_throughput），
外加 `observed_outcome` / `observed_reserved_bytes` 作为真值对照。

重新生成：

```bash
python server_integration/build_rtx4090_test_vectors.py
```

注意它**不能**用 `build_test_vectors.py` 生成 —— `H800UnifiedV3ThroughputV5Predictor`
在构造时就断言运行时 H800 容量等于 V3 artifact 的容量，输出里硬编码
`hardware_id: "h800"`，还叠了 VL / hybrid / packing release 三层 H800 专属逻辑。
4090 的生成器直接跑那两个冻结模型，也正是 Go 要实现的那部分。

### 状态

`joint_card_v3_memory.json` 是 `analysis_only`，**没有**替换出厂的 H800 V3 artifact。
它的已知短板：4090 侧误拒率 24.3%（H800 侧 4.06%），根因是 4090 只有 29 个 source、
3 个模型规模。补数据是正解。
