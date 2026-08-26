# RTX 4090 上 VL 与 Qwen3.5 的显存/吞吐观测结果

**日期**：2026-08-25
**机器**：4×RTX 4090（Ada, sm_89），24564 MiB/卡，PCIe 无 NVLink
**总样本**：640 组作业，成功 386 组，OOM 242 组，软件失败 12 组（全部
`sdpa+packing`，已解释）

---

## 1. 要干的事

VL（视觉-语言）和 Qwen3.5 系列在 H800（Hopper, sm_90）上已完成过大矩阵扫描，
但 4090 侧完全空白。目标是把 4090 上的可行域和最优路径实测清楚，为参数推荐
系统在消费卡场景下的推荐提供证据。

## 2. 现状

640 组作业已跑完并落地：

| 矩阵 | 目的 | 总数 | 成功 | OOM | 失败 |
|---|---|---:|---:|---:|---:|
| VL 多卡矩阵 | Qwen2.5-VL-3B / Qwen3-VL-4B × f∈{1,2,4} × MBS∈{1,2,4} × 卡数 × ZeRO | 108 | 44 | 64 | 0 |
| Qwen3.5 FA2 主矩阵 | 0.8B/4B/9B × 文本+图像 × MBS × GC × ZeRO | 184 | 144 | 40 | 0 |
| Qwen3.5 sdpa 补矩阵 | 同上但用 sdpa 换 FA2，扫大 MBS | 144 | 85 | 59 | 0 |
| Qwen3.5 packing 单卡 | attention × packing 模式 × cutoff | 60 | 33 | 15 | 12 |
| Qwen3.5 packing 多卡 | 同上扩到 2/4 卡 | 144 | 80 | 64 | 0 |
| **合计** |  | **640** | **386** | **242** | **12** |

数据落在：

- `offline_experiments/artifacts/rtx4090_vl_observations_20260823/`
- `offline_experiments/artifacts/rtx4090_qwen35_observations_20260825/`

## 3. 关键结论（先给结论）

### 3.1 Qwen3.5 在 4090 的最优路径是 **FA2 + neat_packing**，不是 sdpa

短样本上比 FA2 基线快 6–26 倍。对比表（`short_512` 数据集，4 卡最佳）：

| 模型 | FA2 无 packing | sdpa MBS=4 | FA2 + neat_packing | 提升倍数 |
|---|---:|---:|---:|---:|
| 0.8B | 1646 tok/s | 8146 tok/s | **43188 tok/s** | 26× / 5.3× |
| 4B | 1369 tok/s | 6439 tok/s | **10010 tok/s** | 7.3× / 1.6× |
| 9B | 1312 tok/s | 1034 tok/s | 全部 OOM | — |

我在早前的中间汇报里两次把结论定为「sdpa+大 MBS 是 4090 最优」，都是错的——
漏掉了 packing 这条线。用户问「4090 上跑 Qwen3.5 不能开 flash attention 吗？」
时我才意识到应该测 FA2 + packing，测出来才是真正的最优路径。这里明确记为
自己看漏。

### 3.2 4090 与 H800 在 Qwen3.5 上不能直接比较，原因是 attention 后端不同

- H800（sm_90）可用 FA3，llamafactory 的 Qwen3.5 patcher **不触发**，MBS 可自由取值
- 4090（sm_89）**用不了 FA3**（FA3 只支持 Hopper），只能 FA2
- FA2 会触发 patcher，把 position_ids 转成 `cu_seqlens` 交给 fla；此时
  fla 强制要求 batch=1
- 因此 H800 上「同一份配置」和 4090 上「同一份配置」跑出来是两个不同数据流

所以 H800 上的最优 MBS 不能外推到 4090。要在 4090 上追回吞吐，只有两条替代路径：
换 sdpa（放弃 FA2 优化，换来 MBS 自由），或开 packing（保留 FA2，把变长序列拼满）。
Packing 是更优的一条。

### 3.3 sdpa + packing 不可行

单卡扫的 12 组 `sdpa+neat_packing` 全部失败，报同一个错：注意力掩码尺寸不匹配
（`expanded size 2120 vs existing 2192`）。sdpa 走的是密集掩码路径，与 packing
生成的变长 `cu_seqlens` 在当前 transformers 版本下不兼容。这条组合已从多卡矩阵中
裁掉，不再占机时。

## 4. 举例说明：为什么 packing 差异这么大

以 **0.8B / multiturn / 单卡** 为例，一张表看完整变化：

| attention | packing | cutoff=2048 | cutoff=4096 | cutoff=8192 |
|---|---|---:|---:|---:|
| FA2 | 无 | 1886 tok/s | 1963 | 1768 |
| FA2 | pack | 3638 | 7006 | 9444 |
| FA2 | neat | 3696 | **7012** | 8980 |
| sdpa | 无（MBS=4） | 7890 (22400 MiB) | OOM | OOM |

两个观察：

1. **不开 packing 时 `cutoff_len` 不驱动吞吐也不驱动显存**——1886 → 1963 →
   1768，波动 ±5% 都算不上；显存也就 6480 → 7478 MiB 打转。这与 VL 单卡矩阵
   看到的规律一致：`cutoff_len` 与 pad 后长度、图片 token 数无关时，把它调大只
   是让 dataloader 分配一个更长的 buffer，不改变实际算子输入。
2. **一开 packing，`cutoff_len` 就成了主要吞吐驱动**——c2048 到 c8192 从 3638
   涨到 9444，2.6 倍。原因是拼接后每条序列的长度直接等于 `cutoff_len`，长
   序列摊薄了每 step 的开销。

对应地，显存也重新被 cutoff 拉起来：c8192 的 17610 MiB 已经吃掉七成显存。所以
4090 上跑 packing 时 `cutoff_len` 是显存与吞吐的联合驱动，得当变量扫，不能定值。

## 5. VL 侧可行域（Qwen2.5-VL-3B / Qwen3-VL-4B）

单卡 24GB 的实测边界，f 是每样本图片数：

| 模型 | 单卡 (MBS=1) | 单卡 (MBS=2) | 单卡 (MBS=4) |
|---|---|---|---|
| Qwen2.5-VL-3B | f≤4 均可 | f≤2 可，f=4 OOM | 全部 OOM |
| Qwen3-VL-4B | f≤2 可，f=4 OOM | 全部 OOM | 全部 OOM |

- 3B 比 4B 多一档 MBS 余量，来源就是 1B 参数量差，权重差近 2 GiB
- 多卡 ZeRO-2 能救回相当一部分单卡 OOM 点；ZeRO-3 有效但很贵（下节）
- 跟 H800 相比，4090 的核心限制是 24GB 上限，不是吞吐；一旦装得下，吞吐差距
  远比想象小

## 6. 三个横向对比（都是可复现观测）

### 6.1 ZeRO-2 vs ZeRO-3 在 PCIe 上：ZeRO-3 是纯亏（除非装不下）

Qwen3.5 9B / sdpa / 4 卡 / short_512：

| 配置 | tok/s | peak MiB |
|---|---:|---:|
| MBS=1 ZeRO-2 | **1034** | 23536 |
| MBS=1 ZeRO-3 | 127 | 14270 |
| MBS=2 ZeRO-3 | 311 | 19572 |

ZeRO-3 慢 8 倍。原因是 4090 之间只有 PCIe 通信，参数 all-gather 每步都要走
一次，慢链路拖住 GPU。所以 ZeRO-3 在 4090 上只用来救「ZeRO-2 装不下」的
边界情况，不当默认。

（H800 走 NVLink，ZeRO-3 代价小很多，这也是两台机器的推荐策略不该共用的
一个具体原因。）

### 6.2 sdpa 与 FA2 同 MBS 时几乎持平

22 组同模型 / 同数据集 / 同 MBS 的对比里，sdpa 与 FA2 的 tok/s 比值集中在
0.98–1.29，中位数 1.05。所以早前我说过的「sdpa 只有 FA2 的 23% 吞吐」是错的，
那是一次 36 tok/s 的测量噪声（复测得到 327.5 tok/s）——写这份文档时已把这
处观测从结论区分离。sdpa 的真实优势不在同 MBS 快，而是**能开 MBS>1**（patcher
只在 FA2 下触发）。

### 6.3 梯度检查点（gradient_checkpointing）的常规代价

Qwen3.5 单卡文本轨的 GC 开/关对照给出的吞吐损失分布：

- 中位数损失 5%–8%
- 少数配置损失接近 15%（cutoff 很短时 recompute 摊不开）
- 显存节省 40%–55%，把三档 MBS 里 OOM 的一档救回是常见收益

结论仍然是「显存紧就开、够用就关」，没有反常。

## 7. 落地位置与如何复用

产物按矩阵切分：

```
offline_experiments/artifacts/
├── rtx4090_vl_observations_20260823/
│   ├── matrix_results.jsonl              # 原 log-regex 版
│   ├── matrix_results_fixed.jsonl        # 重解析后的权威版
│   ├── single_card_sweep.jsonl
│   └── discarded_contaminated_rows.jsonl # 采样污染事故存档
└── rtx4090_qwen35_observations_20260825/
    ├── qwen35_fa2_matrix.jsonl           # 184 组
    ├── qwen35_sdpa_matrix.jsonl          # 144 组
    ├── qwen35_packing_single.jsonl       # 60 组
    ├── qwen35_packing_multi.jsonl        # 144 组
    ├── matrix_qwen35.py                  # sdpa/FA2 主矩阵脚本
    └── matrix_packing.py                 # packing 矩阵脚本
```

**吞吐字段说明**：所有 `tokens_per_s` 都从每个作业的 `trainer_state.json` 重新
解析而来（`source=trainer_state`），覆盖了原先 log-regex 抓错的 24 条记录。
最戏剧的一条：`qwen35_0p8b_short_512_fa2_pack_c8192_mbs1_g4_z2` 实际
42553 tok/s，被 regex 错抓成 4.26 tok/s，规避了近 10000 倍误差。

## 8. 已知踩过的坑（留给下一位维护者）

1. **容器 PID 命名空间不对齐**：nvidia-smi 报的是宿主 PID，容器内 `ps` 是另一
   套，父子树永远对不上。第一版 per-process 采样峰值全记 0。现改为
   「独占锁 + 启动前基线扣除」，见任一矩阵脚本的 `sample_peak`。
2. **GPU 并发污染**：同机上两个矩阵同时占卡会导致假 OOM。所有 4090 矩阵脚本
   共享 `/tmp/rtx4090_matrix.lock` 全局互斥锁。踩过一次，8 组 VL 记录被污染，
   保留在 `discarded_contaminated_rows.jsonl` 供审计。
3. **`train_tokens_per_second` 正则脆弱**：log 里的 progress 行有同名字段带
   小数值，会被 `[^0-9]*([0-9.]+)` 抓错位置。改从 `trainer_state.json` 读，
   见本文档第 7 节。
4. **`sdpa+packing` 必失败**：mask expanded size mismatch，与 transformers 版本
   有关，不是配置问题。已在 packing 矩阵的 `plan()` 里裁掉。

---

**本轮回答的边界**：只覆盖 4090 硬件本身；H800 的对照数据在
`05_泛化与业务验证/H800_Qwen3.5模型泛化验证结论_2026-07-29.md`，两份文档合起来
才能得到「按机型选路径」的完整结论。
