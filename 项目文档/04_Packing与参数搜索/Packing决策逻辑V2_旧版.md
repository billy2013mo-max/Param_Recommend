# Packing 决策逻辑 V2（旧版）

## 背景

neat packing 强制 `per_device_batch_size = 1`,因此在 packing 下 **`cutoff_len` 直接决定每步每卡的 token 数(= cutoff_len),是吞吐量的核心旋钮**。若把 cutoff 只设成"数据集分析得到的最长样本长度",在大显存 GPU 上会严重欠载,发挥不出性能。

## 1. cutoff_len 尽量加长,但上界不是 max_position_embeddings

在候选区间内取若干值作为 cutoff 候选:

```
[data_max_len  ..  上界]
```

- **下界 `data_max_len`**:数据集最长样本长度(向上取整到 512 的倍数),保证不截断真实样本。
- **上界不能盲取 `max_position_embeddings`**。吞吐对 cutoff **不是单调递增**,且越长激活显存越大。上界应取三者的最小值:

```
上界 = min( max_position_embeddings,
            显存可行上限,        # 由显存规划器给出:预测峰值 ≤ 0.95×容量
            GBS 可控上限 )        # 见第 2 节
```

> 「不知道 GPU 型号所以定不了上界」——这正是显存规划器要解决的:给定 GPU 型号,直接用它选出**预测可行的最大 cutoff**,无需盲扫到 max_position_embeddings。
> ⚠️ 注意规划器已知盲区:LoRA + 标准 dense 模型预测可信且偏保守;**全参微调大模型 / VL 视觉模型的显存会被低估**,对这两类选 cutoff 上界时需额外留余量。

## 2. global batch size 如何保证

packing 下 GBS 天然按**样本数**会波动(每个 pack 装的样本数是变的),这是 packing 的固有性质。真正需要稳定的是 **token/step**;样本级 GBS 用梯度累积来逼近:

```
n_pack        ≈ cutoff × pack_utilization / 平均样本长度   # 每个 pack 的样本数(用均值,不用中位数)
grad_accum    = max(1, round( GBS / (n_pack × GPU数) ))
期望样本级 GBS ≈ n_pack × GPU数 × grad_accum
```

**GBS 可控上限(关键约束,即原方案第 3 点)**:当一个优化器步(grad_accum=1)已经超过目标 GBS 时,由于 `grad_accum ≥ 1`,**物理上无法再降到目标 GBS**。因此必须满足:

```
n_pack × GPU数  ≤  GBS × (1 + 允许误差)
```

违反此式的 cutoff 直接淘汰。对小模型,这个上界通常比显存上限更紧。

## 3. 防止 step 数量过小

设每个 pack 平均样本量为 `n_pack`,总 pack 数 `packs`,则真实**优化器步数**为:

```
opt_steps = packs × epochs / (GPU数 × grad_accum)
```

要求 `opt_steps ≥ K`。注意与原方案的两点修正:

- 步数由 pack 数、GPU 数、grad_accum 共同决定,**不能只靠控制 n_pack**。
- 方向修正:cutoff 越长 → `n_pack` 越大 → packs 越少 → **步数越少**。所以"加长 cutoff"(第 1 节要吞吐)与"步数够多"(本节)是**同一个 cutoff 的取舍**,不是两条独立规则。

> `K` 的设定建议以**总优化器步数**为准(涵盖 warmup/decay 所需的最小步数,并考虑 epoch 杠杆),而非只看单个 epoch。具体取值需进一步调研。

## 4. 联合决策(把第 1、2、3 节合并成一次选择)

第 1 节(加长 cutoff 提吞吐)与第 3 节(限制 cutoff 保步数)方向相反,应在**同一循环里联合裁决**,而不是分设两条互相打架的规则:

```
给定 (model, GPU型号, dataset, 目标GBS):
  1. cutoff 候选 = [data_max_len .. 512步进 .. 上界]
  2. 对每个候选 cutoff:
       n_pack     = cutoff × pack_utilization / 平均样本长度
       grad_accum = max(1, round(GBS / (n_pack × GPU数)))
       opt_steps  = packs × epochs / (GPU数 × grad_accum)
       # 三个硬门槛,任一不过则淘汰:
       显存可行:  预测峰值 ≤ margin × 容量   (full-FT/VL 用更小 margin)
       GBS 可控:  n_pack × GPU数 ≤ GBS × (1+误差)
       步数达标:  opt_steps ≥ K
  3. 在存活候选中,选**预测吞吐最高**的 cutoff
```

这样上下界都已在门槛里施加,选吞吐最高者即在"填满 GPU"与"步数够多 / GBS 可控"之间取得平衡。

## 待调研 / 待实测

- `K` 的具体取值(对收敛效果的影响)。
- 样本级 GBS 波动对 SFT 效果是否有实质影响(packing 训练本就如此,预期无害,需实测确认)。
- 长 cutoff 下 `pack_utilization` 是否仍近似恒定(当前假设:cutoff ≫ 平均样本长度时基本持平)。
- 确认 neat packing 使用**块对角注意力 mask**(跨文档互不 attend)——这是正确性前提,也是"attention 代价随 cutoff 近似线性、非平方"的依据。
