# Qwen3-14B 四卡排序验收

固定条件：

- 物理 GPU：4、5、6、7
- 模型：Qwen3-14B Full SFT
- 数据：`baseline_2gpu`
- cutoff：2048
- 普通 Packing 开启，Neat Packing 关闭
- GBS：128
- GC：开启
- optimizer offload：关闭
- 每个候选：45 个 optimizer steps

待验证的 v4b 顺序：

1. ZeRO-3，MBS=16，GA=2
2. ZeRO-2，MBS=8，GA=4
3. ZeRO-3，MBS=8，GA=4

显存门控预测：

| 预测名次 | 候选 | operational P95 |
|---:|---|---:|
| 1 | Z3 / MBS=16 / GA=2 | 104.45 GiB/卡 |
| 2 | Z2 / MBS=8 / GA=4 | 121.77 GiB/卡 |
| 3 | Z3 / MBS=8 / GA=4 | 93.81 GiB/卡 |

验收首先比较第 6 步以后完整 GBS optimizer-step 的聚合 tokens/s；45
步累计吞吐、包含 epoch 边界的稳定吞吐和显存峰值作为辅助指标。若前两名
差距小于 3%，需反序复测后再下结论。
