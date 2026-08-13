# 真实业务数据与资源推荐验证

创建日期：2026-07-30

本目录隔离保存真实业务数据、数据画像、候选配置和预测结果，不修改冻结模型或历史实验结果。目前没有启动 GPU 训练。

## 数据与任务映射

| 数据集 | 本地版本 | 记录数 | 目标任务 |
|---|---|---:|---|
| `dataset-yt0jrk-1775555547` | V2 JSONL | 4,500 | Qwen3-8B LoRA SFT |
| `dataset-flieht-1763953980` | V4 JSONL | 4,500 | Qwen3-14B Full SFT |
| `dataset-pzfj38-1774860803` | V113 ZIP/JSONL | 7,108 | Qwen2.5-VL-7B-Instruct LoRA SFT |
| `dataset-vlhjn4-1760676139` | V1 JSONL | 54,142 | 短视频分享文案偏好学习 |
| `dataset-qype19-1770794488` | V7 ZIP/JSONL | 2,011 | 电商双图质量审核 VL-SFT |
| `dataset-cl63xd-1785209042` | V1 JSONL | 177,870 | 待业务画像 |
| `dataset-bxoblq-1785311859` | V1 JSONL | 71,014 | 待业务画像 |

七份数据均已下载并通过大小、SHA-256、JSON 解析和 schema 校验。V113
和 V7 压缩包只包含 JSONL，图像本体仍由 JSONL 中的内部对象引用提供。

## 已生成的画像

- 两个纯文本数据集使用各自模型的本地 tokenizer 和 LLaMA-Factory `qwen3_nothink` 模板，对全部记录生成 token 画像。
- VL 数据集使用 Qwen2.5-VL tokenizer 和 `qwen2_vl` 模板处理全部 7,108 条文本；对固定随机种子抽取 512 条记录、985 个唯一图片引用读取图片头，并按真实 resize/grid 规则估算视觉 token。
- `vlhjn4` 使用 Qwen3-8B tokenizer 生成参考长度画像；它尚未绑定最终训练模型，因此该长度只用于数据审计。
- `qype19` 使用 Qwen2.5-VL tokenizer 生成全量文本长度画像，并对固定随机种子抽取 256 条记录读取图片头，估算视觉 token。
- 画像产物不复制原始文本和图片 URL；VL 图片引用仅保存 SHA-256。
- 图片媒体没有被批量下载。

## 目录

- `datasets/`：从业务存储下载的原始对象与安全解压文件；
- `download_metadata/`：不含凭据的来源、校验值和格式记录；
- `profiles/`：逐条 token 数值画像，不含原始文本；
- `profile_summaries/`：数据分布和 cutoff 分析；
- `requests/`：提交给 predictor 的候选配置；
- `predictions/`：predictor 原始结果与最终推荐；
- `scripts/`：画像和推荐的可复现脚本；
- `runs/`：预留给后续真实训练回放。

推荐结果和适用边界见
[`真实业务数据参数推荐_2026-07-30.md`](./真实业务数据参数推荐_2026-07-30.md)。
