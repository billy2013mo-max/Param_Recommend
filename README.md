# Param_Recommend

训练效率参数推荐：根据数据画像，联合推荐单卡批大小（MBS）、序列长度、显存策略与卡数，目标是保证不 OOM 的前提下训练更快。

## 目录结构

| 目录 | 说明 |
|---|---|
| `offline_experiments/` | 主实验工作区：显存模型、吞吐模型、Packing 联合搜索的脚本与测试（`scripts/`、`tests/`） |
| `项目文档/` | 人读文档，含项目总览、显存/吞吐模型、Packing 与业务验证（入口见 `00_目录与文档导航.md`） |
| `gpu_full_utilization_scheduler/` | GPU 满利用率调度器 |
| `mfu_qwen3_8b/`、`mfu_qwen3_14b/` | Qwen3 8B / 14B 的 MFU（模型浮点利用率）测量流水线 |

## 约定

- 命名规范与文档归档规则见 `项目文档/00_目录与文档导航.md` 第 9、10 节。
- 实验数据、结果、虚拟环境、内网仓库不进版本库，见 `.gitignore`。
