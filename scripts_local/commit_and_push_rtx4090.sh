#!/usr/bin/env bash
# 提交并推送 4090 建模这一轮的产出。
#
# 为什么需要这个脚本：Claude 所在的环境里没有任何 GitHub 凭据
# （无 credential.helper、无 ~/.git-credentials、无 GH_TOKEN、无 ~/.ssh、无 gh CLI），
# 公开仓库能匿名读但推送必须带 token，所以推送这一步只能由你执行。
#
# 用法：
#   bash scripts_local/commit_and_push_rtx4090.sh              # 只本地提交，不推送
#   bash scripts_local/commit_and_push_rtx4090.sh --push       # 提交后用现有凭据推送
#   bash scripts_local/commit_and_push_rtx4090.sh --push --token <PAT>
#                                                              # 提交后用一次性 PAT 推送
#
# 这个脚本刻意不碰的东西（见末尾输出）：
#   * H800 混合注意力那一批脚本与 artifact —— 不是这轮的工作，混在一起提交会让
#     两条线的历史纠缠在一起
#   * offline_experiments/config/APPROVED_TO_RUN.json —— H800 阶段的审批文件
#   * 根目录那个 .zip 导出包 —— 看着像临时产物，不该进版本库
#   * offline_experiments/campaigns/** —— 被 .gitignore 整目录忽略（既有约定）

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

DO_PUSH=0
TOKEN=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --push)  DO_PUSH=1; shift ;;
    --token) TOKEN="${2:-}"; shift 2 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

echo "仓库   : $REPO_ROOT"
echo "分支   : $BRANCH"
echo "待推送 : $(git rev-list --count origin/"$BRANCH".."$BRANCH" 2>/dev/null || echo '?') 个既有 commit"
echo

# ---------------------------------------------------------------- 提交分组
# 每组是一个独立可回滚的 commit。只列真实存在且未被 ignore 的路径。
stage_group () {
  local title="$1"; shift
  local -a present=()
  for p in "$@"; do
    [[ -e "$REPO_ROOT/$p" ]] || continue
    git check-ignore -q "$p" && continue
    present+=("$p")
  done
  if [[ ${#present[@]} -eq 0 ]]; then
    echo "跳过（无可提交内容）: $title"
    return 1
  fi
  git add -- "${present[@]}"
  if git diff --cached --quiet; then
    echo "跳过（无实际改动）: $title"
    return 1
  fi
  echo "暂存: $title"
  printf '        %s\n' "${present[@]}"
  return 0
}

commit_group () {
  local subject="$1"; local body="$2"
  git commit -q -m "$subject" -m "$body" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
  echo "  → $(git log -1 --format='%h %s')"
  echo
}

# ---- 1) 共享 runner 的硬件认证通用化 ------------------------------------
if stage_group "共享 runner 硬件认证通用化" \
    offline_experiments/scripts/run_job.py; then
  commit_group \
"fix(run_job): 硬件认证从写死 H800 改为通用 SKU 一致性检查" \
"原有 declared_campaign_is_h800_140g 是执行门禁，在任何非 H800 活动上恒为假，
导致 RTX 4090 活动完全无法执行（4090 在 2026-07-17 采集的 1136 行数据早于
这套认证，其 status.json 里没有 runtime_hardware 记录）。

改为 declared_campaign_declares_matching_sku：活动必须声明具体 SKU，且其
training_scope.gpu_type 必须包含该 SKU 名。

H800 行为不变：该项在 H800 上原本恒为真，从 all() 里替换一个恒真项不改变
结果；原判定保留为 manifest 的信息字段，provenance 仍可检索。

安全性不降：活体硬件与声明是否一致，仍由 live_sku_is_declared_h800、
live_memory_matches_declared_sku、assigned_devices_homogeneous、
mig_is_not_enabled、topology_covers_assigned_set 共同保证。已实测两种伪造
场景（声明 4090 而活动写 H800、空声明）仍被拦下。"
fi

# ---- 2) 双卡联合显存建模：探针与拟合器 ----------------------------------
if stage_group "双卡联合显存建模代码" \
    offline_experiments/scripts/probe_joint_card_memory_alignment_v1.py \
    offline_experiments/scripts/probe_rtx4090_v3_memory_contract_v1.py \
    offline_experiments/scripts/fit_joint_card_memory_centre_v1.py \
    offline_experiments/scripts/fit_joint_card_v3_memory_v1.py; then
  commit_group \
"feat(memory): H800 + RTX4090 双卡联合显存模型（平台 V3 契约）" \
"把平台实际读取的 V3 显存契约（physics_anchored_shared_bounded_residual，
38 原始特征 → 144 基函数 + 独立 risk 头）在两张卡上联合拟合，让 Go 侧只需
实现一套公式。

不修改任何出厂模块：V3 那 9 个容量归一化特征原本在
benchmark_h800_memory_center_models_v1._feature_value 里除以一个写死的 H800
容量常量（150142189568），4090 容量差 5.91 倍会让这些特征全错。该文件是 V3
artifact 的 model_math 绑定源，改它会让平台预测器因 source binding drifted
拒绝启动。做法是把全部 38 个特征按每行自己的容量预先算好塞进记录，
_feature_value 对已存在的键原样返回，常量不会被执行到。

位恒等证明：531 行 H800 记录、4779 个特征值，重算值与出厂实现逐位相等，
不通过则 SystemExit。

关键结果（同口径 in-sample source-equal MAPE）：
  出厂 H800-only V3   0.1094
  联合模型 H800       0.1095   （差 0.0001，H800 实质未受影响）
  联合模型 RTX4090    0.1531   （新增覆盖）

发现并修复：出厂 upper_multiplier=1.0171 只在 H800 上标定，直接用于联合模型
会放行 168 个真 OOM 中的 18 个。改为按卡重标定（H800 0.9189 / 4090 1.2535），
折外两张卡均为 OOM 召回 100%、误放行 0；误拒 H800 4.06%、4090 24.29%。

另含 physical_shares 28 特征那版联合拟合（fit_joint_card_memory_centre_v1）。
那版契约选错了——平台不读它——但它证明了两卡数据可共享同一设计矩阵，保留作
方法验证与对照。"
fi

# ---- 3) 平台打包：4090 分支 --------------------------------------------
if stage_group "server_integration 4090 打包" \
    server_integration/artifacts/joint_card_v3_memory.json \
    server_integration/build_rtx4090_test_vectors.py \
    server_integration/testdata/test_vectors_rtx4090.json \
    server_integration/README.md; then
  commit_group \
"feat(server_integration): 打包 RTX 4090 分支（显存联合模型 + 测试向量）" \
"吞吐侧不需要重建模型：throughput_v5.json 本来就是双卡的
（cards: [h800, rtx4090]，带 card_component_raw_offsets /
card_residual_coefficients / card_inverse_efficiency_multipliers 三处 4090
条目）。之前只是没导出成向量、Go 侧没按卡分派。

新增 test_vectors_rtx4090.json：52 条，来自 rtx4090_20260717 实测活动，覆盖
26 种 (训练模式, zero, 梯度检查点, packing, 卡数) 组合、3 个模型规模、
37 条成功 + 15 条 OOM，每条给出两个头的全部中间量。自检：内部一致性 0 处
不符，15 条 OOM 零放行。

⚠ 准入乘子是按卡两个值，Go 侧必须按 card_id 分派，不能用单一常量。

未用 build_test_vectors.py 生成：H800UnifiedV3ThroughputV5Predictor 构造时
即断言运行时 H800 容量等于 V3 artifact 容量，输出里硬编码 hardware_id=h800，
还叠了 VL / hybrid / packing release 三层 H800 专属逻辑。4090 生成器直接跑
那两个冻结模型，也正是 Go 要实现的部分。

README 里「只有 H800」那段已替换为 RTX 4090 章节。"
fi

# ---- 4) 8B 补数的冻结预测 ----------------------------------------------
if stage_group "8B 补数冻结预测" \
    offline_experiments/scripts/freeze_rtx4090_qwen3_8b_supplement_predictions_v1.py \
    offline_experiments/artifacts/rtx4090_qwen3_8b_lora_supplement_frozen_predictions_v1.json; then
  commit_group \
"feat(rtx4090): 冻结 Qwen3-8B LoRA 补数实验的全网格预测" \
"按「先冻结预测、再实测」的规矩，在占卡之前先把联合 V3 模型对这批补数的
预测冻住，使这轮运行是对模型的检验而不是探索。

网格 90 格：29 格预测可放行、61 格预测拒绝；另有配置层排除 72 格
（单卡 zero2/3 与多卡 zero0）。附 8 个最贴近安全线的被拒格作边界探针——
那里模型最没把握，判错代价也最大。

为什么选 8B：4090 侧只有 0.6B/1.7B/4B，与 H800 的跨卡锚点只有 1.7B 和 4B。
8B 是既能向上扩展 4090 覆盖、又与 H800 重叠的最小规模。这是针对联合模型
4090 侧 24.3% 误拒率的对症措施——该诊断（数据覆盖不足而非超参问题）是假设，
需重拟合后用数字检验。"
fi

# ---- 5) 文档 ------------------------------------------------------------
if stage_group "建模文档" \
    "项目文档/02_显存模型/双卡统一显存中心模型V1_2026-08-29.md" \
    "项目文档/02_显存模型/双卡联合V3显存模型_2026-08-29.md" \
    "项目文档/模型几何参数清单_2026-08-27.md"; then
  commit_group \
"docs: 双卡联合显存建模两版记录 + 模型几何参数清单" \
"双卡联合V3显存模型_2026-08-29.md 是当前有效的那版（平台 V3 契约）。

双卡统一显存中心模型V1_2026-08-29.md 是契约选错的那版（physical_shares
28 特征），开头有更正说明，保留作方法验证——它证明了两卡数据能共享同一
设计矩阵、联合拟合不让任何一张卡退化、OOM 零误放行，这些结论对 V3 版仍
有效。文末 §12 记录了三套显存契约并存的事实，以及「判断平台在用哪套必须
从 server_integration 往里追」这条教训。

模型几何参数清单_2026-08-27.md 是 /wanqing-models 下 156 个目录的几何审计，
全部数值来自磁盘上的 config.json 与 safetensors 文件头，未使用任何记忆中的
HF 配置。含 30 个「显式声明 head_dim 与 hidden/heads 不等」的模型清单——
按推导填会全错，最大偏差是 Qwen3-Next-80B（声明 256、推导 128）。"
fi

# ---- 5) VL 与混合注意力：Go 移植规格 -----------------------------------
if stage_group "VL / 混合注意力 Go 移植规格" \
    server_integration/export_rtx4090_predictor_specs.py \
    server_integration/artifacts/rtx4090_vl_predictor_spec.json \
    server_integration/artifacts/rtx4090_qwen35_predictor_spec.json \
    server_integration/testdata/test_vectors_rtx4090_vl.json \
    server_integration/testdata/test_vectors_rtx4090_qwen35.json; then
  commit_group \
"feat(server_integration): 导出 4090 VL 与混合注意力预测器的 Go 移植规格" \
"平台侧不需要调用这两个 Python 预测器，只需要系数和参数以便用 Go 重写。
系数和特征名本来就在冻结 JSON 里，但 Go 实现还需要三样只存在于 Python 源码
里的东西，这次一并导出：

1. 预测公式本身
     center_mib = dot(memory_features, coefficients)   纯线性，不标准化、不取 exp，单位 MiB
     upper_mib  = center_mib + safety_margin_mib
     admitted   = (upper_mib + 800) < card_memory_mib
     tokens_per_second = exp(dot(throughput_features, coefficients))
   显存在 MiB 线性空间、吞吐在 log 空间，搞反差几个数量级。

2. 那个 800 MiB 的准入余量是硬编码在预测器代码里的，不在 artifact 里。
   只读 JSON 的 Go 实现会静默丢掉它，把装不下的配置放行。

3. 前置门必须在预测之前拦，否则会推荐秒崩的配置：
     VL   —— 单卡 zero-2/3 不可行（DeepSpeed 不支持，实测 7 秒框架崩溃）
     混合 —— ① sdpa + packing 必失败（掩码尺寸不匹配，实测 12 组全挂）
             ② 单卡 zero-2/3
             ③ FA2 时 mbs 必须为 1（patcher 转 cu_seqlens 后 fla 强制 batch=1，
                与 packing 无关）。含义：4090 上要 MBS>1 只能走 sdpa 放弃 packing，
                要 packing 只能 fa2+mbs=1。

测试向量：VL 336 条（240 条带完整中间特征向量）、混合 400 条（187 条带）。
建议 Go 侧先对齐中间特征向量再比最终结果——特征对上结果自然对，特征不对能
直接定位到哪一项算错。

自检：只用 spec 里的系数 + 向量里的特征重算（即 Go 要走的同一条路径），
两条线的 center / tok/s / admitted 全部 0 处不符。

另注：混合注意力的请求字段是 cutoff，不是 cutoff_len。"
fi

# ---- 6) 8B 补数的分阶段 v2 审批冻结器 ----------------------------------
if stage_group "8B 补数分阶段 v2 审批冻结器" \
    offline_experiments/scripts/freeze_rtx4090_8b_supplement_stage_approval_v1.py; then
  commit_group \
"feat(rtx4090): 8B 补数的分阶段 v2 审批冻结器（尚未跑通）" \
"validate_setup.py 冻的是 v1 设计（只有 file_sha256 + matrix_summary），而共享的
run_job.verify_approval 现在要求 v2 形状（allowed_job_ids、execution_order、
queue_binding、runtime_fingerprint_sha256、runtime_patch、provenance_binding）。
validate_setup.py 在 git 历史里从未输出过 allowed_job_ids，所以活动流水线路径
无法直接启动。

分阶段是必须的：throughput 与 scaling 队列在 memory 结果物化之前没有 job_id，
无法预先白名单。

⚠ 这个脚本还没跑通。它撞上第四层结构性不匹配：build_provenance_binding 把
provenance 里的源文件路径按活动根目录解析，而 provenance 是在 offline_experiments/
下抓的（路径形如 scripts/common.py），417 个条目全部判为 missing。根因是 v2
审批机制假设活动根目录本身是自包含的项目根（自带 scripts/、README.md、
EXPERIMENT_DESIGN.md），而 4090 活动一直是共享 scripts 的子目录——查过老活动，
没有 scripts/，也没有任何 v2 promote 回执，即从来没有 4090 活动在 v2 门禁下跑过。

先入库留痕；要跑通需要把活动重建成隔离项目副本，那是单独一件事。"
fi

# ---- 7) 本地提交推送脚本 ------------------------------------------------
if stage_group "本地提交推送脚本" scripts_local; then
  commit_group \
"chore: 加提交推送脚本（Claude 环境无 GitHub 凭据，推送需人工执行）" \
"Claude 所在环境没有任何 GitHub 凭据（无 credential.helper、无 ~/.git-credentials、
无 GH_TOKEN、无 ~/.ssh、无 gh CLI），公开仓库能匿名读但推送必须带 token。
这个脚本把工作按逻辑分组提交，并把推送留给人工执行。

刻意不碰：H800 混合注意力那一批（不是同一条线）、H800 的审批文件、
根目录的导出压缩包。末尾列出被 .gitignore 挡住、可能需要放行的两处。"
fi

# ---- 8) 放行联合 V3 的完整验证报告 --------------------------------------
if stage_group "放行联合 V3 完整验证报告" \
    .gitignore \
    offline_experiments/artifacts/joint_card_v3_memory_v1.json; then
  commit_group \
"chore(gitignore): 放行双卡联合 V3 显存模型的完整拟合+验证报告" \
"artifacts 默认全忽略，白名单原本只放行 *frozen* / *safety_upper* /
*model_inventory* / rtx4090_*_predictor 等模式，joint_card_v3_memory_v1.json
不匹配任何一条。

这份 48K 的报告是「为什么选这套系数」的唯一完整记录：40 个候选卡特征集 × alpha
的对比、H800 位恒等检查（531 行 / 4779 个特征值逐位相等）、跨卡外推、留出整个
模型、以及乘子重标定的全过程（出厂 H800-only 值 1.0171 会放行 168 个真 OOM 中的
18 个，按卡重标定为 H800 0.9189 / 4090 1.2535 后两卡均零误放行）。

打包进 server_integration 的是精简版，只带系数与摘要指标，不含这些验证细节。
丢了这份就无法复核选型是否站得住。"
fi

# ---------------------------------------------------------------- 推送
echo "==================== 本地提交完成 ===================="
git --no-pager log --oneline origin/"$BRANCH".."$BRANCH" | sed 's/^/  /'
echo

if [[ $DO_PUSH -eq 0 ]]; then
  cat <<'EOF'
未推送（没加 --push）。推送方式二选一：

  # 1) 已配好凭据
  bash scripts_local/commit_and_push_rtx4090.sh --push

  # 2) 用一次性 PAT（不会写入磁盘，只用于这一次推送）
  bash scripts_local/commit_and_push_rtx4090.sh --push --token <你的PAT>

  # 3) 让 git 记住凭据，之后就能直接 git push
  git config credential.helper store && git push
EOF
else
  if [[ -n "$TOKEN" ]]; then
    URL="$(git remote get-url origin | sed -E "s#https://#https://${TOKEN}@#")"
    echo "推送中（使用一次性 PAT，不写入磁盘）..."
    git push "$URL" "$BRANCH"
  else
    echo "推送中（使用现有凭据）..."
    git push origin "$BRANCH"
  fi
  echo "推送完成。"
fi

echo
cat <<'EOF'
==================== 刻意未提交的内容 ====================
以下都不在本脚本范围内，需要你自己判断：

1. H800 混合注意力那一批（不是这轮工作，单独提交更清晰）
     offline_experiments/scripts/{prepare,freeze,refit,evaluate,analyze}_h800_hybrid_*.py
     offline_experiments/scripts/build_h800_vl_throughput_table_v1.py
     offline_experiments/scripts/{prepare,freeze}_h800_vl_vision_activation_supplement_v1.py
     offline_experiments/artifacts/h800_hybrid_*.json
     offline_experiments/tests/test_hybrid_memory_admission_track.py
     项目文档/06_进展与交接记录/H800准入轨启动与环境修复记录_2026-08-28.md

2. H800 阶段的审批文件（内容我没看过，不该由我代提）
     offline_experiments/config/APPROVED_TO_RUN.json

3. 根目录那个导出压缩包（看着像临时产物）
     音视频效果校验gsb方案2-V6-export-*.zip

4. 被 .gitignore 挡住的两处，可能需要你决定是否放行：
   * offline_experiments/campaigns/**  —— 整目录忽略（既有约定）。
     新活动 rtx4090_20260830_8b 的 config / matrix / runtime 因此只存在于本地，
     包括那个绑定了 design SHA 的审批文件。
   * offline_experiments/artifacts/joint_card_v3_memory_v1.json
     —— 完整的联合 V3 拟合+验证报告（含 40 个候选对比、跨卡外推、留模型验证、
     乘子重标定全过程）。artifacts 默认全忽略，白名单只放行
     *frozen*/*safety_upper*/*model_inventory* 等模式，这个文件名不匹配。
     打包进 server_integration 的是精简版，不含验证细节。
     要入库的话，在 .gitignore 里加一行：
       !offline_experiments/artifacts/joint_card_v3_memory_v*.json
EOF
