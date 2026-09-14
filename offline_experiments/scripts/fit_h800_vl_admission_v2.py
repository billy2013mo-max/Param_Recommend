#!/usr/bin/env python3
"""VL 准入模型 V2：物理加法形式 + 支持域门禁。

这个模型只回答「装不装得下」。「要多少显存」由中心模型（学 allocated）回答，
两者并存、互不替代。

要解决的问题
------------
V6 的准入上界错放 5 个 OOM，全在 mbs=16。根因是两件事咬住：

  一、mbs 高的地方几乎没有成功样本。mbs>=8 只有 15 行成功、24 个 OOM。
      每条路由在 mbs>=8 处只有 0-1 个成功样本，斜率约束不住。
  二、OOM 行被丢弃（右删失），高 mbs 格子里只有「没炸的」进训练集，
      斜率系统性偏平。这是选择偏差。

归因：修好它的是形式，不是删失回归
----------------------------------
做了 2x2 消融，把「换形式」和「加删失」拆开（留一 OOM 交叉验证）：

    对数形式 + 丢弃 OOM   拒住 22/27    mbs=16 预测  57.7 GiB
    对数形式 + 删失       拒住 22/27    mbs=16 预测  90.7 GiB
    加法形式 + 丢弃 OOM   拒住 27/27    mbs=16 预测 437.2 GiB
    加法形式 + 删失       拒住 27/27    mbs=16 预测 437.1 GiB

删失回归确实注入了信息——它把对数形式 mbs=16 的预测抬高 57%（57.7 -> 90.7）——
但没能越过 132.8 GiB 安全线，拒绝数一个没多。真正起作用的是加法形式。

删失项在加法形式下完全失活，原因可算出来：模型已预测 437 GiB，删失点 140 GiB，
z = (log140 - log437)/0.167 = -6.4，1-Phi(-6.4) 约等于 1，似然贡献 log(1) = 0，
梯度也是 0。所以两列数字相同不是巧合。

代码里仍保留删失（它免费、且在形式退化时是唯一的信息来源），但功劳记在形式上。

形式
----
    reserved = P_route + A_route x mbs x (tokens/4096)^delta

    P_route   与 mbs 无关：参数、优化器状态、碎片基底
    A_route   单位微批、单位长度的激活开销
    delta     长度指数，全局共享

「mbs 翻倍则激活翻倍」是硬约束而非待学系数——激活显存正比于 batch x seq，
且不被 ZeRO 分片。V5/V6 那种每路由独立学 log_mbs 的做法，在每路由只有 1 个
高 mbs 样本时必然学歪。

物理验证：P 对上了权重显存
--------------------------
P 是纯从数据学的，从未告知参数量，但它复现了权重的 bf16 占用：

    qwen2p5_vl_7b   权重 15.4 GiB   学出 P = 15.1 / 15.4 / 17.3
    qwen2p5_vl_3b   权重  7.0 GiB   学出 P = 8.5 - 9.3
    qwen3_vl_4b     权重  8.3 GiB   学出 P = 8.9 - 11.1
    qwen3p5_4b      权重  8.7 GiB   学出 P = 9.9 / 10.6

ZeRO-3 路由的 P 是权重/卡数的 1.9-3.8 倍，因为它要为逐层聚合留缓冲。
这说明加法形式不是拟合得巧，是真把「参数部分 + 激活部分」分对了。

支持域
------
同一套检查暴露了边界：只有 2-6 行数据的路由，P 学成 0.0 或 98.1 GiB（权重才 4.0），
完全不可信。所以只在「>=10 行且 >=2 个 mbs 档」的路由上服务，其余返回超出支持域。

门槛为什么是 10 行：做了子采样实验——从现有 22 条厚路由里每条抽 k 行重拟合，
看 P 还对不对得上权重、留一 OOM 还拒不拒得住：

    k=4    P 异常 2.8 条   留一 OOM 拒住 26.8/27   真误拒 1.2%
    k=6    P 异常 0.8 条   留一 OOM 拒住 27.0/27   真误拒 1.8%
    k=8    P 异常 0.2 条   留一 OOM 拒住 27.0/27   真误拒 2.6%
    k=10   P 异常 0.0 条   留一 OOM 拒住 27.0/27   真误拒 2.7%
    全量   P 异常 0.0 条   留一 OOM 拒住 27.0/27   真误拒 2.5%

10 行即等同全量 592 行。原因是每条路由只学 P 和 A 两个参数，而长度指数 delta
是全局共享的、已被现有 619 行钉死——新路由只需辨识自己的 P 和 A。
门槛原为 20，那是照搬 V5（每路由 3-4 个系数）定的，对加法形式过严。
这比按 mbs 卡阈值更准：问题不在 mbs 多大，而在这条路由的 P/A 是否可辨识。

准入判据
--------
    upper = pred x exp(1.645 x sigma)     # P95
    拒绝，如果 upper > 安全线

sigma 是拟合出来的残差尺度，不是扫出来的倍率——前者可解释，后者不可。
"""
from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[1]
TABLE = REPO / "artifacts" / "h800_vl_memory_table_v1.json"
OUT = REPO / "artifacts" / "h800_vl_admission_v2.json"

CAPACITY_BYTES = 150142189568.0
SAFE_LIMIT_BYTES = 142635080089.6
GIB = 1024.0 ** 3
REF_TOKENS = 4096.0
Z_P95 = 1.645

MIN_ROWS = 10
MIN_MBS_LEVELS = 2

# 物理门禁：P 至少要装得下权重。加法形式把 P 解释为「参数 + 优化器状态 +
# 碎片基底」，所以 P 小于权重/卡在物理上不可能。留 10% 余量给测量噪声与
# 权重口径差异（total_parameters 未必含全部 buffer）。
#
# 触发这条门禁的路由不是「样本少」，而是形式在那里不适用：反推可证 P 为负。
# 例如 qwen2p5_vl_7b::g2_z2::gc0，同长度下 mbs 1->2 的 reserved 增量
# (97.32-47.90=49.42 GiB) 大于 mbs=1 时的全部占用 (47.90)，按「mbs 翻倍则
# 激活翻倍」反推得 P = -1.52 GiB。硬约束在该路由上不成立，只能排除。
MIN_P_OVER_WEIGHTS = 0.9


def read_json(path):
    with path.open() as handle:
        return json.load(handle)


def route_of(row):
    return "%s::g%s_z%s::gc%d" % (row["model_id"], row["gpu_count"],
                 row["zero_stage"], int(bool(row["gc"])))


def nll_and_grad(theta, n, obs, cen):
    """负对数似然与解析梯度。

      成功行  L = 0.5 r^2 + log sigma,  r = (log y - log mu)/sigma
              dL/dmu = -r/(sigma mu),   dL/dlogsigma = 1 - r^2
      删失行  L = -log S(z),  z = (log C - log mu)/sigma,  h = phi(z)/S(z)
              dL/dmu = -h/(sigma mu),   dL/dlogsigma = -h z
      共用    dmu/dlogP = P,  dmu/dlogA = mu - P,  dmu/ddelta = (mu-P) ln(t/T)

    有限差分在 76 个参数下每次梯度要 77 次求值，而留一验证要拟合 27 次。
    解析梯度把开销降到 1/77，并且能真收敛（V1 撞到迭代上限）。
    """
    p = np.exp(theta[:n])
    a = np.exp(theta[n:2 * n])
    delta = theta[2 * n]
    log_sigma = theta[2 * n + 1]
    sigma = math.exp(log_sigma)
    grad = np.zeros_like(theta)
    total = 0.0

    for censored, pack in ((False, obs), (True, cen)):
        idx, mbs, tokens, value = pack
        if not len(idx):
            continue
        ratio = tokens / REF_TOKENS
        active = a[idx] * mbs * np.power(ratio, delta)
        mu = np.maximum(p[idx] + active, 1.0)
        gap = (np.log(value) - np.log(mu)) / sigma
        if censored:
            log_sf = norm.logsf(gap)
            total += float(-np.sum(log_sf))
            hazard = np.exp(norm.logpdf(gap) - log_sf)
            dmu = -hazard / (sigma * mu)
            grad[2 * n + 1] += float(-np.sum(hazard * gap))
        else:
            total += float(np.sum(0.5 * gap ** 2 + log_sigma))
            dmu = -gap / (sigma * mu)
            grad[2 * n + 1] += float(np.sum(1.0 - gap ** 2))
        np.add.at(grad, idx, dmu * p[idx])
        np.add.at(grad, n + idx, dmu * active)
        grad[2 * n] += float(np.sum(dmu * active * np.log(ratio)))
    return total, grad


class Admission:
    def __init__(self, routes):
        self.routes = list(routes)
        self.index = {name: i for i, name in enumerate(self.routes)}
        self.n = len(self.routes)

    def pack(self, rows, censored):
        idx = np.array([self.index[route_of(r)] for r in rows], dtype=int)
        mbs = np.array([float(r["mbs"]) for r in rows])
        tokens = np.array([float(r["total_tokens"]) for r in rows])
        value = (np.full(len(rows), CAPACITY_BYTES) if censored
               else np.array([float(r["reserved_bytes"]) for r in rows]))
        return idx, mbs, tokens, value

    def fit(self, success, censored, use_censored=True):
        obs = self.pack(success, False)
        empty = (np.array([], int), np.array([]), np.array([]), np.array([]))
        cen = (self.pack(censored, True)
               if (use_censored and censored) else empty)

        theta = np.zeros(2 * self.n + 2)
        grouped = defaultdict(list)
        for row in success:
            grouped[route_of(row)].append(row["reserved_bytes"])
        for name, i in self.index.items():
            seen = grouped.get(name)
            if seen:
                low, high = min(seen), max(seen)
                theta[i] = math.log(max(low * 0.5, 1e8))
                theta[self.n + i] = math.log(max((high - low * 0.5) * 0.5, 1e7))
            else:
                theta[i] = math.log(1e10)
                theta[self.n + i] = math.log(1e9)
        theta[2 * self.n] = 1.0
        theta[2 * self.n + 1] = math.log(0.15)

        options = {"maxiter": 30000, "maxfun": 60000,
                   "ftol": 1e-16, "gtol": 1e-10}
        result = minimize(nll_and_grad, theta, args=(self.n, obs, cen),
                  jac=True, method="L-BFGS-B", options=options)

        # 容差设得极严（ftol 1e-16 / gtol 1e-10），L-BFGS-B 在逼近最优时会
        # 因线搜索走不动而报 ABNORMAL，此时解可能已经是最优。重启一次会清掉
        # 内部累积的 Hessian 近似，常能越过；若重启也降不动目标函数，说明确
        # 实到了数值极限，保留更优的那个解并如实标 converged=False。
        self.restarts = 0
        while not result.success and self.restarts < 5:
            again = minimize(nll_and_grad, result.x, args=(self.n, obs, cen),
                      jac=True, method="L-BFGS-B", options=options)
            self.restarts += 1
            improved = again.fun < result.fun - 1e-12
            if again.fun < result.fun:
                result = again
            if not improved:
                break

        self.theta = result.x
        self.converged = bool(result.success)
        self.grad_norm = float(np.max(np.abs(result.jac)))
        self.exit_message = str(result.message)
        self.iterations = int(result.nit)
        self.nll = float(result.fun)
        self.used_censored = bool(use_censored and censored)
        return self

    def parameters(self, name):
        i = self.index[name]
        return math.exp(self.theta[i]), math.exp(self.theta[self.n + i])

    def predict(self, row):
        name = route_of(row)
        if name not in self.index:
            return None
        base, act = self.parameters(name)
        return base + act * row["mbs"] * (row["total_tokens"] / REF_TOKENS) ** self.delta

    def upper(self, row, z=Z_P95):
        value = self.predict(row)
        return None if value is None else value * math.exp(z * self.sigma)

    def admits(self, row, z=Z_P95):
        """True 表示放行。超出支持域返回 None（既不放行也不拒绝）。"""
        bound = self.upper(row, z)
        return None if bound is None else bound <= SAFE_LIMIT_BYTES

    @property
    def delta(self):
        return float(self.theta[2 * self.n])

    @property
    def sigma(self):
        return math.exp(self.theta[2 * self.n + 1])


def supported_routes(success):
    '''路由需 >=10 行成功样本、>=2 个 mbs 档。

    为什么以行数为主而不是 mbs 档数：加法形式里 mbs 是线性硬约束，A 不是
    每路由学的斜率，只要长度有变化，2 个 mbs 档就够把 P 和 A 分开。真正
    不可辨识的是只有 2 行数据的路由——那里 P 学成 0.0 或 98.1 GiB（权重才
    4.0 GiB）。先前定 >=3 个 mbs 档，错误排除了 4 条 41-44 行的最优路由。
    '''
    rows_by = Counter(route_of(r) for r in success)
    mbs_by = defaultdict(set)
    for row in success:
        mbs_by[route_of(row)].add(int(row["mbs"]))
    return sorted(name for name, count in rows_by.items()
           if count >= MIN_ROWS and len(mbs_by[name]) >= MIN_MBS_LEVELS)


def accuracy(model, rows):
    errors = [(model.predict(r) - r["reserved_bytes"]) / r["reserved_bytes"]
          for r in rows if model.predict(r) is not None]
    if not errors:
        return None
    errors = np.array(errors)
    return {"rows": len(errors), "mape": float(np.mean(np.abs(errors))),
        "bias": float(np.mean(errors)),
        "p90": float(np.percentile(np.abs(errors), 90)),
        "max_abs": float(np.max(np.abs(errors)))}


def by_mbs(model, rows):
    """按 mbs 分档看精度——外推风险就在这里。"""
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["mbs"])].append(row)
    out = {}
    for level, members in sorted(grouped.items()):
        errors = [(model.predict(r) - r["reserved_bytes"]) / r["reserved_bytes"]
              for r in members]
        out[str(level)] = {
            "rows": len(members),
            "median_actual_gib": statistics.median(
                r["reserved_bytes"] for r in members) / GIB,
            "median_predicted_gib": statistics.median(
                model.predict(r) for r in members) / GIB,
            "mape": float(np.mean(np.abs(errors))),
            "bias": float(np.mean(errors)),
        }
    return out


def admission_metrics(model, success, oom, z=Z_P95):
    """错放与错拒。另外分出「本就跑在安全线之上」的行——拒掉它们是余量该做的事。"""
    false_admit = [r for r in oom
           if model.upper(r, z) is not None
           and model.upper(r, z) <= SAFE_LIMIT_BYTES]
    rejected = [r for r in success
         if model.upper(r, z) is not None
         and model.upper(r, z) > SAFE_LIMIT_BYTES]
    above_line = [r for r in rejected if r["reserved_bytes"] > SAFE_LIMIT_BYTES]
    covered = sum(1 for r in success
          if model.upper(r, z) is not None
          and model.upper(r, z) >= r["reserved_bytes"])
    return {
        "z": z,
        "oom_rows": len(oom),
        "false_admit": len(false_admit),
        "success_rows": len(success),
        "coverage": covered / len(success) if success else None,
        "rejected": len(rejected),
        "reject_rate": len(rejected) / len(success) if success else None,
        "rejected_already_above_safe_line": len(above_line),
        "true_false_reject": len(rejected) - len(above_line),
        "true_false_reject_rate": ((len(rejected) - len(above_line)) / len(success)
                    if success else None),
    }


def leave_one_oom_out(routes, success, oom, use_censored=True, z=Z_P95):
    """留一 OOM 交叉验证——准入唯一诚实的评估。

    训练集上的错放数没有意义：那个 OOM 自己参与了拟合。
    """
    rejected = 0
    missed = []
    for i, held in enumerate(oom):
        rest = [r for j, r in enumerate(oom) if j != i]
        model = Admission(routes).fit(success, rest, use_censored)
        bound = model.upper(held, z)
        if bound is not None and bound > SAFE_LIMIT_BYTES:
            rejected += 1
        else:
            missed.append({"route": route_of(held), "tokens": held["total_tokens"],
                      "mbs": held["mbs"],
                      "upper_gib": bound / GIB if bound else None})
    return {"held_out": len(oom), "rejected": rejected,
        "reject_rate": rejected / len(oom) if oom else None, "missed": missed}


def physical_check(model, success):
    """P 应当复现权重显存。这是形式是否分对的独立证据。"""
    params = {}
    for row in success:
        params.setdefault(row["model_id"], row.get("total_parameters"))
    out = {}
    for name in model.routes:
        model_id, topology, _ = name.split("::")
        gpu_count = int(topology.split("_")[0][1:])
        zero_stage = int(topology.split("_")[1][1:])
        total = params.get(model_id)
        if not total:
            continue
        shard = gpu_count if zero_stage == 3 else 1
        weights = total * 2.0 / shard
        base, act = model.parameters(name)
        out[name] = {"P_gib": base / GIB, "A_gib": act / GIB,
                 "weights_per_card_gib": weights / GIB,
                 "P_over_weights": base / weights}
    return out


def main():
    rows = read_json(TABLE)["rows"]
    all_success = [r for r in rows
           if r.get("reserved_bytes") and r.get("total_tokens")]
    all_oom = [r for r in rows
           if r.get("classification") == "oom" and r.get("total_tokens")]

    # 第一道门禁：数据量。行数太少的路由 P/A 不可辨识。
    data_routes = supported_routes(all_success)

    # 第二道门禁：物理下界。先用第一道的结果试拟一次，把 P 装不下权重的
    # 路由剔掉，再用剩下的重新拟合。必须分两阶段——P 要拟合完才知道。
    probe_keep = set(data_routes)
    probe_success = [r for r in all_success if route_of(r) in probe_keep]
    probe = Admission(data_routes).fit(
        [r for r in probe_success if r["source"] != "prospective_v3"],
        [r for r in all_oom if route_of(r) in probe_keep], True)
    probe_check = physical_check(probe, probe_success)
    physically_rejected = sorted(
        name for name, check in probe_check.items()
        if check["P_over_weights"] < MIN_P_OVER_WEIGHTS)
    unjudged = sorted(probe_keep - set(probe_check))

    routes = [name for name in data_routes if name not in set(physically_rejected)]
    keep = set(routes)
    success = [r for r in all_success if route_of(r) in keep]
    oom = [r for r in all_oom if route_of(r) in keep]
    train = [r for r in success if r["source"] != "prospective_v3"]
    test = [r for r in success if r["source"] == "prospective_v3"]

    print("=== 支持域 ===")
    print("  第一道（数据量）：%d/%d 条通过（>=%d 行且 >=%d 个 mbs 档）"
          % (len(data_routes), len({route_of(r) for r in all_success}),
             MIN_ROWS, MIN_MBS_LEVELS))
    print("  第二道（物理）：剔除 %d 条（P/权重 < %.1f，形式在该路由不适用）"
          % (len(physically_rejected), MIN_P_OVER_WEIGHTS))
    for name in physically_rejected:
        check = probe_check[name]
        print("     %-34s P %5.2fG  权重/卡 %5.2fG  比值 %.3f"
              % (name, check["P_gib"], check["weights_per_card_gib"],
                 check["P_over_weights"]))
    if unjudged:
        print("  无权重数据、物理门禁未判定：%d 条（保留）" % len(unjudged))
    print("  最终支持域 %d 条" % len(routes))
    print("  行数 %d/%d，OOM %d/%d"
          % (len(success), len(all_success), len(oom), len(all_oom)))
    dropped = sorted({route_of(r) for r in all_success} - keep)
    print("  合计排除 %d 条" % len(dropped))

    model = Admission(routes).fit(train, oom, True)
    no_censor = Admission(routes).fit(train, oom, False)

    print("")
    print("=== 拟合 ===")
    print("  参数 %d 个（每路由 P、A + 全局 delta、sigma）" % (2 * len(routes) + 2))
    print("  收敛 %s，梯度无穷范数 %.2e" % (model.converged, model.grad_norm))
    print("  优化器退出：%s（迭代 %d 次，重启 %d 次）" % (model.exit_message, model.iterations, model.restarts))
    print("  负对数似然 %.4f%s" % (
        model.nll,
        "   <- 重启后降不动，解已到数值极限" if (
            not model.converged and model.restarts > 0
            and model.grad_norm < 1e-3) else ""))
    print("  delta %.4f   sigma %.4f" % (model.delta, model.sigma))

    print("")
    print("=== 精度（对 reserved）===")
    print("%-20s%7s%9s%9s%8s%8s" % ("", "行数", "MAPE", "偏差", "P90", "最大"))
    blocks = {}
    for label, rowset in (("训练集", train), ("验收（分布外）", test)):
        block = accuracy(model, rowset)
        blocks[label] = block
        print("%-18s%7d%8.1f%%%+8.1f%%%7.1f%%%7.1f%%" % (
            label, block["rows"], block["mape"] * 100, block["bias"] * 100,
            block["p90"] * 100, block["max_abs"] * 100))

    print("")
    print("=== 按 mbs 分档（外推风险所在）===")
    levels = by_mbs(model, train)
    print("%6s%7s%12s%12s%9s%9s" % (
        "mbs", "行数", "实测中位", "预测中位", "MAPE", "偏差"))
    for level, block in levels.items():
        print("%6s%7d%11.1fG%11.1fG%8.1f%%%+8.1f%%" % (
            level, block["rows"], block["median_actual_gib"],
            block["median_predicted_gib"], block["mape"] * 100,
            block["bias"] * 100))

    print("")
    print("=== 准入 ===")
    metrics = admission_metrics(model, train, oom)
    print("  错放 %d/%d" % (metrics["false_admit"], metrics["oom_rows"]))
    print("  覆盖 %.1f%%（P95 上界，设计上就该约 95%%）"
          % (metrics["coverage"] * 100))
    print("  拒绝 %d 行 = %.1f%%，其中 %d 行本就跑在安全线之上"
          % (metrics["rejected"], metrics["reject_rate"] * 100,
             metrics["rejected_already_above_safe_line"]))
    print("  真误拒 %d 行 = %.1f%%"
          % (metrics["true_false_reject"],
             metrics["true_false_reject_rate"] * 100))

    print("")
    print("=== 留一 OOM 交叉验证 ===")
    loo = leave_one_oom_out(routes, train, oom, True)
    loo_plain = leave_one_oom_out(routes, train, oom, False)
    print("  用删失   拒住 %d/%d = %.1f%%"
          % (loo["rejected"], loo["held_out"], loo["reject_rate"] * 100))
    print("  不用删失 拒住 %d/%d = %.1f%%   <- 两者相同即删失无贡献"
          % (loo_plain["rejected"], loo_plain["held_out"],
             loo_plain["reject_rate"] * 100))
    for item in loo["missed"]:
        print("     漏掉 %-28s tokens %-6d mbs %d"
              % (item["route"], item["tokens"], item["mbs"]))

    print("")
    print("=== 物理验证：P 是否复现权重显存 ===")
    checks = physical_check(model, success)
    ratios = [c["P_over_weights"] for c in checks.values()]
    print("  P/权重 比值：%.2f - %.2f，中位 %.2f"
          % (min(ratios), max(ratios), statistics.median(ratios)))
    for name in list(checks)[:4]:
        c = checks[name]

        print("     %-30s P %5.1fG   权重/卡 %5.1fG   %.2fx"
              % (name, c["P_gib"], c["weights_per_card_gib"],
                 c["P_over_weights"]))

    payload = {
        "schema": "sft_h800_vl_admission/v2_additive_physical",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "candidate_only",
        "production_admission_allowed": False,
        "answers": "装不装得下（reserved 是否超安全线）",
        "does_not_answer": "要多少显存——那是中心模型（学 allocated）的职责",
        "form": "reserved = P_route + A_route * mbs * (tokens/4096)^delta",
        "form_rationale": "显存 = 与 mbs 无关的部分（参数、优化器、碎片基底）"
               " + 激活部分（正比于 mbs，不被 ZeRO 分片）；"
               "mbs 翻倍激活翻倍是硬约束，不是待学系数",
        "attribution": {
            "what_fixed_it": "加法物理形式",
            "what_did_not": "删失回归（Tobit）",
            "ablation_leave_one_oom_out": {
                "log_form_drop_oom": "22/27",
                "log_form_censored": "22/27",
                "additive_drop_oom": "27/27",
                "additive_censored": "27/27",
            },
            "censoring_effect_on_log_form": "把 mbs=16 预测从 57.7 抬到 90.7 GiB（+57%），"
                        "但未越过 132.8 GiB 安全线，拒绝数不变",
            "why_censoring_inactive_here": "加法形式已预测 437 GiB，删失点 140 GiB，"
                       "z=-6.4，似然贡献与梯度均约为 0",
        },
        "support_domain": {
            "rule_1_data": "路由需 >=%d 行成功样本且 >=%d 个 mbs 档" % (MIN_ROWS, MIN_MBS_LEVELS),
            "why_1": "只有 2-6 行的路由 P 学成 0.0 或 98.1 GiB（权重才 4.0），不可辨识",
            "rule_2_physical": "试拟后 P/权重每卡 >= %.1f" % MIN_P_OVER_WEIGHTS,
            "why_2": "P 含参数本身，P 装不下权重说明「mbs 翻倍则激活翻倍」"
                     "这条硬约束在该路由不成立（反推 P 为负），形式不适用，"
                     "留着会在外推时把需求预测得过低而放行会炸的配置",
            "routes_in": routes,
            "routes_out": dropped,
            "rejected_by_data_volume": sorted(
                {route_of(r) for r in all_success} - set(data_routes)),
            "rejected_by_physical_check": physically_rejected,
            "physical_check_unjudged_kept": unjudged,
            "probe_ratios_of_rejected": {
                name: probe_check[name] for name in physically_rejected},
        },
        "delta": model.delta,
        "sigma": model.sigma,
        "converged": model.converged,
        "gradient_inf_norm": model.grad_norm,
        "optimizer_exit": {
            "message": model.exit_message,
            "iterations": model.iterations,
            "restarts": model.restarts,
            "negative_log_likelihood": model.nll,
            "at_numerical_limit": bool(
                not model.converged and model.restarts > 0
                and model.grad_norm < 1e-3),
            "note": "容差设为 ftol 1e-16 / gtol 1e-10，L-BFGS-B 逼近最优时会因"
                    "线搜索走不动报 ABNORMAL。at_numerical_limit=true 表示已从"
                    "该解重启优化且目标函数降不动（重启后迭代 0 次），即解已到"
                    "数值极限，converged=false 是终止方式而非拟合质量问题。"
                    "若 at_numerical_limit=false 而 converged=false，才是真没拟合好。",
        },
        "admission_rule": "reject if pred * exp(%.3f * sigma) > safe_limit" % Z_P95,
        "safe_limit_bytes": SAFE_LIMIT_BYTES,
        "capacity_bytes": CAPACITY_BYTES,
        "route_parameters": {
            name: {"P_bytes": model.parameters(name)[0],
                "A_bytes": model.parameters(name)[1]}
            for name in model.routes},
        "accuracy": blocks,
        "accuracy_by_mbs": levels,
        "admission_train": metrics,
        "acceptance_leave_one_oom_out": loo,
        "acceptance_leave_one_oom_out_without_censoring": loo_plain,
        "physical_check": checks,
        "censoring_kept_because": "免费，且在形式退化时是唯一信息来源；"
                    "但本模型的拒绝能力不依赖它",
    }
    with OUT.open("w") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
    print("")
    print("产物 -> %s" % OUT)


if __name__ == "__main__":
    main()
