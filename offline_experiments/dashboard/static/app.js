const state = { page: "overview", overview: null, activeJob: null, refreshTimer: null, memoryRows: [], campaigns: [] };
const appBaseUrl = new URL(".", document.baseURI);
const renderCache = new WeakMap();

const pages = {
  overview: ["实验总览", "实验实时进度、GPU 状态与异常概览"],
  runs: ["实时任务", "按阶段与配置筛选，点击进入单任务详情"],
  memory: ["显存边界", "最大可行 MBS、显存余量与 OOM 边界"],
  throughput: ["吞吐与 MFU", "重复实验聚合、训练时间和计算效率"],
  scaling: ["多卡扩展", "1/2/4 卡加速比、并行效率与扩卡判断"],
  packing: ["Packing 对照", "严格成对比较 no-packing 与 neat packing"],
  recommendations: ["推荐结果", "基于已完成实测候选生成可解释方案"],
  system: ["环境与设计", "实验范围、运行环境、静态校验与服务状态"],
};

const statusLabels = {
  planned: "等待", running: "运行中", success: "成功", oom: "OOM",
  failed: "失败", incomplete_metrics: "指标不完整", boundary_found: "边界已确定",
  infeasible: "无可行 MBS", launcher_failed: "启动失败",
  conditional_skipped: "条件跳过", family_skipped: "条件跳过",
};
const phaseLabels = { memory: "显存", throughput: "吞吐", throughput_screen: "吞吐初筛", throughput_formal: "正式复验", scaling: "扩展", packing: "Packing", profiler: "Profiler", validation: "验证", preflight: "预检", other: "其他" };
const clockStatusLabels = {
  normal: "频率正常",
  power_limited: "功耗受限",
  thermal_limited: "热限频",
  downclocked: "降频待查",
  insufficient_data: "原因不足",
};

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));
}
function number(value, digits=1) { return value == null || Number.isNaN(Number(value)) ? "—" : Number(value).toLocaleString("zh-CN", {maximumFractionDigits: digits}); }
function percent(value, digits=1) { return value == null ? "—" : `${number(Number(value)*100, digits)}%`; }
function duration(seconds) {
  if (seconds == null) return "—"; seconds = Number(seconds);
  if (seconds < 60) return `${number(seconds,1)}s`;
  if (seconds < 3600) return `${Math.floor(seconds/60)}m ${Math.round(seconds%60)}s`;
  return `${Math.floor(seconds/3600)}h ${Math.round((seconds%3600)/60)}m`;
}
function bytesGiB(value) { return value ? `${number(Number(value)/1024**3,1)} GiB` : "—"; }
function mibGiB(value) { return value ? `${number(Number(value)/1024,1)} GiB` : "—"; }
function memoryCell(metrics={}) {
  const device=mibGiB(metrics.nvidia_smi_peak_mib),allocated=bytesGiB(metrics.max_allocated_bytes);
  if(device==="—"&&allocated==="—")return "—";
  return `<strong title="nvidia-smi 设备峰值">${device}</strong><br><small title="PyTorch max_memory_allocated">${allocated} allocated</small>`;
}
function shortId(value) { value=String(value||""); return value.length > 25 ? `${value.slice(0,13)}…${value.slice(-8)}` : value; }
function timeLabel(unix) { return unix ? new Date(Number(unix)*1000).toLocaleTimeString("zh-CN", {hour12:false}) : "—"; }
function badge(status) { return `<span class="badge ${escapeHtml(status)}">${escapeHtml(statusLabels[status] || status || "未知")}</span>`; }
function clockBadge(metrics={}) {
  const status=metrics.clock_status||"insufficient_data";
  return `<span class="badge clock-${escapeHtml(status)}">${escapeHtml(clockStatusLabels[status]||status)}</span>`;
}
function clockSummary(metrics={}) {
  const p50=number(metrics.busy_clock_p50_mhz,0),healthy=number(metrics.sm_clock_reference_mhz,0),ceiling=number(metrics.sm_clock_spec_max_mhz,0);
  if(p50==="—")return clockBadge(metrics);
  const comparison=healthy!=="—"?` / 健康 ${healthy}`:ceiling!=="—"?` / 上限 ${ceiling}`:"";
  return `${clockBadge(metrics)}<br><small>忙碌 P50 ${p50}${comparison} MHz</small>`;
}
function limiterFlagEvidence(flag={},label="Limiter") {
  const known=Number(flag.known_samples||0),active=Number(flag.active_samples||0);
  if(!known)return `${label}：未采集`;
  return `${label}：${active}/${known}（${percent(flag.active_fraction,1)}）`;
}
function limiterEvidence(metrics={},evidence={}) {
  const status=metrics.clock_status||"insufficient_data",source=metrics.clock_status_source||"missing_throttle_reason";
  const flags=evidence.flags||{},powerFlag=flags.sw_power_cap_active||{},swThermal=flags.sw_thermal_slowdown_active||{},hwThermal=flags.hw_thermal_slowdown_active||{};
  const sourceLabels={
    observed_throttle_reason:"NVIDIA 驱动 limiter 字段（直接证据）",
    inferred_power_cap:"功耗贴墙并相对健康基线降频（组合推断）",
    clock_only:"相对独立健康基线降频（原因未确定）",
    no_observed_queried_limiter:"limiter 字段完整且均未触发",
    partial_throttle_reason_support:"只有部分 limiter 字段",
    missing_throttle_reason:"历史任务未采集 limiter 字段",
  };
  const headlines={
    power_limited:"驱动明确上报 SW Power Cap：该任务受功耗墙约束",
    thermal_limited:"驱动明确上报热限制：该任务发生热限频",
    downclocked:"频率显著低于健康基线，但现有证据不能确定原因",
    normal:"limiter 证据完整，未观察到功耗或热限频",
    insufficient_data:"证据不足，不能仅凭低于驱动最高频率判定降频",
  };
  const powerRatio=metrics.busy_power_p95_w!=null&&metrics.power_limit_w?metrics.busy_power_p95_w/metrics.power_limit_w:null;
  const busy=Number(evidence.busy_samples||metrics.busy_gpu_samples||0),telemetry=Number(evidence.telemetry_samples||metrics.gpu_samples||0);
  const confidence=evidence.reason_coverage_complete?`${busy} 个忙碌采样的三个原因字段覆盖完整${busy<30?"；样本偏少，以长任务为主":""}`:`${busy} 个忙碌采样；limiter 字段覆盖不完整`;
  const cards=[
    ["① 驱动直接原因",`${limiterFlagEvidence(powerFlag,"SW Power Cap")}；${limiterFlagEvidence(swThermal,"SW 热")}；${limiterFlagEvidence(hwThermal,"HW 热")}`],
    ["② 功耗是否贴墙",metrics.busy_power_p95_w==null?"未采集":`忙碌 P95 ${number(metrics.busy_power_p95_w,1)} / ${number(metrics.power_limit_w,0)} W（${percent(powerRatio,1)}）；≥98% 功耗墙 ${percent(metrics.power_limit_busy_fraction,1)}`],
    ["③ 频率响应",metrics.busy_clock_p50_mhz==null?"未采集":`P5 / P50 / P95 = ${number(metrics.busy_clock_p5_mhz,0)} / ${number(metrics.busy_clock_p50_mhz,0)} / ${number(metrics.busy_clock_p95_mhz,0)} MHz；P50/驱动上限 ${percent(metrics.busy_clock_p50_to_spec_max_ratio,1)}`],
    ["④ 排除热限频",metrics.temperature_max_c==null?"历史任务未采集温度":`温度 P95 / Max = ${number(metrics.busy_temperature_p95_c,0)} / ${number(metrics.temperature_max_c,0)} °C；SW/HW 热触发 ${percent(metrics.sw_thermal_slowdown_busy_fraction,1)} / ${percent(metrics.hw_thermal_slowdown_busy_fraction,1)}`],
  ];
  const rows=evidence.per_gpu||[];
  const table=rows.length?`<div class="table-wrap evidence-table"><table><thead><tr><th>GPU</th><th>忙碌样本</th><th>SW Power</th><th>SW/HW 热</th><th>功耗 P95</th><th>SM P5/P50/P95</th><th>温度 P95/Max</th><th>逐卡结论</th></tr></thead><tbody>${rows.map(row=>{const f=row.flags||{};return `<tr><td>GPU ${number(row.gpu_index,0)}</td><td>${number(row.busy_samples,0)}</td><td>${percent(f.sw_power_cap_active?.active_fraction,1)}<br><small>${number(f.sw_power_cap_active?.active_samples,0)}/${number(f.sw_power_cap_active?.known_samples,0)}</small></td><td>${percent(f.sw_thermal_slowdown_active?.active_fraction,1)} / ${percent(f.hw_thermal_slowdown_active?.active_fraction,1)}</td><td>${row.busy_power_p95_w==null?"—":`${number(row.busy_power_p95_w,1)} W`}</td><td>${number(row.busy_clock_p5_mhz,0)} / ${number(row.busy_clock_p50_mhz,0)} / ${number(row.busy_clock_p95_mhz,0)}</td><td>${row.busy_temperature_p95_c==null?"—":`${number(row.busy_temperature_p95_c,0)} / ${number(row.temperature_max_c,0)} °C`}</td><td>${clockBadge(row)}</td></tr>`}).join("")}</tbody></table></div>`:"";
  return `<section class="limiter-evidence evidence-${escapeHtml(status)}"><div class="evidence-head"><div><h3>限频证据链</h3><strong>${escapeHtml(headlines[status]||headlines.insufficient_data)}</strong><p>判定来源：${escapeHtml(sourceLabels[source]||source)} · 遥测 ${number(telemetry,0)} 条 · ${escapeHtml(confidence)}</p></div>${clockBadge(metrics)}</div><div class="evidence-grid">${cards.map(([title,value])=>`<div class="evidence-item"><span>${escapeHtml(title)}</span><strong>${escapeHtml(value)}</strong></div>`).join("")}</div>${table}<p class="evidence-note">判定优先级：NVIDIA 明确热原因 ＞ 明确 SW Power Cap ＞ 有健康基线时的组合推断。驱动最高频率只是硬件上限，不能单独作为“降频”证据。</p></section>`;
}
function liveClockStatus(gpu={},campaign={}) {
  if(Number(gpu.hw_thermal_slowdown_active)===1||Number(gpu.sw_thermal_slowdown_active)===1)return "thermal_limited";
  if(Number(gpu.sw_power_cap_active)===1)return "power_limited";
  const utilization=Number(gpu.utilization_gpu||0),clock=Number(gpu.clock_sm_mhz||0),reference=Number(campaign.healthy_busy_sm_clock_mhz||0);
  if(utilization<90)return "insufficient_data";
  if(reference&&clock<reference*.9)return "downclocked";
  const reasonValues=[gpu.hw_thermal_slowdown_active,gpu.sw_thermal_slowdown_active,gpu.sw_power_cap_active];
  return reasonValues.every(value=>value!=null)?"normal":"insufficient_data";
}
function empty(text) { return `<div class="empty">${escapeHtml(text)}</div>`; }
function appUrl(path) { return new URL(String(path).replace(/^\/+/, ""), appBaseUrl).toString(); }
function setHtml(target, html) {
  const node = typeof target === "string" ? document.querySelector(target) : target;
  if (!node || renderCache.get(node) === html) return false;
  const scrollTop = node.scrollTop;
  const scrollLeft = node.scrollLeft;
  node.innerHTML = html;
  node.scrollTop = scrollTop;
  node.scrollLeft = scrollLeft;
  renderCache.set(node, html);
  return true;
}
function setText(target, value) {
  const node = typeof target === "string" ? document.querySelector(target) : target;
  const text = String(value ?? "");
  if (!node || node.textContent === text) return false;
  node.textContent = text;
  return true;
}
async function api(path) {
  const campaign=document.querySelector("#campaign-filter")?.value;
  if(campaign && (path.startsWith("/api/v1/analysis/") || path==="/api/v1/recommendations" || path==="/api/v1/overview")) {
    path=`${path}${path.includes("?")?"&":"?"}campaign_id=${encodeURIComponent(campaign)}`;
  }
  const response = await fetch(appUrl(path), {cache:"no-store"});
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  return response.json();
}
function kpi(label, value, note="", accent=false) { return `<div class="kpi-card ${accent?"kpi-accent":""}"><div class="kpi-label">${escapeHtml(label)}</div><div class="kpi-value">${escapeHtml(value)}</div><div class="kpi-note">${escapeHtml(note)}</div></div>`; }

function setConnection(ok, label) {
  const dot=document.querySelector("#connection-dot"),className=`status-dot ${ok?"live":"error"}`;
  if(dot.className!==className)dot.className=className;
  setText("#connection-label",label);
  setText("#last-update",ok ? `更新 ${new Date().toLocaleTimeString("zh-CN",{hour12:false})}` : "正在重连");
}

function navigate(page) {
  if (!pages[page]) return;
  state.page = page;
  document.querySelectorAll(".nav-item").forEach(item => item.classList.toggle("active", item.dataset.page === page));
  document.querySelectorAll(".page").forEach(node => node.classList.toggle("active", node.id === `page-${page}`));
  document.querySelector("#page-title").textContent = pages[page][0];
  document.querySelector("#page-subtitle").textContent = pages[page][1];
  location.hash = page;
  loadPage(page);
}

async function refreshOverview(snapshot=null) {
  try {
    const selected=document.querySelector("#campaign-filter")?.value;
    const data = snapshot && !selected ? snapshot : await api("/api/v1/overview"); state.overview = data;
    if (state.page === "overview") renderOverview(data);
    setConnection(true, "实时连接");
  } catch (error) { setConnection(false, "连接中断"); console.error(error); }
}

function renderOverview(data) {
  state.campaigns=data.campaigns||state.campaigns;
  const counts = data.runs_by_status || {};
  const skipped = (counts.conditional_skipped||0)+(counts.family_skipped||0);
  const complete = (counts.success||0)+(counts.oom||0)+(counts.failed||0)+(counts.incomplete_metrics||0)+skipped;
  const total = Object.values(counts).reduce((a,b)=>a+Number(b||0),0);
  const failures = (counts.failed||0)+(counts.incomplete_metrics||0);
  setHtml("#overview-kpis", [
    kpi("当前阶段", data.current_phase_label || "等待", `Revision ${data.revision}`, true),
    kpi("已完成运行", number(complete,0), `已索引 ${total} 个具体运行`),
    kpi("正在运行", number(counts.running||0,0), "任务可并行占用 0–3 卡"),
    kpi("预期 OOM", number(counts.oom||0,0), "边界搜索的正常停止信号"),
    kpi("异常", number(failures,0), failures ? "需要检查日志" : "当前没有运行异常"),
  ].join(""));

  const planRows=Object.values(data.plans||{});
  const planSum=field=>planRows.reduce((sum,row)=>sum+Number(row?.[field]||0),0);
  const phasePlan = {
    memory: planSum("memory_boundary_families"),
    scaling: planSum("strong_scaling_families"),
    packing: planSum("packing_pair_families"),
    profiler: planSum("profiler_calibration_configurations"),
  };
  const standardPhase=phase=>{
    const row=data.phase_counts?.[phase]||{};
    const progress=data.phase_progress?.[phase];
    return {
      phase,label:phaseLabels[phase],
      planned:Number(progress?.planned ?? (phasePlan[phase]||row.indexed||0)),
      done:progress?Number(progress.completed||0):phase==="memory"
        ? Number(row.family_boundary_found||0)+Number(row.family_infeasible||0)
        : Number(row.run_success||0)+Number(row.run_oom||0)+Number(row.run_failed||0)+Number(row.run_incomplete_metrics||0)+Number(row.run_conditional_skipped||0)+Number(row.run_family_skipped||0),
      active:data.current_phase===phase,note:"",
    };
  };
  const throughput=data.throughput_progress||{},screen=throughput.screening||{},formal=throughput.formal||{};
  const phaseCards=[
    standardPhase("memory"),
    {
      phase:"throughput_screen",label:phaseLabels.throughput_screen,
      planned:Number(screen.total||0),done:Number(screen.completed||0),
      active:data.current_phase==="throughput"&&throughput.current_stage==="screening",
      note:screen.total?`已执行 ${number(screen.executed,0)} · 复用 ${number(screen.reused_formal,0)}`:"",
    },
    {
      phase:"throughput_formal",label:phaseLabels.throughput_formal,
      planned:Number(formal.total||0),done:Number(formal.completed||0),
      active:data.current_phase==="throughput"&&throughput.current_stage==="formal",
      waiting:Boolean(formal.waiting_for_screening),
      note:formal.total?`已物化 ${number(formal.materialized,0)} · Top-K 计划上限`:"",
    },
    standardPhase("scaling"),
    standardPhase("packing"),
    standardPhase("profiler"),
  ];
  setHtml("#phase-grid",phaseCards.map(card=>{
    const pct=card.planned?Math.min(100,card.done/card.planned*100):0;
    const status=card.active?"进行中":card.waiting?"待初筛":card.planned&&card.done>=card.planned?"完成":"";
    return `<div class="phase-card"><div class="phase-top"><strong>${escapeHtml(card.label)}</strong><span>${status}</span></div><div class="phase-number">${number(card.done,0)} <small>/ ${card.planned?number(card.planned,0):"—"}</small></div><div class="progress"><span style="width:${pct}%"></span></div>${card.note?`<div class="phase-note">${escapeHtml(card.note)}</div>`:""}</div>`;
  }).join(""));

  const campaigns=data.campaigns||[];
  const campaignMap=Object.fromEntries(campaigns.map(row=>[row.campaign_id,row]));
  const gpuKey=(campaignId,index)=>`${campaignId}:${Number(index)}`;
  const gpuMap=Object.fromEntries((data.gpus||[]).map(row=>[gpuKey(row.campaign_id,row.gpu_index),row]));
  const liveThermal=(data.gpus||[]).filter(gpu=>gpu.live&&(Number(gpu.sw_thermal_slowdown_active)===1||Number(gpu.hw_thermal_slowdown_active)===1));
  const alertParts=[];
  if(failures)alertParts.push(`<div class="alert error">发现 ${failures} 个异常或指标不完整任务，请在“实时任务”中按失败状态筛选。</div>`);
  if(liveThermal.length)alertParts.push(`<div class="alert error">检测到 ${liveThermal.length} 张 GPU 正在触发热限频，请查看温度和任务详情。</div>`);
  setHtml("#overview-alerts",alertParts.join(""));
  const gpuRange=ids=>{
    const values=[...ids].map(Number).sort((a,b)=>a-b);
    const contiguous=values.every((value,index)=>index===0||value===values[index-1]+1);
    return contiguous&&values.length>1 ? `GPU ${values[0]}–${values.at(-1)}` : `GPU ${values.join("、")}`;
  };
  const gpuCard=(campaign,index)=>{
    const gpu=gpuMap[gpuKey(campaign.campaign_id,index)];
    if (!gpu) return `<div class="gpu-card"><div class="gpu-head"><span class="gpu-name">GPU ${index}</span><span class="gpu-stale">等待采样</span></div><div class="gpu-metrics"><div class="gpu-metric"><span>显存</span><strong>—</strong></div><div class="gpu-metric"><span>利用率</span><strong>—</strong></div><div class="gpu-metric"><span>功耗</span><strong>—</strong></div><div class="gpu-metric"><span>温度</span><strong>—</strong></div></div><div class="gpu-job">尚无实验采样</div></div>`;
    const cap=(campaignMap[gpu.campaign_id]?.memory_total_bytes||0)/1024**3;
    const reference=Number(campaign.max_sm_clock_mhz||0),powerLimit=Number(campaign.power_limit_w||0);
    const clockStatus=liveClockStatus(gpu,campaign);
    const clockStatusText=clockStatus==="insufficient_data"?(Number(gpu.utilization_gpu||0)<90?"非忙碌":"原因不足"):clockStatusLabels[clockStatus];
    return `<div class="gpu-card"><div class="gpu-head"><span class="gpu-name">GPU ${index}</span><span class="${gpu.live?"gpu-live":"gpu-stale"}">${gpu.live?"● LIVE":`${number(gpu.stale_seconds,0)}s 前`}</span></div><div class="gpu-metrics"><div class="gpu-metric"><span>显存</span><strong>${number(gpu.memory_used_mib/1024,1)} / ${number(cap,1)}G</strong></div><div class="gpu-metric"><span>利用率</span><strong>${number(gpu.utilization_gpu,0)}%</strong></div><div class="gpu-metric"><span>功耗</span><strong>${number(gpu.power_draw_w,0)}${powerLimit?` / ${number(powerLimit,0)}`:""} W</strong></div><div class="gpu-metric"><span>SM 时钟</span><strong>${number(gpu.clock_sm_mhz,0)}${reference?` / ${number(reference,0)}`:""} MHz</strong></div><div class="gpu-metric"><span>温度</span><strong>${gpu.temperature_gpu_c==null?"—":`${number(gpu.temperature_gpu_c,0)} °C`}</strong></div><div class="gpu-metric"><span>频率状态</span><strong>${clockStatusText}</strong></div></div><div class="gpu-job">${escapeHtml(gpu.model_id||"—")} · ${escapeHtml(gpu.dataset_id||"—")} · Step ${gpu.current_step||0}/${gpu.max_steps||"—"}</div></div>`;
  };
  const gpuGrid=document.querySelector("#gpu-grid");
  const grouped=!data.selected_campaign_id&&campaigns.length>1;
  gpuGrid?.classList.toggle("grouped",grouped);
  if(grouped){
    setText("#gpu-section-title","GPU 状态（按硬件分组）");
    setHtml("#gpu-grid",campaigns.map(campaign=>{
      const ids=(campaign.gpu_ids||[]).map(Number);
      return `<section class="gpu-campaign-group"><div class="gpu-campaign-head"><div><strong>${escapeHtml(campaign.gpu_type||campaign.hardware_id||campaign.campaign_id)}</strong><span>${escapeHtml(campaign.campaign_id)}</span></div><span>${gpuRange(ids)} · ${escapeHtml((campaign.attention_backend||"").toUpperCase())}</span></div><div class="gpu-campaign-cards">${ids.map(index=>gpuCard(campaign,index)).join("")}</div></section>`;
    }).join(""));
  }else{
    const campaign=campaigns[0]||{campaign_id:"",gpu_ids:data.gpu_ids||[0,1,2,3]};
    const ids=(campaign.gpu_ids?.length?campaign.gpu_ids:data.gpu_ids||[0,1,2,3]).map(Number);
    setText("#gpu-section-title",`${gpuRange(ids)} · ${campaign.gpu_type||campaign.hardware_id||""}`);
    setHtml("#gpu-grid",ids.map(index=>gpuCard(campaign,index)).join(""));
  }
  renderRunning(data.running_jobs||[]);
  renderEvents(data.recent_events||[]);
}

function renderRunning(rows) {
  const html = rows.length ? `<table><thead><tr><th>任务</th><th>配置</th><th>Step</th><th>实时吞吐</th></tr></thead><tbody>${rows.map(row=>`<tr class="clickable" data-job="${escapeHtml(row.job_id)}"><td><span class="mono">${escapeHtml(shortId(row.job_id))}</span><br><small>${escapeHtml(row.model_id||"—")}</small></td><td>${escapeHtml(row.dataset_id||"—")}<br><small>${row.gpu_count||"—"}卡 · MBS ${row.mbs||"—"}</small></td><td>${row.current_step||0}/${row.max_steps||"—"}</td><td>${number(row.metrics?.samples_per_second,2)} sample/s<br><small>${number(row.metrics?.effective_tokens_per_second,0)} tok/s</small></td></tr>`).join("")}</tbody></table>` : empty("当前没有运行中的任务");
  if (setHtml("#running-jobs", html) && rows.length) bindJobRows(document.querySelector("#running-jobs"));
}
function renderEvents(rows) {
  setHtml("#recent-events", rows.length ? rows.map(row=>`<div class="event-row"><time>${timeLabel(row.time_unix)}</time><strong>${escapeHtml(shortId(row.job_id||row.event))}</strong><span>${escapeHtml(row.event||"")} ${row.classification?`· ${statusLabels[row.classification]||row.classification}`:""}</span></div>`).join("") : empty("调度器尚未产生运行事件"));
}

function runTable(rows) {
  if (!rows.length) return empty("没有符合条件的任务");
  return `<table><thead><tr><th>状态</th><th>硬件</th><th>任务</th><th>模型 / 类型</th><th>长度</th><th>GPU</th><th>MBS/GBS</th><th>执行策略</th><th>Step</th><th>MFU / samples/s</th><th>频率状态</th><th>设备峰值 / Allocated</th></tr></thead><tbody>${rows.map(row=>`<tr class="clickable" data-job="${escapeHtml(row.job_id)}"><td>${badge(row.status)}</td><td>${escapeHtml(row.gpu_type||row.hardware_id||"—")}<br><small>${escapeHtml(row.campaign_id||"")}</small></td><td class="mono">${escapeHtml(shortId(row.job_id))}</td><td>${escapeHtml(row.model_id||"—")}<br><small>${escapeHtml((row.train_type||"—").toUpperCase())}</small></td><td>${number(row.cutoff_len,0)}</td><td>${row.gpu_count||"—"}</td><td>${row.mbs||"—"} / ${row.target_gbs||"—"}</td><td>${escapeHtml(row.zero_name||"none")} · GC ${row.gc?"On":"Off"}${row.packing?" · Pack":""}</td><td>${row.current_step||0}/${row.max_steps||"—"}</td><td><strong>${percent(row.metrics?.mfu,1)}</strong><br><small>${number(row.metrics?.samples_per_second,2)} samples/s</small></td><td>${clockSummary(row.metrics)}</td><td>${memoryCell(row.metrics)}</td></tr>`).join("")}</tbody></table>`;
}

async function loadRuns() {
  const params=new URLSearchParams();
  [["q","#run-q"],["phase","#run-phase"],["status","#run-status"],["train_type","#run-train-type"],["gpu_count","#run-gpus"],["campaign_id","#run-campaign"]].forEach(([key,sel])=>{const v=document.querySelector(sel)?.value;if(v)params.set(key,v)});
  try { const data=await api(`/api/v1/runs?${params}`); setText("#runs-count",`共 ${data.total} 个具体运行`); if(setHtml("#runs-table",runTable(data.rows)))bindJobRows(document.querySelector("#runs-table")); } catch(e){setHtml("#runs-table",empty(`读取失败：${e.message}`))}
}
function bindJobRows(root) { root.querySelectorAll("[data-job]").forEach(row=>row.addEventListener("click",()=>openRun(row.dataset.job))); }

function drawLineChart(canvas, series, options={}) {
  if (!canvas) return;
  // Keep logical CSS dimensions separate from the DPR-scaled backing buffer.
  // Reading canvas.height after assigning it would multiply the height again on
  // every live refresh (for example 210 -> 420 -> 840 on a 2x display).
  const width=Number(canvas.dataset.chartWidth||480),height=Number(canvas.dataset.chartHeight||240),dpr=window.devicePixelRatio||1;
  canvas.style.width=`${width}px`;
  canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);canvas.style.height=`${height}px`;const ctx=canvas.getContext("2d");ctx.scale(dpr,dpr);ctx.clearRect(0,0,width,height);
  const all=series.flatMap(s=>s.values.filter(v=>Number.isFinite(v.x)&&Number.isFinite(v.y))); if(!all.length){ctx.fillStyle="#8da3aa";ctx.font="12px sans-serif";ctx.fillText("等待足够的实验数据",24,45);return;}
  const pad={l:48,r:20,t:25,b:35}; const xs=all.map(v=>Number(v.x)),ys=all.map(v=>Number(v.y)); let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys); if(xmin===xmax){xmin-=1;xmax+=1} if(ymin===ymax){ymin=Math.max(0,ymin*.8);ymax*=1.2||1}
  ymin=options.zero===false?ymin:Math.min(0,ymin); const sx=x=>pad.l+(x-xmin)/(xmax-xmin)*(width-pad.l-pad.r),sy=y=>height-pad.b-(y-ymin)/(ymax-ymin)*(height-pad.t-pad.b);
  ctx.strokeStyle="#263840";ctx.fillStyle="#70868d";ctx.lineWidth=1;ctx.font="10px sans-serif";for(let i=0;i<=4;i++){const y=pad.t+(height-pad.t-pad.b)*i/4;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(width-pad.r,y);ctx.stroke();const val=ymax-(ymax-ymin)*i/4;ctx.fillText(number(val,options.digits??1),4,y+3)}
  series.forEach((s,si)=>{const values=s.values.filter(v=>Number.isFinite(v.x)&&Number.isFinite(v.y));if(!values.length)return;const color=s.color||["#3dd6a0","#5da9ff","#ffb357","#b49cff","#ff6f73"][si%5];ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=2;ctx.beginPath();values.forEach((v,i)=>{const x=sx(v.x),y=sy(v.y);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();values.forEach(v=>{ctx.beginPath();ctx.arc(sx(v.x),sy(v.y),3,0,Math.PI*2);ctx.fill()});ctx.fillText(s.name||"",pad.l+si*115,13)});
  ctx.fillStyle="#70868d";ctx.fillText(options.xLabel||"配置序号",width/2-25,height-8);
}

async function openRun(jobId, background=false) {
  const modal=document.querySelector("#detail-modal");
  const isVisibleRefresh=background && state.activeJob===jobId && modal.classList.contains("open");
  state.activeJob=jobId;
  if (!isVisibleRefresh) {
    modal.classList.add("open");modal.setAttribute("aria-hidden","false");setText("#detail-title",jobId);setText("#detail-subtitle","正在读取实时指标");setHtml("#detail-content",empty("加载中"));
  }
  try {
    const [detail,series,log]=await Promise.all([api(`/api/v1/runs/${encodeURIComponent(jobId)}`),api(`/api/v1/runs/${encodeURIComponent(jobId)}/series`),fetch(appUrl(`/api/v1/runs/${encodeURIComponent(jobId)}/log`)).then(r=>r.text())]);
    if(state.activeJob!==jobId)return; const j=detail.job,m=j.metrics||{};setHtml("#detail-subtitle",`${badge(j.status)} &nbsp; ${escapeHtml(j.model_id||"—")} · ${escapeHtml(j.dataset_id||"—")}`);
    const clockTriplet=[m.busy_clock_p5_mhz,m.busy_clock_p50_mhz,m.busy_clock_p95_mhz].map(value=>number(value,0)).join(" / ");
    const throttleTriplet=[m.sw_power_cap_busy_fraction,m.sw_thermal_slowdown_busy_fraction,m.hw_thermal_slowdown_busy_fraction].map(value=>percent(value,1)).join(" / ");
    const temperaturePair=[m.busy_temperature_p95_c,m.temperature_max_c].map(value=>value==null?"—":`${number(value,0)} °C`).join(" / ");
    const items=[
      ["阶段",phaseLabels[j.phase]||j.phase],["模型",j.model_id],["训练类型",(j.train_type||"—").toUpperCase()],["数据集",j.dataset_id],
      ["Cutoff",j.cutoff_len],["GPU",`${j.gpu_count||"—"} · ${j.gpu_mask||"—"}`],["MBS / GBS",`${j.mbs||"—"} / ${j.target_gbs||"—"}`],["GC / ZeRO",`${j.gc?"On":"Off"} / ${j.zero_name||"none"}`],
      ["Step",`${j.current_step||0}/${j.max_steps||"—"}`],["Step P50",duration(m.step_p50_seconds)],["Samples/s",number(m.samples_per_second,2)],["Effective tok/s",number(m.effective_tokens_per_second,0)],
      ["MFU",percent(m.mfu,1)],["按频率折算 MFU",percent(m.clock_adjusted_mfu,1)],["Padding效率",percent(m.padding_efficiency,1)],["频率状态",clockStatusLabels[m.clock_status]||"原因不足"],["Limiter 字段",m.throttle_reason_data_complete===true?"完整":m.throttle_reason_data_available===true?"部分支持":"历史任务未采集"],
      ["忙碌 SM P5/P50/P95",clockTriplet==="— / — / —"?"—":`${clockTriplet} MHz`],["P50 / 驱动上限",percent(m.busy_clock_p50_to_spec_max_ratio,1)],["P50 / 健康基线",percent(m.busy_clock_p50_ratio,1)],["低于健康基线 90%",percent(m.busy_clock_below_90pct_fraction,1)],["触及功耗墙",percent(m.power_limit_busy_fraction,1)],
      ["Power / SW热 / HW热",throttleTriplet],["温度 P95 / Max",temperaturePair],["平均功耗",m.average_power_w==null?"—":`${number(m.average_power_w,0)} W`],["忙碌功耗 P95",m.busy_power_p95_w==null?"—":`${number(m.busy_power_p95_w,0)} W`],
      ["设备峰值 (nvidia-smi)",mibGiB(m.nvidia_smi_peak_mib)],["PyTorch allocated 峰值",bytesGiB(m.max_allocated_bytes)],["PyTorch reserved 峰值",bytesGiB(m.max_reserved_bytes)],
    ];
    if(m.oom_requested_bytes!=null||m.oom_free_bytes!=null)items.push(["OOM 申请 / 可用",`${bytesGiB(m.oom_requested_bytes)} / ${bytesGiB(m.oom_free_bytes)}`]);
    const modalCard=modal.querySelector(".modal-card"),scrollTop=modalCard?.scrollTop||0;
    setHtml("#detail-content",`<div class="detail-grid">${items.map(([k,v])=>`<div class="detail-item"><span>${escapeHtml(k)}</span><strong>${escapeHtml(v??"—")}</strong></div>`).join("")}</div><div class="detail-charts"><article class="panel"><div class="panel-head"><h3>Step Time</h3></div><canvas id="detail-step-chart" class="chart detail-chart" data-chart-width="320" data-chart-height="210" width="320" height="210"></canvas></article><article class="panel"><div class="panel-head"><h3>GPU 显存</h3></div><canvas id="detail-gpu-chart" class="chart detail-chart" data-chart-width="320" data-chart-height="210" width="320" height="210"></canvas></article><article class="panel"><div class="panel-head"><div><h3>SM 时钟</h3><p>驱动上限与独立校准的健康基线</p></div></div><canvas id="detail-clock-chart" class="chart detail-chart" data-chart-width="320" data-chart-height="210" width="320" height="210"></canvas></article><article class="panel"><div class="panel-head"><div><h3>GPU 功耗</h3><p>每卡功耗与设备功耗墙</p></div></div><canvas id="detail-power-chart" class="chart detail-chart" data-chart-width="320" data-chart-height="210" width="320" height="210"></canvas></article></div>${j.last_error?`<div class="alert error">${escapeHtml(j.last_error.slice(-1500))}</div>`:""}<pre class="log-view">${escapeHtml(log||"训练日志尚未生成")}</pre>`);
    const detailContentNode=document.querySelector("#detail-content");
    if(detailContentNode&&!detailContentNode.querySelector(".limiter-evidence"))detailContentNode.insertAdjacentHTML("afterbegin",limiterEvidence(m,series.limiter_evidence||{}));
    const detailChartsNode=detailContentNode?.querySelector(".detail-charts");
    if(detailChartsNode&&!detailChartsNode.querySelector("#detail-temperature-chart"))detailChartsNode.insertAdjacentHTML("beforeend",`<article class="panel"><div class="panel-head"><div><h3>GPU 温度</h3><p>结合 SW/HW 热原因字段判断</p></div></div><canvas id="detail-temperature-chart" class="chart detail-chart" data-chart-width="320" data-chart-height="210" width="320" height="210"></canvas></article><article class="panel"><div class="panel-head"><div><h3>Limiter 触发率</h3><p>同一采样时刻各卡 Active 比例</p></div></div><canvas id="detail-limiter-chart" class="chart detail-chart" data-chart-width="320" data-chart-height="210" width="320" height="210"></canvas></article>`);
    if(modalCard)modalCard.scrollTop=scrollTop;
    drawLineChart(document.querySelector("#detail-step-chart"),[{name:"step seconds",values:series.steps.map(x=>({x:x.step,y:x.step_seconds}))}],{xLabel:"Optimizer step",zero:false,digits:2});
    const gpuGroups={},clockGroups={},powerGroups={},temperatureGroups={},limiterGroups={};
    series.gpus.forEach(x=>{
      if(x.memory_used_mib!=null)(gpuGroups[x.gpu_index]??=[]).push({x:x.time_unix,y:x.memory_used_mib/1024});
      if(x.clock_sm_mhz!=null)(clockGroups[x.gpu_index]??=[]).push({x:x.time_unix,y:x.clock_sm_mhz});
      if(x.power_draw_w!=null)(powerGroups[x.gpu_index]??=[]).push({x:x.time_unix,y:x.power_draw_w});
      if(x.temperature_gpu_c!=null)(temperatureGroups[x.gpu_index]??=[]).push({x:x.time_unix,y:x.temperature_gpu_c});
      const key=String(x.time_unix),bucket=limiterGroups[key]??={x:Number(x.time_unix),power:[],swThermal:[],hwThermal:[]};
      if(x.sw_power_cap_active!=null)bucket.power.push(Number(x.sw_power_cap_active));
      if(x.sw_thermal_slowdown_active!=null)bucket.swThermal.push(Number(x.sw_thermal_slowdown_active));
      if(x.hw_thermal_slowdown_active!=null)bucket.hwThermal.push(Number(x.hw_thermal_slowdown_active));
    });
    drawLineChart(document.querySelector("#detail-gpu-chart"),Object.entries(gpuGroups).map(([gpu,values])=>({name:`GPU ${gpu}`,values})),{xLabel:"时间",zero:true,digits:0});
    const times=series.gpus.map(x=>Number(x.time_unix)).filter(Number.isFinite),timeBounds=times.length?[Math.min(...times),Math.max(...times)]:[];
    const clockSeries=Object.entries(clockGroups).map(([gpu,values])=>({name:`GPU ${gpu}`,values}));
    if(m.sm_clock_spec_max_mhz&&timeBounds.length)clockSeries.push({name:"驱动上限",color:"#ff6f73",values:timeBounds.map(x=>({x,y:m.sm_clock_spec_max_mhz}))});
    if(m.sm_clock_reference_mhz&&timeBounds.length)clockSeries.push({name:"健康基线",color:"#ffb357",values:timeBounds.map(x=>({x,y:m.sm_clock_reference_mhz}))});
    drawLineChart(document.querySelector("#detail-clock-chart"),clockSeries,{xLabel:"时间",zero:false,digits:0});
    const powerSeries=Object.entries(powerGroups).map(([gpu,values])=>({name:`GPU ${gpu}`,values}));
    if(m.power_limit_w&&timeBounds.length)powerSeries.push({name:"功耗墙",color:"#ff6f73",values:timeBounds.map(x=>({x,y:m.power_limit_w}))});
    drawLineChart(document.querySelector("#detail-power-chart"),powerSeries,{xLabel:"时间",zero:true,digits:0});
    drawLineChart(document.querySelector("#detail-temperature-chart"),Object.entries(temperatureGroups).map(([gpu,values])=>({name:`GPU ${gpu}`,values})),{xLabel:"时间",zero:false,digits:0});
    const limiterValues=Object.values(limiterGroups).sort((a,b)=>a.x-b.x),mean=values=>values.length?values.reduce((a,b)=>a+b,0)/values.length*100:null;
    drawLineChart(document.querySelector("#detail-limiter-chart"),[
      {name:"SW Power %",color:"#ffb357",values:limiterValues.map(x=>({x:x.x,y:mean(x.power)}))},
      {name:"SW 热 %",color:"#ff6f73",values:limiterValues.map(x=>({x:x.x,y:mean(x.swThermal)}))},
      {name:"HW 热 %",color:"#b49cff",values:limiterValues.map(x=>({x:x.x,y:mean(x.hwThermal)}))},
    ],{xLabel:"时间",zero:true,digits:0});
  } catch(e){setHtml("#detail-content",empty(`详情读取失败：${e.message}`))}
}
function closeModal(){state.activeJob=null;document.querySelector("#detail-modal").classList.remove("open");document.querySelector("#detail-modal").setAttribute("aria-hidden","true")}

function modelSizeBillions(modelId) {
  const matches=[...String(modelId||"").toLowerCase().matchAll(/([0-9]+(?:p[0-9]+)?)b(?![a-z0-9])/g)];
  return matches.length ? Number(matches.at(-1)[1].replace("p",".")) : Number.POSITIVE_INFINITY;
}
function memoryRowCompare(left,right) {
  const sizeLeft=modelSizeBillions(left.model_id),sizeRight=modelSizeBillions(right.model_id);
  if(sizeLeft!==sizeRight)return sizeLeft-sizeRight;
  const modelOrder=String(left.model_id||"").localeCompare(String(right.model_id||""),"zh-CN",{numeric:true});
  if(modelOrder)return modelOrder;
  const lengthOrder=Number(left.cutoff_len||0)-Number(right.cutoff_len||0);
  if(lengthOrder)return lengthOrder;
  return String(left.train_type||"").localeCompare(String(right.train_type||""))
    || Number(left.gpu_count||0)-Number(right.gpu_count||0)
    || String(left.zero||"").localeCompare(String(right.zero||""))
    || Number(Boolean(left.gc))-Number(Boolean(right.gc));
}

function populateMemoryModelFilter(models) {
  const select=document.querySelector("#memory-model-filter"),current=select.value;
  const html=`<option value="">全部模型</option>${models.map(model=>`<option value="${escapeHtml(model)}">${escapeHtml(model)}</option>`).join("")}`;
  if(setHtml(select,html)&&models.includes(current))select.value=current;
}
function renderMemoryTable() {
  const model=document.querySelector("#memory-model-filter")?.value||"";
  const trainType=document.querySelector("#memory-train-filter")?.value||"";
  const rows=state.memoryRows.filter(row=>(!model||row.model_id===model)&&(!trainType||row.train_type===trainType));
  setText("#memory-table-count",`显示 ${rows.length} / ${state.memoryRows.length} 条`);
  setHtml("#memory-table",rows.length?`<table><thead><tr><th>硬件</th><th>模型</th><th>类型</th><th>长度</th><th>GPU</th><th>ZeRO/GC</th><th>最大 MBS</th><th>首个失败</th><th>MFU</th><th>samples/s</th><th>PyTorch Allocated</th><th>设备峰值 / 上限</th></tr></thead><tbody>${rows.map(x=>`<tr><td>${escapeHtml(x.gpu_type||x.hardware_id||"—")}</td><td>${escapeHtml(x.model_id)}</td><td>${escapeHtml((x.train_type||"").toUpperCase())}</td><td>${number(x.cutoff_len,0)}</td><td>${x.gpu_count}</td><td>${escapeHtml(x.zero||"none")} / ${x.gc?"On":"Off"}</td><td>${x.max_feasible_mbs??"—"}</td><td>${x.first_failed_mbs??"—"}</td><td><strong>${percent(x.mfu,1)}</strong></td><td>${number(x.samples_per_second,2)}</td><td>${number(x.max_allocated_gib,1)} GiB</td><td>${number(x.nvidia_smi_peak_gib,1)} / ${number(x.memory_total_gib,1)} GiB</td></tr>`).join("")}</tbody></table>`:empty(state.memoryRows.length?"没有符合当前筛选条件的显存边界":"等待结果"));
}

async function loadMemory() {
  const data=await api("/api/v1/analysis/memory");const rows=[...(data.rows||[])].sort(memoryRowCompare);document.querySelector("#memory-kpis").innerHTML=[kpi("已索引边界族",number(data.families_indexed,0)),kpi("已确定边界",number(data.boundaries_found,0)),kpi("可行 MBS 中位数",number(median(rows.map(x=>x.max_feasible_mbs)),1)),kpi("边界显存中位数",`${number(median(rows.map(x=>x.max_allocated_gib)),1)} GB`)].join("");
  const datasets=[...new Set(rows.map(x=>x.dataset_id))].sort((a,b)=>(rows.find(x=>x.dataset_id===a)?.cutoff_len||0)-(rows.find(x=>x.dataset_id===b)?.cutoff_len||0));const models=[...new Set(rows.map(x=>x.model_id))];
  state.memoryRows=rows;populateMemoryModelFilter(models);renderMemoryTable();
  document.querySelector("#memory-heatmap").innerHTML=rows.length?`<table><thead><tr><th>模型</th>${datasets.map(x=>`<th>${escapeHtml(x.replace("longcontext_",""))}</th>`).join("")}</tr></thead><tbody>${models.map(model=>`<tr><td>${escapeHtml(model)}</td>${datasets.map(ds=>{const vals=rows.filter(x=>x.model_id===model&&x.dataset_id===ds&&x.max_feasible_mbs!=null).map(x=>x.max_feasible_mbs);const val=vals.length?Math.max(...vals):null;const opacity=val?Math.min(.85,.18+Math.log2(val+1)*.17):.08;return `<td><span class="heat-cell" style="background:rgba(61,214,160,${opacity})">${val??"—"}</span></td>`}).join("")}</tr>`).join("")}</tbody></table>`:empty("显存边界实验尚未完成");
  const chartRows=rows.filter(x=>x.max_allocated_gib).slice(0,80);drawLineChart(document.querySelector("#memory-chart"),[{name:"Allocated GiB",values:chartRows.map((x,i)=>({x:i,y:x.max_allocated_gib}))},{name:"设备上限",color:"#ff6f73",values:chartRows.map((x,i)=>({x:i,y:x.memory_total_gib}))}],{digits:0});
}
function median(values){const x=values.filter(v=>v!=null&&Number.isFinite(Number(v))).map(Number).sort((a,b)=>a-b);if(!x.length)return null;return x.length%2?x[(x.length-1)/2]:(x[x.length/2-1]+x[x.length/2])/2}

async function loadThroughput(){
  const data=await api("/api/v1/analysis/throughput"),rows=data.configurations||[],ok=rows.filter(x=>x.successful_runs);
  const thermal=ok.filter(x=>x.clock_status==="thermal_limited"),powerLimited=ok.filter(x=>x.clock_status==="power_limited"),downclocked=ok.filter(x=>x.clock_status==="downclocked"),insufficient=ok.filter(x=>x.clock_status==="insufficient_data");
  const alerts=[];
  if(thermal.length)alerts.push(`<div class="alert error">${thermal.length} 个正式配置观测到热限频；比较吞吐前应优先复验。</div>`);
  if(powerLimited.length)alerts.push(`<div class="alert warning">${powerLimited.length} 个正式配置受功耗墙限制。这不判为任务失败，但频率差异会影响吞吐与标准 MFU。</div>`);
  if(downclocked.length)alerts.push(`<div class="alert warning">${downclocked.length} 个正式配置存在无法由现有历史字段解释的降频。</div>`);
  if(insufficient.length)alerts.push(`<div class="info-banner">${insufficient.length} 个正式配置来自旧的 6 列采样或缺少完整 limiter 字段；展示原始时钟，但不据此断言降频原因。</div>`);
  setHtml("#throughput-alerts",alerts.join(""));
  setHtml("#throughput-kpis",[
    kpi("正式配置",number(rows.length,0),`${data.runs} 次正式运行 · ${data.screening_runs||0} 个初筛任务`),
    kpi("有效配置",number(ok.length,0),`${powerLimited.length+thermal.length+downclocked.length} 个明确受限 · ${insufficient.length} 个原因不足`),
    kpi("Samples/s 中位数",number(median(ok.map(x=>x.samples_per_second)),2)),
    kpi("MFU 中位数",percent(median(ok.map(x=>x.mfu)),1),"标准口径；有独立健康基线时才给出频率折算值"),
  ].join(""));
  const shown=ok.slice(0,80);
  drawLineChart(document.querySelector("#throughput-chart"),[{name:"samples/s",values:shown.map((x,i)=>({x:i,y:x.samples_per_second}))}],{digits:2});
  drawLineChart(document.querySelector("#efficiency-chart"),[{name:"MFU",values:shown.map((x,i)=>({x:i,y:x.mfu==null?null:x.mfu*100}))},{name:"频率折算 MFU",values:shown.map((x,i)=>({x:i,y:x.clock_adjusted_mfu==null?null:x.clock_adjusted_mfu*100}))},{name:"Padding %",values:shown.map((x,i)=>({x:i,y:x.padding_efficiency==null?null:x.padding_efficiency*100}))}],{digits:0});
  setHtml("#throughput-table",rows.length?`<table><thead><tr><th>模型</th><th>类型</th><th>数据</th><th>GPU</th><th>MBS/GBS</th><th>GC/ZeRO</th><th>重复</th><th>samples/s</th><th>有效tok/s</th><th>MFU / 折算</th><th>频率状态</th><th>功耗墙占比</th><th>温度 P95</th><th>Epoch(1000)</th><th>GPU-hours</th><th>CV</th></tr></thead><tbody>${rows.map(x=>`<tr><td>${escapeHtml(x.model_id)}</td><td>${escapeHtml((x.train_type||"").toUpperCase())}</td><td>${escapeHtml(x.dataset_id)}</td><td>${x.gpu_count}</td><td>${x.mbs}/${x.target_gbs}</td><td>${x.gc?"On":"Off"}/${escapeHtml(x.zero_name||"none")}</td><td>${x.successful_runs}/${x.runs}</td><td>${number(x.samples_per_second,2)}</td><td>${number(x.effective_tokens_per_second,0)}</td><td><strong>${percent(x.mfu,1)}</strong><br><small>${percent(x.clock_adjusted_mfu,1)} 折算</small></td><td>${clockSummary(x)}</td><td>${percent(x.power_limit_busy_fraction,1)}</td><td>${x.busy_temperature_p95_c==null?"—":`${number(x.busy_temperature_p95_c,0)} °C`}</td><td>${duration(x.estimated_epoch_seconds_1000_samples)}</td><td>${number(x.gpu_hours_per_1000_samples,3)}</td><td>${percent(x.samples_per_second_cv,1)}</td></tr>`).join("")}</tbody></table>`:empty("正式吞吐实验尚未物化或完成"));
}

async function loadScaling(){const data=await api("/api/v1/analysis/scaling"),families=data.families||[];document.querySelector("#scaling-grid").innerHTML=families.length?families.map(f=>{const max=Math.max(...f.points.map(x=>x.samples_per_second||0),1),base=f.points[0]||{};return `<div class="comparison-card"><div class="comparison-title"><div><strong>${escapeHtml(base.model_id||f.request_id)}</strong><br><small>${escapeHtml(base.dataset_id||"")} · ${escapeHtml((base.train_type||"").toUpperCase())}</small></div><span class="mono">${escapeHtml(shortId(f.request_id))}</span></div>${f.points.map(p=>`<div class="bar-row"><span>${p.gpu_count}卡</span><div class="bar-track"><span class="bar-fill ${p.passes_70_percent_rule?"good":""}" style="width:${(p.samples_per_second||0)/max*100}%"></span></div><strong>${number(p.samples_per_second,2)}/s</strong></div>`).join("")}<div class="decision ${f.points.every(x=>x.passes_70_percent_rule)?"":"off"}"><strong>${f.points.every(x=>x.passes_70_percent_rule)?"扩卡收益达标":"建议在首个未达标点停止"}</strong> · ${f.points.map(x=>`${x.gpu_count}卡 ${x.gain_from_previous==null?"基线":percent(x.gain_from_previous)}`).join(" / ")}</div></div>`}).join(""):empty("多卡扩展实验尚未完成");}

async function loadPacking(){const data=await api("/api/v1/analysis/packing"),pairs=data.pairs||[];document.querySelector("#packing-grid").innerHTML=pairs.length?pairs.map(p=>{const off=p.no_packing,on=p.packing,max=Math.max(off.estimated_epoch_seconds_1000_samples||0,on.estimated_epoch_seconds_1000_samples||0,1);return `<div class="comparison-card"><div class="comparison-title"><div><strong>${escapeHtml(off.model_id)} · ${escapeHtml(off.dataset_id)}</strong><br><small>${escapeHtml((off.train_type||"").toUpperCase())} · ${off.gpu_count}卡 · GBS ${off.target_gbs}</small></div>${badge(p.decision==="on"?"success":"oom")}</div><div class="bar-row"><span>No-pack</span><div class="bar-track"><span class="bar-fill" style="width:${off.estimated_epoch_seconds_1000_samples/max*100}%"></span></div><strong>${duration(off.estimated_epoch_seconds_1000_samples)}</strong></div><div class="bar-row"><span>Packing</span><div class="bar-track"><span class="bar-fill good" style="width:${on.estimated_epoch_seconds_1000_samples/max*100}%"></span></div><strong>${duration(on.estimated_epoch_seconds_1000_samples)}</strong></div><div class="decision ${p.decision==="on"?"":"off"}"><strong>建议 ${p.decision==="on"?"开启":"关闭"} Packing</strong> · 中位收益 ${percent(p.median_paired_time_gain)}，最差重复 ${percent(p.worst_paired_time_gain)}，门槛 ${percent(p.decision_threshold)}；No-pack MBS=${off.mbs}</div></div>`}).join(""):empty("Packing 成对实验尚未完成");}

function recommendationOption(name,row){if(!row)return "";return `<div class="bar-row"><span>${name}</span><div>${row.gpu_count}×${escapeHtml(row.gpu_type||row.hardware_id||"GPU")} · MBS ${row.mbs} · ${row.gc?"GC":"No GC"} · ${escapeHtml(row.zero_name||"none")}</div><strong>${duration(row.estimated_epoch_seconds_1000_samples)}</strong></div>`}
function sameCandidate(left,right){return left&&right&&["gpu_count","zero_name","gc","mbs","target_gbs","packing"].every(key=>left[key]===right[key])}
function candidateState(row){
  const formal=row.formal_status_counts||{},screen=row.screen_status_counts||{};
  if(row.formal_successful_runs)return `<span class="badge success">正式完成</span>`;
  if(row.is_shortlisted&&formal.running)return `<span class="badge running">正式运行</span>`;
  if(row.is_shortlisted&&formal.planned)return `<span class="badge planned">等待正式</span>`;
  if(formal.oom)return `<span class="badge oom">正式 OOM</span>`;
  if(row.screen_successful_runs)return `<span class="badge boundary_found">初筛完成</span>`;
  if(screen.running)return `<span class="badge running">正在初筛</span>`;
  if(screen.planned)return `<span class="badge planned">等待初筛</span>`;
  if(screen.oom)return `<span class="badge oom">初筛 OOM</span>`;
  return badge("failed");
}
function recommendationCandidates(scenario){
  const rows=scenario.candidates||[];
  if(!rows.length)return "";
  return `<details class="candidate-details"><summary>查看全部 ${rows.length} 个候选配置</summary><div class="table-wrap"><table><thead><tr><th>状态</th><th>选择</th><th>GPU</th><th>ZeRO / GC</th><th>MBS</th><th>samples/s</th><th>MFU / 折算</th><th>频率状态</th><th>千样本时间</th><th>GPU-hours</th><th>正式成功</th></tr></thead><tbody>${rows.map(row=>{
    const roles=[];if(sameCandidate(row,scenario.default))roles.push("默认");if(sameCandidate(row,scenario.lowest_resource))roles.push("最省");if(sameCandidate(row,scenario.fastest))roles.push("最快");
    const metrics=row.formal_successful_runs?row:(row.screen_metrics||row);
    return `<tr><td>${candidateState(row)}</td><td>${roles.length?roles.map(role=>`<span class="badge success">${role}</span>`).join(" "):"—"}</td><td>${row.gpu_count}×${escapeHtml(row.gpu_type||row.hardware_id||"GPU")}</td><td>${escapeHtml(row.zero_name||"none")} / ${row.gc?"On":"Off"}</td><td>${row.mbs}</td><td>${number(metrics.samples_per_second,3)}</td><td><strong>${percent(metrics.mfu,1)}</strong><br><small>${percent(metrics.clock_adjusted_mfu,1)} 折算</small></td><td>${clockSummary(metrics)}</td><td>${duration(metrics.estimated_epoch_seconds_1000_samples)}</td><td>${number(metrics.gpu_hours_per_1000_samples,3)}</td><td>${row.formal_successful_runs||0}</td></tr>`;
  }).join("")}</tbody></table></div></details>`;
}
async function loadRecommendations(){const data=await api("/api/v1/recommendations"),rows=data.scenarios||[];setHtml("#recommendation-grid",rows.length?rows.map(x=>`<div class="recommendation-card"><div class="comparison-title"><div><strong>${escapeHtml(x.model_id)} · ${escapeHtml(x.dataset_id)}</strong><br><small>${escapeHtml((x.train_type||"").toUpperCase())} · GBS ${x.target_gbs} · 短测 ${x.screened_candidate_count}/${x.candidate_count} · 正式 ${x.measured_candidate_count}/${x.shortlisted_candidate_count}</small></div><span class="badge ${x.status==="comparison_complete"?"success":"running"}">${x.status==="comparison_complete"?"比较完成":x.status==="formal_in_progress"?"正式复验中":"初筛中"}</span></div>${recommendationOption("默认",x.default)}${recommendationOption("最省资源",x.lowest_resource)}${recommendationOption("最快",x.fastest)}${x.measured_candidate_count?recommendationCandidates(x):`${empty("正式候选尚未完成，短测结果仅用于入围")}${recommendationCandidates(x)}`}<div class="decision"><strong>选择逻辑</strong> · 每场景最多短测 4 个代表项，正式复验最低资源与全局最快 Top-2；最终推荐只使用正式测量。频率状态用于提示复验风险，暂不静默改变推荐排序。</div></div>`).join(""):empty("完成吞吐初筛或正式实验后自动生成推荐结果"));}

async function loadSystem(){const data=await api("/api/v1/system");document.querySelector("#system-grid").innerHTML=`<article class="panel"><div class="panel-head"><div><h3>硬件 Campaign</h3><p>${escapeHtml(data.root)}</p></div></div><pre class="json">${escapeHtml(JSON.stringify(data.campaigns,null,2))}</pre></article><article class="panel"><div class="panel-head"><div><h3>服务与预检</h3><p>Dashboard 只读实验目录</p></div></div><pre class="json">${escapeHtml(JSON.stringify({database:data.database,read_only:data.read_only_experiment_access,ingestor:data.ingestor},null,2))}</pre></article>`;}

async function loadPage(page){try{if(page==="overview")return refreshOverview();if(page==="runs")return loadRuns();if(page==="memory")return loadMemory();if(page==="throughput")return loadThroughput();if(page==="scaling")return loadScaling();if(page==="packing")return loadPacking();if(page==="recommendations")return loadRecommendations();if(page==="system")return loadSystem();}catch(e){console.error(e)}}

function connectEvents(){const events=new EventSource(appUrl("/api/v1/events"));events.addEventListener("snapshot",()=>{setConnection(true,"实时连接");debounced(()=>{refreshOverview();if(state.page!=="overview")loadPage(state.page);if(state.activeJob)openRun(state.activeJob,true)});});events.addEventListener("heartbeat",()=>setConnection(true,"实时连接"));events.onerror=()=>setConnection(false,"正在重连");return events;}
function debounced(fn){clearTimeout(state.refreshTimer);state.refreshTimer=setTimeout(fn,350)}

document.addEventListener("DOMContentLoaded",()=>{
  api("/api/v1/campaigns").then(data=>{state.campaigns=data.rows||[];const options=state.campaigns.map(x=>`<option value="${escapeHtml(x.campaign_id)}">${escapeHtml(x.gpu_type)} · ${escapeHtml(x.attention_backend||"")}</option>`).join("");setHtml("#campaign-filter",`<option value="">全部 campaign</option>${options}`);setHtml("#run-campaign",`<option value="">全部</option>${options}`)}).catch(console.error);
  document.querySelectorAll(".nav-item").forEach(item=>item.addEventListener("click",()=>navigate(item.dataset.page)));
  document.querySelectorAll("[data-goto]").forEach(item=>item.addEventListener("click",()=>navigate(item.dataset.goto)));
  document.querySelectorAll("[data-close-modal]").forEach(item=>item.addEventListener("click",closeModal));
  document.querySelector("#refresh-button").addEventListener("click",()=>loadPage(state.page));
  document.querySelector("#apply-run-filters").addEventListener("click",loadRuns);
  document.querySelector("#campaign-filter").addEventListener("change",event=>{document.querySelector("#run-campaign").value=event.target.value;loadPage(state.page)});
  document.querySelector("#run-q").addEventListener("keydown",event=>{if(event.key==="Enter")loadRuns()});
  document.querySelector("#memory-model-filter").addEventListener("change",renderMemoryTable);
  document.querySelector("#memory-train-filter").addEventListener("change",renderMemoryTable);
  document.addEventListener("keydown",event=>{if(event.key==="Escape")closeModal()});
  const initial=location.hash.slice(1);navigate(pages[initial]?initial:"overview");connectEvents();setInterval(()=>{if(!state.overview)refreshOverview()},5000);
});
