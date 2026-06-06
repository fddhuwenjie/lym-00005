import time
import math
import heapq
import random
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from enum import Enum

app = FastAPI(title="医院急诊分诊离散事件仿真 API", version="1.0.0")


class ESILevel(str, Enum):
    ESI_1 = "ESI_1"
    ESI_2 = "ESI_2"
    ESI_3 = "ESI_3"
    ESI_4 = "ESI_4"
    ESI_5 = "ESI_5"


ESI_PRIORITY = {
    ESILevel.ESI_1: 1, ESILevel.ESI_2: 2, ESILevel.ESI_3: 3,
    ESILevel.ESI_4: 4, ESILevel.ESI_5: 5,
}

ESI_NAMES = {
    ESILevel.ESI_1: "红色-立即抢救", ESILevel.ESI_2: "橙色-紧急",
    ESILevel.ESI_3: "黄色-紧急", ESILevel.ESI_4: "绿色-非紧急",
    ESILevel.ESI_5: "蓝色-非紧急",
}


class DistributionType(str, Enum):
    EXPONENTIAL = "exponential"
    NORMAL = "normal"
    UNIFORM = "uniform"
    GAMMA = "gamma"


class DistributionParams(BaseModel):
    distribution_type: DistributionType = DistributionType.GAMMA
    mean: float = Field(gt=0)
    std: Optional[float] = Field(None, gt=0)
    shape: Optional[float] = Field(None, gt=0)
    scale: Optional[float] = Field(None, gt=0)


class ScenarioConfig(BaseModel):
    name: str
    num_doctors: int = Field(ge=1)
    resuscitation_room_capacity: int = Field(ge=1)
    simulation_duration: float = Field(gt=0)
    arrival_rate: float = Field(gt=0)
    esi_distribution: Dict[ESILevel, float]
    service_time_params: Dict[ESILevel, DistributionParams]
    target_wait_time: Optional[Dict[ESILevel, float]] = None


class SingleSimRequest(BaseModel):
    scenario: ScenarioConfig
    seed: int = 42


class MonteCarloRequest(BaseModel):
    scenario: ScenarioConfig
    n: int = Field(100, ge=1, le=10000)
    base_seed: int = 42


class CompareRequest(BaseModel):
    scenarios: List[ScenarioConfig]
    n: int = Field(100, ge=1, le=10000)
    base_seed: int = 42


class ReportRequest(BaseModel):
    scenario: ScenarioConfig
    monte_carlo_n: Optional[int] = Field(1000, ge=1, le=10000)
    seed: Optional[int] = 42


def _sample_distribution(params: Dict, rng: random.Random) -> float:
    dt = params["distribution_type"]
    mean = params["mean"]
    if dt == "exponential":
        return rng.expovariate(1.0 / mean)
    if dt == "normal":
        std = params.get("std", mean * 0.3)
        val = rng.gauss(mean, std)
        return val if val > 0.1 else 0.1
    if dt == "uniform":
        std = params.get("std", mean * 0.5)
        low = max(0.1, mean - std)
        high = mean + std
        return rng.uniform(low, high)
    shape = params.get("shape")
    scale = params.get("scale")
    if shape is None or scale is None:
        std = params.get("std", mean * 0.5)
        if std > 0:
            shape = (mean / std) ** 2
            scale = (std ** 2) / mean
        else:
            shape, scale = 1.0, mean
    val = rng.gammavariate(shape, scale)
    return val if val > 0.1 else 0.1


def _sample_esi(cum_probs: List[Tuple[ESILevel, float]], rng: random.Random) -> ESILevel:
    r = rng.random()
    for level, cum in cum_probs:
        if r <= cum:
            return level
    return cum_probs[-1][0]


def _run_sim_core(scenario: Dict, seed: int) -> Dict:
    rng = random.Random(seed)
    
    num_docs = scenario["num_doctors"]
    resus_cap = scenario["resuscitation_room_capacity"]
    sim_dur = scenario["simulation_duration"]
    arr_rate = scenario["arrival_rate"]
    service_params = scenario["service_time_params"]
    
    cum_probs = []
    cum = 0.0
    for lvl, p in scenario["esi_distribution"].items():
        cum += p
        cum_probs.append((lvl, cum))
    
    eq: List[Tuple[float, int, int, int, Dict]] = []
    pid = 0
    eid = 0
    
    patients: Dict[int, Dict] = {}
    wait_q: List[Tuple[int, float, int]] = []
    doc_busy: List[Optional[int]] = [None] * num_docs
    doc_busy_end: List[float] = [0.0] * num_docs
    doc_busy_start: List[float] = [0.0] * num_docs
    doc_total: List[float] = [0.0] * num_docs
    
    resus_occ = 0
    resus_peak = 0
    resus_int = 0.0
    last_t = 0.0
    preemptions = 0
    
    first_arr = rng.expovariate(arr_rate)
    if first_arr <= sim_dur:
        heapq.heappush(eq, (first_arr, eid, 0, 0, {"type": "arr"}))
        eid += 1
    
    def try_start(ct: float):
        nonlocal resus_occ, resus_peak, eid
        while wait_q:
            prio, at, pat_id = wait_q[0]
            pat = patients[pat_id]
            pat_prio = ESI_PRIORITY[pat["esi_level"]]
            
            has_bed = resus_occ < resus_cap
            bed_preempt_doc = -1
            
            if not has_bed:
                if pat_prio == 1:
                    lowest_prio = 999
                    for d in range(num_docs):
                        if doc_busy[d] is not None:
                            op = patients[doc_busy[d]]
                            op_prio = ESI_PRIORITY[op["esi_level"]]
                            if op_prio > 1 and op_prio < lowest_prio:
                                lowest_prio = op_prio
                                bed_preempt_doc = d
            
            free_doc = -1
            for d in range(num_docs):
                if doc_busy[d] is None:
                    free_doc = d
                    break
            
            doc_preempt = -1
            if free_doc == -1:
                lowest_prio = 999
                for d in range(num_docs):
                    if doc_busy[d] is not None:
                        op = patients[doc_busy[d]]
                        op_prio = ESI_PRIORITY[op["esi_level"]]
                        if op_prio > pat_prio and op_prio < lowest_prio:
                            lowest_prio = op_prio
                            doc_preempt = d
            
            if bed_preempt_doc >= 0:
                doc_preempt = bed_preempt_doc
                free_doc = doc_preempt
                has_bed = True
            
            if doc_preempt >= 0:
                op_id = doc_busy[doc_preempt]
                op = patients[op_id]
                elapsed = ct - doc_busy_start[doc_preempt]
                total = op["total_service_time"]
                remaining = max(0.1, total - elapsed)
                
                op["remaining"] = remaining
                op["service_start"] = None
                op["service_end"] = None
                op["preempted"] = True
                op["preempt_count"] += 1
                op["completed"] = False
                
                doc_total[doc_preempt] += elapsed
                doc_busy[doc_preempt] = None
                resus_occ -= 1
                
                heapq.heappush(wait_q, (
                    ESI_PRIORITY[op["esi_level"]],
                    op["arrival_time"],
                    op_id
                ))
                free_doc = doc_preempt
            
            if free_doc >= 0 and has_bed:
                heapq.heappop(wait_q)
                if pat["remaining"] is not None:
                    st = pat["remaining"]
                    pat["remaining"] = None
                else:
                    sp = service_params[pat["esi_level"]]
                    st = _sample_distribution(sp, rng)
                    pat["total_service_time"] = st
                
                pat["service_start"] = ct
                pat["wait_time"] = ct - pat["arrival_time"]
                
                doc_busy[free_doc] = pat_id
                doc_busy_start[free_doc] = ct
                end_t = ct + st
                doc_busy_end[free_doc] = end_t
                
                heapq.heappush(eq, (end_t, eid, 2, free_doc, {
                    "type": "end", "doc": free_doc, "pid": pat_id
                }))
                eid += 1
                
                resus_occ += 1
                if resus_occ > resus_peak:
                    resus_peak = resus_occ
            else:
                break
    
    ct = 0.0
    while eq:
        ev_t, _, _, _, ev = heapq.heappop(eq)
        ct = ev_t
        
        if ct > sim_dur and ev["type"] == "arr":
            continue
        
        dt = ct - last_t
        resus_int += resus_occ * dt
        last_t = ct
        
        if ev["type"] == "arr":
            pid += 1
            esi = _sample_esi(cum_probs, rng)
            pat = {
                "patient_id": pid,
                "esi_level": esi,
                "arrival_time": ct,
                "service_start": None,
                "service_end": None,
                "wait_time": None,
                "total_service_time": None,
                "preempted": False,
                "preempt_count": 0,
                "remaining": None,
                "completed": False,
            }
            patients[pid] = pat
            heapq.heappush(wait_q, (ESI_PRIORITY[esi], ct, pid))
            
            next_arr = ct + rng.expovariate(arr_rate)
            if next_arr <= sim_dur:
                heapq.heappush(eq, (next_arr, eid, 0, 0, {"type": "arr"}))
                eid += 1
            
            try_start(ct)
        
        elif ev["type"] == "end":
            d = ev["doc"]
            pat_id = ev["pid"]
            if doc_busy[d] == pat_id:
                pat = patients[pat_id]
                pat["service_end"] = ct
                pat["completed"] = True
                
                elapsed = ct - doc_busy_start[d]
                doc_total[d] += elapsed
                doc_busy[d] = None
                resus_occ -= 1
                
                if pat["preempt_count"] > 0:
                    preemptions += pat["preempt_count"]
                
                try_start(ct)
    
    for d in range(num_docs):
        if doc_busy[d] is not None:
            pat_id = doc_busy[d]
            pat = patients[pat_id]
            elapsed = ct - doc_busy_start[d]
            doc_total[d] += elapsed
            pat["service_start"] = None
    
    total_busy = sum(doc_total)
    util = total_busy / (num_docs * max(ct, 0.001)) if ct > 0 else 0.0
    
    completed = [p for p in patients.values() if p["completed"]]
    waits_esi: Dict[ESILevel, List[float]] = {}
    for p in completed:
        esi = p["esi_level"]
        if esi not in waits_esi:
            waits_esi[esi] = []
        wt = p["wait_time"]
        if wt is not None:
            waits_esi[esi].append(wt)
    
    avg_wait = {}
    max_wait = {}
    for esi in ESILevel:
        w = waits_esi.get(esi, [])
        avg_wait[esi] = sum(w) / len(w) if w else 0.0
        max_wait[esi] = max(w) if w else 0.0
    
    avg_resus = resus_int / max(ct, 0.001) if ct > 0 else 0.0
    
    patient_list = []
    for p in patients.values():
        patient_list.append({
            "patient_id": p["patient_id"],
            "esi_level": p["esi_level"],
            "arrival_time": p["arrival_time"],
            "service_start_time": p["service_start"],
            "service_end_time": p["service_end"],
            "wait_time": p["wait_time"],
            "total_service_time": p["total_service_time"],
            "was_preempted": p["preempted"],
            "preemption_count": p["preempt_count"],
            "remaining_service_time": p["remaining"],
            "is_completed": p["completed"],
        })
    
    return {
        "patients": patient_list,
        "doctor_utilization": util,
        "avg_wait_time_by_esi": avg_wait,
        "max_wait_time_by_esi": max_wait,
        "resuscitation_peak_occupancy": resus_peak,
        "resuscitation_avg_occupancy": avg_resus,
        "total_arrivals": len(patients),
        "total_completed": len(completed),
        "total_preemptions": preemptions,
    }


def _quantile(sorted_data: List[float], q: float) -> float:
    if not sorted_data:
        return 0.0
    n = len(sorted_data)
    pos = (n - 1) * q
    floor = int(pos)
    ceil = min(floor + 1, n - 1)
    if floor == ceil:
        return sorted_data[floor]
    frac = pos - floor
    return sorted_data[floor] * (1 - frac) + sorted_data[ceil] * frac


def _mean(d: List[float]) -> float:
    return sum(d) / len(d) if d else 0.0


def _std(d: List[float]) -> float:
    if len(d) < 2:
        return 0.0
    m = _mean(d)
    var = sum((x - m) ** 2 for x in d) / (len(d) - 1)
    return math.sqrt(var)


def _t_crit(df: int) -> float:
    t = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,
         11:2.201,12:2.179,13:2.160,14:2.145,15:2.131,16:2.120,17:2.110,18:2.101,19:2.093,20:2.086,
         25:2.060,30:2.042,40:2.021,50:2.009,60:2.000,80:1.990,100:1.984,1000:1.962,10000:1.960}
    for k in sorted(t.keys(), reverse=True):
        if df >= k:
            return t[k]
    return 1.96


def _ci(d: List[float]) -> Dict[str, float]:
    if not d:
        return {"mean":0.0,"lower":0.0,"upper":0.0,"std":0.0}
    n = len(d)
    m = _mean(d)
    s = _std(d)
    se = s / math.sqrt(n)
    t = _t_crit(n-1)
    margin = t * se
    return {"mean":m,"lower":m-margin,"upper":m+margin,"std":s}


def _analyze_mc(results: List[Dict], scenario: Dict) -> Dict:
    n = len(results)
    waits_esi: Dict[ESILevel, List[float]] = {e: [] for e in ESILevel}
    utils: List[float] = []
    peaks: List[float] = []
    arrivals: List[float] = []
    
    target_wait = scenario.get("target_wait_time", {})
    exceed: Dict[ESILevel, int] = {e: 0 for e in ESILevel}
    total_esi: Dict[ESILevel, int] = {e: 0 for e in ESILevel}
    
    for r in results:
        utils.append(r["doctor_utilization"])
        peaks.append(r["resuscitation_peak_occupancy"])
        arrivals.append(r["total_arrivals"])
        for p in r["patients"]:
            if p["is_completed"] and p["wait_time"] is not None:
                esi = p["esi_level"]
                waits_esi[esi].append(p["wait_time"])
                total_esi[esi] += 1
                target = target_wait.get(esi)
                if target is not None and p["wait_time"] > target:
                    exceed[esi] += 1
    
    q_levels = ["P50","P75","P90","P95","P99"]
    q_vals = [0.5, 0.75, 0.9, 0.95, 0.99]
    q_res: Dict[ESILevel, Dict[str, float]] = {}
    for esi in ESILevel:
        w = sorted(waits_esi[esi])
        q_res[esi] = {}
        for qn, qv in zip(q_levels, q_vals):
            q_res[esi][qn] = _quantile(w, qv)
    
    util_ci = _ci(utils)
    t_prob: Dict[ESILevel, float] = {}
    for esi in ESILevel:
        t_prob[esi] = exceed[esi] / total_esi[esi] if total_esi[esi] > 0 else 0.0
    
    return {
        "num_simulations": n,
        "wait_time_quantiles": q_res,
        "doctor_utilization_ci": util_ci,
        "target_wait_probability": t_prob,
        "avg_doctor_utilization": util_ci["mean"],
        "avg_resuscitation_peak": _mean(peaks),
        "avg_total_arrivals": _mean(arrivals),
    }


def _config_to_dict(cfg: ScenarioConfig) -> Dict:
    return {
        "name": cfg.name,
        "num_doctors": cfg.num_doctors,
        "resuscitation_room_capacity": cfg.resuscitation_room_capacity,
        "simulation_duration": cfg.simulation_duration,
        "arrival_rate": cfg.arrival_rate,
        "esi_distribution": cfg.esi_distribution,
        "service_time_params": {k: v.model_dump() for k, v in cfg.service_time_params.items()},
        "target_wait_time": cfg.target_wait_time,
    }


def _get_presets() -> List[Dict]:
    def_svc = {
        ESILevel.ESI_1: {"distribution_type":"gamma","mean":45.0,"std":20.0,"shape":5.06,"scale":8.9},
        ESILevel.ESI_2: {"distribution_type":"gamma","mean":30.0,"std":12.0,"shape":6.25,"scale":4.8},
        ESILevel.ESI_3: {"distribution_type":"gamma","mean":20.0,"std":8.0,"shape":6.25,"scale":3.2},
        ESILevel.ESI_4: {"distribution_type":"gamma","mean":12.0,"std":5.0,"shape":5.76,"scale":2.08},
        ESILevel.ESI_5: {"distribution_type":"gamma","mean":8.0,"std":3.0,"shape":7.11,"scale":1.125},
    }
    def_target = {
        ESILevel.ESI_1: 1.0, ESILevel.ESI_2: 10.0, ESILevel.ESI_3: 30.0,
        ESILevel.ESI_4: 60.0, ESILevel.ESI_5: 120.0,
    }
    return [
        {
            "name": "标准场景 - 日间常规",
            "num_doctors": 5, "resuscitation_room_capacity": 8,
            "simulation_duration": 480.0, "arrival_rate": 0.5,
            "esi_distribution": {ESILevel.ESI_1:0.03,ESILevel.ESI_2:0.08,ESILevel.ESI_3:0.25,ESILevel.ESI_4:0.35,ESILevel.ESI_5:0.29},
            "service_time_params": def_svc, "target_wait_time": def_target,
        },
        {
            "name": "高峰场景 - 晚高峰",
            "num_doctors": 6, "resuscitation_room_capacity": 10,
            "simulation_duration": 480.0, "arrival_rate": 0.8,
            "esi_distribution": {ESILevel.ESI_1:0.04,ESILevel.ESI_2:0.12,ESILevel.ESI_3:0.28,ESILevel.ESI_4:0.32,ESILevel.ESI_5:0.24},
            "service_time_params": def_svc, "target_wait_time": def_target,
        },
        {
            "name": "夜间场景 - 低负荷",
            "num_doctors": 3, "resuscitation_room_capacity": 6,
            "simulation_duration": 480.0, "arrival_rate": 0.2,
            "esi_distribution": {ESILevel.ESI_1:0.02,ESILevel.ESI_2:0.05,ESILevel.ESI_3:0.20,ESILevel.ESI_4:0.38,ESILevel.ESI_5:0.35},
            "service_time_params": def_svc, "target_wait_time": def_target,
        },
    ]


def _run_batch(scenario: Dict, n: int, base_seed: int) -> List[Dict]:
    results = []
    for i in range(n):
        results.append(_run_sim_core(scenario, base_seed + i))
    return results


def _gen_report(scenario: Dict, mc_res: Optional[Dict], single_res: Optional[Dict]) -> str:
    lines = []
    lines.append(f"# 急诊分诊仿真报告 - {scenario['name']}")
    lines.append("")
    lines.append("## 1. 输入参数")
    lines.append("")
    lines.append("### 1.1 资源配置")
    lines.append("")
    lines.append("| 参数 | 值 |")
    lines.append("|------|-----|")
    lines.append(f"| 医生数量 | {scenario['num_doctors']} 名 |")
    lines.append(f"| 抢救室容量 | {scenario['resuscitation_room_capacity']} 床 |")
    lines.append(f"| 仿真时长 | {scenario['simulation_duration']:.0f} 分钟 |")
    lines.append(f"| 患者到达率 | {scenario['arrival_rate']:.3f} 患者/分钟 |")
    lines.append("")
    
    lines.append("### 1.2 ESI 五级分诊比例")
    lines.append("")
    lines.append("| 级别 | 名称 | 比例 |")
    lines.append("|------|------|------|")
    for esi in ESILevel:
        pct = scenario['esi_distribution'].get(esi, 0) * 100
        lines.append(f"| {esi.value} | {ESI_NAMES[esi]} | {pct:.1f}% |")
    lines.append("")
    
    lines.append("### 1.3 服务时间分布")
    lines.append("")
    lines.append("| 级别 | 分布类型 | 均值(分钟) | 标准差(分钟) |")
    lines.append("|------|----------|------------|--------------|")
    for esi in ESILevel:
        sp = scenario['service_time_params'].get(esi, {})
        lines.append(f"| {esi.value} | {sp.get('distribution_type','N/A')} | {sp.get('mean',0):.1f} | {sp.get('std','N/A')} |")
    lines.append("")
    
    tw = scenario.get('target_wait_time')
    if tw:
        lines.append("### 1.4 目标等待时间")
        lines.append("")
        lines.append("| 级别 | 目标等待时间(分钟) |")
        lines.append("|------|-------------------|")
        for esi in ESILevel:
            t = tw.get(esi)
            if t is not None:
                lines.append(f"| {esi.value} | {t:.0f} |")
        lines.append("")
    
    if single_res:
        lines.append("## 2. 单次仿真结果")
        lines.append("")
        lines.append(f"- 总到达患者数: {single_res['total_arrivals']}")
        lines.append(f"- 完成服务患者数: {single_res['total_completed']}")
        lines.append(f"- 医生平均利用率: {single_res['doctor_utilization'] * 100:.1f}%")
        lines.append(f"- 抢救室峰值占用率: {single_res['resuscitation_peak_occupancy']}/{scenario['resuscitation_room_capacity']}")
        lines.append(f"- 总抢占次数: {single_res['total_preemptions']}")
        lines.append("")
    
    if mc_res:
        lines.append(f"## 3. 蒙特卡洛仿真结果 (N={mc_res['num_simulations']})")
        lines.append("")
        lines.append("### 3.1 等待时长分位数")
        lines.append("")
        lines.append("| 级别 | P50(分钟) | P75(分钟) | P90(分钟) | P95(分钟) | P99(分钟) |")
        lines.append("|------|-----------|-----------|-----------|-----------|-----------|")
        for esi in ESILevel:
            q = mc_res['wait_time_quantiles'].get(esi, {})
            lines.append(f"| {esi.value} | {q.get('P50',0):.1f} | {q.get('P75',0):.1f} | {q.get('P90',0):.1f} | {q.get('P95',0):.1f} | {q.get('P99',0):.1f} |")
        lines.append("")
        ci = mc_res['doctor_utilization_ci']
        lines.append(f"### 3.2 医生利用率: {ci['mean']*100:.1f}% (95% CI: [{ci['lower']*100:.1f}%, {ci['upper']*100:.1f}%])")
        lines.append("")
        if tw:
            lines.append("### 3.3 超目标等待时长概率")
            lines.append("")
            lines.append("| 级别 | 目标(分钟) | 超目标概率 | 达标率 |")
            lines.append("|------|-----------|-----------|--------|")
            for esi in ESILevel:
                t = tw.get(esi)
                if t is not None:
                    p = mc_res['target_wait_probability'].get(esi, 0)
                    lines.append(f"| {esi.value} | {t:.0f} | {p*100:.1f}% | {(1-p)*100:.1f}% |")
            lines.append("")
    
    lines.append("## 4. 改进建议")
    lines.append("")
    if mc_res:
        um = mc_res['avg_doctor_utilization']
        if um > 0.92:
            lines.append(f"- ⚠️ 医生利用率过高 ({um*100:.1f}%)，建议增加 1-2 名医生")
        elif um < 0.6:
            lines.append(f"- ⚠️ 医生利用率偏低 ({um*100:.1f}%)，可考虑减少 1 名医生")
        else:
            lines.append(f"- ✅ 医生利用率合理 ({um*100:.1f}%)")
        if tw:
            for esi in ESILevel:
                t = tw.get(esi)
                if t is not None:
                    p = mc_res['target_wait_probability'].get(esi, 0)
                    if p > 0.1:
                        lines.append(f"- ⚠️ {esi.value} 等待不达标，超标概率 {p*100:.1f}%")
    else:
        lines.append("- 建议运行蒙特卡洛仿真获得更稳健结论")
    
    return "\n".join(lines)


@app.get("/api/scenarios", summary="获取预置标准场景列表")
async def get_preset_scenarios():
    return {"scenarios": _get_presets()}


@app.post("/api/scenarios", summary="创建自定义仿真场景")
async def create_scenario(scenario: ScenarioConfig):
    total_p = sum(scenario.esi_distribution.values())
    if abs(total_p - 1.0) > 0.01:
        raise HTTPException(400, f"ESI 比例之和应为 1.0，当前为 {total_p}")
    return _config_to_dict(scenario)


@app.post("/api/simulate/single", summary="运行单次仿真")
async def simulate_single(req: SingleSimRequest):
    t0 = time.time()
    scenario = _config_to_dict(req.scenario)
    result = _run_sim_core(scenario, req.seed)
    result["elapsed_ms"] = int((time.time() - t0) * 1000)
    return result


@app.post("/api/simulate/monte-carlo", summary="运行蒙特卡洛批量仿真")
async def simulate_monte_carlo(req: MonteCarloRequest):
    t0 = time.time()
    scenario = _config_to_dict(req.scenario)
    
    results = _run_batch(scenario, req.n, req.base_seed)
    
    analysis = _analyze_mc(results, scenario)
    analysis["elapsed_ms"] = int((time.time() - t0) * 1000)
    analysis["per_sim_ms"] = int((time.time() - t0) * 1000 / req.n) if req.n > 0 else 0
    return analysis


@app.post("/api/simulate/compare", summary="多场景对比分析")
async def compare_scenarios(req: CompareRequest):
    t0 = time.time()
    if len(req.scenarios) < 2:
        raise HTTPException(400, "至少需要 2 个场景进行对比")
    
    scenario_dicts = [_config_to_dict(s) for s in req.scenarios]
    mc_results = []
    
    for sc in scenario_dicts:
        results = _run_batch(sc, req.n, req.base_seed)
        mc_results.append(_analyze_mc(results, sc))
    
    names = [s.get("name", f"场景{i+1}") for i, s in enumerate(scenario_dicts)]
    
    metrics = []
    for i, (sc, mc) in enumerate(zip(scenario_dicts, mc_results)):
        m = {
            "scenario_name": names[i],
            "num_doctors": sc["num_doctors"],
            "doctor_utilization_mean": mc["avg_doctor_utilization"],
            "doctor_utilization_lower": mc["doctor_utilization_ci"]["lower"],
            "doctor_utilization_upper": mc["doctor_utilization_ci"]["upper"],
        }
        tw = sc.get("target_wait_time", {})
        for esi in ESILevel:
            t = tw.get(esi)
            if t is not None:
                m[f"{esi.value}_wait_p95"] = mc["wait_time_quantiles"][esi]["P95"]
                m[f"{esi.value}_meet_target_prob"] = 1.0 - mc["target_wait_probability"][esi]
        metrics.append(m)
    
    rec_lines = ["## 医生数配置建议", ""]
    sorted_idx = sorted(range(len(scenario_dicts)), key=lambda i: scenario_dicts[i]["num_doctors"])
    for idx in sorted_idx:
        sc = scenario_dicts[idx]
        mc = mc_results[idx]
        nd = sc["num_doctors"]
        util = mc["avg_doctor_utilization"]
        if util > 0.92:
            rec_lines.append(f"- **{nd} 名医生**: 过载 (利用率 {util:.1%})，建议增加医生")
        elif util < 0.6:
            rec_lines.append(f"- **{nd} 名医生**: 资源闲置 (利用率 {util:.1%})，可考虑减少")
        else:
            rec_lines.append(f"- **{nd} 名医生**: 合理区间 (利用率 {util:.1%})")
        tw = sc.get("target_wait_time", {})
        if tw:
            for esi in ESILevel:
                t = tw.get(esi)
                if t is not None:
                    prob = 1.0 - mc["target_wait_probability"][esi]
                    rec_lines.append(f"  * {esi.value}: {prob:.1%} 达标")
    
    rec_lines.append("")
    best_score = -1
    best_idx = 0
    for i, sc in enumerate(scenario_dicts):
        mc = mc_results[i]
        util = mc["avg_doctor_utilization"]
        score = 0.0
        if 0.6 <= util <= 0.92:
            score += 50
        elif util < 0.6:
            score += util * 80
        else:
            score += max(0, (1.0 - util) * 80)
        tw = sc.get("target_wait_time", {})
        if tw:
            for esi, t in tw.items():
                prob = 1.0 - mc["target_wait_probability"][esi]
                if prob >= 0.95:
                    score += 10
        if score > best_score:
            best_score = score
            best_idx = i
    
    best = scenario_dicts[best_idx]
    rec_lines.append(f"## 最优配置推荐: {best['name']}")
    rec_lines.append(f"- 医生数: {best['num_doctors']} 名")
    rec_lines.append(f"- 抢救室容量: {best['resuscitation_room_capacity']} 床")
    recommendation = "\n".join(rec_lines)
    
    return {
        "scenario_names": names,
        "base_seed": req.base_seed,
        "metrics": metrics,
        "recommendation": recommendation,
        "elapsed_ms": int((time.time() - t0) * 1000),
    }


@app.post("/api/report/markdown", summary="生成 Markdown 仿真报告", response_class=PlainTextResponse)
async def generate_report(req: ReportRequest):
    scenario = _config_to_dict(req.scenario)
    single_res = None
    mc_res = None
    
    if req.monte_carlo_n:
        results = _run_batch(scenario, req.monte_carlo_n, req.seed)
        mc_res = _analyze_mc(results, scenario)
    else:
        single_res = _run_sim_core(scenario, req.seed)
    
    return _gen_report(scenario, mc_res, single_res)


@app.get("/api/health", summary="健康检查")
async def health():
    return {"status": "ok", "port": 8005}


if __name__ == "__main__":
    import uvicorn
    print("🚀 正在启动急诊分诊仿真服务...")
    print(f"📋 服务地址: http://localhost:8005")
    print(f"📚 API 文档: http://localhost:8005/docs")
    uvicorn.run(app, host="0.0.0.0", port=8005)
