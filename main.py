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


class TimeSegment(BaseModel):
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    arrival_rate: float = Field(gt=0)


class ScenarioConfig(BaseModel):
    name: str
    num_doctors: int = Field(ge=1)
    resuscitation_room_capacity: int = Field(ge=1)
    simulation_duration: float = Field(gt=0)
    arrival_rate: float = Field(gt=0)
    esi_distribution: Dict[ESILevel, float]
    service_time_params: Dict[ESILevel, DistributionParams]
    target_wait_time: Optional[Dict[ESILevel, float]] = None
    warmup_duration: float = Field(0.0, ge=0)
    cooldown_duration: float = Field(0.0, ge=0)
    escalation_thresholds: Optional[Dict[ESILevel, float]] = None
    arrival_rate_schedule: Optional[List[TimeSegment]] = None
    patience_limit: Optional[Dict[ESILevel, float]] = None


class CapacitySearchTarget(BaseModel):
    esi_level: ESILevel
    metric: str
    operator: str
    threshold: float


class CapacitySearchRequest(BaseModel):
    name: str
    max_doctors: int = Field(ge=1)
    max_resuscitation_beds: int = Field(ge=1)
    simulation_duration: float = Field(gt=0)
    arrival_rate: float = Field(gt=0)
    esi_distribution: Dict[ESILevel, float]
    service_time_params: Dict[ESILevel, DistributionParams]
    target_wait_time: Optional[Dict[ESILevel, float]] = None
    targets: List[CapacitySearchTarget]
    warmup_duration: float = Field(0.0, ge=0)
    cooldown_duration: float = Field(0.0, ge=0)
    escalation_thresholds: Optional[Dict[ESILevel, float]] = None
    arrival_rate_schedule: Optional[List[TimeSegment]] = None
    patience_limit: Optional[Dict[ESILevel, float]] = None
    n_mc: int = Field(200, ge=10, le=1000)
    base_seed: int = 42
    doctor_cost_per_hour: float = Field(100.0, gt=0)


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


def _get_arrival_rate(scenario: Dict, t: float) -> float:
    schedule = scenario.get("arrival_rate_schedule")
    if schedule:
        for seg in schedule:
            if seg["start_time"] <= t < seg["end_time"]:
                return seg["arrival_rate"]
    return scenario["arrival_rate"]


def _run_sim_core(scenario: Dict, seed: int) -> Dict:
    rng = random.Random(seed)
    
    num_docs = scenario["num_doctors"]
    resus_cap = scenario["resuscitation_room_capacity"]
    sim_dur = scenario["simulation_duration"]
    service_params = scenario["service_time_params"]
    warmup = scenario.get("warmup_duration", 0.0)
    cooldown = scenario.get("cooldown_duration", 0.0)
    steady_start = warmup
    steady_end = sim_dur - cooldown
    esc_thresh = scenario.get("escalation_thresholds")
    patience = scenario.get("patience_limit")
    
    cum_probs = []
    cum = 0.0
    for lvl, p in scenario["esi_distribution"].items():
        cum += p
        cum_probs.append((lvl, cum))
    
    eq: List[Tuple[float, int, int, int, Dict]] = []
    pid = 0
    eid = 0
    
    patients: Dict[int, Dict] = {}
    wait_q_set: Dict[int, Tuple[int, float, int]] = {}
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
    
    total_escalations = 0
    escalation_in: Dict[str, int] = {e.value: 0 for e in ESILevel}
    escalation_out: Dict[str, int] = {e.value: 0 for e in ESILevel}
    escalated_patients_wait: List[float] = []
    non_escalated_wait: Dict[str, List[float]] = {e.value: [] for e in ESILevel}
    
    lwbs_count = 0
    lwbs_by_esi: Dict[str, int] = {e.value: 0 for e in ESILevel}
    total_patients_by_esi: Dict[str, int] = {e.value: 0 for e in ESILevel}
    
    resus_occ_steady = 0
    resus_int_steady = 0.0
    doc_total_steady: List[float] = [0.0] * num_docs
    last_t_steady = steady_start
    steady_ct = 0.0
    
    def _push_wait(pat_id: int):
        pat = patients[pat_id]
        entry = (ESI_PRIORITY[pat["esi_level"]], pat["arrival_time"], pat_id)
        wait_q_set[pat_id] = entry
        heapq.heappush(wait_q, entry)
    
    def _pop_wait() -> Optional[int]:
        while wait_q:
            prio, at, pat_id = heapq.heappop(wait_q)
            if pat_id in wait_q_set and wait_q_set[pat_id] == (prio, at, pat_id):
                del wait_q_set[pat_id]
                return pat_id
        return None
    
    def _requeue_patient(pat_id: int, new_esi: ESILevel):
        nonlocal total_escalations
        pat = patients[pat_id]
        old_esi = pat["esi_level"]
        pat["esi_level"] = new_esi
        pat["was_escalated"] = True
        pat["escalation_count"] = pat.get("escalation_count", 0) + 1
        total_escalations += 1
        escalation_out[old_esi.value] += 1
        escalation_in[new_esi.value] += 1
        if pat_id in wait_q_set:
            del wait_q_set[pat_id]
        _push_wait(pat_id)
    
    def _schedule_next_events(pat_id: int, ct: float):
        nonlocal eid
        pat = patients[pat_id]
        current_esi = pat["esi_level"]
        
        if esc_thresh:
            thresh = esc_thresh.get(current_esi)
            if thresh is not None and ESI_PRIORITY[current_esi] > 1:
                esc_time = ct + thresh
                if esc_time <= sim_dur:
                    next_prio = ESI_PRIORITY[current_esi] - 1
                    for lvl, p in ESI_PRIORITY.items():
                        if p == next_prio:
                            next_esi = lvl
                            break
                    heapq.heappush(eq, (esc_time, eid, 1, 0, {
                        "type": "escalate", "pid": pat_id, "to_esi": next_esi
                    }))
                    eid += 1
        
        if patience:
            pat_limit = patience.get(current_esi)
            if pat_limit is not None:
                leave_time = ct + pat_limit
                if leave_time <= sim_dur:
                    heapq.heappush(eq, (leave_time, eid, 1, 0, {
                        "type": "leave", "pid": pat_id
                    }))
                    eid += 1
    
    def _cancel_patient_events(pat_id: int):
        pass
    
    initial_rate = _get_arrival_rate(scenario, 0.0)
    first_arr = rng.expovariate(initial_rate)
    if first_arr <= sim_dur:
        heapq.heappush(eq, (first_arr, eid, 0, 0, {"type": "arr"}))
        eid += 1
    
    def try_start(ct: float):
        nonlocal resus_occ, resus_peak, eid, resus_occ_steady
        while True:
            pat_id = _pop_wait()
            if pat_id is None:
                break
            pat = patients[pat_id]
            if pat.get("left_without_seen"):
                continue
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
                if steady_start <= ct <= steady_end:
                    doc_total_steady[doc_preempt] += elapsed
                doc_busy[doc_preempt] = None
                resus_occ -= 1
                if steady_start <= ct <= steady_end:
                    resus_occ_steady -= 1
                
                _push_wait(op_id)
                free_doc = doc_preempt
                preemptions += 1
            
            if free_doc >= 0 and has_bed:
                if pat["remaining"] is not None:
                    st = pat["remaining"]
                    pat["remaining"] = None
                else:
                    sp = service_params[pat["esi_level"]]
                    st = _sample_distribution(sp, rng)
                    pat["total_service_time"] = st
                
                pat["service_start"] = ct
                pat["wait_time"] = ct - pat["arrival_time"]
                
                if pat.get("was_escalated"):
                    escalated_patients_wait.append(pat["wait_time"])
                else:
                    non_escalated_wait[pat["esi_level"].value].append(pat["wait_time"])
                
                doc_busy[free_doc] = pat_id
                doc_busy_start[free_doc] = ct
                end_t = ct + st
                doc_busy_end[free_doc] = end_t
                
                heapq.heappush(eq, (end_t, eid, 2, free_doc, {
                    "type": "end", "doc": free_doc, "pid": pat_id
                }))
                eid += 1
                
                resus_occ += 1
                if steady_start <= ct <= steady_end:
                    resus_occ_steady += 1
                if resus_occ > resus_peak:
                    resus_peak = resus_occ
            else:
                _push_wait(pat_id)
                break
    
    ct = 0.0
    while eq:
        ev_t, _, _, _, ev = heapq.heappop(eq)
        ct = ev_t
        
        if ct > sim_dur and ev["type"] in ("arr", "escalate", "leave"):
            continue
        
        dt = ct - last_t
        resus_int += resus_occ * dt
        if steady_start <= last_t and ct <= steady_end:
            resus_int_steady += resus_occ_steady * dt
            steady_ct += dt
        elif last_t < steady_start and ct > steady_start:
            resus_int_steady += resus_occ_steady * (ct - steady_start)
            steady_ct += (ct - steady_start)
        elif last_t < steady_end and ct > steady_end:
            resus_int_steady += resus_occ_steady * (steady_end - last_t)
            steady_ct += (steady_end - last_t)
        last_t = ct
        
        if ev["type"] == "arr":
            pid += 1
            esi = _sample_esi(cum_probs, rng)
            pat = {
                "patient_id": pid,
                "esi_level": esi,
                "original_esi": esi,
                "arrival_time": ct,
                "service_start": None,
                "service_end": None,
                "wait_time": None,
                "total_service_time": None,
                "preempted": False,
                "preempt_count": 0,
                "remaining": None,
                "completed": False,
                "was_escalated": False,
                "escalation_count": 0,
                "left_without_seen": False,
            }
            patients[pid] = pat
            total_patients_by_esi[esi.value] += 1
            _push_wait(pid)
            _schedule_next_events(pid, ct)
            
            current_rate = _get_arrival_rate(scenario, ct)
            next_arr = ct + rng.expovariate(max(current_rate, 0.0001))
            if next_arr <= sim_dur:
                heapq.heappush(eq, (next_arr, eid, 0, 0, {"type": "arr"}))
                eid += 1
            
            try_start(ct)
        
        elif ev["type"] == "escalate":
            pat_id = ev["pid"]
            pat = patients.get(pat_id)
            if pat and not pat["completed"] and not pat.get("left_without_seen") and pat_id in wait_q_set:
                current_prio = ESI_PRIORITY[pat["esi_level"]]
                if current_prio > 1:
                    for lvl, p in ESI_PRIORITY.items():
                        if p == current_prio - 1:
                            _requeue_patient(pat_id, lvl)
                            _schedule_next_events(pat_id, ct)
                            break
            try_start(ct)
        
        elif ev["type"] == "leave":
            pat_id = ev["pid"]
            pat = patients.get(pat_id)
            if pat and not pat["completed"] and not pat.get("left_without_seen"):
                pat["left_without_seen"] = True
                pat["left_time"] = ct
                esi_key = pat["original_esi"].value
                lwbs_count += 1
                lwbs_by_esi[esi_key] += 1
                if pat_id in wait_q_set:
                    del wait_q_set[pat_id]
                else:
                    for d in range(num_docs):
                        if doc_busy[d] == pat_id:
                            elapsed = ct - doc_busy_start[d]
                            doc_total[d] += elapsed
                            if steady_start <= ct <= steady_end:
                                doc_total_steady[d] += elapsed
                            doc_busy[d] = None
                            resus_occ -= 1
                            if steady_start <= ct <= steady_end:
                                resus_occ_steady -= 1
                            break
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
                if steady_start <= ct <= steady_end:
                    doc_total_steady[d] += elapsed
                doc_busy[d] = None
                resus_occ -= 1
                if steady_start <= ct <= steady_end:
                    resus_occ_steady -= 1
                
                try_start(ct)
    
    for d in range(num_docs):
        if doc_busy[d] is not None:
            pat_id = doc_busy[d]
            pat = patients[pat_id]
            elapsed = ct - doc_busy_start[d]
            doc_total[d] += elapsed
            if steady_start <= ct <= steady_end:
                doc_total_steady[d] += elapsed
            pat["service_start"] = None
    
    total_busy = sum(doc_total)
    util = total_busy / (num_docs * max(ct, 0.001)) if ct > 0 else 0.0
    
    total_busy_steady = sum(doc_total_steady)
    util_steady = total_busy_steady / (num_docs * max(steady_ct, 0.001)) if steady_ct > 0 else util
    
    completed = [p for p in patients.values() if p["completed"]]
    
    steady_patients = []
    for p in patients.values():
        if p["completed"] and p["service_start"] is not None:
            if steady_start <= p["service_start"] <= steady_end:
                steady_patients.append(p)
    
    waits_esi: Dict[str, List[float]] = {e.value: [] for e in ESILevel}
    for p in steady_patients:
        esi = p["esi_level"]
        wt = p["wait_time"]
        if wt is not None:
            waits_esi[esi.value].append(wt)
    
    avg_wait = {}
    max_wait = {}
    for esi in ESILevel:
        w = waits_esi.get(esi.value, [])
        avg_wait[esi] = sum(w) / len(w) if w else 0.0
        max_wait[esi] = max(w) if w else 0.0
    
    avg_resus = resus_int / max(ct, 0.001) if ct > 0 else 0.0
    avg_resus_steady = resus_int_steady / max(steady_ct, 0.001) if steady_ct > 0 else avg_resus
    
    lwbs_rate = {}
    for esi in ESILevel:
        total = total_patients_by_esi.get(esi.value, 0)
        lwbs = lwbs_by_esi.get(esi.value, 0)
        lwbs_rate[esi] = lwbs / total if total > 0 else 0.0
    
    escalated_avg = sum(escalated_patients_wait) / len(escalated_patients_wait) if escalated_patients_wait else 0.0
    non_escalated_avg = {}
    for esi in ESILevel:
        w = non_escalated_wait.get(esi.value, [])
        non_escalated_avg[esi] = sum(w) / len(w) if w else 0.0
    
    time_slices = []
    if scenario.get("arrival_rate_schedule"):
        for seg in scenario["arrival_rate_schedule"]:
            seg_start = seg["start_time"]
            seg_end = seg["end_time"]
            seg_waits = []
            seg_util = 0.0
            seg_resus = 0.0
            seg_count = 0
            for p in steady_patients:
                if seg_start <= p["service_start"] < seg_end:
                    if p["wait_time"] is not None:
                        seg_waits.append(p["wait_time"])
                        seg_count += 1
            seg_waits_sorted = sorted(seg_waits)
            time_slices.append({
                "start_time": seg_start,
                "end_time": seg_end,
                "arrival_rate": seg["arrival_rate"],
                "sample_size": seg_count,
                "wait_p50": _quantile(seg_waits_sorted, 0.5),
                "wait_p95": _quantile(seg_waits_sorted, 0.95),
            })
    
    patient_list = []
    for p in patients.values():
        patient_list.append({
            "patient_id": p["patient_id"],
            "esi_level": p["esi_level"],
            "original_esi_level": p["original_esi"],
            "arrival_time": p["arrival_time"],
            "service_start_time": p["service_start"],
            "service_end_time": p["service_end"],
            "wait_time": p["wait_time"],
            "total_service_time": p["total_service_time"],
            "was_preempted": p["preempted"],
            "preemption_count": p["preempt_count"],
            "remaining_service_time": p["remaining"],
            "is_completed": p["completed"],
            "was_escalated": p["was_escalated"],
            "escalation_count": p["escalation_count"],
            "left_without_seen": p["left_without_seen"],
            "left_time": p.get("left_time"),
        })
    
    return {
        "patients": patient_list,
        "doctor_utilization": util,
        "doctor_utilization_steady": util_steady,
        "avg_wait_time_by_esi": avg_wait,
        "max_wait_time_by_esi": max_wait,
        "resuscitation_peak_occupancy": resus_peak,
        "resuscitation_avg_occupancy": avg_resus,
        "resuscitation_avg_occupancy_steady": avg_resus_steady,
        "total_arrivals": len(patients),
        "total_completed": len(completed),
        "total_preemptions": preemptions,
        "steady_state": {
            "start_time": steady_start,
            "end_time": steady_end,
            "sample_size": len(steady_patients),
            "duration": steady_ct,
        },
        "escalation": {
            "total_escalations": total_escalations,
            "escalation_in": escalation_in,
            "escalation_out": escalation_out,
            "escalated_avg_wait": escalated_avg,
            "non_escalated_avg_wait": non_escalated_avg,
        },
        "lwbs": {
            "total_lwbs": lwbs_count,
            "lwbs_by_esi": lwbs_by_esi,
            "lwbs_rate_by_esi": lwbs_rate,
        },
        "time_slices": time_slices,
        "total_patients_by_esi": total_patients_by_esi,
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
    utils_steady: List[float] = []
    peaks: List[float] = []
    arrivals: List[float] = []
    
    target_wait = scenario.get("target_wait_time", {})
    exceed: Dict[ESILevel, int] = {e: 0 for e in ESILevel}
    total_esi: Dict[ESILevel, int] = {e: 0 for e in ESILevel}
    
    total_escalations = 0
    escalation_in: Dict[str, float] = {e.value: 0.0 for e in ESILevel}
    escalation_out: Dict[str, float] = {e.value: 0.0 for e in ESILevel}
    escalated_waits: List[float] = []
    non_escalated_waits: Dict[str, List[float]] = {e.value: [] for e in ESILevel}
    
    total_lwbs = 0
    lwbs_by_esi: Dict[str, int] = {e.value: 0 for e in ESILevel}
    total_patients_esi: Dict[str, int] = {e.value: 0 for e in ESILevel}
    
    time_slice_stats: Dict[Tuple[float, float], Dict] = {}
    if scenario.get("arrival_rate_schedule"):
        for seg in scenario["arrival_rate_schedule"]:
            key = (seg["start_time"], seg["end_time"])
            time_slice_stats[key] = {
                "arrival_rate": seg["arrival_rate"],
                "waits": [],
                "utils": [],
                "resus": [],
            }
    
    steady_samples = []
    steady_durations = []
    
    for r in results:
        utils.append(r["doctor_utilization"])
        utils_steady.append(r["doctor_utilization_steady"])
        peaks.append(r["resuscitation_peak_occupancy"])
        arrivals.append(r["total_arrivals"])
        
        steady = r.get("steady_state", {})
        steady_samples.append(steady.get("sample_size", 0))
        steady_durations.append(steady.get("duration", 0.0))
        
        esc = r.get("escalation", {})
        total_escalations += esc.get("total_escalations", 0)
        for k, v in esc.get("escalation_in", {}).items():
            escalation_in[k] += v
        for k, v in esc.get("escalation_out", {}).items():
            escalation_out[k] += v
        if esc.get("escalated_avg_wait", 0) > 0:
            escalated_waits.append(esc["escalated_avg_wait"])
        for k, v in esc.get("non_escalated_avg_wait", {}).items():
            if v > 0:
                non_escalated_waits[k].append(v)
        
        lwbs = r.get("lwbs", {})
        total_lwbs += lwbs.get("total_lwbs", 0)
        for k, v in lwbs.get("lwbs_by_esi", {}).items():
            lwbs_by_esi[k] += v
        for k, v in r.get("total_patients_by_esi", {}).items():
            total_patients_esi[k] += v
        
        for ts in r.get("time_slices", []):
            key = (ts["start_time"], ts["end_time"])
            if key in time_slice_stats:
                if ts["sample_size"] > 0:
                    time_slice_stats[key]["waits"].append(ts["wait_p50"])
                    time_slice_stats[key]["waits"].append(ts["wait_p95"])
        
        for p in r["patients"]:
            if p["is_completed"] and not p.get("left_without_seen", False) and p["wait_time"] is not None:
                ss = r.get("steady_state", {})
                steady_s = ss.get("start_time", 0)
                steady_e = ss.get("end_time", float('inf'))
                svc_start = p.get("service_start_time")
                if svc_start is not None and steady_s <= svc_start <= steady_e:
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
    
    util_ci = _ci(utils_steady)
    t_prob: Dict[ESILevel, float] = {}
    for esi in ESILevel:
        t_prob[esi] = exceed[esi] / total_esi[esi] if total_esi[esi] > 0 else 0.0
    
    lwbs_rate = {}
    for esi in ESILevel:
        total = total_patients_esi.get(esi.value, 0)
        lwbs = lwbs_by_esi.get(esi.value, 0)
        lwbs_rate[esi] = lwbs / total if total > 0 else 0.0
    
    time_slices_result = []
    max_congestion_p95 = -1
    max_congestion_idx = 0
    for i, seg in enumerate(scenario.get("arrival_rate_schedule", [])):
        key = (seg["start_time"], seg["end_time"])
        stats = time_slice_stats.get(key, {"waits": []})
        waits_sorted = sorted(stats["waits"])
        p50 = _quantile(waits_sorted, 0.5) if waits_sorted else 0.0
        p95 = _quantile(waits_sorted, 0.95) if waits_sorted else 0.0
        if p95 > max_congestion_p95:
            max_congestion_p95 = p95
            max_congestion_idx = len(time_slices_result)
        time_slices_result.append({
            "start_time": seg["start_time"],
            "end_time": seg["end_time"],
            "arrival_rate": seg["arrival_rate"],
            "wait_p50": p50,
            "wait_p95": p95,
        })
    
    for ts in time_slices_result:
        ts["is_most_congested"] = (ts["start_time"] == time_slices_result[max_congestion_idx]["start_time"] 
                                  and ts["end_time"] == time_slices_result[max_congestion_idx]["end_time"])
    
    escalated_avg = _mean(escalated_waits) if escalated_waits else 0.0
    non_esc_avg = {}
    for k, v in non_escalated_waits.items():
        non_esc_avg[k] = _mean(v) if v else 0.0
    
    return {
        "num_simulations": n,
        "wait_time_quantiles": q_res,
        "doctor_utilization_ci": util_ci,
        "target_wait_probability": t_prob,
        "avg_doctor_utilization": util_ci["mean"],
        "avg_resuscitation_peak": _mean(peaks),
        "avg_total_arrivals": _mean(arrivals),
        "steady_state": {
            "start_time": scenario.get("warmup_duration", 0.0),
            "end_time": scenario.get("simulation_duration", 0) - scenario.get("cooldown_duration", 0.0),
            "avg_sample_size": _mean(steady_samples),
            "avg_duration": _mean(steady_durations),
        },
        "escalation": {
            "total_escalations": total_escalations,
            "avg_escalations_per_run": total_escalations / n if n > 0 else 0,
            "escalation_in": {k: v / n if n > 0 else 0 for k, v in escalation_in.items()},
            "escalation_out": {k: v / n if n > 0 else 0 for k, v in escalation_out.items()},
            "avg_escalated_wait": escalated_avg,
            "avg_non_escalated_wait": non_esc_avg,
        },
        "lwbs": {
            "total_lwbs": total_lwbs,
            "avg_lwbs_per_run": total_lwbs / n if n > 0 else 0,
            "lwbs_by_esi": {k: v for k, v in lwbs_by_esi.items()},
            "lwbs_rate_by_esi": lwbs_rate,
        },
        "time_slices": time_slices_result,
        "most_congested_period": time_slices_result[max_congestion_idx] if time_slices_result else None,
    }


def _config_to_dict(cfg: ScenarioConfig) -> Dict:
    schedule = None
    if cfg.arrival_rate_schedule:
        schedule = [s.model_dump() for s in cfg.arrival_rate_schedule]
    return {
        "name": cfg.name,
        "num_doctors": cfg.num_doctors,
        "resuscitation_room_capacity": cfg.resuscitation_room_capacity,
        "simulation_duration": cfg.simulation_duration,
        "arrival_rate": cfg.arrival_rate,
        "esi_distribution": cfg.esi_distribution,
        "service_time_params": {k: v.model_dump() for k, v in cfg.service_time_params.items()},
        "target_wait_time": cfg.target_wait_time,
        "warmup_duration": cfg.warmup_duration,
        "cooldown_duration": cfg.cooldown_duration,
        "escalation_thresholds": cfg.escalation_thresholds,
        "arrival_rate_schedule": schedule,
        "patience_limit": cfg.patience_limit,
    }


def _capacity_search_config_to_dict(cfg: CapacitySearchRequest) -> Dict:
    schedule = None
    if cfg.arrival_rate_schedule:
        schedule = [s.model_dump() for s in cfg.arrival_rate_schedule]
    return {
        "name": cfg.name,
        "simulation_duration": cfg.simulation_duration,
        "arrival_rate": cfg.arrival_rate,
        "esi_distribution": cfg.esi_distribution,
        "service_time_params": {k: v.model_dump() for k, v in cfg.service_time_params.items()},
        "target_wait_time": cfg.target_wait_time,
        "warmup_duration": cfg.warmup_duration,
        "cooldown_duration": cfg.cooldown_duration,
        "escalation_thresholds": cfg.escalation_thresholds,
        "arrival_rate_schedule": schedule,
        "patience_limit": cfg.patience_limit,
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
    def_esc = {
        ESILevel.ESI_5: 30.0, ESILevel.ESI_4: 25.0, ESILevel.ESI_3: 20.0,
        ESILevel.ESI_2: 15.0, ESILevel.ESI_1: 9999.0,
    }
    def_patience = {
        ESILevel.ESI_5: 60.0, ESILevel.ESI_4: 90.0, ESILevel.ESI_3: 120.0,
        ESILevel.ESI_2: 180.0, ESILevel.ESI_1: 9999.0,
    }
    return [
        {
            "name": "标准场景 - 日间常规",
            "num_doctors": 5, "resuscitation_room_capacity": 8,
            "simulation_duration": 480.0, "arrival_rate": 0.5,
            "esi_distribution": {ESILevel.ESI_1:0.03,ESILevel.ESI_2:0.08,ESILevel.ESI_3:0.25,ESILevel.ESI_4:0.35,ESILevel.ESI_5:0.29},
            "service_time_params": def_svc, "target_wait_time": def_target,
            "warmup_duration": 30.0, "cooldown_duration": 30.0,
            "escalation_thresholds": def_esc, "patience_limit": def_patience,
            "arrival_rate_schedule": None,
        },
        {
            "name": "高峰场景 - 晚高峰",
            "num_doctors": 6, "resuscitation_room_capacity": 10,
            "simulation_duration": 480.0, "arrival_rate": 0.8,
            "esi_distribution": {ESILevel.ESI_1:0.04,ESILevel.ESI_2:0.12,ESILevel.ESI_3:0.28,ESILevel.ESI_4:0.32,ESILevel.ESI_5:0.24},
            "service_time_params": def_svc, "target_wait_time": def_target,
            "warmup_duration": 30.0, "cooldown_duration": 30.0,
            "escalation_thresholds": def_esc, "patience_limit": def_patience,
            "arrival_rate_schedule": [
                {"start_time": 0.0, "end_time": 60.0, "arrival_rate": 0.4},
                {"start_time": 60.0, "end_time": 180.0, "arrival_rate": 0.8},
                {"start_time": 180.0, "end_time": 300.0, "arrival_rate": 1.2},
                {"start_time": 300.0, "end_time": 420.0, "arrival_rate": 0.8},
                {"start_time": 420.0, "end_time": 480.0, "arrival_rate": 0.4},
            ],
        },
        {
            "name": "夜间场景 - 低负荷",
            "num_doctors": 3, "resuscitation_room_capacity": 6,
            "simulation_duration": 480.0, "arrival_rate": 0.2,
            "esi_distribution": {ESILevel.ESI_1:0.02,ESILevel.ESI_2:0.05,ESILevel.ESI_3:0.20,ESILevel.ESI_4:0.38,ESILevel.ESI_5:0.35},
            "service_time_params": def_svc, "target_wait_time": def_target,
            "warmup_duration": 30.0, "cooldown_duration": 30.0,
            "escalation_thresholds": def_esc, "patience_limit": def_patience,
            "arrival_rate_schedule": None,
        },
    ]


def _run_batch(scenario: Dict, n: int, base_seed: int) -> List[Dict]:
    results = []
    for i in range(n):
        results.append(_run_sim_core(scenario, base_seed + i))
    return results


def _check_targets(mc_res: Dict, targets: List[Dict]) -> Tuple[bool, List[str]]:
    all_met = True
    failures = []
    
    for t in targets:
        esi = t["esi_level"]
        metric = t["metric"]
        op = t["operator"]
        threshold = t["threshold"]
        
        actual = None
        metric_name = ""
        
        if metric == "wait_p95":
            q = mc_res.get("wait_time_quantiles", {}).get(esi, {})
            actual = q.get("P95", 0)
            metric_name = f"{esi.value} P95等待"
        elif metric == "wait_p50":
            q = mc_res.get("wait_time_quantiles", {}).get(esi, {})
            actual = q.get("P50", 0)
            metric_name = f"{esi.value} P50等待"
        elif metric == "doctor_utilization":
            actual = mc_res.get("avg_doctor_utilization", 0)
            metric_name = "医生利用率"
        elif metric == "target_exceed_prob":
            actual = mc_res.get("target_wait_probability", {}).get(esi, 0)
            metric_name = f"{esi.value} 超目标概率"
        elif metric == "lwbs_rate":
            actual = mc_res.get("lwbs", {}).get("lwbs_rate_by_esi", {}).get(esi, 0)
            metric_name = f"{esi.value} 离院率"
        
        if actual is None:
            all_met = False
            failures.append(f"{metric_name}: 无法获取指标值")
            continue
        
        met = False
        if op == "<":
            met = actual < threshold
        elif op == "<=":
            met = actual <= threshold
        elif op == ">":
            met = actual > threshold
        elif op == ">=":
            met = actual >= threshold
        elif op == "==":
            met = actual == threshold
        
        if not met:
            all_met = False
            failures.append(f"{metric_name}: {actual:.3f} {op} {threshold} 不满足")
    
    return all_met, failures


def _search_capacity_configs(req: Dict) -> Dict:
    base_cfg = _capacity_search_config_to_dict(CapacitySearchRequest(**req))
    max_docs = req["max_doctors"]
    max_beds = req["max_resuscitation_beds"]
    targets = [t.model_dump() if hasattr(t, 'model_dump') else t for t in req["targets"]]
    n_mc = req["n_mc"]
    base_seed = req["base_seed"]
    doc_cost = req["doctor_cost_per_hour"]
    
    candidates = []
    
    for nd in range(1, max_docs + 1):
        for nb in range(1, max_beds + 1):
            scenario = dict(base_cfg)
            scenario["num_doctors"] = nd
            scenario["resuscitation_room_capacity"] = nb
            scenario["name"] = f"{base_cfg['name']}_D{nd}_B{nb}"
            
            results = _run_batch(scenario, n_mc, base_seed)
            mc_res = _analyze_mc(results, scenario)
            
            all_met, failures = _check_targets(mc_res, targets)
            
            cost = nd * doc_cost
            
            candidate = {
                "num_doctors": nd,
                "resuscitation_room_capacity": nb,
                "total_cost_per_hour": cost,
                "meets_all_targets": all_met,
                "failures": failures,
                "key_metrics": {
                    "avg_doctor_utilization": mc_res["avg_doctor_utilization"],
                    "avg_resuscitation_peak": mc_res["avg_resuscitation_peak"],
                    "wait_time_quantiles": mc_res["wait_time_quantiles"],
                    "target_wait_probability": mc_res["target_wait_probability"],
                    "lwbs": mc_res.get("lwbs", {}),
                },
                "verification_n": n_mc,
            }
            
            if all_met:
                candidates.append(candidate)
    
    candidates.sort(key=lambda c: c["total_cost_per_hour"])
    top_3 = candidates[:3]
    
    return {
        "search_name": base_cfg["name"],
        "search_space": {
            "max_doctors": max_docs,
            "max_resuscitation_beds": max_beds,
            "total_configs_checked": max_docs * max_beds,
        },
        "targets": targets,
        "feasible_count": len(candidates),
        "top_candidates": top_3,
    }


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
    lines.append(f"| 热身期时长 | {scenario.get('warmup_duration', 0):.0f} 分钟 |")
    lines.append(f"| 冷却期时长 | {scenario.get('cooldown_duration', 0):.0f} 分钟 |")
    lines.append(f"| 稳态区间 | [{scenario.get('warmup_duration', 0):.0f}, {scenario['simulation_duration'] - scenario.get('cooldown_duration', 0):.0f}] 分钟 |")
    lines.append(f"| 患者到达率 | {scenario['arrival_rate']:.3f} 患者/分钟 |")
    lines.append("")
    
    if scenario.get('arrival_rate_schedule'):
        lines.append("### 1.1.1 时变到达率配置")
        lines.append("")
        lines.append("| 时段(分钟) | 到达率(患者/分钟) |")
        lines.append("|------------|------------------|")
        for seg in scenario['arrival_rate_schedule']:
            lines.append(f"| [{seg['start_time']:.0f}, {seg['end_time']:.0f}) | {seg['arrival_rate']:.3f} |")
        lines.append("")
    
    if scenario.get('escalation_thresholds'):
        lines.append("### 1.1.2 病情升级阈值")
        lines.append("")
        lines.append("| 级别 | 升级阈值(分钟) |")
        lines.append("|------|---------------|")
        for esi in ESILevel:
            t = scenario['escalation_thresholds'].get(esi)
            if t is not None and t < 9999:
                lines.append(f"| {esi.value} | {t:.0f} |")
        lines.append("")
    
    if scenario.get('patience_limit'):
        lines.append("### 1.1.3 离院耐心上限")
        lines.append("")
        lines.append("| 级别 | 耐心上限(分钟) |")
        lines.append("|------|---------------|")
        for esi in ESILevel:
            t = scenario['patience_limit'].get(esi)
            if t is not None and t < 9999:
                lines.append(f"| {esi.value} | {t:.0f} |")
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
        ss = single_res.get('steady_state', {})
        lines.append(f"### 2.1 稳态区间信息")
        lines.append(f"- 稳态区间: [{ss.get('start_time', 0):.1f}, {ss.get('end_time', 0):.1f}] 分钟")
        lines.append(f"- 稳态样本量: {ss.get('sample_size', 0)} 名患者")
        lines.append(f"- 稳态持续时长: {ss.get('duration', 0):.1f} 分钟")
        lines.append("")
        lines.append(f"### 2.2 总体指标")
        lines.append(f"- 总到达患者数: {single_res['total_arrivals']}")
        lines.append(f"- 完成服务患者数: {single_res['total_completed']}")
        lines.append(f"- 医生平均利用率(全时段): {single_res['doctor_utilization'] * 100:.1f}%")
        lines.append(f"- 医生平均利用率(稳态): {single_res['doctor_utilization_steady'] * 100:.1f}%")
        lines.append(f"- 抢救室峰值占用: {single_res['resuscitation_peak_occupancy']}/{scenario['resuscitation_room_capacity']}")
        lines.append(f"- 抢救室平均占用(全时段): {single_res['resuscitation_avg_occupancy']:.1f} 床")
        lines.append(f"- 抢救室平均占用(稳态): {single_res['resuscitation_avg_occupancy_steady']:.1f} 床")
        lines.append(f"- 总抢占次数: {single_res['total_preemptions']}")
        lines.append("")
        
        esc = single_res.get('escalation', {})
        if esc.get('total_escalations', 0) > 0:
            lines.append(f"### 2.3 病情升级统计")
            lines.append(f"- 队列内升级总次数: {esc['total_escalations']}")
            lines.append("")
            lines.append("| 级别 | 升级入(次) | 升级出(次) |")
            lines.append("|------|-----------|-----------|")
            for esi in ESILevel:
                e_in = esc['escalation_in'].get(esi.value, 0)
                e_out = esc['escalation_out'].get(esi.value, 0)
                lines.append(f"| {esi.value} | {e_in} | {e_out} |")
            lines.append("")
            lines.append(f"- 升级患者平均等待: {esc['escalated_avg_wait']:.1f} 分钟")
            lines.append("")
            lines.append("| 级别 | 未升级患者平均等待(分钟) |")
            lines.append("|------|-------------------------|")
            for esi in ESILevel:
                avg = esc['non_escalated_avg_wait'].get(esi, 0)
                if avg > 0:
                    lines.append(f"| {esi.value} | {avg:.1f} |")
            lines.append("")
        
        lwbs = single_res.get('lwbs', {})
        if lwbs.get('total_lwbs', 0) > 0:
            lines.append(f"### 2.4 LWBS (未就诊离院) 统计")
            lines.append(f"- 总LWBS数: {lwbs['total_lwbs']}")
            lines.append("")
            lines.append("| 级别 | LWBS数 | 放弃率 |")
            lines.append("|------|--------|--------|")
            for esi in ESILevel:
                cnt = lwbs['lwbs_by_esi'].get(esi.value, 0)
                rate = lwbs['lwbs_rate_by_esi'].get(esi, 0)
                lines.append(f"| {esi.value} | {cnt} | {rate*100:.1f}% |")
            lines.append("")
        
        if single_res.get('time_slices'):
            lines.append(f"### 2.5 时段切片统计")
            lines.append("")
            lines.append("| 时段(分钟) | 到达率 | 样本量 | P50等待 | P95等待 |")
            lines.append("|------------|--------|--------|---------|---------|")
            for ts in single_res['time_slices']:
                lines.append(f"| [{ts['start_time']:.0f}, {ts['end_time']:.0f}) | {ts['arrival_rate']:.2f} | {ts['sample_size']} | {ts['wait_p50']:.1f} | {ts['wait_p95']:.1f} |")
            lines.append("")
    
    if mc_res:
        lines.append(f"## 3. 蒙特卡洛仿真结果 (N={mc_res['num_simulations']})")
        lines.append("")
        
        ss_mc = mc_res.get('steady_state', {})
        lines.append(f"### 3.1 稳态区间信息")
        lines.append(f"- 稳态区间: [{ss_mc.get('start_time', 0):.1f}, {ss_mc.get('end_time', 0):.1f}] 分钟")
        lines.append(f"- 平均稳态样本量: {ss_mc.get('avg_sample_size', 0):.1f} 名患者")
        lines.append(f"- 平均稳态持续时长: {ss_mc.get('avg_duration', 0):.1f} 分钟")
        lines.append("")
        
        lines.append("### 3.2 等待时长分位数 (稳态区间)")
        lines.append("")
        lines.append("| 级别 | P50(分钟) | P75(分钟) | P90(分钟) | P95(分钟) | P99(分钟) |")
        lines.append("|------|-----------|-----------|-----------|-----------|-----------|")
        for esi in ESILevel:
            q = mc_res['wait_time_quantiles'].get(esi, {})
            lines.append(f"| {esi.value} | {q.get('P50',0):.1f} | {q.get('P75',0):.1f} | {q.get('P90',0):.1f} | {q.get('P95',0):.1f} | {q.get('P99',0):.1f} |")
        lines.append("")
        
        ci = mc_res['doctor_utilization_ci']
        lines.append(f"### 3.3 医生利用率(稳态): {ci['mean']*100:.1f}% (95% CI: [{ci['lower']*100:.1f}%, {ci['upper']*100:.1f}%])")
        lines.append("")
        
        if tw:
            lines.append("### 3.4 超目标等待时长概率 (稳态区间)")
            lines.append("")
            lines.append("| 级别 | 目标(分钟) | 超目标概率 | 达标率 |")
            lines.append("|------|-----------|-----------|--------|")
            for esi in ESILevel:
                t = tw.get(esi)
                if t is not None:
                    p = mc_res['target_wait_probability'].get(esi, 0)
                    lines.append(f"| {esi.value} | {t:.0f} | {p*100:.1f}% | {(1-p)*100:.1f}% |")
            lines.append("")
        
        esc_mc = mc_res.get('escalation', {})
        if esc_mc.get('total_escalations', 0) > 0:
            lines.append(f"### 3.5 病情升级统计")
            lines.append(f"- 总升级次数: {esc_mc['total_escalations']}")
            lines.append(f"- 平均每轮升级次数: {esc_mc['avg_escalations_per_run']:.2f}")
            lines.append("")
            lines.append("| 级别 | 平均升级入(次/轮) | 平均升级出(次/轮) |")
            lines.append("|------|-----------------|-----------------|")
            for esi in ESILevel:
                e_in = esc_mc['escalation_in'].get(esi.value, 0)
                e_out = esc_mc['escalation_out'].get(esi.value, 0)
                lines.append(f"| {esi.value} | {e_in:.2f} | {e_out:.2f} |")
            lines.append("")
            lines.append(f"- 升级患者平均等待: {esc_mc['avg_escalated_wait']:.1f} 分钟")
            lines.append("")
            lines.append("| 级别 | 未升级患者平均等待(分钟) |")
            lines.append("|------|-------------------------|")
            for esi in ESILevel:
                avg = esc_mc['avg_non_escalated_wait'].get(esi.value, 0)
                if avg > 0:
                    lines.append(f"| {esi.value} | {avg:.1f} |")
            lines.append("")
        
        lwbs_mc = mc_res.get('lwbs', {})
        if lwbs_mc.get('total_lwbs', 0) > 0:
            lines.append(f"### 3.6 LWBS (未就诊离院) 统计")
            lines.append(f"- 总LWBS数: {lwbs_mc['total_lwbs']}")
            lines.append(f"- 平均每轮LWBS数: {lwbs_mc['avg_lwbs_per_run']:.2f}")
            lines.append("")
            lines.append("| 级别 | LWBS数 | 放弃率 |")
            lines.append("|------|--------|--------|")
            for esi in ESILevel:
                cnt = lwbs_mc['lwbs_by_esi'].get(esi.value, 0)
                rate = lwbs_mc['lwbs_rate_by_esi'].get(esi, 0)
                lines.append(f"| {esi.value} | {cnt} | {rate*100:.1f}% |")
            lines.append("")
        
        if mc_res.get('time_slices'):
            lines.append(f"### 3.7 时段切片统计")
            lines.append("")
            lines.append("| 时段(分钟) | 到达率 | P50等待 | P95等待 | 最拥堵 |")
            lines.append("|------------|--------|---------|---------|--------|")
            for ts in mc_res['time_slices']:
                marker = "🔥 是" if ts.get('is_most_congested') else ""
                lines.append(f"| [{ts['start_time']:.0f}, {ts['end_time']:.0f}) | {ts['arrival_rate']:.2f} | {ts['wait_p50']:.1f} | {ts['wait_p95']:.1f} | {marker} |")
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
        
        lwbs_mc = mc_res.get('lwbs', {})
        for esi in ESILevel:
            rate = lwbs_mc.get('lwbs_rate_by_esi', {}).get(esi, 0)
            if rate > 0.05:
                lines.append(f"- ⚠️ {esi.value} 离院率过高 ({rate*100:.1f}%)，建议增加资源或优化流程")
        
        if mc_res.get('most_congested_period'):
            mc = mc_res['most_congested_period']
            lines.append(f"- ⚠️ 最拥堵时段: [{mc['start_time']:.0f}, {mc['end_time']:.0f}) 分钟，P95等待 {mc['wait_p95']:.1f} 分钟，建议考虑该时段增派资源")
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
    
    if scenario.warmup_duration + scenario.cooldown_duration >= scenario.simulation_duration:
        raise HTTPException(400, "热身期+冷却期时长不能超过仿真总时长")
    
    if scenario.arrival_rate_schedule:
        for i, seg in enumerate(scenario.arrival_rate_schedule):
            if seg.start_time >= seg.end_time:
                raise HTTPException(400, f"时段{i}的开始时间必须小于结束时间")
            if seg.arrival_rate <= 0:
                raise HTTPException(400, f"时段{i}的到达率必须大于0")
    
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


@app.post("/api/capacity/search", summary="经济配置搜索 - 自动搜索满足目标的最优资源配置")
async def search_capacity(req: CapacitySearchRequest):
    t0 = time.time()
    
    if req.warmup_duration + req.cooldown_duration >= req.simulation_duration:
        raise HTTPException(400, "热身期+冷却期时长不能超过仿真总时长")
    
    if req.arrival_rate_schedule:
        for i, seg in enumerate(req.arrival_rate_schedule):
            if seg.start_time >= seg.end_time:
                raise HTTPException(400, f"时段{i}的开始时间必须小于结束时间")
            if seg.arrival_rate <= 0:
                raise HTTPException(400, f"时段{i}的到达率必须大于0")
    
    if not req.targets:
        raise HTTPException(400, "至少需要指定一个达标目标")
    
    valid_metrics = {"wait_p95", "wait_p50", "doctor_utilization", "target_exceed_prob", "lwbs_rate"}
    valid_ops = {"<", "<=", ">", ">=", "=="}
    for i, t in enumerate(req.targets):
        if t.metric not in valid_metrics:
            raise HTTPException(400, f"目标{i}的metric必须为: {valid_metrics}")
        if t.operator not in valid_ops:
            raise HTTPException(400, f"目标{i}的operator必须为: {valid_ops}")
    
    req_dict = req.model_dump()
    result = _search_capacity_configs(req_dict)
    result["elapsed_ms"] = int((time.time() - t0) * 1000)
    
    return result


@app.get("/api/health", summary="健康检查")
async def health():
    return {"status": "ok", "port": 8005}


if __name__ == "__main__":
    import uvicorn
    print("🚀 正在启动急诊分诊仿真服务...")
    print(f"📋 服务地址: http://localhost:8005")
    print(f"📚 API 文档: http://localhost:8005/docs")
    uvicorn.run(app, host="0.0.0.0", port=8005)
