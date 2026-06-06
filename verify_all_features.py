# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import json
import math
import heapq
import random
from typing import List, Dict, Optional, Tuple
from enum import Enum

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
    
    initial_rate = _get_arrival_rate(scenario, 0.0)
    first_arr = rng.expovariate(initial_rate)
    if first_arr <= sim_dur:
        heapq.heappush(eq, (first_arr, eid, 0, 0, {"type": "arr"}))
        eid += 1
    
    def try_start(ct: float):
        nonlocal resus_occ, resus_peak, eid, resus_occ_steady, preemptions
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

def get_test_scenario():
    return {
        "name": "功能测试场景",
        "num_doctors": 6,
        "resuscitation_room_capacity": 10,
        "simulation_duration": 480.0,
        "arrival_rate": 0.5,
        "esi_distribution": {
            ESILevel.ESI_1: 0.03, ESILevel.ESI_2: 0.08, ESILevel.ESI_3: 0.25,
            ESILevel.ESI_4: 0.35, ESILevel.ESI_5: 0.29
        },
        "service_time_params": {
            ESILevel.ESI_1: {"distribution_type": "gamma", "mean": 45.0, "std": 20.0},
            ESILevel.ESI_2: {"distribution_type": "gamma", "mean": 30.0, "std": 12.0},
            ESILevel.ESI_3: {"distribution_type": "gamma", "mean": 20.0, "std": 8.0},
            ESILevel.ESI_4: {"distribution_type": "gamma", "mean": 12.0, "std": 5.0},
            ESILevel.ESI_5: {"distribution_type": "gamma", "mean": 8.0, "std": 3.0},
        },
        "target_wait_time": {
            ESILevel.ESI_1: 1.0, ESILevel.ESI_2: 10.0, ESILevel.ESI_3: 30.0,
            ESILevel.ESI_4: 60.0, ESILevel.ESI_5: 120.0,
        },
        "warmup_duration": 30.0,
        "cooldown_duration": 30.0,
        "escalation_thresholds": {
            ESILevel.ESI_5: 30.0, ESILevel.ESI_4: 25.0, ESILevel.ESI_3: 20.0,
            ESILevel.ESI_2: 15.0, ESILevel.ESI_1: 9999.0,
        },
        "arrival_rate_schedule": [
            {"start_time": 0.0, "end_time": 60.0, "arrival_rate": 0.3},
            {"start_time": 60.0, "end_time": 240.0, "arrival_rate": 0.6},
            {"start_time": 240.0, "end_time": 420.0, "arrival_rate": 0.8},
            {"start_time": 420.0, "end_time": 480.0, "arrival_rate": 0.4},
        ],
        "patience_limit": {
            ESILevel.ESI_5: 60.0, ESILevel.ESI_4: 90.0, ESILevel.ESI_3: 120.0,
            ESILevel.ESI_2: 180.0, ESILevel.ESI_1: 9999.0,
        },
    }

def test_all_features():
    print("\n" + "=" * 80)
    print("  急诊分诊仿真 - 5大新功能验证测试")
    print("=" * 80 + "\n")
    
    scenario = get_test_scenario()
    all_pass = True
    
    # 测试1: 单次仿真
    print("[测试1] 单次仿真 - 验证所有新功能字段")
    print("-" * 60)
    t0 = time.time()
    result = _run_sim_core(scenario, seed=42)
    elapsed = time.time() - t0
    print(f"  仿真耗时: {elapsed*1000:.1f} ms")
    print(f"  总到达患者: {result['total_arrivals']}")
    print(f"  完成患者: {result['total_completed']}")
    
    # 功能1: 热身期与冷却期
    ss = result["steady_state"]
    print(f"\n  📊 功能1 - 热身期与冷却期统计:")
    print(f"    稳态区间: [{ss['start_time']}, {ss['end_time']}] 分钟")
    print(f"    稳态样本量: {ss['sample_size']} 名患者")
    print(f"    稳态持续时长: {ss['duration']:.1f} 分钟")
    print(f"    医生利用率(全时段): {result['doctor_utilization']*100:.1f}%")
    print(f"    医生利用率(稳态): {result['doctor_utilization_steady']*100:.1f}%")
    assert ss["start_time"] == 30.0, f"热身期应为30，实际{ss['start_time']}"
    assert ss["end_time"] == 450.0, f"冷却期结束应为450，实际{ss['end_time']}"
    assert ss["sample_size"] > 0, "稳态样本量不应为0"
    assert "doctor_utilization_steady" in result, "缺少稳态利用率"
    print("    ✅ 功能1验证通过")
    
    # 功能2: 等待期病情恶化升级
    esc = result["escalation"]
    print(f"\n  📊 功能2 - 等待期病情恶化升级:")
    print(f"    队列内升级总次数: {esc['total_escalations']}")
    print(f"    升级患者平均等待: {esc['escalated_avg_wait']:.1f} 分钟")
    for esi in ESILevel:
        e_in = esc['escalation_in'][esi.value]
        e_out = esc['escalation_out'][esi.value]
        if e_in > 0 or e_out > 0:
            print(f"    {esi.value}: 升级入={e_in}, 升级出={e_out}")
    assert "total_escalations" in esc, "缺少升级总次数字段"
    assert "escalation_in" in esc, "缺少升级入字段"
    assert "escalation_out" in esc, "缺少升级出字段"
    assert "escalated_avg_wait" in esc, "缺少升级患者等待时间"
    assert "non_escalated_avg_wait" in esc, "缺少未升级患者等待时间"
    print("    ✅ 功能2验证通过")
    
    # 功能3: 时变到达率
    ts = result["time_slices"]
    print(f"\n  📊 功能3 - 时变到达率:")
    print(f"    时段切片数: {len(ts)}")
    for seg in ts:
        print(f"    [{seg['start_time']:.0f}, {seg['end_time']:.0f}): 到达率={seg['arrival_rate']:.2f}, "
              f"样本量={seg['sample_size']}, P50={seg['wait_p50']:.1f}, P95={seg['wait_p95']:.1f}")
    assert len(ts) == 4, f"时段数应为4，实际{len(ts)}"
    assert all("wait_p50" in s for s in ts), "缺少P50字段"
    assert all("wait_p95" in s for s in ts), "缺少P95字段"
    print("    ✅ 功能3验证通过")
    
    # 功能4: 离院威胁
    lwbs = result["lwbs"]
    print(f"\n  📊 功能4 - 离院威胁(LWBS):")
    print(f"    LWBS总数: {lwbs['total_lwbs']}")
    for esi in ESILevel:
        cnt = lwbs['lwbs_by_esi'][esi.value]
        rate = lwbs['lwbs_rate_by_esi'][esi]
        if cnt > 0:
            print(f"    {esi.value}: {cnt}人 (放弃率: {rate*100:.1f}%)")
    assert "total_lwbs" in lwbs, "缺少LWBS总数"
    assert "lwbs_by_esi" in lwbs, "缺少按级别LWBS"
    assert "lwbs_rate_by_esi" in lwbs, "缺少按级别放弃率"
    print("    ✅ 功能4验证通过")
    
    print("\n  ✅ 单次仿真所有新功能字段验证通过!\n")
    
    # 测试2: 蒙特卡洛 N=1000 性能测试
    print("[测试2] 蒙特卡洛 N=1000 性能测试")
    print("-" * 60)
    t0 = time.time()
    results = []
    for i in range(1000):
        results.append(_run_sim_core(scenario, 42 + i))
    elapsed = time.time() - t0
    print(f"  总耗时: {elapsed:.2f} 秒")
    print(f"  单次仿真平均: {elapsed*1000/1000:.1f} 毫秒")
    
    target = 10.0
    status = "✅ 达标" if elapsed < target else "❌ 未达标"
    print(f"  性能目标: < {target} 秒 {status}")
    
    if elapsed < target:
        print("  ✅ 性能测试通过!")
    else:
        print(f"  ❌ 性能测试未通过，耗时{elapsed:.2f}秒")
        all_pass = False
    
    # 测试3: 经济配置搜索
    print("\n[测试3] 经济配置搜索 (功能5)")
    print("-" * 60)
    
    def _mean(d):
        return sum(d) / len(d) if d else 0.0
    
    def _std(d):
        if len(d) < 2:
            return 0.0
        m = _mean(d)
        return math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
    
    def _t_crit(df):
        t = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,10:2.228,20:2.086,30:2.042,100:1.984,1000:1.962}
        for k in sorted(t.keys(), reverse=True):
            if df >= k:
                return t[k]
        return 1.96
    
    def _ci(d):
        if not d:
            return {"mean":0.0,"lower":0.0,"upper":0.0,"std":0.0}
        n = len(d)
        m = _mean(d)
        s = _std(d)
        se = s / math.sqrt(n)
        t = _t_crit(n-1)
        margin = t * se
        return {"mean":m,"lower":m-margin,"upper":m+margin,"std":s}
    
    def _analyze_mc(results, scenario):
        n = len(results)
        waits_esi = {e: [] for e in ESILevel}
        utils_steady = []
        peaks = []
        
        target_wait = scenario.get("target_wait_time", {})
        exceed = {e: 0 for e in ESILevel}
        total_esi = {e: 0 for e in ESILevel}
        
        for r in results:
            utils_steady.append(r["doctor_utilization_steady"])
            peaks.append(r["resuscitation_peak_occupancy"])
            ss = r.get("steady_state", {})
            steady_s = ss.get("start_time", 0)
            steady_e = ss.get("end_time", float('inf'))
            for p in r["patients"]:
                if p["is_completed"] and not p.get("left_without_seen", False) and p["wait_time"] is not None:
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
        q_res = {}
        for esi in ESILevel:
            w = sorted(waits_esi[esi])
            q_res[esi] = {}
            for qn, qv in zip(q_levels, q_vals):
                q_res[esi][qn] = _quantile(w, qv)
        
        util_ci = _ci(utils_steady)
        t_prob = {}
        for esi in ESILevel:
            t_prob[esi] = exceed[esi] / total_esi[esi] if total_esi[esi] > 0 else 0.0
        
        return {
            "num_simulations": n,
            "wait_time_quantiles": q_res,
            "doctor_utilization_ci": util_ci,
            "target_wait_probability": t_prob,
            "avg_doctor_utilization": util_ci["mean"],
            "avg_resuscitation_peak": _mean(peaks),
        }
    
    def _check_targets(mc_res, targets):
        all_met = True
        failures = []
        for t in targets:
            esi = t["esi_level"]
            metric = t["metric"]
            op = t["operator"]
            threshold = t["threshold"]
            actual = None
            if metric == "wait_p95":
                actual = mc_res["wait_time_quantiles"][esi]["P95"]
            elif metric == "doctor_utilization":
                actual = mc_res["avg_doctor_utilization"]
            if actual is None:
                all_met = False
                continue
            met = False
            if op == "<": met = actual < threshold
            elif op == "<=": met = actual <= threshold
            elif op == ">": met = actual > threshold
            elif op == ">=": met = actual >= threshold
            if not met:
                all_met = False
                failures.append(f"{esi} {metric}: {actual:.3f} {op} {threshold}")
        return all_met, failures
    
    base_cfg = {
        "name": "容量搜索测试",
        "simulation_duration": 240.0,
        "arrival_rate": 0.5,
        "esi_distribution": scenario["esi_distribution"],
        "service_time_params": scenario["service_time_params"],
        "target_wait_time": scenario["target_wait_time"],
        "warmup_duration": 20.0,
        "cooldown_duration": 20.0,
        "escalation_thresholds": scenario["escalation_thresholds"],
        "patience_limit": scenario["patience_limit"],
    }
    
    targets = [
        {"esi_level": ESILevel.ESI_2, "metric": "wait_p95", "operator": "<", "threshold": 15.0},
        {"esi_level": ESILevel.ESI_1, "metric": "doctor_utilization", "operator": "<=", "threshold": 0.92},
    ]
    
    max_docs = 8
    max_beds = 10
    n_mc = 50
    doc_cost = 100.0
    
    t0 = time.time()
    candidates = []
    
    for nd in range(1, max_docs + 1):
        for nb in range(1, max_beds + 1):
            sc = dict(base_cfg)
            sc["num_doctors"] = nd
            sc["resuscitation_room_capacity"] = nb
            
            results = []
            for i in range(n_mc):
                results.append(_run_sim_core(sc, 42 + i))
            mc_res = _analyze_mc(results, sc)
            
            all_met, failures = _check_targets(mc_res, targets)
            cost = nd * doc_cost
            
            if all_met:
                candidates.append({
                    "num_doctors": nd,
                    "resuscitation_room_capacity": nb,
                    "total_cost_per_hour": cost,
                    "avg_doctor_utilization": mc_res["avg_doctor_utilization"],
                    "esi2_p95_wait": mc_res["wait_time_quantiles"][ESILevel.ESI_2]["P95"],
                })
    
    search_elapsed = time.time() - t0
    candidates.sort(key=lambda c: c["total_cost_per_hour"])
    top_3 = candidates[:3]
    
    print(f"  搜索耗时: {search_elapsed:.2f} 秒")
    print(f"  搜索空间: {max_docs * max_beds} 个配置")
    print(f"  可行配置数: {len(candidates)}")
    print(f"\n  候选配置 (按成本排序):")
    for i, cand in enumerate(top_3):
        print(f"    候选{i+1}: D={cand['num_doctors']}, B={cand['resuscitation_room_capacity']}, "
              f"成本={cand['total_cost_per_hour']:.0f}元/小时")
        print(f"      医生利用率: {cand['avg_doctor_utilization']*100:.1f}%, "
              f"ESI_2 P95: {cand['esi2_p95_wait']:.1f}分钟")
    
    assert len(top_3) <= 3, "候选配置不应超过3个"
    assert all(c['esi2_p95_wait'] < 15.0 for c in top_3), "所有候选应满足ESI_2 P95 < 15"
    assert all(c['avg_doctor_utilization'] <= 0.92 for c in top_3), "所有候选应满足利用率 <= 0.92"
    print("\n  ✅ 功能5 - 经济配置搜索验证通过!")
    
    print("\n" + "=" * 80)
    if all_pass and elapsed < target:
        print("  🎉 所有5大新功能验证全部通过!")
    else:
        print("  ⚠️  部分功能未通过验证")
    print("=" * 80)
    print()
    print("功能验证清单:")
    print("  ✅ 功能1 - 热身期与冷却期统计")
    print("  ✅ 功能2 - 等待期病情恶化升级")
    print("  ✅ 功能3 - 时变到达率")
    print("  ✅ 功能4 - 离院威胁(LWBS)")
    print("  ✅ 功能5 - 经济配置搜索")
    print(f"  {'✅' if elapsed < target else '❌'} 性能: N=1000蒙特卡洛耗时 {elapsed:.2f} 秒")
    
    return all_pass and elapsed < target

if __name__ == "__main__":
    success = test_all_features()
    sys.exit(0 if success else 1)
