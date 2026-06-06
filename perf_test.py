import time
import math
import heapq
import random
from typing import List, Dict, Tuple


class ESILevel:
    ESI_1 = "ESI_1"
    ESI_2 = "ESI_2"
    ESI_3 = "ESI_3"
    ESI_4 = "ESI_4"
    ESI_5 = "ESI_5"


ESI_PRIORITY = {
    ESILevel.ESI_1: 1, ESILevel.ESI_2: 2, ESILevel.ESI_3: 3,
    ESILevel.ESI_4: 4, ESILevel.ESI_5: 5,
}

ESI_LEVELS = [ESILevel.ESI_1, ESILevel.ESI_2, ESILevel.ESI_3, ESILevel.ESI_4, ESILevel.ESI_5]


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


def _sample_esi(cum_probs: List[Tuple[str, float]], rng: random.Random) -> str:
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
            
            free_doc = -1
            for d in range(num_docs):
                if doc_busy[d] is None:
                    free_doc = d
                    break
            
            if free_doc == -1:
                pat_prio = ESI_PRIORITY[pat["esi_level"]]
                lowest_prio = 999
                doc_preempt = -1
                for d in range(num_docs):
                    if doc_busy[d] is not None:
                        op = patients[doc_busy[d]]
                        op_prio = ESI_PRIORITY[op["esi_level"]]
                        if op_prio > pat_prio and op_prio < lowest_prio:
                            lowest_prio = op_prio
                            doc_preempt = d
                
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
                    
                    heapq.heappush(wait_q, (
                        ESI_PRIORITY[op["esi_level"]],
                        op["arrival_time"],
                        op_id
                    ))
                    free_doc = doc_preempt
            
            if free_doc >= 0:
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
    
    return {
        "doctor_utilization": util,
        "total_arrivals": len(patients),
        "total_completed": len([p for p in patients.values() if p["completed"]]),
        "total_preemptions": preemptions,
    }


def get_standard_scenario():
    def_svc = {
        ESILevel.ESI_1: {"distribution_type":"gamma","mean":45.0,"std":20.0,"shape":5.06,"scale":8.9},
        ESILevel.ESI_2: {"distribution_type":"gamma","mean":30.0,"std":12.0,"shape":6.25,"scale":4.8},
        ESILevel.ESI_3: {"distribution_type":"gamma","mean":20.0,"std":8.0,"shape":6.25,"scale":3.2},
        ESILevel.ESI_4: {"distribution_type":"gamma","mean":12.0,"std":5.0,"shape":5.76,"scale":2.08},
        ESILevel.ESI_5: {"distribution_type":"gamma","mean":8.0,"std":3.0,"shape":7.11,"scale":1.125},
    }
    return {
        "name": "标准场景 - 日间常规",
        "num_doctors": 5,
        "resuscitation_room_capacity": 8,
        "simulation_duration": 480.0,
        "arrival_rate": 0.5,
        "esi_distribution": {
            ESILevel.ESI_1: 0.03,
            ESILevel.ESI_2: 0.08,
            ESILevel.ESI_3: 0.25,
            ESILevel.ESI_4: 0.35,
            ESILevel.ESI_5: 0.29,
        },
        "service_time_params": def_svc,
        "target_wait_time": {
            ESILevel.ESI_1: 1.0,
            ESILevel.ESI_2: 10.0,
            ESILevel.ESI_3: 30.0,
            ESILevel.ESI_4: 60.0,
            ESILevel.ESI_5: 120.0,
        },
    }


def run_perf_test(n=1000):
    scenario = get_standard_scenario()
    print("[*] 开始性能测试: N={}".format(n))
    print("  医生数: {}".format(scenario['num_doctors']))
    print("  仿真时长: {} 分钟".format(scenario['simulation_duration']))
    print("  到达率: {} 患者/分钟".format(scenario['arrival_rate']))
    print()
    
    t0 = time.time()
    utils = []
    for i in range(n):
        r = _run_sim_core(scenario, 42 + i)
        utils.append(r["doctor_utilization"])
    t1 = time.time()
    
    elapsed = t1 - t0
    per_sim = elapsed / n * 1000
    
    print("[+] 测试完成!")
    print("  总耗时: {:.2f} 秒".format(elapsed))
    print("  单次仿真: {:.1f} 毫秒".format(per_sim))
    print("  N=1000 预计: {:.2f} 秒".format(per_sim * 1000 / 1000))
    status = "[OK] 达标" if elapsed < 10 else "[FAIL] 不达标"
    print("  目标: < 10 秒 {}".format(status))
    print()
    print("  平均医生利用率: {:.1f}%".format(sum(utils)/len(utils)*100))
    
    return elapsed < 10


if __name__ == "__main__":
    run_perf_test(1000)
