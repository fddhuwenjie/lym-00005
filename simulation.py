import heapq
import math
import random
from typing import Dict, List, Tuple, Optional
from models import (
    ScenarioConfig, ESILevel, ESI_PRIORITY, Patient,
    SingleSimulationResult, Event, EventType, DistributionType
)


def _sample_distribution(params: Dict, rng: random.Random) -> float:
    dist_type = params["distribution_type"]
    if dist_type == DistributionType.EXPONENTIAL:
        return rng.expovariate(1.0 / params["mean"])
    elif dist_type == DistributionType.NORMAL:
        std = params.get("std", params["mean"] * 0.3)
        return max(0.1, rng.gauss(params["mean"], std))
    elif dist_type == DistributionType.UNIFORM:
        std = params.get("std", params["mean"] * 0.5)
        return max(0.1, rng.uniform(params["mean"] - std, params["mean"] + std))
    elif dist_type == DistributionType.GAMMA:
        shape = params.get("shape")
        scale = params.get("scale")
        if shape is None or scale is None:
            mean = params["mean"]
            std = params.get("std", mean * 0.5)
            shape = (mean / std) ** 2 if std > 0 else 1.0
            scale = (std ** 2) / mean if std > 0 else mean
        return max(0.1, rng.gammavariate(shape, scale))
    return params["mean"]


def _sample_esi_level(esi_dist: Dict[ESILevel, float], rng: random.Random) -> ESILevel:
    levels = list(esi_dist.keys())
    probs = [esi_dist[l] for l in levels]
    r = rng.random()
    cumulative = 0.0
    for level, prob in zip(levels, probs):
        cumulative += prob
        if r <= cumulative:
            return level
    return levels[-1]


def run_single_simulation(scenario: Dict, seed: int) -> Dict:
    rng = random.Random(seed)
    
    num_doctors = scenario["num_doctors"]
    resus_capacity = scenario["resuscitation_room_capacity"]
    sim_duration = scenario["simulation_duration"]
    arrival_rate = scenario["arrival_rate"]
    esi_dist = scenario["esi_distribution"]
    service_params = scenario["service_time_params"]
    
    event_queue: List[Tuple[float, int, int, Dict]] = []
    patient_counter = 0
    event_counter = 0
    
    patients: Dict[int, Dict] = {}
    waiting_queue: List[Tuple[int, float, int]] = []
    doctors_busy: Dict[int, Optional[int]] = {i: None for i in range(num_doctors)}
    doctors_busy_until: Dict[int, float] = {i: 0.0 for i in range(num_doctors)}
    doctor_total_busy_time: Dict[int, float] = {i: 0.0 for i in range(num_doctors)}
    
    resus_occupancy = 0
    resus_peak = 0
    resus_integral = 0.0
    last_resus_time = 0.0
    total_preemptions = 0
    
    inter_arrival_mean = 1.0 / arrival_rate
    first_arrival = rng.expovariate(arrival_rate)
    heapq.heappush(event_queue, (first_arrival, event_counter, 0, {"type": EventType.ARRIVAL}))
    event_counter += 1
    
    def push_event(time: float, event_type: EventType, data: Dict = None):
        nonlocal event_counter
        heapq.heappush(event_queue, (time, event_counter, 0, {"type": event_type, **(data or {})}))
        event_counter += 1
    
    def assign_doctor(current_time: float) -> Optional[int]:
        for doc_id in range(num_doctors):
            if doctors_busy[doc_id] is None:
                return doc_id
        return None
    
    def preempt_lowest_priority(current_time: float, new_patient_priority: int) -> Optional[int]:
        lowest_priority = 999
        doc_to_preempt = None
        patient_to_preempt = None
        
        for doc_id, patient_id in doctors_busy.items():
            if patient_id is not None:
                p = patients[patient_id]
                p_priority = ESI_PRIORITY[p["esi_level"]]
                if p_priority > new_patient_priority and p_priority < lowest_priority:
                    lowest_priority = p_priority
                    doc_to_preempt = doc_id
                    patient_to_preempt = patient_id
        
        if doc_to_preempt is not None:
            p = patients[patient_to_preempt]
            elapsed = current_time - p["service_start_time"]
            total_needed = p["total_service_time"]
            remaining = max(0.1, total_needed - elapsed)
            
            p["remaining_service_time"] = remaining
            p["service_end_time"] = None
            p["service_start_time"] = None
            p["was_preempted"] = True
            p["preemption_count"] += 1
            p["is_completed"] = False
            
            doctor_total_busy_time[doc_to_preempt] += elapsed
            doctors_busy[doc_to_preempt] = None
            
            heapq.heappush(waiting_queue, (
                ESI_PRIORITY[p["esi_level"]],
                p["arrival_time"],
                patient_to_preempt
            ))
            
            return doc_to_preempt
        return None
    
    def try_start_service(current_time: float):
        while waiting_queue:
            priority, arr_time, patient_id = waiting_queue[0]
            
            free_doc = assign_doctor(current_time)
            if free_doc is None:
                patient = patients[patient_id]
                p_priority = ESI_PRIORITY[patient["esi_level"]]
                free_doc = preempt_lowest_priority(current_time, p_priority)
                if free_doc is None:
                    break
            
            if free_doc is not None:
                heapq.heappop(waiting_queue)
                patient = patients[patient_id]
                
                if patient["remaining_service_time"] is not None:
                    service_time = patient["remaining_service_time"]
                    patient["remaining_service_time"] = None
                else:
                    sp = service_params[patient["esi_level"]]
                    service_time = _sample_distribution(sp, rng)
                    patient["total_service_time"] = service_time
                
                patient["service_start_time"] = current_time
                patient["wait_time"] = current_time - patient["arrival_time"]
                
                doctors_busy[free_doc] = patient_id
                end_time = current_time + service_time
                doctors_busy_until[free_doc] = end_time
                
                push_event(end_time, EventType.END_SERVICE, {
                    "doctor_id": free_doc,
                    "patient_id": patient_id
                })
                
                nonlocal resus_occupancy, resus_peak
                resus_occupancy += 1
                if resus_occupancy > resus_peak:
                    resus_peak = resus_occupancy
            else:
                break
    
    current_time = 0.0
    
    while event_queue:
        event_time, _, _, event = heapq.heappop(event_queue)
        current_time = event_time
        
        if current_time > sim_duration and event["type"] == EventType.ARRIVAL:
            continue
        
        time_delta = current_time - last_resus_time
        resus_integral += resus_occupancy * time_delta
        last_resus_time = current_time
        
        if event["type"] == EventType.ARRIVAL:
            patient_counter += 1
            esi = _sample_esi_level(esi_dist, rng)
            
            patient = {
                "patient_id": patient_counter,
                "esi_level": esi,
                "arrival_time": current_time,
                "service_start_time": None,
                "service_end_time": None,
                "wait_time": None,
                "total_service_time": None,
                "was_preempted": False,
                "preemption_count": 0,
                "remaining_service_time": None,
                "is_completed": False,
            }
            patients[patient_counter] = patient
            
            heapq.heappush(waiting_queue, (
                ESI_PRIORITY[esi],
                current_time,
                patient_counter
            ))
            
            next_arrival = current_time + rng.expovariate(arrival_rate)
            if next_arrival <= sim_duration:
                push_event(next_arrival, EventType.ARRIVAL)
            
            try_start_service(current_time)
        
        elif event["type"] == EventType.END_SERVICE:
            doctor_id = event["doctor_id"]
            patient_id = event["patient_id"]
            
            if doctors_busy[doctor_id] == patient_id:
                patient = patients[patient_id]
                patient["service_end_time"] = current_time
                patient["is_completed"] = True
                
                elapsed = current_time - patient["service_start_time"]
                doctor_total_busy_time[doctor_id] += elapsed
                doctors_busy[doctor_id] = None
                resus_occupancy -= 1
                
                if patient["preemption_count"] > 0:
                    total_preemptions += patient["preemption_count"]
                
                try_start_service(current_time)
    
    for doc_id in range(num_doctors):
        if doctors_busy[doc_id] is not None:
            patient_id = doctors_busy[doc_id]
            patient = patients[patient_id]
            elapsed = current_time - patient["service_start_time"]
            doctor_total_busy_time[doc_id] += elapsed
            patient["service_start_time"] = None
    
    total_busy = sum(doctor_total_busy_time.values())
    doctor_utilization = total_busy / (num_doctors * max(current_time, 0.001)) if current_time > 0 else 0.0
    
    completed_patients = [p for p in patients.values() if p["is_completed"]]
    wait_by_esi: Dict[ESILevel, List[float]] = {}
    for p in completed_patients:
        esi = p["esi_level"]
        if esi not in wait_by_esi:
            wait_by_esi[esi] = []
        if p["wait_time"] is not None:
            wait_by_esi[esi].append(p["wait_time"])
    
    avg_wait = {}
    max_wait = {}
    for esi in ESILevel:
        waits = wait_by_esi.get(esi, [])
        avg_wait[esi] = sum(waits) / len(waits) if waits else 0.0
        max_wait[esi] = max(waits) if waits else 0.0
    
    avg_resus = resus_integral / max(current_time, 0.001) if current_time > 0 else 0.0
    
    patient_models = []
    for p in patients.values():
        patient_models.append(Patient(**p))
    
    return {
        "patients": patient_models,
        "doctor_utilization": doctor_utilization,
        "avg_wait_time_by_esi": avg_wait,
        "max_wait_time_by_esi": max_wait,
        "resuscitation_peak_occupancy": min(resus_peak, resus_capacity),
        "resuscitation_avg_occupancy": avg_resus,
        "total_arrivals": len(patients),
        "total_completed": len(completed_patients),
        "total_preemptions": total_preemptions,
    }


def run_monte_carlo(scenario: Dict, n: int, base_seed: int) -> List[Dict]:
    results = []
    for i in range(n):
        seed = base_seed + i
        result = run_single_simulation(scenario, seed)
        results.append(result)
    return results
