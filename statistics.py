import math
from typing import List, Dict, Tuple
from models import ESILevel, MonteCarloResult, ScenarioConfig


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


def _mean(data: List[float]) -> float:
    return sum(data) / len(data) if data else 0.0


def _std(data: List[float]) -> float:
    if len(data) < 2:
        return 0.0
    m = _mean(data)
    variance = sum((x - m) ** 2 for x in data) / (len(data) - 1)
    return math.sqrt(variance)


def _t_critical(df: int, confidence: float = 0.95) -> float:
    p = (1 + confidence) / 2
    t_values = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        25: 2.060, 30: 2.042, 40: 2.021, 50: 2.009, 60: 2.000,
        80: 1.990, 100: 1.984, 1000: 1.962, 10000: 1.960
    }
    for df_key in sorted(t_values.keys(), reverse=True):
        if df >= df_key:
            return t_values[df_key]
    return 1.96


def compute_confidence_interval(data: List[float], confidence: float = 0.95) -> Dict[str, float]:
    if not data:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "std": 0.0}
    n = len(data)
    m = _mean(data)
    s = _std(data)
    se = s / math.sqrt(n) if n > 0 else 0
    t = _t_critical(n - 1, confidence)
    margin = t * se
    return {
        "mean": m,
        "lower": m - margin,
        "upper": m + margin,
        "std": s
    }


def analyze_monte_carlo_results(
    results: List[Dict],
    scenario: Dict
) -> MonteCarloResult:
    n = len(results)
    
    wait_times_by_esi: Dict[ESILevel, List[float]] = {esi: [] for esi in ESILevel}
    doctor_utils: List[float] = []
    resus_peaks: List[float] = []
    total_arrivals_list: List[float] = []
    
    target_wait = scenario.get("target_wait_time", {})
    exceed_count: Dict[ESILevel, int] = {esi: 0 for esi in ESILevel}
    total_patients_by_esi: Dict[ESILevel, int] = {esi: 0 for esi in ESILevel}
    
    for result in results:
        doctor_utils.append(result["doctor_utilization"])
        resus_peaks.append(result["resuscitation_peak_occupancy"])
        total_arrivals_list.append(result["total_arrivals"])
        
        for patient in result["patients"]:
            if patient.is_completed and patient.wait_time is not None:
                esi = patient.esi_level
                wait_times_by_esi[esi].append(patient.wait_time)
                total_patients_by_esi[esi] += 1
                
                target = target_wait.get(esi)
                if target is not None and patient.wait_time > target:
                    exceed_count[esi] += 1
    
    quantile_levels = ["P50", "P75", "P90", "P95", "P99"]
    quantile_values = [0.5, 0.75, 0.9, 0.95, 0.99]
    
    wait_time_quantiles: Dict[ESILevel, Dict[str, float]] = {}
    for esi in ESILevel:
        waits = sorted(wait_times_by_esi[esi])
        wait_time_quantiles[esi] = {}
        for q_name, q_val in zip(quantile_levels, quantile_values):
            wait_time_quantiles[esi][q_name] = _quantile(waits, q_val)
    
    doc_util_ci = compute_confidence_interval(doctor_utils, 0.95)
    
    target_prob: Dict[ESILevel, float] = {}
    for esi in ESILevel:
        if total_patients_by_esi[esi] > 0:
            target_prob[esi] = exceed_count[esi] / total_patients_by_esi[esi]
        else:
            target_prob[esi] = 0.0
    
    return MonteCarloResult(
        num_simulations=n,
        wait_time_quantiles=wait_time_quantiles,
        doctor_utilization_ci=doc_util_ci,
        target_wait_probability=target_prob,
        avg_doctor_utilization=doc_util_ci["mean"],
        avg_resuscitation_peak=_mean(resus_peaks),
        avg_total_arrivals=_mean(total_arrivals_list)
    )


def compare_scenarios(
    scenarios: List[Dict],
    mc_results: List[MonteCarloResult],
    base_seed: int
) -> Dict:
    names = [s.get("name", f"场景{i+1}") for i, s in enumerate(scenarios)]
    
    metrics = []
    for i, (scenario, mc) in enumerate(zip(scenarios, mc_results)):
        metric = {
            "scenario_name": names[i],
            "num_doctors": scenario["num_doctors"],
            "resus_capacity": scenario["resuscitation_room_capacity"],
            "arrival_rate": scenario["arrival_rate"],
            "doctor_utilization_mean": mc.avg_doctor_utilization,
            "doctor_utilization_lower": mc.doctor_utilization_ci["lower"],
            "doctor_utilization_upper": mc.doctor_utilization_ci["upper"],
        }
        
        target_wait = scenario.get("target_wait_time", {})
        for esi in ESILevel:
            target = target_wait.get(esi)
            if target is not None:
                metric[f"{esi.value}_wait_p95"] = mc.wait_time_quantiles[esi]["P95"]
                metric[f"{esi.value}_exceed_prob"] = mc.target_wait_probability[esi]
                metric[f"{esi.value}_meet_target_prob"] = 1.0 - mc.target_wait_probability[esi]
        
        metrics.append(metric)
    
    recommendation = _generate_comparison_recommendation(scenarios, mc_results, metrics)
    
    return {
        "scenario_names": names,
        "base_seed": base_seed,
        "metrics": metrics,
        "recommendation": recommendation
    }


def _generate_comparison_recommendation(
    scenarios: List[Dict],
    mc_results: List[MonteCarloResult],
    metrics: List[Dict]
) -> str:
    lines = []
    
    critical_esi = ESILevel.ESI_1
    target_wait = scenarios[0].get("target_wait_time", {})
    has_targets = len(target_wait) > 0
    
    lines.append("## 医生数配置建议")
    lines.append("")
    
    sorted_by_doctors = sorted(enumerate(scenarios), key=lambda x: x[1]["num_doctors"])
    
    for idx, (orig_idx, scenario) in enumerate(sorted_by_doctors):
        mc = mc_results[orig_idx]
        num_docs = scenario["num_doctors"]
        util = mc.avg_doctor_utilization
        
        meet_all = True
        if has_targets:
            for esi, target in target_wait.items():
                prob = 1.0 - mc.target_wait_probability[esi]
                if prob < 0.95:
                    meet_all = False
                    break
        
        if util > 0.92:
            status = "过载"
            advice = f"医生数 {num_docs} 明显不足，利用率 {util:.1%} 过高，建议增加医生"
        elif util < 0.60:
            status = "资源闲置"
            advice = f"医生数 {num_docs} 过多，利用率 {util:.1%} 偏低，可考虑减少"
        else:
            status = "合理区间"
            advice = f"医生数 {num_docs} 配置合理，利用率 {util:.1%} 在理想区间 (60%-92%)"
        
        lines.append(f"- **{num_docs} 名医生**: {status}")
        lines.append(f"  - 医生利用率: {util:.1%} (95% CI: [{mc.doctor_utilization_ci['lower']:.1%}, {mc.doctor_utilization_ci['upper']:.1%}])")
        
        if has_targets:
            lines.append(f"  - 等待达标率:")
            for esi in ESILevel:
                target = target_wait.get(esi)
                if target is not None:
                    prob = 1.0 - mc.target_wait_probability[esi]
                    p95 = mc.wait_time_quantiles[esi]["P95"]
                    lines.append(f"    * {esi.value}: {prob:.1%} 达标 (P95等待: {p95:.1f}分钟, 目标: {target}分钟)")
        
        lines.append(f"  - 建议: {advice}")
        lines.append("")
    
    optimal_idx = None
    best_score = -1
    
    for i, scenario in enumerate(scenarios):
        mc = mc_results[i]
        util = mc.avg_doctor_utilization
        
        score = 0.0
        if 0.60 <= util <= 0.92:
            score += 50
        elif util < 0.60:
            score += util * 80
        else:
            score += max(0, (1.0 - util) * 80)
        
        if has_targets:
            for esi, target in target_wait.items():
                prob = 1.0 - mc.target_wait_probability[esi]
                if prob >= 0.95:
                    score += 10
                elif prob >= 0.80:
                    score += 5
        
        if score > best_score:
            best_score = score
            optimal_idx = i
    
    if optimal_idx is not None:
        optimal = scenarios[optimal_idx]
        lines.append(f"## 最优配置推荐")
        lines.append("")
        lines.append(f"推荐采用 **{optimal['name']}** 配置方案:")
        lines.append(f"- 医生数: {optimal['num_doctors']} 名")
        lines.append(f"- 抢救室容量: {optimal['resuscitation_room_capacity']} 床")
        lines.append(f"- 可应对到达率: {optimal['arrival_rate']} 患者/分钟")
    
    return "\n".join(lines)
