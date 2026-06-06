from typing import Dict, Optional
from models import ESILevel, ESI_NAMES, MonteCarloResult, SingleSimulationResult


def generate_markdown_report(
    scenario: Dict,
    monte_carlo_result: Optional[MonteCarloResult] = None,
    single_result: Optional[SingleSimulationResult] = None,
) -> str:
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
        dist_type = sp.get('distribution_type', 'N/A')
        mean = sp.get('mean', 0)
        std = sp.get('std', 'N/A')
        lines.append(f"| {esi.value} | {dist_type} | {mean:.1f} | {std} |")
    lines.append("")
    
    target_wait = scenario.get('target_wait_time')
    if target_wait:
        lines.append("### 1.4 目标等待时间")
        lines.append("")
        lines.append("| 级别 | 目标等待时间(分钟) |")
        lines.append("|------|-------------------|")
        for esi in ESILevel:
            target = target_wait.get(esi)
            if target is not None:
                lines.append(f"| {esi.value} | {target:.0f} |")
        lines.append("")
    
    if single_result:
        lines.append("## 2. 单次仿真结果")
        lines.append("")
        lines.append("### 2.1 总体指标")
        lines.append("")
        lines.append(f"- 总到达患者数: {single_result.total_arrivals}")
        lines.append(f"- 完成服务患者数: {single_result.total_completed}")
        lines.append(f"- 医生平均利用率: {single_result.doctor_utilization * 100:.1f}%")
        lines.append(f"- 抢救室峰值占用率: {single_result.resuscitation_peak_occupancy}/{scenario['resuscitation_room_capacity']} ({single_result.resuscitation_peak_occupancy/scenario['resuscitation_room_capacity']*100:.1f}%)")
        lines.append(f"- 抢救室平均占用率: {single_result.resuscitation_avg_occupancy:.1f} 床")
        lines.append(f"- 总抢占次数: {single_result.total_preemptions}")
        lines.append("")
        
        lines.append("### 2.2 各 ESI 级别等待时长")
        lines.append("")
        lines.append("| 级别 | 平均等待(分钟) | 最大等待(分钟) |")
        lines.append("|------|---------------|---------------|")
        for esi in ESILevel:
            avg_wait = single_result.avg_wait_time_by_esi.get(esi, 0)
            max_wait = single_result.max_wait_time_by_esi.get(esi, 0)
            lines.append(f"| {esi.value} | {avg_wait:.1f} | {max_wait:.1f} |")
        lines.append("")
    
    if monte_carlo_result:
        lines.append("## 3. 蒙特卡洛仿真结果 (N=" + str(monte_carlo_result.num_simulations) + ")")
        lines.append("")
        
        lines.append("### 3.1 等待时长分位数")
        lines.append("")
        lines.append("| 级别 | P50(分钟) | P75(分钟) | P90(分钟) | P95(分钟) | P99(分钟) |")
        lines.append("|------|-----------|-----------|-----------|-----------|-----------|")
        for esi in ESILevel:
            q = monte_carlo_result.wait_time_quantiles.get(esi, {})
            lines.append(
                f"| {esi.value} | {q.get('P50', 0):.1f} | {q.get('P75', 0):.1f} | "
                f"{q.get('P90', 0):.1f} | {q.get('P95', 0):.1f} | {q.get('P99', 0):.1f} |"
            )
        lines.append("")
        
        lines.append("### 3.2 医生利用率")
        lines.append("")
        ci = monte_carlo_result.doctor_utilization_ci
        lines.append(f"- 均值: {ci['mean'] * 100:.1f}%")
        lines.append(f"- 95% 置信区间: [{ci['lower'] * 100:.1f}%, {ci['upper'] * 100:.1f}%]")
        lines.append(f"- 标准差: {ci['std'] * 100:.1f}%")
        lines.append("")
        
        if target_wait:
            lines.append("### 3.3 超目标等待时长概率")
            lines.append("")
            lines.append("| 级别 | 目标(分钟) | 超目标概率 | 达标率 |")
            lines.append("|------|-----------|-----------|--------|")
            for esi in ESILevel:
                target = target_wait.get(esi)
                if target is not None:
                    prob = monte_carlo_result.target_wait_probability.get(esi, 0)
                    lines.append(f"| {esi.value} | {target:.0f} | {prob * 100:.1f}% | {(1-prob) * 100:.1f}% |")
            lines.append("")
        
        lines.append("### 3.4 其他指标")
        lines.append("")
        lines.append(f"- 平均抢救室峰值占用: {monte_carlo_result.avg_resuscitation_peak:.1f} 床")
        lines.append(f"- 平均总到达患者: {monte_carlo_result.avg_total_arrivals:.1f}")
        lines.append("")
    
    lines.append("## 4. 改进建议")
    lines.append("")
    
    suggestions = []
    
    if monte_carlo_result:
        util_mean = monte_carlo_result.avg_doctor_utilization
        if util_mean > 0.92:
            suggestions.append(
                f"- ⚠️ **医生利用率过高 ({util_mean * 100:.1f}%)**: 建议增加 1-2 名医生，"
                f"以降低等待时间和提升服务质量。当前配置下医生持续高负荷运转，"
                f"可能导致医疗差错风险增加。"
            )
        elif util_mean < 0.60:
            suggestions.append(
                f"- ⚠️ **医生利用率偏低 ({util_mean * 100:.1f}%)**: 当前配置下医生资源有闲置，"
                f"可考虑减少 1 名医生或适当扩展服务范围。"
            )
        else:
            suggestions.append(
                f"- ✅ **医生利用率合理 ({util_mean * 100:.1f}%)**: 当前医生配置在理想区间内，"
                f"既能保证服务质量，又能有效利用资源。"
            )
        
        resus_peak = monte_carlo_result.avg_resuscitation_peak
        resus_cap = scenario['resuscitation_room_capacity']
        if resus_peak / resus_cap > 0.9:
            suggestions.append(
                f"- ⚠️ **抢救室资源紧张**: 平均峰值占用 {resus_peak:.1f} 床，"
                f"已接近容量上限 ({resus_cap} 床)。建议增加抢救室床位或优化收治流程。"
            )
        else:
            suggestions.append(
                f"- ✅ **抢救室资源充足**: 平均峰值占用 {resus_peak:.1f} 床，"
                f"容量裕度为 {(1 - resus_peak / resus_cap) * 100:.1f}%。"
            )
        
        if target_wait:
            for esi in ESILevel:
                target = target_wait.get(esi)
                if target is not None:
                    prob = monte_carlo_result.target_wait_probability.get(esi, 0)
                    p95 = monte_carlo_result.wait_time_quantiles[esi]['P95']
                    if prob > 0.10:
                        suggestions.append(
                            f"- ⚠️ **{esi.value} ({ESI_NAMES[esi]}) 等待不达标**: "
                            f"超目标概率 {prob * 100:.1f}%，P95 等待 {p95:.1f} 分钟 "
                            f"(目标 {target} 分钟)。建议优先保障该级别患者的资源配置。"
                        )
    else:
        suggestions.append("- 建议运行蒙特卡洛仿真 (N=1000) 以获得更稳健的统计结论。")
    
    for s in suggestions:
        lines.append(s)
    lines.append("")
    
    if monte_carlo_result and target_wait:
        all_meet = True
        for esi in ESILevel:
            target = target_wait.get(esi)
            if target is not None:
                prob = monte_carlo_result.target_wait_probability.get(esi, 0)
                if prob > 0.05:
                    all_meet = False
                    break
        
        if all_meet:
            lines.append("**总体评价**: 🎉 当前配置可满足所有级别患者的等待时间目标，系统运行良好。")
        else:
            lines.append("**总体评价**: ⚠️ 当前配置下部分级别患者等待时间超标，建议参考上述改进意见调整资源配置。")
    
    return "\n".join(lines)
