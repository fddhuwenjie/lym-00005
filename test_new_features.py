import sys
sys.path.insert(0, r'e:\solo\项目\lym-00005')

import json
import time
from main import (
    _run_sim_core, _analyze_mc, _config_to_dict,
    _get_presets, _run_batch, ScenarioConfig, ESILevel,
    DistributionParams, DistributionType, TimeSegment,
    _search_capacity_configs, CapacitySearchRequest,
    CapacitySearchTarget
)

def test_scenario_config():
    print("=" * 60)
    print("测试1: 场景配置 - 新字段验证")
    print("=" * 60)
    
    scenario_dict = {
        "name": "测试场景",
        "num_doctors": 5,
        "resuscitation_room_capacity": 8,
        "simulation_duration": 480.0,
        "arrival_rate": 0.5,
        "esi_distribution": {
            ESILevel.ESI_1: 0.03, ESILevel.ESI_2: 0.08,
            ESILevel.ESI_3: 0.25, ESILevel.ESI_4: 0.35, ESILevel.ESI_5: 0.29
        },
        "service_time_params": {
            ESILevel.ESI_1: DistributionParams(distribution_type=DistributionType.GAMMA, mean=45.0, std=20.0),
            ESILevel.ESI_2: DistributionParams(distribution_type=DistributionType.GAMMA, mean=30.0, std=12.0),
            ESILevel.ESI_3: DistributionParams(distribution_type=DistributionType.GAMMA, mean=20.0, std=8.0),
            ESILevel.ESI_4: DistributionParams(distribution_type=DistributionType.GAMMA, mean=12.0, std=5.0),
            ESILevel.ESI_5: DistributionParams(distribution_type=DistributionType.GAMMA, mean=8.0, std=3.0),
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
            TimeSegment(start_time=0.0, end_time=240.0, arrival_rate=0.4),
            TimeSegment(start_time=240.0, end_time=480.0, arrival_rate=0.6),
        ],
        "patience_limit": {
            ESILevel.ESI_5: 60.0, ESILevel.ESI_4: 90.0, ESILevel.ESI_3: 120.0,
            ESILevel.ESI_2: 180.0, ESILevel.ESI_1: 9999.0,
        }
    }
    
    cfg = ScenarioConfig(**scenario_dict)
    d = _config_to_dict(cfg)
    
    assert d["warmup_duration"] == 30.0
    assert d["cooldown_duration"] == 30.0
    assert d["escalation_thresholds"][ESILevel.ESI_5] == 30.0
    assert len(d["arrival_rate_schedule"]) == 2
    assert d["patience_limit"][ESILevel.ESI_5] == 60.0
    
    print("[PASS] 场景配置新字段验证通过")
    print()
    return d


def test_single_simulation(scenario_dict):
    print("=" * 60)
    print("测试2: 单次仿真 - 所有新功能")
    print("=" * 60)
    
    t0 = time.time()
    result = _run_sim_core(scenario_dict, seed=42)
    elapsed = time.time() - t0
    
    print(f"仿真耗时: {elapsed*1000:.1f} ms")
    print(f"总到达患者: {result['total_arrivals']}")
    print(f"完成患者: {result['total_completed']}")
    
    ss = result["steady_state"]
    print(f"\n稳态区间: [{ss['start_time']}, {ss['end_time']}]")
    print(f"稳态样本量: {ss['sample_size']}")
    print(f"稳态持续时长: {ss['duration']:.1f} 分钟")
    
    esc = result["escalation"]
    print(f"\n病情升级总次数: {esc['total_escalations']}")
    if esc['total_escalations'] > 0:
        print("升级入/出:")
        for esi in ESILevel:
            print(f"  {esi.value}: 入={esc['escalation_in'][esi.value]}, 出={esc['escalation_out'][esi.value]}")
        print(f"升级患者平均等待: {esc['escalated_avg_wait']:.1f} 分钟")
    
    lwbs = result["lwbs"]
    print(f"\nLWBS总数: {lwbs['total_lwbs']}")
    if lwbs['total_lwbs'] > 0:
        print("LWBS按级别:")
        for esi in ESILevel:
            print(f"  {esi.value}: {lwbs['lwbs_by_esi'][esi.value]} (放弃率: {lwbs['lwbs_rate_by_esi'][esi]*100:.1f}%)")
    
    if result["time_slices"]:
        print(f"\n时段切片统计:")
        for ts in result["time_slices"]:
            print(f"  [{ts['start_time']:.0f}, {ts['end_time']:.0f}): 到达率={ts['arrival_rate']:.2f}, "
                  f"样本量={ts['sample_size']}, P50={ts['wait_p50']:.1f}, P95={ts['wait_p95']:.1f}")
    
    print(f"\n医生利用率(全时段): {result['doctor_utilization']*100:.1f}%")
    print(f"医生利用率(稳态): {result['doctor_utilization_steady']*100:.1f}%")
    
    assert "steady_state" in result
    assert "escalation" in result
    assert "lwbs" in result
    assert "time_slices" in result
    
    print("\n[PASS] 单次仿真新功能验证通过")
    print()
    return result


def test_monte_carlo(scenario_dict, n=100):
    print("=" * 60)
    print(f"测试3: 蒙特卡洛仿真 N={n}")
    print("=" * 60)
    
    t0 = time.time()
    results = _run_batch(scenario_dict, n, base_seed=42)
    mc_res = _analyze_mc(results, scenario_dict)
    elapsed = time.time() - t0
    
    print(f"蒙特卡洛耗时: {elapsed*1000:.1f} ms")
    print(f"每轮平均耗时: {elapsed*1000/n:.1f} ms")
    print(f"医生利用率(稳态): {mc_res['avg_doctor_utilization']*100:.1f}%")
    
    ss = mc_res["steady_state"]
    print(f"\n平均稳态样本量: {ss['avg_sample_size']:.1f}")
    
    esc = mc_res["escalation"]
    print(f"\n平均每轮升级次数: {esc['avg_escalations_per_run']:.2f}")
    
    lwbs = mc_res["lwbs"]
    print(f"平均每轮LWBS数: {lwbs['avg_lwbs_per_run']:.2f}")
    
    if mc_res["time_slices"]:
        print(f"\n最拥堵时段: [{mc_res['most_congested_period']['start_time']:.0f}, "
              f"{mc_res['most_congested_period']['end_time']:.0f}) "
              f"P95={mc_res['most_congested_period']['wait_p95']:.1f}分钟")
    
    assert "steady_state" in mc_res
    assert "escalation" in mc_res
    assert "lwbs" in mc_res
    assert "time_slices" in mc_res
    assert "most_congested_period" in mc_res
    
    print("\n[PASS] 蒙特卡洛新功能验证通过")
    print()
    return mc_res, elapsed


def test_performance(scenario_dict):
    print("=" * 60)
    print("测试4: 性能测试 - N=1000 目标 < 10秒")
    print("=" * 60)
    
    t0 = time.time()
    results = _run_batch(scenario_dict, 1000, base_seed=42)
    mc_res = _analyze_mc(results, scenario_dict)
    elapsed = time.time() - t0
    
    print(f"N=1000 总耗时: {elapsed:.2f} 秒")
    print(f"每轮平均: {elapsed*1000/1000:.1f} ms")
    
    if elapsed < 10.0:
        print(f"\n[PASS] 性能达标: {elapsed:.2f}秒 < 10秒")
    else:
        print(f"\n[FAIL] 性能不达标: {elapsed:.2f}秒 > 10秒")
    
    print()
    return elapsed


def test_capacity_search():
    print("=" * 60)
    print("测试5: 经济配置搜索")
    print("=" * 60)
    
    req_dict = {
        "name": "容量搜索测试",
        "max_doctors": 8,
        "max_resuscitation_beds": 10,
        "simulation_duration": 240.0,
        "arrival_rate": 0.5,
        "esi_distribution": {
            ESILevel.ESI_1: 0.03, ESILevel.ESI_2: 0.08,
            ESILevel.ESI_3: 0.25, ESILevel.ESI_4: 0.35, ESILevel.ESI_5: 0.29
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
        "targets": [
            {"esi_level": ESILevel.ESI_2, "metric": "wait_p95", "operator": "<", "threshold": 15.0},
            {"esi_level": ESILevel.ESI_1, "metric": "doctor_utilization", "operator": "<=", "threshold": 0.92},
        ],
        "warmup_duration": 20.0,
        "cooldown_duration": 20.0,
        "n_mc": 50,
        "base_seed": 42,
        "doctor_cost_per_hour": 100.0,
    }
    
    t0 = time.time()
    result = _search_capacity_configs(req_dict)
    elapsed = time.time() - t0
    
    print(f"搜索耗时: {elapsed:.2f} 秒")
    print(f"搜索空间: {result['search_space']['total_configs_checked']} 个配置")
    print(f"可行配置数: {result['feasible_count']}")
    print(f"\n候选配置 (按成本排序):")
    for i, cand in enumerate(result["top_candidates"]):
        print(f"\n  候选 {i+1}: D={cand['num_doctors']}, B={cand['resuscitation_room_capacity']}, "
              f"成本={cand['total_cost_per_hour']:.0f}元/小时")
        print(f"    医生利用率: {cand['key_metrics']['avg_doctor_utilization']*100:.1f}%")
        print(f"    ESI_2 P95等待: {cand['key_metrics']['wait_time_quantiles'][ESILevel.ESI_2]['P95']:.1f} 分钟")
    
    assert len(result["top_candidates"]) <= 3
    assert all(c["meets_all_targets"] for c in result["top_candidates"])
    
    print("\n[PASS] 经济配置搜索验证通过")
    print()
    return result


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("开始急诊分诊仿真新功能测试")
    print("=" * 60 + "\n")
    
    try:
        scenario = test_scenario_config()
        test_single_simulation(scenario)
        mc_res, mc_time = test_monte_carlo(scenario, n=100)
        perf_time = test_performance(scenario)
        test_capacity_search()
        
        print("=" * 60)
        print("所有测试通过! ✅")
        print("=" * 60)
        print(f"\n性能摘要:")
        print(f"  N=100 蒙特卡洛: {mc_time:.2f} 秒")
        print(f"  N=1000 蒙特卡洛: {perf_time:.2f} 秒")
        print(f"  目标: < 10 秒, {'✅ 达标' if perf_time < 10 else '❌ 未达标'}")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
