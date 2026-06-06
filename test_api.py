import json
import urllib.request
import urllib.parse

BASE_URL = "http://localhost:8005"


def make_request(url, data=None, method="GET"):
    headers = {"Content-Type": "application/json"}
    if data:
        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method=method)
    else:
        req = urllib.request.Request(url, headers=headers, method=method)
    
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_health():
    print("[1/6] Testing /api/health ...")
    result = make_request(f"{BASE_URL}/api/health")
    print(f"  -> Status: {result['status']}, Port: {result['port']}")
    assert result["status"] == "ok"
    print("  [OK]")
    print()


def test_get_scenarios():
    print("[2/6] Testing GET /api/scenarios ...")
    result = make_request(f"{BASE_URL}/api/scenarios")
    scenarios = result["scenarios"]
    print(f"  -> Found {len(scenarios)} preset scenarios")
    assert len(scenarios) >= 3
    for s in scenarios:
        print(f"    * {s['name']}")
    print("  [OK]")
    print()
    return scenarios[0]


def test_create_scenario():
    print("[3/6] Testing POST /api/scenarios ...")
    scenario = {
        "name": "自定义测试场景",
        "num_doctors": 4,
        "resuscitation_room_capacity": 6,
        "simulation_duration": 480.0,
        "arrival_rate": 0.4,
        "esi_distribution": {
            "ESI_1": 0.03,
            "ESI_2": 0.07,
            "ESI_3": 0.25,
            "ESI_4": 0.35,
            "ESI_5": 0.30
        },
        "service_time_params": {
            "ESI_1": {"distribution_type": "gamma", "mean": 45.0, "std": 20.0},
            "ESI_2": {"distribution_type": "gamma", "mean": 30.0, "std": 12.0},
            "ESI_3": {"distribution_type": "gamma", "mean": 20.0, "std": 8.0},
            "ESI_4": {"distribution_type": "gamma", "mean": 12.0, "std": 5.0},
            "ESI_5": {"distribution_type": "gamma", "mean": 8.0, "std": 3.0}
        },
        "target_wait_time": {
            "ESI_1": 1.0,
            "ESI_2": 10.0,
            "ESI_3": 30.0,
            "ESI_4": 60.0,
            "ESI_5": 120.0
        }
    }
    result = make_request(f"{BASE_URL}/api/scenarios", data=scenario, method="POST")
    print(f"  -> Created: {result['name']}")
    assert result["num_doctors"] == 4
    print("  [OK]")
    print()
    return result


def test_single_simulation(scenario):
    print("[4/6] Testing POST /api/simulate/single ...")
    req_data = {
        "scenario": scenario,
        "seed": 42
    }
    result = make_request(f"{BASE_URL}/api/simulate/single", data=req_data, method="POST")
    
    print(f"  -> Total arrivals: {result['total_arrivals']}")
    print(f"  -> Completed: {result['total_completed']}")
    print(f"  -> Doctor utilization: {result['doctor_utilization']:.2%}")
    print(f"  -> Resus peak: {result['resuscitation_peak_occupancy']}")
    print(f"  -> Preemptions: {result['total_preemptions']}")
    print(f"  -> Elapsed: {result['elapsed_ms']}ms")
    
    assert len(result["patients"]) > 0
    has_preempted = any(p["was_preempted"] for p in result["patients"])
    print(f"  -> Has preempted patients: {has_preempted}")
    
    for esi in ["ESI_1", "ESI_2", "ESI_3", "ESI_4", "ESI_5"]:
        avg = result["avg_wait_time_by_esi"].get(esi, 0)
        print(f"    {esi}: avg wait = {avg:.1f} min")
    
    print("  [OK]")
    print()
    return result


def test_monte_carlo(scenario, n=100):
    print(f"[5/6] Testing POST /api/simulate/monte-carlo (N={n}) ...")
    req_data = {
        "scenario": scenario,
        "n": n,
        "base_seed": 42
    }
    result = make_request(f"{BASE_URL}/api/simulate/monte-carlo", data=req_data, method="POST")
    
    print(f"  -> Simulations: {result['num_simulations']}")
    print(f"  -> Avg doctor utilization: {result['avg_doctor_utilization']:.2%}")
    print(f"  -> CI: [{result['doctor_utilization_ci']['lower']:.2%}, {result['doctor_utilization_ci']['upper']:.2%}]")
    
    for esi in ["ESI_1", "ESI_2", "ESI_3", "ESI_4", "ESI_5"]:
        q = result["wait_time_quantiles"].get(esi, {})
        p50 = q.get("P50", 0)
        p95 = q.get("P95", 0)
        exceed = result["target_wait_probability"].get(esi, 0)
        print(f"    {esi}: P50={p50:.1f}min, P95={p95:.1f}min, exceed_prob={exceed:.1%}")
    
    print(f"  -> Total elapsed: {result['elapsed_ms']}ms")
    print(f"  -> Per simulation: {result['per_sim_ms']}ms")
    
    assert result["num_simulations"] == n
    print("  [OK]")
    print()
    return result


def test_compare(scenarios):
    print("[6/6] Testing POST /api/simulate/compare ...")
    req_data = {
        "scenarios": scenarios[:3],
        "n": 50,
        "base_seed": 42
    }
    result = make_request(f"{BASE_URL}/api/simulate/compare", data=req_data, method="POST")
    
    print(f"  -> Compared {len(result['scenario_names'])} scenarios")
    print(f"  -> Base seed: {result['base_seed']}")
    print(f"  -> Elapsed: {result['elapsed_ms']}ms")
    
    for metric in result["metrics"]:
        name = metric["scenario_name"]
        util = metric["doctor_utilization_mean"]
        print(f"    * {name}: util={util:.1%}")
    
    print(f"  -> Recommendation length: {len(result['recommendation'])} chars")
    print("  [OK]")
    print()
    return result


def test_report(scenario, n=100):
    print("[EXTRA] Testing POST /api/report/markdown ...")
    req_data = {
        "scenario": scenario,
        "monte_carlo_n": n,
        "seed": 42
    }
    import urllib.request
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(
        f"{BASE_URL}/api/report/markdown",
        data=json.dumps(req_data).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        content = resp.read().decode("utf-8")
    
    print(f"  -> Report length: {len(content)} chars")
    assert "# 急诊分诊仿真报告" in content
    assert "## 1. 输入参数" in content
    assert "## 4. 改进建议" in content
    print("  [OK]")
    print()
    return content


def test_reproducibility(scenario):
    print("[EXTRA] Testing reproducibility (same seed = same result) ...")
    
    req_data = {"scenario": scenario, "seed": 12345}
    r1 = make_request(f"{BASE_URL}/api/simulate/single", data=req_data, method="POST")
    r2 = make_request(f"{BASE_URL}/api/simulate/single", data=req_data, method="POST")
    
    assert r1["total_arrivals"] == r2["total_arrivals"]
    assert r1["total_completed"] == r2["total_completed"]
    assert r1["patients"][0]["arrival_time"] == r2["patients"][0]["arrival_time"]
    
    print(f"  -> Same seed gives same results: [OK]")
    print(f"    Arrivals: {r1['total_arrivals']} == {r2['total_arrivals']}")
    print()


def test_performance(scenario):
    print("[EXTRA] Testing N=1000 performance ...")
    import time
    req_data = {"scenario": scenario, "n": 1000, "base_seed": 42}
    
    t0 = time.time()
    result = make_request(f"{BASE_URL}/api/simulate/monte-carlo", data=req_data, method="POST")
    t1 = time.time()
    
    elapsed = t1 - t0
    print(f"  -> N=1000 elapsed: {elapsed:.2f} seconds")
    print(f"  -> Target: < 10 seconds [{elapsed < 10 and 'OK' or 'FAIL'}]")
    assert elapsed < 10
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("  医院急诊分诊仿真 API 接口测试")
    print("=" * 60)
    print()
    
    test_health()
    preset_scenarios = test_get_scenarios()
    custom_scenario = test_create_scenario()
    single_result = test_single_simulation(custom_scenario)
    
    test_reproducibility(custom_scenario)
    
    mc_result = test_monte_carlo(custom_scenario, n=100)
    compare_result = test_compare(preset_scenarios)
    report_content = test_report(custom_scenario, n=50)
    
    test_performance(custom_scenario)
    
    print("=" * 60)
    print("  所有测试通过! [OK]")
    print("=" * 60)
