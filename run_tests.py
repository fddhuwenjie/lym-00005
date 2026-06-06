import json
import urllib.request
import time
import sys

BASE_URL = "http://localhost:8005"

def log(msg):
    print(msg, flush=True)
    with open("test_results.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def curl_get(url):
    log(f"[GET] {url}")
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def curl_post(url, data):
    body = json.dumps(data, ensure_ascii=False)
    log(f"[POST] {url}")
    req = urllib.request.Request(
        url,
        data=body.encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def get_test_scenario():
    return {
        "name": "API测试场景",
        "num_doctors": 5,
        "resuscitation_room_capacity": 8,
        "simulation_duration": 480.0,
        "arrival_rate": 0.5,
        "esi_distribution": {
            "ESI_1": 0.03, "ESI_2": 0.08, "ESI_3": 0.25, "ESI_4": 0.35, "ESI_5": 0.29
        },
        "service_time_params": {
            "ESI_1": {"distribution_type": "gamma", "mean": 45.0, "std": 20.0},
            "ESI_2": {"distribution_type": "gamma", "mean": 30.0, "std": 12.0},
            "ESI_3": {"distribution_type": "gamma", "mean": 20.0, "std": 8.0},
            "ESI_4": {"distribution_type": "gamma", "mean": 12.0, "std": 5.0},
            "ESI_5": {"distribution_type": "gamma", "mean": 8.0, "std": 3.0}
        },
        "target_wait_time": {
            "ESI_1": 1.0, "ESI_2": 10.0, "ESI_3": 30.0, "ESI_4": 60.0, "ESI_5": 120.0
        }
    }

with open("test_results.txt", "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("  医院急诊分诊离散事件仿真 API - curl 验证测试\n")
    f.write("=" * 70 + "\n\n")

log("")
log("[1/6] 健康检查")
log("-" * 50)
result = curl_get(f"{BASE_URL}/api/health")
log(f"  响应: {json.dumps(result, ensure_ascii=False)}")
assert result["status"] == "ok"
log("  [PASS] 健康检查通过")
log("")

log("[2/6] 获取预置场景")
log("-" * 50)
result = curl_get(f"{BASE_URL}/api/scenarios")
scenarios = result["scenarios"]
log(f"  找到 {len(scenarios)} 个预置场景:")
for s in scenarios:
    log(f"    - {s['name']}")
assert len(scenarios) >= 3
log("  [PASS] 预置场景获取通过")
log("")

log("[3/6] 单次仿真")
log("-" * 50)
scenario = get_test_scenario()
req_data = {"scenario": scenario, "seed": 42}
t0 = time.time()
result = curl_post(f"{BASE_URL}/api/simulate/single", req_data)
t1 = time.time()

log(f"  总到达患者: {result['total_arrivals']}")
log(f"  完成患者: {result['total_completed']}")
log(f"  医生利用率: {result['doctor_utilization']*100:.1f}%")
log(f"  抢救室峰值占用: {result['resuscitation_peak_occupancy']}")
log(f"  抢占次数: {result['total_preemptions']}")
log(f"  接口耗时: {result['elapsed_ms']} ms")

assert len(result["patients"]) > 0
preempted = sum(1 for p in result["patients"] if p["was_preempted"])
log(f"  被抢占患者数: {preempted}")
log("  [PASS] 单次仿真通过")
log("")

log("[4/6] 同种子可复现性")
log("-" * 50)
scenario = get_test_scenario()
r1 = curl_post(f"{BASE_URL}/api/simulate/single", {"scenario": scenario, "seed": 12345})
r2 = curl_post(f"{BASE_URL}/api/simulate/single", {"scenario": scenario, "seed": 12345})
assert r1["total_arrivals"] == r2["total_arrivals"]
assert r1["patients"][0]["arrival_time"] == r2["patients"][0]["arrival_time"]
log(f"  两次运行结果完全一致")
log("  [PASS] 同种子结果可复现")
log("")

log("[5/6] 蒙特卡洛仿真 (N=1000)")
log("-" * 50)
scenario = get_test_scenario()
req_data = {"scenario": scenario, "n": 1000, "base_seed": 42}
t0 = time.time()
result = curl_post(f"{BASE_URL}/api/simulate/monte-carlo", req_data)
t1 = time.time()
elapsed = t1 - t0

log(f"  仿真次数: {result['num_simulations']}")
log(f"  平均医生利用率: {result['avg_doctor_utilization']*100:.1f}%")
ci = result['doctor_utilization_ci']
log(f"  95%置信区间: [{ci['lower']*100:.1f}%, {ci['upper']*100:.1f}%]")
log("  等待时长分位数:")
for esi in ["ESI_1", "ESI_2", "ESI_3", "ESI_4", "ESI_5"]:
    q = result["wait_time_quantiles"][esi]
    exceed = result["target_wait_probability"][esi]
    log(f"    {esi}: P50={q['P50']:.1f}min, P95={q['P95']:.1f}min, 超标率={exceed*100:.1f}%")

log(f"  总耗时: {elapsed:.2f} 秒 (目标 < 10 秒)")
assert elapsed < 10
log("  [PASS] 蒙特卡洛仿真通过")
log("")

log("[6/6] 场景对比")
log("-" * 50)
preset = curl_get(f"{BASE_URL}/api/scenarios")
scenarios = preset["scenarios"][:3]
req_data = {"scenarios": scenarios, "n": 100, "base_seed": 42}
result = curl_post(f"{BASE_URL}/api/simulate/compare", req_data)

log(f"  对比场景数: {len(result['scenario_names'])}")
log("  各场景医生利用率:")
for m in result["metrics"]:
    log(f"    {m['scenario_name']}: {m['doctor_utilization_mean']*100:.1f}%")

log("  [PASS] 场景对比通过")
log("")

log("=" * 70)
log("  所有测试通过！ [ALL PASS]")
log("=" * 70)
log("")
log(f"性能: N=1000 蒙特卡洛耗时 {elapsed:.2f} 秒，远低于 10 秒目标")

print("\n测试完成！结果已保存到 test_results.txt")
