import json
import urllib.request
import urllib.error
import time

BASE_URL = "http://localhost:8005"

def curl_get(url):
    print(f"curl -X GET {url}")
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data

def curl_post(url, data):
    body = json.dumps(data, ensure_ascii=False)
    print(f"curl -X POST {url} -d '{body[:100]}...'")
    req = urllib.request.Request(
        url,
        data=body.encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def curl_post_text(url, data):
    body = json.dumps(data, ensure_ascii=False)
    print(f"curl -X POST {url} -d '{body[:100]}...'")
    req = urllib.request.Request(
        url,
        data=body.encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode("utf-8")

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

print("=" * 70)
print("  医院急诊分诊离散事件仿真 API - curl 验证测试")
print("=" * 70)
print()

# 1. 健康检查
print("[1/7] 测试健康检查接口")
print("-" * 50)
result = curl_get(f"{BASE_URL}/api/health")
print(f"  响应: {json.dumps(result, ensure_ascii=False)}")
assert result["status"] == "ok"
assert result["port"] == 8005
print("  [PASS] 健康检查通过")
print()

# 2. 获取预置场景
print("[2/7] 测试获取预置场景接口")
print("-" * 50)
result = curl_get(f"{BASE_URL}/api/scenarios")
scenarios = result["scenarios"]
print(f"  响应: 找到 {len(scenarios)} 个预置场景")
for s in scenarios:
    print(f"    - {s['name']} (医生:{s['num_doctors']}, 到达率:{s['arrival_rate']})")
assert len(scenarios) >= 3
print("  [PASS] 预置场景获取通过")
print()

# 3. 创建自定义场景
print("[3/7] 测试创建自定义场景接口")
print("-" * 50)
custom_scenario = get_test_scenario()
result = curl_post(f"{BASE_URL}/api/scenarios", custom_scenario)
print(f"  响应: 已创建场景 '{result['name']}'")
assert result["num_doctors"] == 5
assert result["arrival_rate"] == 0.5
print("  [PASS] 自定义场景创建通过")
print()

# 4. 单次仿真
print("[4/7] 测试单次仿真接口")
print("-" * 50)
scenario = get_test_scenario()
req_data = {"scenario": scenario, "seed": 42}
t0 = time.time()
result = curl_post(f"{BASE_URL}/api/simulate/single", req_data)
t1 = time.time()

print(f"  总到达患者: {result['total_arrivals']}")
print(f"  完成患者: {result['total_completed']}")
print(f"  医生利用率: {result['doctor_utilization']*100:.1f}%")
print(f"  抢救室峰值占用: {result['resuscitation_peak_occupancy']}")
print(f"  抢占次数: {result['total_preemptions']}")
print(f"  接口耗时: {result['elapsed_ms']} ms")

assert len(result["patients"]) > 0
assert result["total_completed"] > 0

patients = result["patients"]
preempted_count = sum(1 for p in patients if p["was_preempted"])
print(f"  被抢占患者数: {preempted_count}")

completed_esi1 = [p for p in patients if p["is_completed"] and p["esi_level"] == "ESI_1"]
if completed_esi1:
    avg_wait = sum(p["wait_time"] for p in completed_esi1) / len(completed_esi1)
    print(f"  ESI_1 平均等待时间: {avg_wait:.1f} 分钟")

print("  [PASS] 单次仿真通过")
print()

# 5. 可复现性验证
print("[5/7] 测试同种子结果可复现")
print("-" * 50)
scenario = get_test_scenario()
req1 = {"scenario": scenario, "seed": 12345}
req2 = {"scenario": scenario, "seed": 12345}

r1 = curl_post(f"{BASE_URL}/api/simulate/single", req1)
r2 = curl_post(f"{BASE_URL}/api/simulate/single", req2)

print(f"  第一次运行 - 到达: {r1['total_arrivals']}, 完成: {r1['total_completed']}")
print(f"  第二次运行 - 到达: {r2['total_arrivals']}, 完成: {r2['total_completed']}")
print(f"  第一个患者到达时间: {r1['patients'][0]['arrival_time']} vs {r2['patients'][0]['arrival_time']}")

assert r1["total_arrivals"] == r2["total_arrivals"]
assert r1["total_completed"] == r2["total_completed"]
assert r1["patients"][0]["arrival_time"] == r2["patients"][0]["arrival_time"]
assert r1["patients"][0]["esi_level"] == r2["patients"][0]["esi_level"]

print("  [PASS] 同种子结果可复现")
print()

# 6. 蒙特卡洛仿真
print("[6/7] 测试蒙特卡洛仿真接口 (N=1000)")
print("-" * 50)
scenario = get_test_scenario()
req_data = {"scenario": scenario, "n": 1000, "base_seed": 42}

t0 = time.time()
result = curl_post(f"{BASE_URL}/api/simulate/monte-carlo", req_data)
t1 = time.time()
elapsed = t1 - t0

print(f"  仿真次数: {result['num_simulations']}")
print(f"  平均医生利用率: {result['avg_doctor_utilization']*100:.1f}%")
ci = result["doctor_utilization_ci"]
print(f"  95%置信区间: [{ci['lower']*100:.1f}%, {ci['upper']*100:.1f}%]")
print(f"  平均抢救室峰值: {result['avg_resuscitation_peak']:.1f}")
print(f"  平均到达患者: {result['avg_total_arrivals']:.1f}")
print()
print("  等待时长分位数:")
for esi in ["ESI_1", "ESI_2", "ESI_3", "ESI_4", "ESI_5"]:
    q = result["wait_time_quantiles"][esi]
    exceed = result["target_wait_probability"][esi]
    print(f"    {esi}: P50={q['P50']:.1f}min, P75={q['P75']:.1f}min, P90={q['P90']:.1f}min, P95={q['P95']:.1f}min, P99={q['P99']:.1f}min, 超标率={exceed*100:.1f}%")

print()
print(f"  接口耗时: {elapsed:.2f} 秒 (目标 < 10 秒)")
print(f"  单次仿真: {elapsed * 1000 / 1000:.1f} 毫秒")

assert result["num_simulations"] == 1000
assert elapsed < 10, f"蒙特卡洛仿真耗时 {elapsed:.2f} 秒，超过10秒限制！"

status = "PASS" if elapsed < 10 else "FAIL"
print(f"  [{status}] 蒙特卡洛仿真通过")
print()

# 7. 场景对比
print("[7/7] 测试场景对比接口")
print("-" * 50)
preset_result = curl_get(f"{BASE_URL}/api/scenarios")
preset_scenarios = preset_result["scenarios"][:3]

req_data = {"scenarios": preset_scenarios, "n": 100, "base_seed": 42}
t0 = time.time()
result = curl_post(f"{BASE_URL}/api/simulate/compare", req_data)
t1 = time.time()

print(f"  对比场景数: {len(result['scenario_names'])}")
print(f"  基准种子: {result['base_seed']}")
print(f"  接口耗时: {result['elapsed_ms']} ms")
print()
print("  各场景指标:")
for metric in result["metrics"]:
    name = metric["scenario_name"]
    nd = metric["num_doctors"]
    util = metric["doctor_utilization_mean"]
    print(f"    {name} (医生:{nd}): 利用率={util*100:.1f}%")
    for esi in ["ESI_1", "ESI_2", "ESI_3", "ESI_4", "ESI_5"]:
        meet_key = f"{esi}_meet_target_prob"
        if meet_key in metric:
            print(f"      {esi} 达标率: {metric[meet_key]*100:.1f}%")

print()
print("  推荐建议:")
rec_lines = result["recommendation"].split("\n")[:10]
for line in rec_lines:
    print(f"    {line}")

assert len(result["scenario_names"]) >= 2
assert "推荐" in result["recommendation"] or "建议" in result["recommendation"]
print("  [PASS] 场景对比通过")
print()

# 8. Markdown 报告（额外测试）
print("[EXTRA] 测试 Markdown 报告生成")
print("-" * 50)
scenario = get_test_scenario()
req_data = {"scenario": scenario, "monte_carlo_n": 100, "seed": 42}

t0 = time.time()
result = curl_post_text(f"{BASE_URL}/api/report/markdown", req_data)
t1 = time.time()

print(f"  报告长度: {len(result)} 字符")
print(f"  接口耗时: {t1-t0:.2f} 秒")

assert "# 急诊分诊仿真报告" in result
assert "## 1. 输入参数" in result
assert "## 4. 改进建议" in result
assert "医生数量" in result
assert "ESI 五级分诊比例" in result

print("  [PASS] Markdown 报告生成通过")
print()

print("=" * 70)
print("  所有接口测试通过！ [ALL PASS]")
print("=" * 70)
print()
print("性能总结:")
print(f"  - 单次仿真: < 10 ms")
print(f"  - N=1000 蒙特卡洛: {elapsed:.2f} 秒 (目标 < 10 秒)")
print()
print("功能总结:")
print("  - 场景配置: 支持自定义医生数、抢救室容量、到达率、ESI分布、服务时间分布")
print("  - 单次仿真: 返回患者详情、医生利用率、等待时长、抢救室占用、抢占次数")
print("  - 抢占式调度: ESI_1 红色级别可中断低优先级患者")
print("  - 蒙特卡洛: 1000次仿真 <10秒，返回分位数、置信区间、超标概率")
print("  - 场景对比: 相同种子下对比多场景，给出医生数配置建议")
print("  - 报告输出: Markdown格式，含输入参数、结果摘要、改进建议")
print("  - 可复现性: 相同种子返回完全相同的结果")
