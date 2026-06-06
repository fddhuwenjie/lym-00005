import urllib.request
import urllib.error
import json
import time

BASE_URL = "http://localhost:8005"

def api_get(endpoint):
    try:
        url = f"{BASE_URL}{endpoint}"
        req = urllib.request.Request(url, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.read().decode('utf-8')}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def api_post(endpoint, data):
    try:
        url = f"{BASE_URL}{endpoint}"
        json_data = json.dumps(data).encode('utf-8')
        req = urllib.request.Request(url, data=json_data, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.read().decode('utf-8')}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

print("=" * 60)
print("测试 1: 健康检查")
print("=" * 60)
result = api_get("/api/health")
print(f"响应: {json.dumps(result, ensure_ascii=False, indent=2)}")
print()

print("=" * 60)
print("测试 2: 获取预置场景")
print("=" * 60)
result = api_get("/api/scenarios")
if result:
    first = result[0] if result else {}
    print(f"场景数量: {len(result)}")
    print(f"热身期字段: {'warmup_duration' in first}")
    print(f"冷却期字段: {'cooldown_duration' in first}")
    print(f"升级阈值字段: {'escalation_thresholds' in first}")
    print(f"时变到达率字段: {'arrival_rate_schedule' in first}")
    print(f"耐心上限字段: {'patience_limit' in first}")
    if 'warmup_duration' in first:
        print(f"热身期默认值: {first['warmup_duration']}")
        print(f"冷却期默认值: {first['cooldown_duration']}")
print()

print("=" * 60)
print("测试 3: 创建带所有新功能的场景")
print("=" * 60)
test_scenario = {
    "name": "综合功能测试场景",
    "num_doctors": 3,
    "resuscitation_room_capacity": 2,
    "simulation_duration": 480,
    "arrival_rate": 1.5,
    "warmup_duration": 60,
    "cooldown_duration": 60,
    "esi_distribution": {
        "ESI_1": 0.05,
        "ESI_2": 0.15,
        "ESI_3": 0.40,
        "ESI_4": 0.25,
        "ESI_5": 0.15
    },
    "service_time_params": {
        "ESI_1": {"type": "normal", "mean": 30, "std": 10},
        "ESI_2": {"type": "normal", "mean": 25, "std": 8},
        "ESI_3": {"type": "normal", "mean": 20, "std": 6},
        "ESI_4": {"type": "normal", "mean": 15, "std": 5},
        "ESI_5": {"type": "normal", "mean": 10, "std": 3}
    },
    "target_wait_time": {
        "ESI_1": 1,
        "ESI_2": 10,
        "ESI_3": 30,
        "ESI_4": 60,
        "ESI_5": 90
    },
    "escalation_thresholds": {
        "ESI_5": 20,
        "ESI_4": 15,
        "ESI_3": 10,
        "ESI_2": 5
    },
    "arrival_rate_schedule": [
        {"start_time": 0, "end_time": 120, "arrival_rate": 1.0},
        {"start_time": 120, "end_time": 240, "arrival_rate": 1.5},
        {"start_time": 240, "end_time": 360, "arrival_rate": 2.0},
        {"start_time": 360, "end_time": 480, "arrival_rate": 1.2}
    ],
    "patience_limit": {
        "ESI_1": 99999,
        "ESI_2": 99999,
        "ESI_3": 60,
        "ESI_4": 45,
        "ESI_5": 30
    }
}
result = api_post("/api/scenarios", test_scenario)
if result:
    print(f"创建成功: {result.get('name')}")
    print(f"热身期: {result.get('warmup_duration')}")
    print(f"冷却期: {result.get('cooldown_duration')}")
    print(f"升级阈值: {result.get('escalation_thresholds')}")
    print(f"时变到达率时段数: {len(result.get('arrival_rate_schedule', []))}")
    print(f"耐心上限: {result.get('patience_limit')}")
print()

print("=" * 60)
print("测试 4: 单次仿真")
print("=" * 60)
sim_request = {"scenario": test_scenario}
result = api_post("/api/simulate/single", sim_request)
if result:
    print(f"稳态区间: {result.get('steady_state_start')} - {result.get('steady_state_end')}")
    print(f"稳态样本量: {result.get('steady_state_patients')}")
    print(f"队列内升级总次数: {result.get('total_escalations')}")
    print(f"升级入流量: {result.get('escalation_in')}")
    print(f"升级出流量: {result.get('escalation_out')}")
    print(f"LWBS总数: {result.get('total_lwbs')}")
    print(f"各级别LWBS率: {result.get('lwbs_rates')}")
    wait_quantiles = result.get('wait_time_quantiles', {})
    print(f"等待时长P50/P95: {wait_quantiles}")
print()

print("=" * 60)
print("测试 5: 蒙特卡洛仿真 (N=100, 快速测试)")
print("=" * 60)
mc_request = {"scenario": test_scenario, "num_runs": 100}
result = api_post("/api/simulate/monte-carlo", mc_request)
if result:
    print(f"稳态区间: {result.get('steady_state_start')} - {result.get('steady_state_end')}")
    print(f"平均稳态样本量: {result.get('avg_steady_state_patients')}")
    print(f"平均队列升级次数: {result.get('avg_total_escalations')}")
    print(f"平均LWBS总数: {result.get('avg_total_lwbs')}")
    print(f"时段切片数: {len(result.get('time_slices', []))}")
    if result.get('time_slices'):
        print(f"最拥堵时段: {result.get('most_congested_time')}")
print()

print("=" * 60)
print("测试 6: 蒙特卡洛性能测试 (N=1000)")
print("=" * 60)
mc_request = {"scenario": test_scenario, "num_runs": 1000}
start_time = time.time()
result = api_post("/api/simulate/monte-carlo", mc_request)
elapsed = time.time() - start_time
if result:
    print(f"蒙特卡洛N=1000耗时: {elapsed:.2f}秒")
    print(f"是否达标 (<=10秒): {'是' if elapsed <= 10 else '否'}")
print()

print("=" * 60)
print("测试 7: 经济配置搜索")
print("=" * 60)
search_request = {
    "base_scenario": test_scenario,
    "max_doctors": 6,
    "max_resuscitation_rooms": 4,
    "targets": {
        "ESI_2": {"wait_time_p95": 10, "type": "wait_time"},
        "any": {"utilization": 0.92, "type": "utilization"}
    }
}
result = api_post("/api/capacity/search", search_request)
if result:
    print(f"可行配置数: {result.get('feasible_count')}")
    print(f"候选配置数: {len(result.get('top_candidates', []))}")
    for i, candidate in enumerate(result.get('top_candidates', []), 1):
        print(f"  候选 {i}: 医生={candidate['config']['num_doctors']}, 抢救室={candidate['config']['resuscitation_room_capacity']}, 成本={candidate['cost']}")
        print(f"    ESI_2 P95等待: {candidate['metrics']['wait_time_p95_by_level'].get('ESI_2', 'N/A')}")
        print(f"    医生利用率: {candidate['metrics'].get('doctor_utilization', 'N/A')}")
print()

print("=" * 60)
print("所有测试完成!")
print("=" * 60)
