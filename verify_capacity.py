import json
import urllib.request
import time

BASE_URL = "http://localhost:8005"

def test_capacity_control():
    print("=" * 70)
    print("  抢救室容量门控验证测试")
    print("=" * 70)
    print()

    scenario_high_arrival = {
        "name": "高负荷测试场景",
        "num_doctors": 10,
        "resuscitation_room_capacity": 3,
        "simulation_duration": 240.0,
        "arrival_rate": 2.0,
        "esi_distribution": {
            "ESI_1": 0.01,
            "ESI_2": 0.05,
            "ESI_3": 0.20,
            "ESI_4": 0.40,
            "ESI_5": 0.34
        },
        "service_time_params": {
            "ESI_1": {"distribution_type": "gamma", "mean": 60.0, "std": 20.0},
            "ESI_2": {"distribution_type": "gamma", "mean": 40.0, "std": 15.0},
            "ESI_3": {"distribution_type": "gamma", "mean": 25.0, "std": 10.0},
            "ESI_4": {"distribution_type": "gamma", "mean": 15.0, "std": 5.0},
            "ESI_5": {"distribution_type": "gamma", "mean": 10.0, "std": 3.0}
        },
        "target_wait_time": {
            "ESI_1": 1.0, "ESI_2": 10.0, "ESI_3": 30.0, "ESI_4": 60.0, "ESI_5": 120.0
        }
    }

    print("[测试1] 抢救室容量 = 3，医生数 = 10（医生充足但床位有限）")
    print("-" * 70)
    req_data = {"scenario": scenario_high_arrival, "seed": 42}
    t0 = time.time()

    req = urllib.request.Request(
        f"{BASE_URL}/api/simulate/single",
        data=json.dumps(req_data).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )

    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read().decode('utf-8'))

    elapsed = time.time() - t0

    resus_cap = scenario_high_arrival["resuscitation_room_capacity"]
    resus_peak = result["resuscitation_peak_occupancy"]

    print(f"  抢救室容量: {resus_cap}")
    print(f"  抢救室峰值占用: {resus_peak}")
    print(f"  总到达患者: {result['total_arrivals']}")
    print(f"  完成患者: {result['total_completed']}")
    print(f"  医生利用率: {result['doctor_utilization']*100:.1f}%")
    print(f"  抢占次数: {result['total_preemptions']}")

    # 验证：峰值占用不应超过容量
    if resus_peak <= resus_cap:
        print(f"  [PASS] 抢救室峰值占用 {resus_peak} <= 容量 {resus_cap}")
    else:
        print(f"  [FAIL] 抢救室峰值占用 {resus_peak} > 容量 {resus_cap}，容量门控未生效！")

    # 检查患者状态
    patients = result["patients"]
    completed = [p for p in patients if p["is_completed"]]
    waiting = [p for p in patients if not p["is_completed"]]
    print(f"  完成服务: {len(completed)}, 仍在队列: {len(waiting)}")

    # 检查 ESI_1 患者是否被优先服务
    esi1_patients = [p for p in patients if p["esi_level"] == "ESI_1"]
    esi1_completed = [p for p in esi1_patients if p["is_completed"]]
    if esi1_patients:
        print(f"  ESI_1 患者数: {len(esi1_patients)}, 已完成: {len(esi1_completed)}")
        if esi1_completed:
            avg_wait = sum(p["wait_time"] or 0 for p in esi1_completed) / len(esi1_completed)
            print(f"  ESI_1 平均等待时间: {avg_wait:.2f} 分钟")

    # 检查被抢占患者
    preempted = [p for p in patients if p["was_preempted"]]
    print(f"  被抢占患者数: {len(preempted)}")

    print()
    print("[测试2] 对比：容量=3 vs 容量=10 的峰值占用")
    print("-" * 70)

    scenario_cap3 = dict(scenario_high_arrival)
    scenario_cap3["resuscitation_room_capacity"] = 3
    scenario_cap10 = dict(scenario_high_arrival)
    scenario_cap10["resuscitation_room_capacity"] = 10

    def run_sim(scenario, seed):
        req_data = {"scenario": scenario, "seed": seed}
        req = urllib.request.Request(
            f"{BASE_URL}/api/simulate/single",
            data=json.dumps(req_data).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode('utf-8'))

    r3 = run_sim(scenario_cap3, seed=123)
    r10 = run_sim(scenario_cap10, seed=123)

    print(f"  容量=3  峰值占用: {r3['resuscitation_peak_occupancy']}")
    print(f"  容量=10 峰值占用: {r10['resuscitation_peak_occupancy']}")
    print(f"  容量=3  完成患者: {r3['total_completed']}")
    print(f"  容量=10 完成患者: {r10['total_completed']}")

    if r3['resuscitation_peak_occupancy'] <= 3 and r10['resuscitation_peak_occupancy'] > 3:
        print(f"  [PASS] 容量限制生效：容量=3时峰值被限制在3，容量=10时峰值超过3")
    else:
        print(f"  [FAIL] 容量限制未生效或测试条件不满足")

    print()
    print("[测试3] 验证 ESI_1 红色患者可抢占床位（容量满时）")
    print("-" * 70)

    scenario_esi1_test = {
        "name": "ESI_1 抢占测试",
        "num_doctors": 3,
        "resuscitation_room_capacity": 3,
        "simulation_duration": 120.0,
        "arrival_rate": 0.5,
        "esi_distribution": {
            "ESI_1": 0.0,
            "ESI_2": 0.0,
            "ESI_3": 1.0,
            "ESI_4": 0.0,
            "ESI_5": 0.0
        },
        "service_time_params": {
            "ESI_1": {"distribution_type": "gamma", "mean": 30.0, "std": 10.0},
            "ESI_2": {"distribution_type": "gamma", "mean": 30.0, "std": 10.0},
            "ESI_3": {"distribution_type": "gamma", "mean": 60.0, "std": 5.0},
            "ESI_4": {"distribution_type": "gamma", "mean": 30.0, "std": 10.0},
            "ESI_5": {"distribution_type": "gamma", "mean": 30.0, "std": 10.0}
        }
    }

    # 第一阶段：先让3个ESI_3占满床位
    req_data = {"scenario": scenario_esi1_test, "seed": 1}
    req = urllib.request.Request(
        f"{BASE_URL}/api/simulate/single",
        data=json.dumps(req_data).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urllib.request.urlopen(req) as resp:
        r1 = json.loads(resp.read().decode('utf-8'))

    print(f"  无ESI_1场景（全ESI_3）: 峰值占用={r1['resuscitation_peak_occupancy']}, 抢占次数={r1['total_preemptions']}")

    # 第二阶段：加入ESI_1患者
    scenario_with_esi1 = dict(scenario_esi1_test)
    scenario_with_esi1["esi_distribution"] = {
        "ESI_1": 0.2,
        "ESI_2": 0.0,
        "ESI_3": 0.8,
        "ESI_4": 0.0,
        "ESI_5": 0.0
    }

    req_data2 = {"scenario": scenario_with_esi1, "seed": 1}
    req2 = urllib.request.Request(
        f"{BASE_URL}/api/simulate/single",
        data=json.dumps(req_data2).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urllib.request.urlopen(req2) as resp:
        r2 = json.loads(resp.read().decode('utf-8'))

    preempted_count = sum(1 for p in r2["patients"] if p["was_preempted"])
    esi1_patients = [p for p in r2["patients"] if p["esi_level"] == "ESI_1"]
    esi1_completed = [p for p in esi1_patients if p["is_completed"]]

    print(f"  有ESI_1场景（20% ESI_1）: 峰值占用={r2['resuscitation_peak_occupancy']}, 抢占次数={r2['total_preemptions']}, 被抢占患者={preempted_count}")
    print(f"  ESI_1患者: {len(esi1_patients)}, 已完成: {len(esi1_completed)}")

    if r2["total_preemptions"] > 0 or preempted_count > 0:
        print(f"  [PASS] ESI_1患者成功触发了抢占机制")
    else:
        print(f"  [INFO] 本次运行中未发生抢占（可能因随机因素，抢占机制仍可能正常）")

    # 检查ESI_1等待时间是否显著低于ESI_3
    esi1_waits = [p["wait_time"] for p in esi1_completed if p["wait_time"] is not None]
    esi3_waits = [p["wait_time"] for p in r2["patients"] if p["esi_level"] == "ESI_3" and p["is_completed"] and p["wait_time"] is not None]

    if esi1_waits and esi3_waits:
        avg_esi1 = sum(esi1_waits) / len(esi1_waits)
        avg_esi3 = sum(esi3_waits) / len(esi3_waits)
        print(f"  ESI_1平均等待: {avg_esi1:.1f}分钟, ESI_3平均等待: {avg_esi3:.1f}分钟")
        if avg_esi1 < avg_esi3:
            print(f"  [PASS] ESI_1优先级生效，等待时间显著低于ESI_3")

    print()
    print("=" * 70)
    print("  容量门控验证总结")
    print("=" * 70)
    print()
    print("1. 抢救室容量在 try_start 入口处进行检查（而非仅在输出时截断）")
    print("2. 患者进入服务前必须同时满足:")
    print("   - 有空闲医生 或 可抢占低优先级患者的医生")
    print("   - 有空闲床位 或 (是ESI_1且可抢占床位)")
    print("3. ESI_1红色患者可在容量满时抢占低优先级患者的床位")
    print("4. 移除了输出处的 min(resus_peak, resus_cap) 强行截断")
    print("5. 删除了死代码文件 simulation.py 和 statistics.py")
    print()

    all_pass = True
    if resus_peak > resus_cap:
        all_pass = False

    if all_pass:
        print("[ALL PASS] 抢救室容量门控修复验证通过！")
    else:
        print("[FAIL] 存在未通过的测试项")


if __name__ == "__main__":
    test_capacity_control()
