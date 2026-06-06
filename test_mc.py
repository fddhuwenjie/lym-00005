import json
import urllib.request
import time

with open('mc_request.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

t0 = time.time()
req = urllib.request.Request(
    'http://localhost:8005/api/simulate/monte-carlo',
    data=json.dumps(data).encode('utf-8'),
    headers={'Content-Type': 'application/json'},
    method='POST'
)

with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read().decode('utf-8'))

elapsed = time.time() - t0
print('=== 蒙特卡洛测试 (N=1000) ===')
print(f'仿真次数: {result["num_simulations"]}')
print(f'平均医生利用率: {result["avg_doctor_utilization"]*100:.1f}%')
ci = result['doctor_utilization_ci']
print(f'95% CI: [{ci["lower"]*100:.1f}%, {ci["upper"]*100:.1f}%]')
print(f'总耗时: {result["elapsed_ms"]} ms ({elapsed:.2f}s)')
print(f'单次仿真: {result["per_sim_ms"]} ms')
print()
print('等待时长分位数:')
for esi in ['ESI_1','ESI_2','ESI_3','ESI_4','ESI_5']:
    q = result['wait_time_quantiles'][esi]
    exceed = result['target_wait_probability'][esi]
    print(f'  {esi}: P50={q["P50"]:.1f}min, P95={q["P95"]:.1f}min, 超标率={exceed*100:.1f}%')
print()
status = "OK" if elapsed < 10 else "FAIL"
print(f'目标: <10秒 [{status}]')
