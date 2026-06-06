$baseUrl = "http://localhost:8005"
$headers = @{"Content-Type" = "application/json"}

function Invoke-API-Get($endpoint) {
    try {
        $url = "$baseUrl$endpoint"
        $response = Invoke-WebRequest -Uri $url -Method Get -Headers $headers -UseBasicParsing
        return $response.Content | ConvertFrom-Json
    } catch {
        Write-Host "Error: $_"
        if ($_.Exception.Response) {
            $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
            Write-Host "Response: $($reader.ReadToEnd())"
        }
        return $null
    }
}

function Invoke-API-Post($endpoint, $data) {
    try {
        $url = "$baseUrl$endpoint"
        $body = $data | ConvertTo-Json -Depth 10
        $response = Invoke-WebRequest -Uri $url -Method Post -Headers $headers -Body $body -UseBasicParsing
        return $response.Content | ConvertFrom-Json
    } catch {
        Write-Host "Error: $_"
        if ($_.Exception.Response) {
            $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
            Write-Host "Response: $($reader.ReadToEnd())"
        }
        return $null
    }
}

Write-Host "=" * 60
Write-Host "测试 1: 健康检查"
Write-Host "=" * 60
$result = Invoke-API-Get "/api/health"
if ($result) {
    Write-Host "响应: $($result | ConvertTo-Json -Depth 2)"
}
Write-Host ""

Write-Host "=" * 60
Write-Host "测试 2: 获取预置场景"
Write-Host "=" * 60
$result = Invoke-API-Get "/api/scenarios"
if ($result) {
    $first = $result[0]
    Write-Host "场景数量: $($result.Length)"
    $props = $first | Get-Member -MemberType NoteProperty | Select-Object -ExpandProperty Name
    Write-Host "热身期字段: $($props -contains 'warmup_duration')"
    Write-Host "冷却期字段: $($props -contains 'cooldown_duration')"
    Write-Host "升级阈值字段: $($props -contains 'escalation_thresholds')"
    Write-Host "时变到达率字段: $($props -contains 'arrival_rate_schedule')"
    Write-Host "耐心上限字段: $($props -contains 'patience_limit')"
    if ($props -contains 'warmup_duration') {
        Write-Host "热身期默认值: $($first.warmup_duration)"
        Write-Host "冷却期默认值: $($first.cooldown_duration)"
    }
}
Write-Host ""

Write-Host "=" * 60
Write-Host "测试 3: 创建带所有新功能的场景"
Write-Host "=" * 60
$testScenario = @{
    name = "综合功能测试场景"
    num_doctors = 3
    resuscitation_room_capacity = 2
    simulation_duration = 480
    arrival_rate = 1.5
    warmup_duration = 60
    cooldown_duration = 60
    esi_distribution = @{
        ESI_1 = 0.05
        ESI_2 = 0.15
        ESI_3 = 0.40
        ESI_4 = 0.25
        ESI_5 = 0.15
    }
    service_time_params = @{
        ESI_1 = @{type = "normal"; mean = 30; std = 10}
        ESI_2 = @{type = "normal"; mean = 25; std = 8}
        ESI_3 = @{type = "normal"; mean = 20; std = 6}
        ESI_4 = @{type = "normal"; mean = 15; std = 5}
        ESI_5 = @{type = "normal"; mean = 10; std = 3}
    }
    target_wait_time = @{
        ESI_1 = 1
        ESI_2 = 10
        ESI_3 = 30
        ESI_4 = 60
        ESI_5 = 90
    }
    escalation_thresholds = @{
        ESI_5 = 20
        ESI_4 = 15
        ESI_3 = 10
        ESI_2 = 5
    }
    arrival_rate_schedule = @(
        @{start_time = 0; end_time = 120; arrival_rate = 1.0}
        @{start_time = 120; end_time = 240; arrival_rate = 1.5}
        @{start_time = 240; end_time = 360; arrival_rate = 2.0}
        @{start_time = 360; end_time = 480; arrival_rate = 1.2}
    )
    patience_limit = @{
        ESI_1 = 99999
        ESI_2 = 99999
        ESI_3 = 60
        ESI_4 = 45
        ESI_5 = 30
    }
}
$result = Invoke-API-Post "/api/scenarios" $testScenario
if ($result) {
    Write-Host "创建成功: $($result.name)"
    Write-Host "热身期: $($result.warmup_duration)"
    Write-Host "冷却期: $($result.cooldown_duration)"
    Write-Host "升级阈值: $($result.escalation_thresholds | ConvertTo-Json -Compress)"
    Write-Host "时变到达率时段数: $($result.arrival_rate_schedule.Length)"
    Write-Host "耐心上限: $($result.patience_limit | ConvertTo-Json -Compress)"
}
Write-Host ""

Write-Host "=" * 60
Write-Host "测试 4: 单次仿真"
Write-Host "=" * 60
$simRequest = @{scenario = $testScenario}
$result = Invoke-API-Post "/api/simulate/single" $simRequest
if ($result) {
    Write-Host "稳态区间: $($result.steady_state_start) - $($result.steady_state_end)"
    Write-Host "稳态样本量: $($result.steady_state_patients)"
    Write-Host "队列内升级总次数: $($result.total_escalations)"
    Write-Host "升级入流量: $($result.escalation_in | ConvertTo-Json -Compress)"
    Write-Host "升级出流量: $($result.escalation_out | ConvertTo-Json -Compress)"
    Write-Host "LWBS总数: $($result.total_lwbs)"
    Write-Host "各级别LWBS率: $($result.lwbs_rates | ConvertTo-Json -Compress)"
    Write-Host "等待时长P50/P95: $($result.wait_time_quantiles | ConvertTo-Json -Compress)"
}
Write-Host ""

Write-Host "=" * 60
Write-Host "测试 5: 蒙特卡洛仿真 (N=100, 快速测试)"
Write-Host "=" * 60
$mcRequest = @{scenario = $testScenario; num_runs = 100}
$result = Invoke-API-Post "/api/simulate/monte-carlo" $mcRequest
if ($result) {
    Write-Host "稳态区间: $($result.steady_state_start) - $($result.steady_state_end)"
    Write-Host "平均稳态样本量: $($result.avg_steady_state_patients)"
    Write-Host "平均队列升级次数: $($result.avg_total_escalations)"
    Write-Host "平均LWBS总数: $($result.avg_total_lwbs)"
    Write-Host "时段切片数: $($result.time_slices.Length)"
    if ($result.time_slices -and $result.time_slices.Length -gt 0) {
        Write-Host "最拥堵时段: $($result.most_congested_time)"
    }
}
Write-Host ""

Write-Host "=" * 60
Write-Host "测试 6: 蒙特卡洛性能测试 (N=1000)"
Write-Host "=" * 60
$mcRequest = @{scenario = $testScenario; num_runs = 1000}
$startTime = Get-Date
$result = Invoke-API-Post "/api/simulate/monte-carlo" $mcRequest
$elapsed = (Get-Date) - $startTime
if ($result) {
    Write-Host "蒙特卡洛N=1000耗时: $($elapsed.TotalSeconds.ToString('F2'))秒"
    Write-Host "是否达标 (`<=10秒): $($elapsed.TotalSeconds -le 10 ? '是' : '否')"
}
Write-Host ""

Write-Host "=" * 60
Write-Host "测试 7: 经济配置搜索"
Write-Host "=" * 60
$searchRequest = @{
    base_scenario = $testScenario
    max_doctors = 6
    max_resuscitation_rooms = 4
    targets = @{
        ESI_2 = @{wait_time_p95 = 10; type = "wait_time"}
        any = @{utilization = 0.92; type = "utilization"}
    }
}
$result = Invoke-API-Post "/api/capacity/search" $searchRequest
if ($result) {
    Write-Host "可行配置数: $($result.feasible_count)"
    Write-Host "候选配置数: $($result.top_candidates.Length)"
    $i = 1
    foreach ($candidate in $result.top_candidates) {
        Write-Host "  候选 $($i): 医生=$($candidate.config.num_doctors), 抢救室=$($candidate.config.resuscitation_room_capacity), 成本=$($candidate.cost)"
        Write-Host "    ESI_2 P95等待: $($candidate.metrics.wait_time_p95_by_level.ESI_2)"
        Write-Host "    医生利用率: $($candidate.metrics.doctor_utilization)"
        $i++
    }
}
Write-Host ""

Write-Host "=" * 60
Write-Host "所有测试完成!"
Write-Host "=" * 60
