from typing import Dict, List
from models import ESILevel, DistributionType, DistributionParams, ScenarioConfig


def get_default_esi_distribution() -> Dict[ESILevel, float]:
    return {
        ESILevel.ESI_1: 0.03,
        ESILevel.ESI_2: 0.08,
        ESILevel.ESI_3: 0.25,
        ESILevel.ESI_4: 0.35,
        ESILevel.ESI_5: 0.29,
    }


def get_default_service_params() -> Dict[ESILevel, DistributionParams]:
    return {
        ESILevel.ESI_1: DistributionParams(
            distribution_type=DistributionType.GAMMA,
            mean=45.0,
            std=20.0,
            shape=5.06,
            scale=8.9
        ),
        ESILevel.ESI_2: DistributionParams(
            distribution_type=DistributionType.GAMMA,
            mean=30.0,
            std=12.0,
            shape=6.25,
            scale=4.8
        ),
        ESILevel.ESI_3: DistributionParams(
            distribution_type=DistributionType.GAMMA,
            mean=20.0,
            std=8.0,
            shape=6.25,
            scale=3.2
        ),
        ESILevel.ESI_4: DistributionParams(
            distribution_type=DistributionType.GAMMA,
            mean=12.0,
            std=5.0,
            shape=5.76,
            scale=2.08
        ),
        ESILevel.ESI_5: DistributionParams(
            distribution_type=DistributionType.GAMMA,
            mean=8.0,
            std=3.0,
            shape=7.11,
            scale=1.125
        ),
    }


def get_default_target_wait() -> Dict[ESILevel, float]:
    return {
        ESILevel.ESI_1: 1.0,
        ESILevel.ESI_2: 10.0,
        ESILevel.ESI_3: 30.0,
        ESILevel.ESI_4: 60.0,
        ESILevel.ESI_5: 120.0,
    }


def get_standard_scenarios() -> List[Dict]:
    scenarios = []
    
    scenario1 = {
        "name": "标准场景 - 日间常规",
        "num_doctors": 5,
        "resuscitation_room_capacity": 8,
        "simulation_duration": 480.0,
        "arrival_rate": 0.5,
        "esi_distribution": get_default_esi_distribution(),
        "service_time_params": {k: v.model_dump() for k, v in get_default_service_params().items()},
        "target_wait_time": get_default_target_wait(),
    }
    scenarios.append(scenario1)
    
    scenario2 = {
        "name": "高峰场景 - 晚高峰",
        "num_doctors": 6,
        "resuscitation_room_capacity": 10,
        "simulation_duration": 480.0,
        "arrival_rate": 0.8,
        "esi_distribution": {
            ESILevel.ESI_1: 0.04,
            ESILevel.ESI_2: 0.12,
            ESILevel.ESI_3: 0.28,
            ESILevel.ESI_4: 0.32,
            ESILevel.ESI_5: 0.24,
        },
        "service_time_params": {k: v.model_dump() for k, v in get_default_service_params().items()},
        "target_wait_time": get_default_target_wait(),
    }
    scenarios.append(scenario2)
    
    scenario3 = {
        "name": "夜间场景 - 低负荷",
        "num_doctors": 3,
        "resuscitation_room_capacity": 6,
        "simulation_duration": 480.0,
        "arrival_rate": 0.2,
        "esi_distribution": {
            ESILevel.ESI_1: 0.02,
            ESILevel.ESI_2: 0.05,
            ESILevel.ESI_3: 0.20,
            ESILevel.ESI_4: 0.38,
            ESILevel.ESI_5: 0.35,
        },
        "service_time_params": {k: v.model_dump() for k, v in get_default_service_params().items()},
        "target_wait_time": get_default_target_wait(),
    }
    scenarios.append(scenario3)
    
    scenario4 = {
        "name": "资源紧张场景",
        "num_doctors": 4,
        "resuscitation_room_capacity": 6,
        "simulation_duration": 480.0,
        "arrival_rate": 0.6,
        "esi_distribution": {
            ESILevel.ESI_1: 0.05,
            ESILevel.ESI_2: 0.10,
            ESILevel.ESI_3: 0.30,
            ESILevel.ESI_4: 0.30,
            ESILevel.ESI_5: 0.25,
        },
        "service_time_params": {k: v.model_dump() for k, v in get_default_service_params().items()},
        "target_wait_time": get_default_target_wait(),
    }
    scenarios.append(scenario4)
    
    return scenarios


def scenario_dict_to_config(scenario_dict: Dict) -> ScenarioConfig:
    service_params = {}
    for k, v in scenario_dict["service_time_params"].items():
        if isinstance(v, dict):
            service_params[k] = DistributionParams(**v)
        else:
            service_params[k] = v
    
    return ScenarioConfig(
        name=scenario_dict["name"],
        num_doctors=scenario_dict["num_doctors"],
        resuscitation_room_capacity=scenario_dict["resuscitation_room_capacity"],
        simulation_duration=scenario_dict["simulation_duration"],
        arrival_rate=scenario_dict["arrival_rate"],
        esi_distribution=scenario_dict["esi_distribution"],
        service_time_params=service_params,
        target_wait_time=scenario_dict.get("target_wait_time"),
    )


def config_to_scenario_dict(config: ScenarioConfig) -> Dict:
    return {
        "name": config.name,
        "num_doctors": config.num_doctors,
        "resuscitation_room_capacity": config.resuscitation_room_capacity,
        "simulation_duration": config.simulation_duration,
        "arrival_rate": config.arrival_rate,
        "esi_distribution": config.esi_distribution,
        "service_time_params": {k: v.model_dump() for k, v in config.service_time_params.items()},
        "target_wait_time": config.target_wait_time,
    }
