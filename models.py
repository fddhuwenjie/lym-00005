from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field
from enum import Enum


class ESILevel(str, Enum):
    ESI_1 = "ESI_1"
    ESI_2 = "ESI_2"
    ESI_3 = "ESI_3"
    ESI_4 = "ESI_4"
    ESI_5 = "ESI_5"


ESI_PRIORITY = {
    ESILevel.ESI_1: 1,
    ESILevel.ESI_2: 2,
    ESILevel.ESI_3: 3,
    ESILevel.ESI_4: 4,
    ESILevel.ESI_5: 5,
}

ESI_NAMES = {
    ESILevel.ESI_1: "红色-立即抢救",
    ESILevel.ESI_2: "橙色-紧急",
    ESILevel.ESI_3: "黄色-紧急",
    ESILevel.ESI_4: "绿色-非紧急",
    ESILevel.ESI_5: "蓝色-非紧急",
}


class DistributionType(str, Enum):
    EXPONENTIAL = "exponential"
    NORMAL = "normal"
    UNIFORM = "uniform"
    GAMMA = "gamma"


class DistributionParams(BaseModel):
    distribution_type: DistributionType = DistributionType.GAMMA
    mean: float = Field(gt=0)
    std: Optional[float] = Field(None, gt=0)
    shape: Optional[float] = Field(None, gt=0)
    scale: Optional[float] = Field(None, gt=0)


class ScenarioConfig(BaseModel):
    name: str
    num_doctors: int = Field(ge=1)
    resuscitation_room_capacity: int = Field(ge=1)
    simulation_duration: float = Field(gt=0)
    arrival_rate: float = Field(gt=0)
    esi_distribution: Dict[ESILevel, float]
    service_time_params: Dict[ESILevel, DistributionParams]
    target_wait_time: Optional[Dict[ESILevel, float]] = None


class EventType(str, Enum):
    ARRIVAL = "arrival"
    START_SERVICE = "start_service"
    END_SERVICE = "end_service"
    PREEMPTION = "preemption"


class Patient(BaseModel):
    patient_id: int
    esi_level: ESILevel
    arrival_time: float
    service_start_time: Optional[float] = None
    service_end_time: Optional[float] = None
    wait_time: Optional[float] = None
    total_service_time: Optional[float] = None
    was_preempted: bool = False
    preemption_count: int = 0
    remaining_service_time: Optional[float] = None
    is_completed: bool = False


class Event(BaseModel):
    event_type: EventType
    event_time: float
    patient_id: Optional[int] = None
    doctor_id: Optional[int] = None


class SingleSimulationResult(BaseModel):
    patients: List[Patient]
    doctor_utilization: float
    avg_wait_time_by_esi: Dict[ESILevel, float]
    max_wait_time_by_esi: Dict[ESILevel, float]
    resuscitation_peak_occupancy: int
    resuscitation_avg_occupancy: float
    total_arrivals: int
    total_completed: int
    total_preemptions: int


class MonteCarloResult(BaseModel):
    num_simulations: int
    wait_time_quantiles: Dict[ESILevel, Dict[str, float]]
    doctor_utilization_ci: Dict[str, float]
    target_wait_probability: Dict[ESILevel, float]
    avg_doctor_utilization: float
    avg_resuscitation_peak: float
    avg_total_arrivals: float


class ComparisonResult(BaseModel):
    scenario_names: List[str]
    metrics: List[Dict[str, float]]
    recommendation: str


class MarkdownReportRequest(BaseModel):
    scenario: ScenarioConfig
    monte_carlo_result: Optional[MonteCarloResult] = None
    single_result: Optional[SingleSimulationResult] = None
