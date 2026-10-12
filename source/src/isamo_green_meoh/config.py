
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple


MODEL_VERSION = "V29_FINAL"
DEFAULT_SEED = 42


@dataclass
class VariableSpec:
    name: str
    unit: str
    lower: float
    upper: float


@dataclass
class DataConfig:
    target_converged_samples: int = 3000
    raw_sobol_samples: int = 3000
    initial_sobol_target: int = 3000
    active_learning_rounds: int = 4
    active_learning_batch_size: int = 384
    boundary_seed_pool_size: int = 160
    boundary_margin_quantile: float = 0.35
    boundary_perturb_scale: float = 0.06
    pareto_neighbour_rounds: int = 0
    pareto_neighbour_batch_size: int = 220
    pareto_neighbour_perturb_scale: float = 0.035
    max_train_pool_size: int = 3000
    pareto_validation_sample_size: int = 0
    train_fraction: float = 0.8
    seed: int = DEFAULT_SEED


@dataclass
class SurrogateConfig:
    prefer_xgboost: bool = True
    random_state: int = DEFAULT_SEED
    k_folds: int = 3
    max_iter: int = 1200
    learning_rate: float = 0.05
    max_depth: int = 5
    min_samples_leaf: int = 10
    xgb_n_estimators: int = 1800
    xgb_subsample: float = 0.9
    xgb_colsample_bytree: float = 0.9
    xgb_reg_lambda: float = 1.0
    xgb_early_stopping_rounds: int = 60
    xgb_n_jobs: int = 4
    ensemble_size: int = 5
    ensemble_bootstrap_fraction: float = 0.85


@dataclass
class ExplainConfig:
    eval_samples: int = 24
    background_samples: int = 64
    n_permutations: int = 64
    random_state: int = DEFAULT_SEED


@dataclass
class OptimizerConfig:
    pop_size: int = 256
    n_generations: int = 300
    al_pop_size: int = 48
    al_generations: int = 12
    n_reference_partitions: int = 20
    crossover_prob: float = 0.9
    mutation_prob: float = 0.2
    mutation_scale: float = 0.08
    uncertainty_penalty_lambda: float = 0.0
    feasibility_prob_threshold: float = 0.55
    feasibility_penalty_scale: float = 1.5
    random_state: int = DEFAULT_SEED


@dataclass
class DecisionConfig:
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    disagreement_quantile: float = 0.10


@dataclass
class PipelineConfig:
    variables: List[VariableSpec] = field(default_factory=lambda: [
        VariableSpec("P_Cap", "bar", 3.000732422, 5.998535156),
        VariableSpec("J_eCO2R", "A/cm2", 0.200292969, 0.799853516),
        VariableSpec("T_eCO2R", "K", 323.1597656, 363.1304688),
        VariableSpec("N_eCO2R", "count", 321.0, 1280.0),
        VariableSpec("P_WE", "MW", 1.001953125, 4.999023438),
        VariableSpec("N_WE", "count", 100.0, 400.0),
        VariableSpec("P_MeOH", "bar", 40.01464844, 69.99267578),
        VariableSpec("T_utility", "K", 493.1695313, 533.1402344),
    ])
    data: DataConfig = field(default_factory=DataConfig)
    surrogate: SurrogateConfig = field(default_factory=SurrogateConfig)
    explain: ExplainConfig = field(default_factory=ExplainConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)

    @property
    def feature_names(self) -> List[str]:
        return [variable.name for variable in self.variables]

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        return [(variable.lower, variable.upper) for variable in self.variables]

    @property
    def objective_names(self) -> List[str]:
        return ["carbon_utilization", "specific_energy_kwh_per_kg_meoh", "capex_proxy"]


def default_config() -> PipelineConfig:
    return PipelineConfig()


def synchronize_seed(cfg: PipelineConfig, seed: Optional[int] = None) -> int:
    master_seed = int(cfg.data.seed if seed is None else seed)
    cfg.data.seed = master_seed
    cfg.surrogate.random_state = master_seed
    cfg.explain.random_state = master_seed
    cfg.optimizer.random_state = master_seed
    return master_seed


def seed_manifest(cfg: PipelineConfig, master_seed: int) -> Dict[str, int]:
    return {
        "master_seed": int(master_seed),
        "data_seed": int(cfg.data.seed),
        "surrogate_random_state": int(cfg.surrogate.random_state),
        "explain_random_state": int(cfg.explain.random_state),
        "optimizer_random_state": int(cfg.optimizer.random_state),
    }


def config_manifest(cfg: PipelineConfig) -> Dict[str, object]:
    return asdict(cfg)
