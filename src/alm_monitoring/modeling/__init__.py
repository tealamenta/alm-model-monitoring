"""Modélisation : benchmark, sélection du champion, explicabilité, champion-challenger."""

from alm_monitoring.modeling.benchmark import (
    CandidateResult,
    ModelMetrics,
    benchmark_table,
    build_candidates,
    compute_metrics,
    run_benchmark,
    select_champion,
)
from alm_monitoring.modeling.challenger import ChallengerDecision, compare_by_window, decide
from alm_monitoring.modeling.explain import (
    economic_sign_check,
    partial_dependence_table,
    permutation_importance_table,
)

__all__ = [
    "CandidateResult",
    "ChallengerDecision",
    "ModelMetrics",
    "benchmark_table",
    "build_candidates",
    "compare_by_window",
    "compute_metrics",
    "decide",
    "economic_sign_check",
    "partial_dependence_table",
    "permutation_importance_table",
    "run_benchmark",
    "select_champion",
]
