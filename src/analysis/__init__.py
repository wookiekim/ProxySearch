"""Analysis modules for ProxySearch experiments."""

from .rank_correlation import RankCorrelationStudy
from .scaling_curves import ScalingCurveAnalyzer, compute_scaling_curve, plot_scaling

__all__ = [
    "RankCorrelationStudy",
    "ScalingCurveAnalyzer",
    "compute_scaling_curve",
    "plot_scaling",
]
