"""Tests for evaluation metrics."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.pareto import compute_pareto_frontier, compute_area_under_pareto


class TestParetoFrontier:
    """Tests for Pareto frontier computation."""

    def test_simple_pareto(self):
        """Test simple Pareto frontier computation."""
        # Points: (time, quality)
        x = np.array([1.0, 2.0, 3.0, 2.5])
        y = np.array([0.5, 0.7, 0.8, 0.6])

        pareto_x, pareto_y, pareto_idx = compute_pareto_frontier(
            x, y, maximize_x=False, maximize_y=True
        )

        # Point (2.5, 0.6) is dominated by (2.0, 0.7)
        assert len(pareto_idx) == 3
        assert 3 not in pareto_idx  # (2.5, 0.6) should not be on frontier

    def test_all_pareto_optimal(self):
        """Test when all points are Pareto-optimal."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([0.5, 0.7, 0.9])  # Strictly increasing

        pareto_x, pareto_y, pareto_idx = compute_pareto_frontier(
            x, y, maximize_x=False, maximize_y=True
        )

        assert len(pareto_idx) == 3

    def test_single_point(self):
        """Test with single point."""
        x = np.array([1.0])
        y = np.array([0.5])

        pareto_x, pareto_y, pareto_idx = compute_pareto_frontier(
            x, y, maximize_x=False, maximize_y=True
        )

        assert len(pareto_idx) == 1
        assert pareto_idx[0] == 0

    def test_maximize_both(self):
        """Test maximizing both dimensions."""
        x = np.array([1.0, 2.0, 1.5, 0.5])
        y = np.array([0.5, 0.7, 0.8, 0.6])

        pareto_x, pareto_y, pareto_idx = compute_pareto_frontier(
            x, y, maximize_x=True, maximize_y=True
        )

        # (0.5, 0.6) is dominated, others might be on frontier
        assert 3 not in pareto_idx

    def test_sorted_output(self):
        """Test that output is sorted by x."""
        x = np.array([3.0, 1.0, 2.0])
        y = np.array([0.9, 0.5, 0.7])

        pareto_x, pareto_y, pareto_idx = compute_pareto_frontier(
            x, y, maximize_x=False, maximize_y=True
        )

        # Should be sorted by x
        assert np.all(pareto_x[:-1] <= pareto_x[1:])


class TestParetoArea:
    """Tests for area under Pareto curve."""

    def test_simple_area(self):
        """Test simple area computation."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([0.5, 0.7, 0.8])

        area = compute_area_under_pareto(x, y)
        assert area > 0

    def test_normalized_area(self):
        """Test normalized area computation."""
        x = np.array([0.0, 1.0])
        y = np.array([1.0, 1.0])  # Constant function

        area = compute_area_under_pareto(x, y, x_range=(0.0, 1.0))
        assert abs(area - 1.0) < 0.01  # Should be 1.0

    def test_empty_input(self):
        """Test with single point (no area)."""
        x = np.array([1.0])
        y = np.array([0.5])

        area = compute_area_under_pareto(x, y)
        assert area == 0.0


class TestScalingCurveFit:
    """Tests for scaling curve fitting."""

    def test_log_fit(self):
        """Test logarithmic curve fitting."""
        from src.analysis.scaling_curves import compute_scaling_curve

        # Generate log-like data
        n_values = [64, 128, 256, 512, 1024]
        scores = [0.1 * np.log(n) for n in n_values]

        result = compute_scaling_curve(n_values, scores, fit_model="log")

        assert "error" not in result
        assert result["r_squared"] > 0.99  # Should fit perfectly

    def test_power_fit(self):
        """Test power law curve fitting."""
        from src.analysis.scaling_curves import compute_scaling_curve

        # Generate power-law-like data
        n_values = [64, 128, 256, 512]
        scores = [n**0.3 for n in n_values]

        result = compute_scaling_curve(n_values, scores, fit_model="power")

        assert "error" not in result
        assert result["r_squared"] > 0.95


class TestCorrelationMetrics:
    """Tests for correlation metric computation."""

    def test_perfect_correlation(self):
        """Test with perfect correlation."""
        from scipy import stats

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        rho, p = stats.spearmanr(x, y)
        assert abs(rho - 1.0) < 0.01
        assert p < 0.05

    def test_no_correlation(self):
        """Test with uncorrelated data."""
        from scipy import stats

        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)

        rho, p = stats.spearmanr(x, y)
        assert abs(rho) < 0.3  # Should be near zero

    def test_negative_correlation(self):
        """Test with negative correlation."""
        from scipy import stats

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

        rho, p = stats.spearmanr(x, y)
        assert abs(rho - (-1.0)) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
