"""Pareto frontier analysis for quality vs. compute trade-offs."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def compute_pareto_frontier(
    x: np.ndarray,
    y: np.ndarray,
    maximize_x: bool = False,
    maximize_y: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Pareto frontier from a set of points.

    A point is Pareto-optimal if no other point dominates it
    (i.e., is better in both dimensions).

    Args:
        x: Array of x-values (e.g., wall-clock time).
        y: Array of y-values (e.g., quality metric).
        maximize_x: Whether to maximize x (default: minimize, e.g., time).
        maximize_y: Whether to maximize y (default: maximize, e.g., quality).

    Returns:
        Tuple of:
            - pareto_x: x-values of Pareto-optimal points
            - pareto_y: y-values of Pareto-optimal points
            - pareto_indices: Indices of Pareto-optimal points
    """
    points = np.column_stack([x, y])
    n = len(points)

    # Apply sign transformation for maximization
    transformed = points.copy()
    if not maximize_x:
        transformed[:, 0] = -transformed[:, 0]
    if not maximize_y:
        transformed[:, 1] = -transformed[:, 1]

    # Find Pareto-optimal points
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i != j:
                # Check if j dominates i
                if (
                    transformed[j, 0] >= transformed[i, 0]
                    and transformed[j, 1] >= transformed[i, 1]
                    and (
                        transformed[j, 0] > transformed[i, 0]
                        or transformed[j, 1] > transformed[i, 1]
                    )
                ):
                    is_pareto[i] = False
                    break

    pareto_indices = np.where(is_pareto)[0]

    # Sort by x-value for plotting
    sorted_indices = pareto_indices[np.argsort(x[pareto_indices])]

    return x[sorted_indices], y[sorted_indices], sorted_indices


def compute_area_under_pareto(
    x: np.ndarray,
    y: np.ndarray,
    x_range: Optional[Tuple[float, float]] = None,
) -> float:
    """
    Compute the area under the Pareto frontier.

    This can be used as a single metric to compare different methods.

    Args:
        x: x-values of Pareto frontier (sorted).
        y: y-values of Pareto frontier.
        x_range: Optional (min, max) range for normalization.

    Returns:
        Area under the Pareto curve.
    """
    if len(x) < 2:
        return 0.0

    # Use trapezoidal integration
    # np.trapz was removed in NumPy 2.x in favour of np.trapezoid
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    area = trapezoid(y, x)

    # Normalize by x range if provided
    if x_range is not None:
        x_min, x_max = x_range
        area = area / (x_max - x_min)

    return float(area)


class ParetoAnalyzer:
    """
    Analyzer for quality vs. compute Pareto frontiers.

    Computes and visualizes the trade-off between generation quality
    and computational cost for different methods.
    """

    def __init__(self):
        """Initialize the Pareto analyzer."""
        self.methods: Dict[str, Dict[str, Any]] = {}

    def add_method(
        self,
        name: str,
        times: Union[List[float], np.ndarray],
        scores: Union[List[float], np.ndarray],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add results for a method.

        Args:
            name: Method name.
            times: List of wall-clock times per image.
            scores: List of quality scores per image.
            metadata: Optional metadata (config, etc.).
        """
        times = np.array(times)
        scores = np.array(scores)

        self.methods[name] = {
            "times": times,
            "scores": scores,
            "time_mean": float(np.mean(times)),
            "time_std": float(np.std(times)),
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "metadata": metadata or {},
        }

    def add_scaling_curve(
        self,
        name: str,
        n_values: List[int],
        times: List[float],
        scores: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add scaling curve data for a method (quality vs. N).

        Args:
            name: Method name.
            n_values: List of N (number of candidates) values.
            times: Corresponding wall-clock times.
            scores: Corresponding quality scores.
            metadata: Optional metadata.
        """
        self.methods[name] = {
            "n_values": np.array(n_values),
            "times": np.array(times),
            "scores": np.array(scores),
            "is_scaling_curve": True,
            "metadata": metadata or {},
        }

    def compute_pareto_frontier(
        self,
        method_names: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute Pareto frontier across methods.

        Args:
            method_names: Methods to include (default: all).

        Returns:
            Dictionary with Pareto frontier points and metadata.
        """
        if method_names is None:
            method_names = list(self.methods.keys())

        # Collect all points
        all_times = []
        all_scores = []
        all_labels = []

        for name in method_names:
            method = self.methods[name]

            if method.get("is_scaling_curve"):
                # Add each point from scaling curve
                for t, s in zip(method["times"], method["scores"]):
                    all_times.append(t)
                    all_scores.append(s)
                    all_labels.append(name)
            else:
                # Add mean point for method
                all_times.append(method["time_mean"])
                all_scores.append(method["score_mean"])
                all_labels.append(name)

        all_times = np.array(all_times)
        all_scores = np.array(all_scores)

        # Compute Pareto frontier (minimize time, maximize score)
        pareto_x, pareto_y, pareto_idx = compute_pareto_frontier(
            all_times, all_scores, maximize_x=False, maximize_y=True
        )

        pareto_labels = [all_labels[i] for i in pareto_idx]

        return {
            "times": pareto_x,
            "scores": pareto_y,
            "labels": pareto_labels,
            "all_times": all_times,
            "all_scores": all_scores,
            "all_labels": all_labels,
        }

    def compute_dominance(
        self,
        method1: str,
        method2: str,
    ) -> Dict[str, Any]:
        """
        Check if method1 dominates method2.

        Args:
            method1: Name of first method.
            method2: Name of second method.

        Returns:
            Dictionary with dominance analysis.
        """
        m1 = self.methods[method1]
        m2 = self.methods[method2]

        # Compare means
        time_better = m1["time_mean"] < m2["time_mean"]
        score_better = m1["score_mean"] > m2["score_mean"]

        dominates = time_better and score_better
        dominated = (not time_better) and (not score_better)

        # Compute relative improvements
        time_ratio = m2["time_mean"] / m1["time_mean"] if m1["time_mean"] > 0 else float("inf")
        score_diff = m1["score_mean"] - m2["score_mean"]

        return {
            "method1": method1,
            "method2": method2,
            "method1_dominates": dominates,
            "method1_dominated": dominated,
            "time_speedup": time_ratio,
            "score_improvement": score_diff,
        }

    def plot_pareto(
        self,
        save_path: Optional[Union[str, Path]] = None,
        title: str = "Quality vs. Wall-Clock Time",
        x_label: str = "Wall-Clock Time (s)",
        y_label: str = "Quality Score",
        figsize: Tuple[int, int] = (10, 8),
        show_frontier: bool = True,
        log_x: bool = False,
    ) -> Optional[Any]:
        """
        Plot Pareto frontier visualization.

        Args:
            save_path: Path to save figure (optional).
            title: Plot title.
            x_label: X-axis label.
            y_label: Y-axis label.
            figsize: Figure size.
            show_frontier: Whether to draw the Pareto frontier line.
            log_x: Whether to use log scale for x-axis.

        Returns:
            Matplotlib figure if available, None otherwise.
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for plotting")

        pareto = self.compute_pareto_frontier()

        fig, ax = plt.subplots(figsize=figsize)

        # Plot all points
        colors = sns.color_palette("husl", len(self.methods))
        method_colors = dict(zip(self.methods.keys(), colors))

        for name, method in self.methods.items():
            color = method_colors[name]

            if method.get("is_scaling_curve"):
                # Plot scaling curve as connected points
                ax.plot(
                    method["times"],
                    method["scores"],
                    "o-",
                    color=color,
                    label=name,
                    markersize=8,
                    linewidth=2,
                )
                # Annotate N values
                for t, s, n in zip(method["times"], method["scores"], method["n_values"]):
                    ax.annotate(
                        f"N={n}",
                        (t, s),
                        textcoords="offset points",
                        xytext=(5, 5),
                        fontsize=8,
                    )
            else:
                # Plot single point with error bars
                ax.errorbar(
                    method["time_mean"],
                    method["score_mean"],
                    xerr=method["time_std"],
                    yerr=method["score_std"],
                    fmt="o",
                    color=color,
                    label=name,
                    markersize=10,
                    capsize=5,
                )

        # Plot Pareto frontier
        if show_frontier:
            ax.plot(
                pareto["times"],
                pareto["scores"],
                "k--",
                linewidth=2,
                alpha=0.5,
                label="Pareto Frontier",
            )

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

        if log_x:
            ax.set_xscale("log")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_scaling_curve(
        self,
        method_name: str,
        save_path: Optional[Union[str, Path]] = None,
        metric: str = "scores",
        x_axis: str = "n_values",
        figsize: Tuple[int, int] = (8, 6),
        log_x: bool = True,
    ) -> Optional[Any]:
        """
        Plot scaling curve for a single method.

        Args:
            method_name: Name of the method.
            save_path: Path to save figure.
            metric: Y-axis metric ("scores" or "times").
            x_axis: X-axis variable ("n_values" or "times").
            figsize: Figure size.
            log_x: Whether to use log scale for x-axis.

        Returns:
            Matplotlib figure if available.
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for plotting")

        method = self.methods[method_name]
        if not method.get("is_scaling_curve"):
            raise ValueError(f"{method_name} is not a scaling curve")

        fig, ax = plt.subplots(figsize=figsize)

        x_data = method[x_axis]
        y_data = method[metric]

        ax.plot(x_data, y_data, "o-", markersize=8, linewidth=2)

        ax.set_xlabel(x_axis.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel(metric.title(), fontsize=12)
        ax.set_title(f"{method_name} Scaling Curve", fontsize=14)
        ax.grid(True, alpha=0.3)

        if log_x:
            ax.set_xscale("log")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def generate_comparison_table(
        self,
        method_names: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a comparison table in markdown format.

        Args:
            method_names: Methods to include (default: all).

        Returns:
            Markdown table string.
        """
        if method_names is None:
            method_names = list(self.methods.keys())

        headers = ["Method", "Time (s)", "Score", "Time/Score"]
        rows = []

        for name in method_names:
            method = self.methods[name]
            if method.get("is_scaling_curve"):
                # Report best point
                best_idx = np.argmax(method["scores"])
                time_str = f"{method['times'][best_idx]:.2f}"
                score_str = f"{method['scores'][best_idx]:.4f}"
                ratio = method["times"][best_idx] / (method["scores"][best_idx] + 1e-8)
            else:
                time_str = f"{method['time_mean']:.2f} ± {method['time_std']:.2f}"
                score_str = f"{method['score_mean']:.4f} ± {method['score_std']:.4f}"
                ratio = method["time_mean"] / (method["score_mean"] + 1e-8)

            rows.append([name, time_str, score_str, f"{ratio:.2f}"])

        # Format as markdown table
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]

        def format_row(row):
            return "| " + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) + " |"

        lines = [
            format_row(headers),
            "|" + "|".join("-" * (w + 2) for w in col_widths) + "|",
        ]
        for row in rows:
            lines.append(format_row(row))

        return "\n".join(lines)
