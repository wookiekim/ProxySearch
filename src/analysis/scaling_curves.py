"""Scaling curve analysis for quality vs. number of candidates."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import optimize

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def compute_scaling_curve(
    n_values: List[int],
    scores: List[float],
    fit_model: str = "log",
) -> Dict[str, Any]:
    """
    Fit a scaling curve to quality vs. N data.

    Args:
        n_values: List of N (number of candidates) values.
        scores: Corresponding quality scores.
        fit_model: Model to fit ("log", "power", "linear").

    Returns:
        Dictionary with fitted parameters and predictions.
    """
    n_values = np.array(n_values)
    scores = np.array(scores)

    if fit_model == "log":
        # y = a * log(x) + b
        def model(x, a, b):
            return a * np.log(x) + b

        try:
            popt, pcov = optimize.curve_fit(model, n_values, scores)
            predictions = model(n_values, *popt)
            r_squared = 1 - np.sum((scores - predictions) ** 2) / np.sum(
                (scores - np.mean(scores)) ** 2
            )
            return {
                "model": "log",
                "params": {"a": float(popt[0]), "b": float(popt[1])},
                "r_squared": float(r_squared),
                "predictions": predictions.tolist(),
                "formula": f"y = {popt[0]:.4f} * log(x) + {popt[1]:.4f}",
            }
        except Exception as e:
            return {"model": "log", "error": str(e)}

    elif fit_model == "power":
        # y = a * x^b + c
        def model(x, a, b, c):
            return a * np.power(x, b) + c

        try:
            popt, pcov = optimize.curve_fit(
                model, n_values, scores, p0=[1, 0.5, 0], maxfev=5000
            )
            predictions = model(n_values, *popt)
            r_squared = 1 - np.sum((scores - predictions) ** 2) / np.sum(
                (scores - np.mean(scores)) ** 2
            )
            return {
                "model": "power",
                "params": {"a": float(popt[0]), "b": float(popt[1]), "c": float(popt[2])},
                "r_squared": float(r_squared),
                "predictions": predictions.tolist(),
                "formula": f"y = {popt[0]:.4f} * x^{popt[1]:.4f} + {popt[2]:.4f}",
            }
        except Exception as e:
            return {"model": "power", "error": str(e)}

    elif fit_model == "linear":
        # y = a * x + b
        def model(x, a, b):
            return a * x + b

        try:
            popt, pcov = optimize.curve_fit(model, n_values, scores)
            predictions = model(n_values, *popt)
            r_squared = 1 - np.sum((scores - predictions) ** 2) / np.sum(
                (scores - np.mean(scores)) ** 2
            )
            return {
                "model": "linear",
                "params": {"a": float(popt[0]), "b": float(popt[1])},
                "r_squared": float(r_squared),
                "predictions": predictions.tolist(),
                "formula": f"y = {popt[0]:.6f} * x + {popt[1]:.4f}",
            }
        except Exception as e:
            return {"model": "linear", "error": str(e)}

    else:
        raise ValueError(f"Unknown fit model: {fit_model}")


def plot_scaling(
    n_values: List[int],
    scores: List[float],
    std_devs: Optional[List[float]] = None,
    fit_curve: bool = True,
    fit_model: str = "log",
    title: str = "Quality vs. Number of Candidates",
    x_label: str = "Number of Candidates (N)",
    y_label: str = "Quality Score",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6),
    log_x: bool = True,
) -> Optional[Any]:
    """
    Plot scaling curve with optional fit.

    Args:
        n_values: List of N values.
        scores: Corresponding scores.
        std_devs: Optional standard deviations for error bars.
        fit_curve: Whether to fit and plot a curve.
        fit_model: Model to fit.
        title: Plot title.
        x_label: X-axis label.
        y_label: Y-axis label.
        save_path: Path to save figure.
        figsize: Figure size.
        log_x: Whether to use log scale for x-axis.

    Returns:
        Matplotlib figure if available.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    fig, ax = plt.subplots(figsize=figsize)

    n_values = np.array(n_values)
    scores = np.array(scores)

    # Plot data points
    if std_devs is not None:
        ax.errorbar(
            n_values,
            scores,
            yerr=std_devs,
            fmt="o",
            markersize=10,
            capsize=5,
            label="Data",
        )
    else:
        ax.scatter(n_values, scores, s=100, label="Data", zorder=3)

    # Fit and plot curve
    if fit_curve:
        fit_result = compute_scaling_curve(n_values, scores, fit_model)

        if "error" not in fit_result:
            # Generate smooth curve
            x_smooth = np.logspace(
                np.log10(min(n_values)),
                np.log10(max(n_values)),
                100,
            )

            if fit_model == "log":
                y_smooth = fit_result["params"]["a"] * np.log(x_smooth) + fit_result["params"]["b"]
            elif fit_model == "power":
                y_smooth = (
                    fit_result["params"]["a"]
                    * np.power(x_smooth, fit_result["params"]["b"])
                    + fit_result["params"]["c"]
                )
            elif fit_model == "linear":
                y_smooth = fit_result["params"]["a"] * x_smooth + fit_result["params"]["b"]

            ax.plot(
                x_smooth,
                y_smooth,
                "--",
                linewidth=2,
                alpha=0.8,
                label=f"Fit: {fit_result['formula']}\n(R² = {fit_result['r_squared']:.3f})",
            )

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if log_x:
        ax.set_xscale("log")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


class ScalingCurveAnalyzer:
    """
    Analyzer for scaling behavior across different methods and metrics.
    """

    def __init__(self):
        """Initialize the scaling curve analyzer."""
        self.curves: Dict[str, Dict[str, Any]] = {}

    def add_curve(
        self,
        name: str,
        n_values: List[int],
        scores: List[float],
        std_devs: Optional[List[float]] = None,
        times: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a scaling curve for a method.

        Args:
            name: Method name.
            n_values: List of N values.
            scores: Corresponding quality scores.
            std_devs: Optional standard deviations.
            times: Optional wall-clock times.
            metadata: Optional metadata.
        """
        self.curves[name] = {
            "n_values": np.array(n_values),
            "scores": np.array(scores),
            "std_devs": np.array(std_devs) if std_devs else None,
            "times": np.array(times) if times else None,
            "metadata": metadata or {},
        }

    def fit_all_curves(
        self,
        fit_model: str = "log",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fit scaling curves to all methods.

        Args:
            fit_model: Model to fit.

        Returns:
            Dictionary of fit results for each method.
        """
        results = {}
        for name, curve in self.curves.items():
            results[name] = compute_scaling_curve(
                curve["n_values"].tolist(),
                curve["scores"].tolist(),
                fit_model,
            )
        return results

    def find_inflection_point(
        self,
        name: str,
        threshold_ratio: float = 0.9,
    ) -> Dict[str, Any]:
        """
        Find the inflection point where returns diminish.

        Uses the point where incremental improvement falls below
        a threshold of the maximum improvement.

        Args:
            name: Method name.
            threshold_ratio: Ratio of max improvement to use as threshold.

        Returns:
            Dictionary with inflection point analysis.
        """
        curve = self.curves[name]
        n_values = curve["n_values"]
        scores = curve["scores"]

        # Compute incremental improvements
        improvements = np.diff(scores)
        max_improvement = np.max(improvements)
        threshold = threshold_ratio * max_improvement

        # Find first point where improvement drops below threshold
        inflection_idx = np.argmax(improvements < threshold)

        if inflection_idx == 0 and improvements[0] >= threshold:
            # Never drops below threshold
            inflection_n = n_values[-1]
            inflection_score = scores[-1]
        else:
            inflection_n = n_values[inflection_idx]
            inflection_score = scores[inflection_idx]

        return {
            "inflection_n": int(inflection_n),
            "inflection_score": float(inflection_score),
            "improvements": improvements.tolist(),
            "max_improvement": float(max_improvement),
            "threshold": float(threshold),
        }

    def compare_methods_at_n(
        self,
        n_value: int,
    ) -> Dict[str, float]:
        """
        Compare methods at a specific N value.

        Args:
            n_value: N value to compare at.

        Returns:
            Dictionary of method names to scores (interpolated if necessary).
        """
        results = {}

        for name, curve in self.curves.items():
            n_values = curve["n_values"]
            scores = curve["scores"]

            if n_value in n_values:
                idx = np.where(n_values == n_value)[0][0]
                results[name] = float(scores[idx])
            else:
                # Interpolate
                results[name] = float(np.interp(n_value, n_values, scores))

        return results

    def compute_efficiency(
        self,
        name: str,
    ) -> Dict[str, Any]:
        """
        Compute efficiency metrics for a method.

        Args:
            name: Method name.

        Returns:
            Dictionary with efficiency metrics.
        """
        curve = self.curves[name]
        n_values = curve["n_values"]
        scores = curve["scores"]
        times = curve["times"]

        if times is None:
            return {"error": "No timing data available"}

        # Compute score per second
        score_per_second = scores / times

        # Compute marginal efficiency (score improvement per second)
        marginal_score = np.diff(scores)
        marginal_time = np.diff(times)
        marginal_efficiency = marginal_score / marginal_time

        return {
            "score_per_second": score_per_second.tolist(),
            "marginal_efficiency": marginal_efficiency.tolist(),
            "best_efficiency_n": int(n_values[np.argmax(score_per_second)]),
        }

    def plot_comparison(
        self,
        method_names: Optional[List[str]] = None,
        save_path: Optional[Union[str, Path]] = None,
        title: str = "Scaling Curve Comparison",
        figsize: Tuple[int, int] = (12, 8),
        log_x: bool = True,
    ) -> Optional[Any]:
        """
        Plot scaling curves for multiple methods.

        Args:
            method_names: Methods to include (default: all).
            save_path: Path to save figure.
            title: Plot title.
            figsize: Figure size.
            log_x: Whether to use log scale for x-axis.

        Returns:
            Matplotlib figure if available.
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for plotting")

        if method_names is None:
            method_names = list(self.curves.keys())

        fig, ax = plt.subplots(figsize=figsize)

        colors = sns.color_palette("husl", len(method_names))

        for name, color in zip(method_names, colors):
            curve = self.curves[name]

            if curve["std_devs"] is not None:
                ax.errorbar(
                    curve["n_values"],
                    curve["scores"],
                    yerr=curve["std_devs"],
                    fmt="o-",
                    color=color,
                    label=name,
                    markersize=8,
                    capsize=4,
                    linewidth=2,
                )
            else:
                ax.plot(
                    curve["n_values"],
                    curve["scores"],
                    "o-",
                    color=color,
                    label=name,
                    markersize=8,
                    linewidth=2,
                )

        ax.set_xlabel("Number of Candidates (N)", fontsize=12)
        ax.set_ylabel("Quality Score", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        if log_x:
            ax.set_xscale("log")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def generate_table(
        self,
        n_values_to_show: Optional[List[int]] = None,
    ) -> str:
        """
        Generate comparison table in markdown format.

        Args:
            n_values_to_show: N values to include in table.

        Returns:
            Markdown table string.
        """
        if not self.curves:
            return "No curves available."

        # Get all unique N values if not specified
        if n_values_to_show is None:
            all_n = set()
            for curve in self.curves.values():
                all_n.update(curve["n_values"].tolist())
            n_values_to_show = sorted(all_n)

        # Build table
        headers = ["Method"] + [f"N={n}" for n in n_values_to_show]
        rows = []

        for name, curve in self.curves.items():
            row = [name]
            for n in n_values_to_show:
                if n in curve["n_values"]:
                    idx = np.where(curve["n_values"] == n)[0][0]
                    score = curve["scores"][idx]
                    row.append(f"{score:.4f}")
                else:
                    row.append("-")
            rows.append(row)

        # Format as markdown
        col_widths = [
            max(len(str(row[i])) for row in [headers] + rows)
            for i in range(len(headers))
        ]

        def format_row(row):
            return (
                "| "
                + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
                + " |"
            )

        lines = [
            format_row(headers),
            "|" + "|".join("-" * (w + 2) for w in col_widths) + "|",
        ]
        for row in rows:
            lines.append(format_row(row))

        return "\n".join(lines)
