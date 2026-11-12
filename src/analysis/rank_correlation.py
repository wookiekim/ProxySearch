"""Rank correlation analysis between proxy and teacher models."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers import FluxPipeline
from PIL import Image
from scipy import stats
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def compute_rank_ensemble(per_verifier_scores: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Compute rank-based ensemble scores (Ma et al. CVPR 2025 style).

    Each verifier ranks the N candidates independently, then average ranks
    are used as the ensemble score. This avoids scale issues across verifiers
    (e.g., aesthetic ~5-7 vs CLIP ~0.2-0.4).

    Args:
        per_verifier_scores: Dict mapping verifier names to score arrays of shape (N,).

    Returns:
        Average rank array of shape (N,). Lower rank = better candidate.
    """
    all_ranks = []
    for scores in per_verifier_scores.values():
        # rankdata: rank 1 = smallest. We want high score = low (best) rank,
        # so rank the negated scores.
        ranks = stats.rankdata(-np.asarray(scores))
        all_ranks.append(ranks)
    return np.mean(all_ranks, axis=0)


class RankCorrelationStudy:
    """
    Study to validate proxy-teacher rank correlation.

    This is a core empirical contribution of ProxySearch: we need to
    validate that noise vectors scoring high under the proxy also
    score high under the teacher model.
    """

    def __init__(
        self,
        proxy_pipeline: Any,
        teacher_pipeline: Any,
        verifier: Any,
        noise_sampler: Any,
        device: str = "cuda",
    ):
        """
        Initialize the rank correlation study.

        Args:
            proxy_pipeline: Loaded proxy model pipeline.
            teacher_pipeline: Loaded teacher model pipeline.
            verifier: VerifierEnsemble instance for scoring.
            noise_sampler: NoiseSampler instance.
            device: Target device.
        """
        self.proxy_pipeline = proxy_pipeline
        self.teacher_pipeline = teacher_pipeline
        self.verifier = verifier
        self.noise_sampler = noise_sampler
        self.device = device

        # Results storage
        self.results: List[Dict[str, Any]] = []

    def _prepare_latents(self, pipeline: Any, latents: torch.Tensor) -> torch.Tensor:
        """Prepare latents for the pipeline, handling FLUX packing if needed."""
        if isinstance(pipeline, FluxPipeline):
            B, C, H, W = latents.shape
            return pipeline._pack_latents(latents, B, C, H, W)
        return latents

    def _generate_images(
        self,
        pipeline: Any,
        noises: torch.Tensor,
        prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        image_size: int,
        true_cfg_scale: float = 1.0,
    ) -> List[Image.Image]:
        """Generate images for all noise vectors using a single pipeline."""
        images = []

        # Build extra kwargs for true CFG (FLUX only)
        extra_kwargs = {}
        if isinstance(pipeline, FluxPipeline) and true_cfg_scale > 1.0:
            extra_kwargs["true_cfg_scale"] = true_cfg_scale
            extra_kwargs["negative_prompt"] = ""

        for i in range(noises.shape[0]):
            latents = self._prepare_latents(pipeline, noises[i : i + 1])
            with torch.no_grad():
                output = pipeline(
                    prompt=prompt,
                    latents=latents,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=image_size,
                    width=image_size,
                    output_type="pil",
                    **extra_kwargs,
                )
            images.append(output.images[0])
            torch.cuda.empty_cache()
        return images

    def run_single_prompt(
        self,
        prompt: str,
        num_noises: int = 64,
        seed: Optional[int] = None,
        proxy_steps: int = 4,
        teacher_steps: int = 28,
        proxy_guidance: float = 0.0,
        teacher_guidance: float = 3.5,
        image_size: int = 1024,
    ) -> Dict[str, Any]:
        """
        Run correlation study for a single prompt.

        Requires that only one of proxy_pipeline/teacher_pipeline is None
        at a time (for sequential loading), or both loaded (if memory allows).
        """
        noises, _ = self.noise_sampler.sample_with_ids(
            num_samples=num_noises,
            base_seed=seed or 0,
            prompt_id=prompt,
        )

        proxy_images = self._generate_images(
            self.proxy_pipeline, noises, prompt,
            proxy_steps, proxy_guidance, image_size,
        )
        proxy_scores = self.verifier(proxy_images, prompt)

        teacher_images = self._generate_images(
            self.teacher_pipeline, noises, prompt,
            teacher_steps, teacher_guidance, image_size,
        )
        teacher_scores = self.verifier(teacher_images, prompt)

        spearman_rho, spearman_p = stats.spearmanr(proxy_scores, teacher_scores)
        pearson_r, pearson_p = stats.pearsonr(proxy_scores, teacher_scores)
        kendall_tau, kendall_p = stats.kendalltau(proxy_scores, teacher_scores)

        result = {
            "prompt": prompt,
            "num_noises": num_noises,
            "proxy_scores": proxy_scores,
            "teacher_scores": teacher_scores,
            "spearman_rho": float(spearman_rho),
            "spearman_p": float(spearman_p),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "kendall_tau": float(kendall_tau),
            "kendall_p": float(kendall_p),
        }

        self.results.append(result)
        return result

    def run_study_sequential(
        self,
        prompts: List[str],
        proxy_loader: Any,
        teacher_loader: Any,
        num_noises: int = 64,
        seed: int = 42,
        verbose: bool = True,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        proxy_steps: int = 4,
        teacher_steps: int = 28,
        proxy_guidance: float = 0.0,
        teacher_guidance: float = 3.5,
        teacher_true_cfg_scale: float = 1.0,
        image_size: int = 1024,
    ) -> Dict[str, Any]:
        """
        Run correlation study with sequential model loading to save GPU memory.

        Phase 1: Load proxy, generate+score all proxy images for all prompts, unload proxy.
        Phase 2: Load teacher, generate+score all teacher images for all prompts, unload teacher.
        Phase 3: Compute correlations.

        Args:
            prompts: List of prompts to test.
            proxy_loader: ProxyLoader instance (will load/unload).
            teacher_loader: TeacherLoader instance (will load/unload).
            num_noises: Number of noise vectors per prompt.
            seed: Base random seed.
            verbose: Whether to show progress.
            checkpoint_dir: Directory for checkpoint file.
            proxy_steps, teacher_steps, proxy_guidance, teacher_guidance, image_size:
                Generation parameters.

        Returns:
            Aggregated results dictionary.
        """
        import json as _json

        self.results = []
        checkpoint_path = None
        completed_data = {}

        # Load checkpoint if exists
        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir) / "checkpoint.jsonl"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            if checkpoint_path.exists():
                with open(checkpoint_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            r = _json.loads(line)
                        except _json.JSONDecodeError:
                            continue  # skip truncated last line from interrupted write
                        r["proxy_scores"] = np.array(r["proxy_scores"])
                        r["teacher_scores"] = np.array(r["teacher_scores"])
                        self.results.append(r)
                        completed_data[r["prompt"]] = r
                if verbose:
                    print(f"Resumed from checkpoint: {len(completed_data)} prompts already done")

        # Identify remaining prompts
        remaining = [p for p in prompts if p not in completed_data]
        if not remaining:
            if verbose:
                print("All prompts already completed.")
            return self.aggregate_results()

        # Pre-generate all noise vectors (cheap, CPU-only)
        all_noises = {}
        for i, prompt in enumerate(prompts):
            if prompt in completed_data:
                continue
            prompt_seed = seed + i * 1000
            noises, _ = self.noise_sampler.sample_with_ids(
                num_samples=num_noises,
                base_seed=prompt_seed,
                prompt_id=prompt,
            )
            all_noises[prompt] = noises

        # Phase 1: Proxy generation
        if verbose:
            print(f"Phase 1: Generating proxy images for {len(remaining)} prompts...")
        proxy_pipeline = proxy_loader.load()
        proxy_scores_map = {}

        iterator = remaining
        if verbose:
            iterator = tqdm(remaining, desc="Proxy generation")

        for prompt in iterator:
            images = self._generate_images(
                proxy_pipeline, all_noises[prompt], prompt,
                proxy_steps, proxy_guidance, image_size,
                true_cfg_scale=1.0,  # proxy never uses true CFG
            )
            # Score with each verifier individually
            proxy_scores_map[prompt] = self.verifier.score_individual(images, prompt)
            del images
            torch.cuda.empty_cache()

        del proxy_pipeline
        self.proxy_pipeline = None
        proxy_loader.unload()
        import gc; gc.collect()
        torch.cuda.empty_cache()
        if verbose:
            print("Proxy unloaded.")

        # Phase 2: Teacher generation
        if verbose:
            print(f"Phase 2: Generating teacher images for {len(remaining)} prompts...")
        teacher_pipeline = teacher_loader.load()

        iterator = remaining
        if verbose:
            iterator = tqdm(remaining, desc="Teacher generation")

        for prompt in iterator:
            images = self._generate_images(
                teacher_pipeline, all_noises[prompt], prompt,
                teacher_steps, teacher_guidance, image_size,
                true_cfg_scale=teacher_true_cfg_scale,
            )
            teacher_individual = self.verifier.score_individual(images, prompt)
            proxy_individual = proxy_scores_map[prompt]
            del images
            torch.cuda.empty_cache()

            # Compute per-verifier correlations
            per_verifier = {}
            for vname in proxy_individual:
                ps = proxy_individual[vname]
                ts = teacher_individual[vname]
                rho, rho_p = stats.spearmanr(ps, ts)
                r, r_p = stats.pearsonr(ps, ts)
                tau, tau_p = stats.kendalltau(ps, ts)
                per_verifier[vname] = {
                    "proxy_scores": ps,
                    "teacher_scores": ts,
                    "spearman_rho": float(rho),
                    "spearman_p": float(rho_p),
                    "pearson_r": float(r),
                    "pearson_p": float(r_p),
                    "kendall_tau": float(tau),
                    "kendall_p": float(tau_p),
                }

            # Also compute ensemble (mean of all verifier scores)
            proxy_ensemble = np.mean([proxy_individual[v] for v in proxy_individual], axis=0)
            teacher_ensemble = np.mean([teacher_individual[v] for v in teacher_individual], axis=0)
            ens_rho, ens_rho_p = stats.spearmanr(proxy_ensemble, teacher_ensemble)
            ens_r, ens_r_p = stats.pearsonr(proxy_ensemble, teacher_ensemble)
            ens_tau, ens_tau_p = stats.kendalltau(proxy_ensemble, teacher_ensemble)

            result = {
                "prompt": prompt,
                "num_noises": num_noises,
                "per_verifier": per_verifier,
                "proxy_scores": proxy_ensemble,
                "teacher_scores": teacher_ensemble,
                "spearman_rho": float(ens_rho),
                "spearman_p": float(ens_rho_p),
                "pearson_r": float(ens_r),
                "pearson_p": float(ens_r_p),
                "kendall_tau": float(ens_tau),
                "kendall_p": float(ens_tau_p),
            }
            self.results.append(result)

            # Checkpoint
            if checkpoint_path:
                sr = result.copy()
                sr["proxy_scores"] = result["proxy_scores"].tolist()
                sr["teacher_scores"] = result["teacher_scores"].tolist()
                # Serialize per-verifier scores
                sr["per_verifier"] = {}
                for vname, vdata in per_verifier.items():
                    sv = vdata.copy()
                    sv["proxy_scores"] = vdata["proxy_scores"].tolist()
                    sv["teacher_scores"] = vdata["teacher_scores"].tolist()
                    sr["per_verifier"][vname] = sv
                with open(checkpoint_path, "a") as f:
                    f.write(_json.dumps(sr) + "\n")

        del teacher_pipeline
        self.teacher_pipeline = None
        teacher_loader.unload()
        import gc; gc.collect()
        torch.cuda.empty_cache()
        if verbose:
            print("Teacher unloaded.")

        return self.aggregate_results()

    def run_study(
        self,
        prompts: List[str],
        num_noises: int = 64,
        seed: int = 42,
        verbose: bool = True,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run correlation study across multiple prompts.

        Args:
            prompts: List of prompts to test.
            num_noises: Number of noise vectors per prompt.
            seed: Base random seed.
            verbose: Whether to show progress.
            checkpoint_dir: Directory for JSONL checkpoint file. If provided,
                           enables resume from interrupted runs.
            **kwargs: Additional arguments passed to run_single_prompt.

        Returns:
            Aggregated results dictionary.
        """
        import json as _json

        self.results = []
        completed_prompts = set()
        checkpoint_path = None

        # Load checkpoint if exists
        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir) / "checkpoint.jsonl"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            if checkpoint_path.exists():
                with open(checkpoint_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            r = _json.loads(line)
                        except _json.JSONDecodeError:
                            continue  # skip truncated last line from interrupted write
                        r["proxy_scores"] = np.array(r["proxy_scores"])
                        r["teacher_scores"] = np.array(r["teacher_scores"])
                        self.results.append(r)
                        completed_prompts.add(r["prompt"])
                if verbose:
                    print(f"Resumed from checkpoint: {len(completed_prompts)} prompts already done")

        iterator = prompts
        if verbose:
            iterator = tqdm(prompts, desc="Running correlation study")

        for i, prompt in enumerate(iterator):
            if prompt in completed_prompts:
                continue

            prompt_seed = seed + i * 1000
            result = self.run_single_prompt(
                prompt=prompt,
                num_noises=num_noises,
                seed=prompt_seed,
                **kwargs,
            )

            # Checkpoint after each prompt
            if checkpoint_path:
                sr = result.copy()
                sr["proxy_scores"] = result["proxy_scores"].tolist()
                sr["teacher_scores"] = result["teacher_scores"].tolist()
                with open(checkpoint_path, "a") as f:
                    f.write(_json.dumps(sr) + "\n")

        return self.aggregate_results()

    def aggregate_results(self) -> Dict[str, Any]:
        """
        Aggregate results across all prompts.

        Returns:
            Dictionary with aggregate statistics (ensemble + per-verifier).
        """
        if not self.results:
            return {}

        spearman_rhos = [r["spearman_rho"] for r in self.results]
        pearson_rs = [r["pearson_r"] for r in self.results]
        kendall_taus = [r["kendall_tau"] for r in self.results]

        spearman_significant = sum(
            1 for r in self.results if r["spearman_p"] < 0.05
        )
        pearson_significant = sum(
            1 for r in self.results if r["pearson_p"] < 0.05
        )

        agg = {
            "num_prompts": len(self.results),
            "ensemble": {
                "spearman": {
                    "mean": float(np.mean(spearman_rhos)),
                    "std": float(np.std(spearman_rhos)),
                    "median": float(np.median(spearman_rhos)),
                    "min": float(np.min(spearman_rhos)),
                    "max": float(np.max(spearman_rhos)),
                    "significant_count": spearman_significant,
                    "significant_ratio": spearman_significant / len(self.results),
                },
                "pearson": {
                    "mean": float(np.mean(pearson_rs)),
                    "std": float(np.std(pearson_rs)),
                    "median": float(np.median(pearson_rs)),
                    "significant_count": pearson_significant,
                    "significant_ratio": pearson_significant / len(self.results),
                },
                "kendall": {
                    "mean": float(np.mean(kendall_taus)),
                    "std": float(np.std(kendall_taus)),
                    "median": float(np.median(kendall_taus)),
                },
            },
            # Keep flat keys for backwards compatibility
            "spearman": {
                "mean": float(np.mean(spearman_rhos)),
                "std": float(np.std(spearman_rhos)),
                "median": float(np.median(spearman_rhos)),
                "min": float(np.min(spearman_rhos)),
                "max": float(np.max(spearman_rhos)),
                "significant_count": spearman_significant,
                "significant_ratio": spearman_significant / len(self.results),
            },
            "pearson": {
                "mean": float(np.mean(pearson_rs)),
                "std": float(np.std(pearson_rs)),
                "median": float(np.median(pearson_rs)),
                "significant_count": pearson_significant,
                "significant_ratio": pearson_significant / len(self.results),
            },
            "kendall": {
                "mean": float(np.mean(kendall_taus)),
                "std": float(np.std(kendall_taus)),
                "median": float(np.median(kendall_taus)),
            },
            "detailed_results": self.results,
        }

        # Per-verifier aggregation (if available)
        if self.results and "per_verifier" in self.results[0]:
            verifier_names = list(self.results[0]["per_verifier"].keys())
            per_verifier_agg = {}
            for vname in verifier_names:
                v_rhos = [r["per_verifier"][vname]["spearman_rho"] for r in self.results if "per_verifier" in r]
                v_sig = sum(1 for r in self.results if "per_verifier" in r and r["per_verifier"][vname]["spearman_p"] < 0.05)
                per_verifier_agg[vname] = {
                    "spearman_mean": float(np.mean(v_rhos)),
                    "spearman_std": float(np.std(v_rhos)),
                    "spearman_median": float(np.median(v_rhos)),
                    "significant_count": v_sig,
                    "significant_ratio": v_sig / len(v_rhos) if v_rhos else 0,
                }
            agg["per_verifier"] = per_verifier_agg

        return agg

    def plot_correlation_scatter(
        self,
        prompt_idx: int = 0,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (8, 8),
    ) -> Optional[Any]:
        """
        Plot scatter plot of proxy vs. teacher scores for one prompt.

        Args:
            prompt_idx: Index of prompt to plot.
            save_path: Path to save figure.
            figsize: Figure size.

        Returns:
            Matplotlib figure if available.
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for plotting")

        if prompt_idx >= len(self.results):
            raise IndexError(f"prompt_idx {prompt_idx} out of range")

        result = self.results[prompt_idx]

        fig, ax = plt.subplots(figsize=figsize)

        ax.scatter(
            result["proxy_scores"],
            result["teacher_scores"],
            alpha=0.6,
            s=50,
        )

        # Add regression line
        z = np.polyfit(result["proxy_scores"], result["teacher_scores"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(
            min(result["proxy_scores"]),
            max(result["proxy_scores"]),
            100,
        )
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

        # Add correlation annotation
        ax.annotate(
            f"Spearman ρ = {result['spearman_rho']:.3f}\n"
            f"Pearson r = {result['pearson_r']:.3f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        ax.set_xlabel("Proxy Score", fontsize=12)
        ax.set_ylabel("Teacher Score", fontsize=12)
        ax.set_title(f"Proxy vs. Teacher Score Correlation\n{result['prompt'][:50]}...", fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_correlation_distribution(
        self,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (10, 6),
    ) -> Optional[Any]:
        """
        Plot distribution of correlation coefficients across prompts.

        Args:
            save_path: Path to save figure.
            figsize: Figure size.

        Returns:
            Matplotlib figure if available.
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for plotting")

        if not self.results:
            raise ValueError("No results to plot")

        spearman_rhos = [r["spearman_rho"] for r in self.results]

        fig, ax = plt.subplots(figsize=figsize)

        ax.hist(spearman_rhos, bins=20, edgecolor="black", alpha=0.7)

        ax.axvline(
            np.mean(spearman_rhos),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean = {np.mean(spearman_rhos):.3f}",
        )

        ax.set_xlabel("Spearman Correlation (ρ)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Distribution of Proxy-Teacher Rank Correlations", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def generate_table(self) -> str:
        """
        Generate summary table in markdown format.

        Returns:
            Markdown table string.
        """
        agg = self.aggregate_results()
        if not agg:
            return "No results available."

        lines = [
            "Ensemble (all verifiers combined):",
            "| Metric | Mean | Std | Median | Significant (p<0.05) |",
            "|--------|------|-----|--------|---------------------|",
            f"| Spearman ρ | {agg['spearman']['mean']:.3f} | {agg['spearman']['std']:.3f} | {agg['spearman']['median']:.3f} | {agg['spearman']['significant_ratio']:.1%} |",
            f"| Pearson r | {agg['pearson']['mean']:.3f} | {agg['pearson']['std']:.3f} | {agg['pearson']['median']:.3f} | {agg['pearson']['significant_ratio']:.1%} |",
            f"| Kendall τ | {agg['kendall']['mean']:.3f} | {agg['kendall']['std']:.3f} | {agg['kendall']['median']:.3f} | - |",
        ]

        if "per_verifier" in agg:
            lines.append("")
            lines.append("Per-verifier Spearman ρ:")
            lines.append("| Verifier | Mean | Std | Median | Significant |")
            lines.append("|----------|------|-----|--------|-------------|")
            for vname, vstats in agg["per_verifier"].items():
                lines.append(
                    f"| {vname} | {vstats['spearman_mean']:.3f} | {vstats['spearman_std']:.3f} | {vstats['spearman_median']:.3f} | {vstats['significant_ratio']:.1%} |"
                )

        return "\n".join(lines)

    def save_results(self, path: Union[str, Path]) -> None:
        """
        Save results to JSON file.

        Args:
            path: Path to save results.
        """
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = []
        for r in self.results:
            sr = r.copy()
            sr["proxy_scores"] = r["proxy_scores"].tolist()
            sr["teacher_scores"] = r["teacher_scores"].tolist()
            serializable_results.append(sr)

        class _NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.floating, np.integer)):
                    return obj.item()
                return super().default(obj)

        with open(path, "w") as f:
            json.dump(
                {
                    "aggregate": self.aggregate_results(),
                    "detailed_results": serializable_results,
                },
                f,
                indent=2,
                cls=_NumpyEncoder,
            )

    def load_results(self, path: Union[str, Path]) -> None:
        """
        Load results from JSON file.

        Args:
            path: Path to load results from.
        """
        import json

        with open(path, "r") as f:
            data = json.load(f)

        self.results = data.get("detailed_results", [])

        # Convert lists back to numpy arrays
        for r in self.results:
            r["proxy_scores"] = np.array(r["proxy_scores"])
            r["teacher_scores"] = np.array(r["teacher_scores"])
