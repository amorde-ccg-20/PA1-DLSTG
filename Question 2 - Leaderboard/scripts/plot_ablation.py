"""Plot the matched Question 2 external-data ablations from saved validation runs.

From the Question 2 directory:
    python -m scripts.plot_ablation
    python -m scripts.plot_ablation --origins 40000 43000 --seed 7

The script accepts either the runner's outputs/ layout or this checkout's
imported outputs/outputs/ layout. It never reruns or changes a model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONFIGS = {
    "A": "a_log_target",
    "B": "b_log_external",
    "C": "c_raw_target",
    "D": "d_raw_external",
}
HORIZON = 168


def find_results_root(project_root: Path, supplied: Path | None) -> Path:
    candidates = [supplied] if supplied is not None else [
        project_root / "outputs" / "outputs",
        project_root / "outputs",
    ]
    for candidate in candidates:
        if candidate is not None and all(
            (candidate / name / "validate_summary.csv").is_file()
            for name in CONFIGS.values()
        ):
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find all four A-D validate_summary.csv files. "
        "Pass --results-root pointing to their parent directory."
    )


def load_summaries(root: Path) -> dict[str, pd.DataFrame]:
    summaries = {}
    pair_sets = {}
    for label, folder in CONFIGS.items():
        path = root / folder / "validate_summary.csv"
        frame = pd.read_csv(path)
        required = {"origin", "seed", "mae", "rmse", "smape"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path}: missing columns {sorted(required - set(frame.columns))}")
        if frame[["origin", "seed"]].duplicated().any():
            raise ValueError(f"{path}: duplicate origin/seed pair")
        if not np.isfinite(frame[["mae", "rmse", "smape"]].to_numpy(dtype=float)).all():
            raise ValueError(f"{path}: non-finite metric")
        frame = frame.copy()
        frame["origin"] = frame["origin"].astype(int)
        frame["seed"] = frame["seed"].astype(int)
        frame = frame.set_index(["origin", "seed"]).sort_index()
        summaries[label] = frame
        pair_sets[label] = set(frame.index)
    if any(pairs != pair_sets["A"] for pairs in pair_sets.values()):
        raise ValueError("A-D summaries do not contain the same origin/seed pairs")
    if len(pair_sets["A"]) < 2 or len({seed for _, seed in pair_sets["A"]}) < 2:
        raise ValueError("Ablation requires matched runs across multiple seeds")
    return summaries


def load_forecast(root: Path, label: str, origin: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    path = root / CONFIGS[label] / f"validate_origin{origin}_seed{seed}" / "validation_forecast.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        actual = np.asarray(archive["actual"], dtype=float).reshape(-1)
        predicted = np.asarray(archive["prediction"], dtype=float).reshape(-1)
    if actual.size != HORIZON or predicted.size != HORIZON:
        raise ValueError(f"{path}: expected {HORIZON} actual and predicted values")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError(f"{path}: non-finite forecast data")
    return actual, predicted


def plot_forecasts(
    root: Path, summaries: dict[str, pd.DataFrame], origins: list[int], seed: int
) -> plt.Figure:
    fig, axes = plt.subplots(
        len(origins), 2, figsize=(12, 3.55 * len(origins) + 0.6),
        squeeze=False, constrained_layout=True,
    )
    pairs = [
        ("A", "B", "Log target: A vs B", "#2070a8", "#d07724"),
        ("C", "D", "Raw target: C vs D", "#168371", "#a34285"),
    ]
    for row, origin in enumerate(origins):
        if (origin, seed) not in summaries["A"].index:
            raise ValueError(f"No matched saved runs for origin={origin}, seed={seed}")
        index = origin + np.arange(1, HORIZON + 1)
        for col, (without, with_external, title, color_without, color_with) in enumerate(pairs):
            actual_0, prediction_0 = load_forecast(root, without, origin, seed)
            actual_1, prediction_1 = load_forecast(root, with_external, origin, seed)
            if not np.allclose(actual_0, actual_1, rtol=0, atol=1e-3):
                raise ValueError(f"Actual targets differ for {without}/{with_external} at {origin}/{seed}")
            for label, prediction in ((without, prediction_0), (with_external, prediction_1)):
                recalculated = float(np.sqrt(np.mean((actual_0 - prediction) ** 2)))
                recorded = float(summaries[label].loc[(origin, seed), "rmse"])
                if not np.isclose(recalculated, recorded, rtol=1e-5, atol=1e-3):
                    raise ValueError(f"Forecast and summary RMSE disagree for {label} at {origin}/{seed}")
            ax = axes[row, col]
            ax.plot(index, actual_0, color="#20242a", linewidth=1.9, label="Observed", zorder=3)
            ax.plot(index, prediction_0, color=color_without, linewidth=1.45,
                    label=f"{without}: no external", alpha=0.95)
            ax.plot(index, prediction_1, color=color_with, linewidth=1.45,
                    label=f"{with_external}: external", alpha=0.95)
            ax.set_title(f"{title} | origin {origin}, seed {seed}", loc="left", fontsize=10)
            ax.set_xlabel("time_idx")
            ax.set_ylabel("Target value")
            ax.grid(alpha=0.22, linewidth=0.6)
            ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.suptitle(
        "External-data ablation: matched validation forecasts\n"
        "Each panel is one seed and origin; aggregate evidence is in the paired metrics.",
        fontsize=12,
    )
    return fig


def plot_rmse_differences(summaries: dict[str, pd.DataFrame]) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7), sharex=True, constrained_layout=True)
    origins = sorted({origin for origin, _ in summaries["A"].index})
    comparisons = [
        ("B", "A", "Log target: B - A", "#c2671b"),
        ("D", "C", "Raw target: D - C", "#8b3b79"),
    ]
    for ax, (with_external, without, title, color) in zip(axes, comparisons):
        delta = summaries[with_external]["rmse"] - summaries[without]["rmse"]
        origin_means = []
        for x, origin in enumerate(origins):
            points = delta.xs(origin, level="origin").sort_index()
            offsets = np.linspace(-0.09, 0.09, len(points))
            ax.scatter(np.full(len(points), x) + offsets, points.to_numpy(),
                       color=color, alpha=0.55, s=26, zorder=3,
                       label="individual seed" if x == 0 else None)
            origin_means.append(float(points.mean()))
        ax.plot(range(len(origins)), origin_means, color=color, linewidth=1.7,
                marker="D", markersize=7, label="origin mean", zorder=4)
        ax.axhline(0, color="#30343a", linewidth=1.1)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        wins = int((delta < 0).sum())
        ax.set_title(
            f"{title} | overall mean {delta.mean():+.2f} RMSE; external wins {wins}/{len(delta)}",
            loc="left", fontsize=10,
        )
        ax.set_ylabel("RMSE difference")
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xticks(range(len(origins)), [str(origin) for origin in origins])
    axes[-1].set_xlabel("Chronological validation origin")
    fig.suptitle(
        "Paired external-data effect across all saved origins and seeds\n"
        "External minus target-only RMSE; below zero favors external data.",
        fontsize=12,
    )
    return fig


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        destination = output / f"{stem}{suffix}"
        fig.savefig(destination, dpi=190, bbox_inches="tight")
        print(destination.resolve())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, help="Parent directory of A-D result folders")
    parser.add_argument("--output-dir", type=Path, help="Figure destination (default: Report/figures)")
    parser.add_argument("--origins", type=int, nargs="+", default=[40000, 43000],
                        help="Forecast origins for example traces (default: 40000 43000)")
    parser.add_argument("--seed", type=int, default=7, help="Seed for example traces (default: 7)")
    args = parser.parse_args()
    if len(args.origins) != len(set(args.origins)):
        parser.error("--origins must not contain duplicates")
    project_root = Path(__file__).resolve().parents[1]
    root = find_results_root(project_root, args.results_root)
    output = args.output_dir or project_root.parent / "Report" / "figures"
    summaries = load_summaries(root)
    save_figure(plot_forecasts(root, summaries, args.origins, args.seed), output,
                "q2_ablation_forecasts")
    save_figure(plot_rmse_differences(summaries), output, "q2_ablation_rmse")


if __name__ == "__main__":
    main()
