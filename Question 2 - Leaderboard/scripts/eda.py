from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from q2_forecasting.data import HORIZON, load_frames


def autocorrelation(values: np.ndarray, max_lag: int) -> np.ndarray:
    centered = values - values.mean()
    denominator = np.dot(centered, centered)
    return np.array([1.0 if lag == 0 else np.dot(centered[:-lag], centered[lag:]) / denominator for lag in range(max_lag + 1)])


def local_peaks(acf: np.ndarray, minimum_lag: int = 2, count: int = 10) -> list[tuple[int, float]]:
    candidates = [(lag, acf[lag]) for lag in range(minimum_lag, len(acf) - 1) if acf[lag] > acf[lag - 1] and acf[lag] >= acf[lag + 1]]
    return sorted(candidates, key=lambda pair: pair[1], reverse=True)[:count]


def save_plots(train: pd.DataFrame, acf: np.ndarray, associations: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    time, value = train.time_idx.to_numpy(), train.value.to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, constrained_layout=True)
    axes[0].plot(time, value, linewidth=0.45, color="#155b8a")
    axes[0].set(title="Target over time", ylabel="value")
    axes[1].plot(time, np.log1p(value), linewidth=0.45, color="#a84a00")
    axes[1].set(title="log1p(target) over time", xlabel="time_idx", ylabel="log1p(value)")
    fig.savefig(output / "target_overview.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].hist(value, bins=100, color="#155b8a")
    axes[0].set(title="Target distribution", xlabel="value", ylabel="count")
    axes[1].hist(np.log1p(value), bins=100, color="#a84a00")
    axes[1].set(title="log1p(target) distribution", xlabel="log1p(value)", ylabel="count")
    fig.savefig(output / "target_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ax.plot(np.arange(len(acf)), acf, linewidth=0.8, color="#155b8a")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set(title="Target autocorrelation", xlabel="lag", ylabel="correlation")
    fig.savefig(output / "autocorrelation.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    plotted = associations.set_index("feature")[["corr_target_same_time", "corr_target_plus_168"]]
    plotted.plot.barh(ax=ax, color=["#155b8a", "#a84a00"])
    ax.set(title="External-feature Pearson associations on observed training rows", xlabel="correlation")
    fig.savefig(output / "external_associations.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="Data")
    parser.add_argument("--output-dir", default="outputs/eda")
    parser.add_argument("--max-lag", type=int, default=1000)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train, test, external = load_frames(args.data_dir)
    target = train.value.to_numpy(dtype=float)
    acf = autocorrelation(target, min(args.max_lag, len(target) - 1))
    peaks = local_peaks(acf)
    features = external.iloc[:len(train)].drop(columns="time_idx")
    future_target = train.value.shift(-HORIZON)
    rows = []
    for column in features:
        series = features[column]
        rows.append({
            "feature": column,
            "unique_values": int(series.nunique()),
            "binary": bool(series.dropna().isin([0, 1]).all()),
            "corr_target_same_time": series.corr(train.value),
            "corr_target_plus_168": series.iloc[:-HORIZON].corr(future_target.iloc[:-HORIZON]),
        })
    associations = pd.DataFrame(rows).sort_values("corr_target_plus_168", key=lambda s: s.abs(), ascending=False)
    associations.to_csv(output / "external_associations.csv", index=False)
    binary_columns = associations.loc[associations.binary, "feature"].tolist()
    binary_sum = features[binary_columns].sum(axis=1) if binary_columns else pd.Series(dtype=float)
    save_plots(train, acf, associations, output)
    quantiles = np.quantile(target, [0, .01, .05, .25, .5, .75, .95, .99, 1])
    suggested_origins = [40000, 41000, 42000, 43000]
    if any(origin + HORIZON > len(train) for origin in suggested_origins):
        raise RuntimeError("suggested validation origins exceed observed data")
    summary = [
        "# Local EDA summary",
        "",
        "## Data checks",
        f"- Observed target rows: {len(train)}; held-out forecast rows: {len(test)}; external rows: {len(external)}.",
        f"- Target range: {target.min():.6g} to {target.max():.6g}; mean {target.mean():.6g}; median {np.median(target):.6g}.",
        f"- Zero targets: {(target == 0).sum()} ({100 * (target == 0).mean():.3f}%).",
        "- Target quantiles (0, 1, 5, 25, 50, 75, 95, 99, 100%): " + ", ".join(f"{v:.6g}" for v in quantiles) + ".",
        f"- Four binary features: {', '.join(binary_columns)}; rows where their sum is exactly one: {int((binary_sum == 1).sum())}/{len(binary_sum)}.",
        "",
        "## Autocorrelation",
        "- Strongest local ACF peaks (lag, correlation): " + ", ".join(f"({lag}, {score:.3f})" for lag, score in peaks) + ".",
        f"- ACF at lag 1: {acf[1]:.3f}; lag 24: {acf[24]:.3f}; lag 168: {acf[168]:.3f}.",
        "",
        "## External features",
        "- `corr_target_plus_168` is a simple training-only screening correlation between feature[t] and target[t+168]. It is not evidence of predictive value; validate feature choices chronologically across seeds.",
        "- Full feature table: `external_associations.csv`.",
        "",
        "## Candidate validation origins",
        "- " + ", ".join(str(origin) for origin in suggested_origins) + ". Each forecasts the next 168 observed targets; use the same origins and seeds for target-only and external-data runs.",
    ]
    (output / "eda_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
