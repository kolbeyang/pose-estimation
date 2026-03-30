"""Plot experiment results: MPJPE vs training time, latency breakdown, etc.

Reads all JSON files from motionbert-pose/experiments/ and generates comparison plots.
"""

import json
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

EXPERIMENTS_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")


def load_experiments() -> dict[str, dict[str, Any]]:
    """Load all experiment JSON files."""
    experiments: dict[str, dict[str, Any]] = {}
    for fname in sorted(os.listdir(EXPERIMENTS_DIR)):
        if fname.endswith(".json"):
            fpath: str = os.path.join(EXPERIMENTS_DIR, fname)
            with open(fpath) as f:
                data = json.load(f)
            name: str = data.get("experiment_name", fname.replace(".json", ""))
            experiments[name] = data
    return experiments


def plot_mpjpe_vs_time(experiments: dict[str, dict[str, Any]], output_dir: str) -> None:
    """MPJPE (y-axis) vs total optimization time (x-axis) for each experiment."""
    fig, ax = plt.subplots(figsize=(12, 8))

    for name, data in experiments.items():
        if "aggregate" not in data:
            continue
        agg = data["aggregate"]
        opt_time: float = agg.get("mean_optimization_time_s", 0)
        opt_mpjpe: float = agg.get("mean_opt_mpjpe", 0) * 100  # cm
        det_mpjpe: float = agg.get("mean_det_mpjpe", 0) * 100
        n_examples: int = agg.get("n_examples", 0)

        ax.scatter(opt_time, opt_mpjpe, s=100, zorder=5)
        ax.annotate(
            f"{name}\n({n_examples}ex)",
            (opt_time, opt_mpjpe),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=8,
        )

    ax.set_xlabel("Mean Optimization Time (s)", fontsize=12)
    ax.set_ylabel("Mean Opt MPJPE (cm)", fontsize=12)
    ax.set_title("MPJPE vs Optimization Time", fontsize=14)
    ax.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "mpjpe_vs_optimization_time.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved mpjpe_vs_optimization_time.png")


def plot_mpjpe_vs_total_time(experiments: dict[str, dict[str, Any]], output_dir: str) -> None:
    """MPJPE vs total end-to-end time (includes detection)."""
    fig, ax = plt.subplots(figsize=(12, 8))

    for name, data in experiments.items():
        if "aggregate" not in data:
            continue
        agg = data["aggregate"]
        total_time: float = agg.get("mean_total_time_s", 0)
        opt_mpjpe: float = agg.get("mean_opt_mpjpe", 0) * 100
        n_examples: int = agg.get("n_examples", 0)

        ax.scatter(total_time, opt_mpjpe, s=100, zorder=5)
        ax.annotate(
            f"{name}\n({n_examples}ex)",
            (total_time, opt_mpjpe),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=8,
        )

    ax.set_xlabel("Mean Total Time per Example (s)", fontsize=12)
    ax.set_ylabel("Mean Opt MPJPE (cm)", fontsize=12)
    ax.set_title("MPJPE vs Total End-to-End Time", fontsize=14)
    ax.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "mpjpe_vs_total_time.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved mpjpe_vs_total_time.png")


def plot_latency_breakdown(experiments: dict[str, dict[str, Any]], output_dir: str) -> None:
    """Stacked bar chart of detection vs optimization vs evaluation time."""
    names: list[str] = []
    det_times: list[float] = []
    opt_times: list[float] = []
    eval_times: list[float] = []

    for name, data in experiments.items():
        if "aggregate" not in data:
            continue
        agg = data["aggregate"]
        names.append(name)
        det_times.append(agg.get("mean_detection_time_s", 0))
        opt_times.append(agg.get("mean_optimization_time_s", 0))
        eval_times.append(agg.get("mean_evaluation_time_s", 0))

    if not names:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(names))
    width = 0.6

    bars_det = ax.bar(x, det_times, width, label="Detection", color="#2196F3")
    bars_opt = ax.bar(x, opt_times, width, bottom=det_times, label="Optimization", color="#FF9800")
    bars_eval = ax.bar(
        x, eval_times, width,
        bottom=[d + o for d, o in zip(det_times, opt_times)],
        label="Evaluation", color="#4CAF50",
    )

    ax.set_ylabel("Time (s)", fontsize=12)
    ax.set_title("Latency Breakdown by Pipeline Stage", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "latency_breakdown.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved latency_breakdown.png")


def plot_loss_curves(experiments: dict[str, dict[str, Any]], output_dir: str) -> None:
    """Loss curves for first example across experiments."""
    fig, ax = plt.subplots(figsize=(12, 8))

    for name, data in experiments.items():
        examples = data.get("examples", [])
        if not examples or "loss_history" not in examples[0]:
            continue
        loss: list[float] = examples[0]["loss_history"]
        ax.plot(loss, label=name, alpha=0.8)

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Loss Curves (First Example)", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved loss_curves.png")


def plot_per_example_comparison(experiments: dict[str, dict[str, Any]], output_dir: str) -> None:
    """Bar chart of per-example MPJPE for key experiments."""
    # Only include experiments with full data
    full_exps: dict[str, dict[str, Any]] = {
        k: v for k, v in experiments.items()
        if "aggregate" in v and v["aggregate"].get("n_examples", 0) >= 10
    }
    if not full_exps:
        return

    fig, ax = plt.subplots(figsize=(16, 8))

    exp_names = list(full_exps.keys())
    n_exp = len(exp_names)
    width = 0.8 / n_exp

    for idx, name in enumerate(exp_names):
        data = full_exps[name]
        examples = [e for e in data["examples"] if e.get("opt_mpjpe") is not None]
        ex_names = [e["name"][:20] for e in examples]
        mpjpes = [e["opt_mpjpe"] * 100 for e in examples]
        x = np.arange(len(examples))
        ax.bar(x + idx * width, mpjpes, width, label=name, alpha=0.8)

    ax.set_xlabel("Example", fontsize=12)
    ax.set_ylabel("Opt MPJPE (cm)", fontsize=12)
    ax.set_title("Per-Example MPJPE Comparison", fontsize=14)
    if examples:
        ax.set_xticks(np.arange(len(examples)) + width * (n_exp - 1) / 2)
        ax.set_xticklabels(ex_names, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "per_example_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved per_example_comparison.png")


def main() -> None:
    """Generate all experiment comparison plots."""
    experiments = load_experiments()
    print(f"Loaded {len(experiments)} experiments: {list(experiments.keys())}")

    output_dir: str = EXPERIMENTS_DIR
    plot_mpjpe_vs_time(experiments, output_dir)
    plot_mpjpe_vs_total_time(experiments, output_dir)
    plot_latency_breakdown(experiments, output_dir)
    plot_loss_curves(experiments, output_dir)
    plot_per_example_comparison(experiments, output_dir)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
