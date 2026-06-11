"""Plot training histories for model benchmark runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_COLORS = [
    "#4e79a7",
    "#b07aa1",
    "#59a14f",
    "#9c755f",
    "#f28e2b",
    "#76b7b2",
    "#79706e",
]


def plot_combined_training_loss(
    history_csv: Path,
    out_prefix: Path,
    *,
    model_order: list[str] | None = None,
    note: str | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    df = pd.read_csv(history_csv).sort_values(["model", "epoch"])
    if df.empty:
        raise ValueError(f"empty training history: {history_csv}")

    models = model_order or sorted(str(x) for x in df["model"].dropna().unique())
    present = [m for m in models if m in set(df["model"])]
    colors = {model: DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i, model in enumerate(present)}

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for model in present:
        g = df[df["model"] == model].sort_values("epoch")
        color = colors[model]
        ax.plot(g["epoch"], g["train_loss"], color=color, linestyle="-", marker="o", linewidth=2.3, markersize=5)
        ax.plot(g["epoch"], g["val_loss"], color=color, linestyle="--", marker="s", linewidth=2.3, markersize=5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and validation loss by model")
    ax.grid(True, alpha=0.25)
    ax.set_xticks(sorted(df["epoch"].unique()))

    model_handles = [Line2D([0], [0], color=colors[m], lw=2.5, label=m) for m in present]
    split_handles = [
        Line2D([0], [0], color="#111827", lw=2.5, linestyle="-", label="train loss"),
        Line2D([0], [0], color="#111827", lw=2.5, linestyle="--", label="valid loss"),
    ]
    legend1 = ax.legend(handles=model_handles, title="Model color", loc="upper right", frameon=False)
    ax.add_artist(legend1)
    ax.legend(handles=split_handles, title="Line style", loc="lower left", frameon=False)

    if note:
        ax.text(0.01, -0.18, note, transform=ax.transAxes, fontsize=8, color="#6b7280")
    fig.tight_layout()

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".svg"))
    fig.savefig(out_prefix.with_suffix(".png"), dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-csv", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--models", nargs="*", default=None, help="Optional display order.")
    parser.add_argument("--note", default=None)
    args = parser.parse_args()

    plot_combined_training_loss(
        args.history_csv,
        args.out_prefix,
        model_order=args.models,
        note=args.note,
    )


if __name__ == "__main__":
    main()
