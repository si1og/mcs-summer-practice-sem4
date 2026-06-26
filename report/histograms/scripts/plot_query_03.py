from pathlib import Path
import os

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib.pyplot as plt

DATA_PATH = ROOT / "data" / "query_03.csv"
OUTPUT_PATH = ROOT / "query-03-histogram.png"

plt.rcParams.update(
    {
        "axes.titlesize": 28,
        "axes.labelsize": 23,
        "xtick.labelsize": 19,
        "ytick.labelsize": 19,
        "legend.fontsize": 18,
    }
)


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    positions = range(len(data))
    width = 0.36

    fig, left_axis = plt.subplots(figsize=(15.5, 8.8))
    right_axis = left_axis.twinx()

    typeface_bars = left_axis.bar(
        [position - width / 2 for position in positions],
        data["typeface_count"],
        width=width,
        color="#4f7cac",
        edgecolor="#263d59",
        label="Typeface count",
    )

    symbol_bars = right_axis.bar(
        [position + width / 2 for position in positions],
        data["symbol_count"],
        width=width,
        color="#d08c60",
        edgecolor="#7b4d32",
        label="Symbol count",
    )

    fig.suptitle("Typeface and symbol counts by format", fontsize=30, y=0.98)
    left_axis.set_xlabel("Format")
    left_axis.set_ylabel("Typeface count")
    right_axis.set_ylabel("Symbol count")
    left_axis.set_xticks(list(positions))
    left_axis.set_xticklabels(data["format_name"], rotation=15, ha="right")
    left_axis.grid(axis="y", linestyle="--", alpha=0.35)

    left_axis.bar_label(typeface_bars, padding=4, fontsize=16)
    right_axis.bar_label(symbol_bars, padding=4, fontsize=16)

    handles_left, labels_left = left_axis.get_legend_handles_labels()
    handles_right, labels_right = right_axis.get_legend_handles_labels()
    fig.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.88),
        ncol=2,
        framealpha=0.92,
    )

    left_axis.margins(x=0.05)
    fig.subplots_adjust(top=0.72, bottom=0.18, left=0.09, right=0.91)
    plt.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight", pad_inches=0.25)


if __name__ == "__main__":
    main()
