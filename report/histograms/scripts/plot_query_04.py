from pathlib import Path
import os

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib.pyplot as plt

DATA_PATH = ROOT / "data" / "query_04.csv"
OUTPUT_PATH = ROOT / "query-04-histogram.png"

plt.rcParams.update(
    {
        "axes.titlesize": 28,
        "axes.labelsize": 23,
        "xtick.labelsize": 19,
        "ytick.labelsize": 19,
    }
)


def main() -> None:
    data = pd.read_csv(DATA_PATH)

    plt.figure(figsize=(15.5, 8.6))
    plt.bar(
        data["symbol_count"],
        data["typeface_count"],
        width=0.9,
        color="#557a46",
        edgecolor="#2e3f29",
        linewidth=0.6,
    )

    plt.title("Typeface count by number of symbols", pad=22)
    plt.xlabel("Symbol count")
    plt.ylabel("Typeface count")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight", pad_inches=0.25)


if __name__ == "__main__":
    main()
