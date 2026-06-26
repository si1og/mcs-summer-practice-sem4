import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib.pyplot as plt

DATA_PATH = ROOT / "data" / "query_08.csv"
OUTPUT_PATH = ROOT / "query-08-histogram.png"
OUTPUT_3D_PATH = ROOT / "query-08-histogram-3d.png"

plt.rcParams.update(
    {
        "axes.titlesize": 28,
        "axes.labelsize": 23,
        "xtick.labelsize": 19,
        "ytick.labelsize": 19,
        "legend.fontsize": 17,
    }
)


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    matrix = data.pivot(
        index="category_name",
        columns="language_name",
        values="symbol_count",
    )
    category_positions = range(len(matrix.index))
    bar_width = 0.075
    colors = plt.cm.tab10.colors

    fig, axis = plt.subplots(figsize=(22, 11.0))

    for language_index, language in enumerate(matrix.columns):
        offsets = [
            position + (language_index - (len(matrix.columns) - 1) / 2) * bar_width
            for position in category_positions
        ]
        axis.bar(
            offsets,
            matrix[language],
            width=bar_width,
            label=language,
            color=colors[language_index % len(colors)],
        )

    axis.set_title("Symbol counts by category and language", pad=30)
    axis.set_xlabel("Category")
    axis.set_ylabel("Symbol count")
    axis.set_xticks(list(category_positions))
    axis.set_xticklabels(matrix.index, rotation=25, ha="right")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.legend(ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5))
    axis.margins(x=0.01)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight", pad_inches=0.3)

    plot_3d_histogram(matrix)


def plot_3d_histogram(matrix: pd.DataFrame) -> None:
    category_count = len(matrix.index)
    language_count = len(matrix.columns)

    x_positions = []
    y_positions = []
    heights = []
    bar_colors = []
    category_colors = plt.cm.tab20.colors

    for category_index, category in enumerate(matrix.index):
        for language_index, language in enumerate(matrix.columns):
            x_positions.append(category_index)
            y_positions.append(language_index)
            heights.append(matrix.loc[category, language])
            bar_colors.append(category_colors[category_index % len(category_colors)])

    fig = plt.figure(figsize=(15, 15))
    axis = fig.add_subplot(111, projection="3d")

    axis.bar3d(
        x_positions,
        y_positions,
        [0] * len(heights),
        0.62,
        0.62,
        heights,
        color=bar_colors,
        edgecolor="#2f2f2f",
        linewidth=0.7,
        shade=True,
        alpha=0.96,
    )

    axis.set_title("Symbol counts by category and language", pad=32, fontsize=30)
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.set_zlabel("Symbol count", labelpad=34, fontsize=25)

    axis.set_xticks([position + 0.31 for position in range(category_count)])
    axis.set_xticklabels(matrix.index, rotation=38, ha="right", fontsize=18)
    axis.set_yticks([position + 0.31 for position in range(language_count)])
    axis.set_yticklabels(matrix.columns, fontsize=18)
    axis.tick_params(axis="z", labelsize=18, pad=7)

    axis.view_init(elev=27, azim=-58)
    axis.set_box_aspect((category_count * 1.35, language_count, 8))
    axis.margins(x=0.02, y=0.02)

    fig.subplots_adjust(left=0.02, right=0.88, bottom=0.08, top=0.92)
    fig.savefig(OUTPUT_3D_PATH, dpi=220, pad_inches=0.35)


if __name__ == "__main__":
    main()
