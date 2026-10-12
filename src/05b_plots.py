"""Stage 5 figures: confusion matrices and the cross-validation comparison.

What it does
    Draws the figures of the evaluation from the tables written by
    src/05_evaluate.py. Nothing is recomputed, so the figures always match
    the CSV files.

Inputs (results/evaluation/)
    confusion_test.csv, test_results.csv, cv_results.csv

Outputs (report/figures/)
    confusion_{A,B}_{config}.png  one per configuration scored on test
    cv_comparison_{A,B}.png       F1-macro of every configuration in
                                  cross-validation

How to run
    .venv/bin/python src/05b_plots.py
"""
import matplotlib

matplotlib.use("Agg")  # write files without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from config import EVALUATION_DIR, EXTERNAL_LABELS, FIGURES_DIR, LAB_LABELS

# One hue, light to dark: the cell color encodes a magnitude (a share), so a
# single-hue ramp is used and a zero cell stays the color of the background.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
             "#0d366b"]
SURFACE = "#ffffff"
INK = "#1f2328"
MUTED = "#6b7280"
GRID = "#e5e7eb"
SERIES = "#2a78d6"

LABEL_ORDER = ["benign"] + EXTERNAL_LABELS + LAB_LABELS

plt.rcParams.update({
    "font.size": 9, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": GRID,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def plot_confusion(part, dataset, config, n_features, path):
    """Draw one confusion matrix.

    Rows are normalized: the color is the share of each true class, because
    the classes go from 6 to about 25,000 test rows and raw counts would make
    every small class invisible. The number written in a cell is the raw row
    count. Labels are ordered by data source so that confusions between the
    external and the lab labels would show up outside the two blocks.
    """
    matrix = (part.pivot(index="true_label", columns="predicted_label",
                         values="rows")
              .reindex(index=LABEL_ORDER, columns=LABEL_ORDER))
    counts = matrix.to_numpy()
    share = counts / counts.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(9.2, 7.6))
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    cmap.set_bad(SURFACE)
    # Zero cells are masked so they keep the background color.
    mesh = ax.pcolormesh(np.ma.masked_where(counts == 0, share), cmap=cmap,
                         vmin=0, vmax=1, edgecolors=SURFACE, linewidth=2)
    ax.invert_yaxis()
    ax.set_aspect("equal")

    for i in range(len(LABEL_ORDER)):
        for j in range(len(LABEL_ORDER)):
            if counts[i, j]:
                ax.text(j + 0.5, i + 0.5, f"{counts[i, j]:,}", ha="center",
                        va="center", fontsize=7.5,
                        color=SURFACE if share[i, j] > 0.55 else INK)

    ticks = np.arange(len(LABEL_ORDER)) + 0.5
    row_labels = [f"{lab}  (n={int(n):,})"
                  for lab, n in zip(LABEL_ORDER, counts.sum(axis=1))]
    ax.set_xticks(ticks, LABEL_ORDER, rotation=45, ha="right")
    ax.set_yticks(ticks, row_labels)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Separators between the three source groups of labels.
    for edge in (1, 1 + len(EXTERNAL_LABELS)):
        ax.axhline(edge, color=MUTED, linewidth=0.8)
        ax.axvline(edge, color=MUTED, linewidth=0.8)
    for start, size, text in ((0, 1, "both"),
                              (1, len(EXTERNAL_LABELS), "external labels"),
                              (1 + len(EXTERNAL_LABELS), len(LAB_LABELS),
                               "lab labels")):
        ax.text(start + size / 2, -0.25, text, ha="center", va="bottom",
                fontsize=8, color=MUTED)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label (test rows)")
    fig.suptitle(f"Confusion matrix on test, dataset {dataset}: {config} "
                 f"({n_features} features)", x=0.02, ha="left", fontsize=11,
                 fontweight="bold")
    ax.set_title("Number = test rows. Color = share of the true class.",
                 loc="left", fontsize=8.5, color=MUTED, pad=22)
    bar = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.02)
    bar.set_label("Share of the true class")
    bar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_cv_comparison(part, dataset, path):
    """Draw the cross-validation F1-macro of every configuration.

    One series, so one color and no legend. A filled marker is the setting
    selected in its family; the bars are one standard error over the folds.
    """
    part = part.sort_values("f1_macro_mean")
    labels = [f"{c}  ({n:.0f})" for c, n in zip(part["config"],
                                                part["n_features"])]
    y = np.arange(len(part))
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    ax.errorbar(part["f1_macro_mean"], y, xerr=part["f1_macro_se"], fmt="none",
                ecolor=SERIES, elinewidth=2, capsize=0)
    selected = part["selected"].to_numpy()
    ax.scatter(part["f1_macro_mean"][selected], y[selected], s=34,
               color=SERIES, edgecolors=SERIES, linewidths=1.5, zorder=3,
               label="selected in its family")
    ax.scatter(part["f1_macro_mean"][~selected], y[~selected], s=34,
               facecolors=SURFACE, edgecolors=SERIES, linewidths=1.5, zorder=3,
               label="not selected")
    ax.set_yticks(y, labels)
    ax.set_xlabel("F1-macro, mean of 5 folds inside train "
                  "(bar = one standard error)")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    # The upper left corner is empty: every configuration but one scores
    # above 0.85, so the legend cannot cover a mark there.
    ax.legend(frameon=False, loc="upper left")
    fig.suptitle(f"Cross-validation F1-macro by configuration, dataset "
                 f"{dataset}", x=0.02, ha="left", fontsize=11,
                 fontweight="bold")
    ax.set_title("Number of features in parentheses.", loc="left",
                 fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    """Draw every figure for the datasets present in the result tables."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    confusion = pd.read_csv(EVALUATION_DIR / "confusion_test.csv")
    test = pd.read_csv(EVALUATION_DIR / "test_results.csv")
    cv = pd.read_csv(EVALUATION_DIR / "cv_results.csv")

    for row in test.itertuples():
        part = confusion[(confusion["dataset"] == row.dataset)
                         & (confusion["config"] == row.config)]
        path = FIGURES_DIR / f"confusion_{row.dataset}_{row.config}.png"
        plot_confusion(part, row.dataset, row.config, row.n_features, path)
        print(f"written {path.name}")
    for dataset, part in cv.groupby("dataset"):
        path = FIGURES_DIR / f"cv_comparison_{dataset}.png"
        plot_cv_comparison(part, dataset, path)
        print(f"written {path.name}")


if __name__ == "__main__":
    main()
