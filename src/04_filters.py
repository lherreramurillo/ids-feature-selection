"""Stage 4: the three filter methods (Low Variance, Pearson, Chi-square).

What it does
    Scores every feature with three filter methods and records which features
    each method keeps at several thresholds. Filters are not models: nothing
    is trained and no feature is removed from the data here. The subsets are
    evaluated with a classifier in stage 5.
    Everything is computed on the TRAINING rows only, separately for A and B,
    so the test rows have no influence on which features are selected.

    1. Low Variance, two rules:
       - quasi-constant: share of rows taken by the most frequent value;
       - variance after MinMax scaling (sklearn VarianceThreshold).
    2. Pearson: absolute pairwise correlation; from every group of features
       correlated above the threshold only the one most related to the target
       (highest ANOVA F score) is kept.
    3. Chi-square: features are discretized into quantile bins and scored with
       sklearn chi2 against the label; the top k features are kept.

Inputs
    data/interim/{A,B}_clean.parquet and data/interim/{A,B}_train_idx.csv
    Thresholds from src/config.py

Outputs (results/filters/)
    feature_scores_{A,B}.csv   one row per feature with every score
    pearson_pairs_{A,B}.csv    feature pairs with |r| above the lowest threshold
    feature_subsets_{A,B}.csv  long table: method, setting, kept feature
    filter_summary.csv         features kept and removed per method and setting

How to run
    .venv/bin/python src/04_filters.py

Deterministic: no random step is used (the discretizer does not subsample).
"""
import numpy as np
import pandas as pd
from scipy import stats

from config import (CHI2_N_BINS, FEATURE_SCORES_FILE, FEATURE_SUBSETS_FILE,
                    FILTER_SUMMARY_FILE, FILTERS_DIR, PEARSON_PAIRS_FILE,
                    PEARSON_THRESHOLDS)
from data_io import load_split
from filters import (chi2_scores, chi2_subsets, correlated_pairs,
                     correlation_matrix, pearson_keep, relevance_scores,
                     variance_scores, variance_subsets)


# --------------------------------------------------------------------------
def run_dataset(name):
    """Run the three filters on the training rows of one dataset."""
    X, y = load_split(name, "train")
    print(f"\n{'=' * 70}\nDataset {name}: train {X.shape[0]} rows, "
          f"{X.shape[1]} features\n{'=' * 70}")

    var = variance_scores(X, y)
    relevance = relevance_scores(X, y).rename("f_classif")
    corr = correlation_matrix(X)
    chi = chi2_scores(X, y)

    subsets = variance_subsets(var)
    for thr in PEARSON_THRESHOLDS:
        subsets[("pearson", thr)] = pearson_keep(corr, relevance, thr)
    subsets.update(chi2_subsets(chi))

    scores = var.join(relevance).join(chi)
    scores["max_abs_r_with_other"] = (
        corr.where(~np.eye(len(corr), dtype=bool)).max(axis=1))
    scores.to_csv(FILTERS_DIR / FEATURE_SCORES_FILE.format(name=name))
    correlated_pairs(corr, min(PEARSON_THRESHOLDS)).to_csv(
        FILTERS_DIR / PEARSON_PAIRS_FILE.format(name=name), index=False)
    pd.DataFrame(
        [(m, s, f) for (m, s), feats in subsets.items() for f in feats],
        columns=["method", "setting", "feature"],
    ).to_csv(FILTERS_DIR / FEATURE_SUBSETS_FILE.format(name=name), index=False)

    all_features = list(X.columns)
    summary = pd.DataFrame([{
        "dataset": name, "method": m, "setting": s,
        "n_kept": len(feats), "n_removed": len(all_features) - len(feats),
        "removed": "; ".join(f for f in all_features if f not in set(feats))
        if m != "chi2_top_k" else "",
    } for (m, s), feats in subsets.items()])

    print("\nFeatures kept / removed per filter setting:")
    print(summary.drop(columns=["dataset", "removed"]).to_string(index=False))

    print("\nLow variance, features with top value share >= 0.95:")
    low = var[var["top_value_share"] >= 0.95].sort_values(
        "top_value_share", ascending=False)
    print(low[["top_value_share", "variance_minmax", "unusual_rows",
               "unusual_top_label", "unusual_top_label_coverage_pct"]]
          .to_string(float_format=lambda v: f"{v:.6g}"))

    for thr in PEARSON_THRESHOLDS:
        print(f"\nPearson > {thr}: "
              f"{len(correlated_pairs(corr, thr))} pairs above the threshold")
    print("\nChi-square, top 15:")
    print(chi.sort_values("chi2_rank").head(15)[
        ["chi2", "chi2_bins", "cramers_v"]].join(relevance)
        .to_string(float_format=lambda v: f"{v:.4g}"))
    rho = stats.spearmanr(chi["chi2"], chi["cramers_v"])[0]
    print(f"Spearman correlation between chi2 and Cramer's V rankings: {rho:.3f}")
    print(f"Features with fewer than {CHI2_N_BINS} bins: "
          f"{(chi['chi2_bins'] < CHI2_N_BINS).sum()} of {len(chi)}")
    return summary


def main():
    """Run the filters on A and B and write the combined summary."""
    FILTERS_DIR.mkdir(parents=True, exist_ok=True)
    summaries = [run_dataset(name) for name in ("A", "B")]
    pd.concat(summaries, ignore_index=True).to_csv(
        FILTERS_DIR / FILTER_SUMMARY_FILE, index=False)
    print(f"\nWritten: {FILTERS_DIR}")


if __name__ == "__main__":
    main()
