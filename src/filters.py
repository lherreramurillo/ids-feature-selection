"""Filter methods shared by stage 4 (scoring) and stage 5 (evaluation).

The three filters (Low Variance, Pearson, Chi-square) live here as plain
functions so that stage 5 can refit them inside every cross-validation fold
with exactly the same code stage 4 uses on the full training set.
Every function receives the rows it must be fitted on; none of them loads
data, so the caller decides which rows a filter is allowed to see.
"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import VarianceThreshold, chi2, f_classif
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler

from config import (CHI2_N_BINS, CHI2_TOP_K, COMBINED_TOP_K,
                    EVAL_QUASI_CONSTANT_THRESHOLD, PEARSON_THRESHOLDS,
                    QUASI_CONSTANT_THRESHOLDS, SCALED_VARIANCE_THRESHOLDS)

# --------------------------------------------------------------------------
# 1. Low Variance
# --------------------------------------------------------------------------
def variance_scores(X, y):
    """Compute the two low-variance measures for every feature.

    Besides the measures, it reports where the "unusual" rows (those that do
    not hold the most frequent value) are: a feature that is 99.9% zeros looks
    useless, but if its few non-zero rows all belong to one rare class it is
    exactly the signal that class needs. With this imbalance that check cannot
    be skipped.
    """
    # Raw variance is not comparable across features (microseconds vs packet
    # counts), so it is only reported. The threshold is applied after scaling
    # every feature to [0, 1].
    scaled = MinMaxScaler().fit_transform(X)
    scaled_var = VarianceThreshold().fit(scaled).variances_

    class_sizes = y.value_counts()
    rows = []
    for i, col in enumerate(X.columns):
        counts = X[col].value_counts()
        unusual = y[X[col] != counts.index[0]]
        if len(unusual):
            by_label = unusual.value_counts()
            top_label = by_label.index[0]
            # Share of that label's rows that carry an unusual value.
            coverage = by_label.iloc[0] / class_sizes[top_label] * 100
        else:
            top_label, coverage = "", 0.0
        rows.append({
            "feature": col,
            "variance_raw": X[col].var(ddof=0),
            "variance_minmax": scaled_var[i],
            "top_value_share": counts.iloc[0] / len(X),
            "unusual_rows": len(unusual),
            "unusual_top_label": top_label,
            "unusual_top_label_coverage_pct": round(coverage, 2),
        })
    return pd.DataFrame(rows).set_index("feature")


def variance_subsets(scores):
    """Return {(method, setting): kept features} for both low-variance rules."""
    subsets = {}
    for thr in QUASI_CONSTANT_THRESHOLDS:
        keep = scores.index[scores["top_value_share"] < thr]
        subsets[("variance_quasi_constant", thr)] = list(keep)
    for thr in SCALED_VARIANCE_THRESHOLDS:
        # Same rule as VarianceThreshold(threshold=thr): keep variance > thr.
        keep = scores.index[scores["variance_minmax"] > thr]
        subsets[("variance_minmax", thr)] = list(keep)
    return subsets


# --------------------------------------------------------------------------
# 2. Pearson correlation
# --------------------------------------------------------------------------
def relevance_scores(X, y):
    """ANOVA F score of every feature against the label.

    Used only to decide which feature of a correlated pair survives. F is the
    natural partner of Pearson: both measure linear separation. Constant
    features have no F score (0/0), which is set to 0, the lowest relevance.
    """
    with warnings.catch_warnings():
        # Constant features trigger a division warning; handled just below.
        warnings.simplefilter("ignore")
        f_scores, _ = f_classif(X, y)
    return pd.Series(np.nan_to_num(f_scores, nan=0.0), index=X.columns)


def correlation_matrix(X):
    """Absolute Pearson correlation between all features.

    Constant features have no defined correlation (zero variance). They are
    set to 0, so the Pearson filter never removes them: catching constants is
    the job of the Low Variance filter.
    """
    return X.corr(method="pearson").abs().fillna(0.0)


def pearson_keep(corr, relevance, threshold):
    """Select features so that no kept pair has |r| above ``threshold``.

    Features are visited from most to least related to the target; a feature
    is kept only if it is not correlated above the threshold with any feature
    already kept. This always keeps the more relevant member of a redundant
    pair and guarantees the result has no remaining pair above the threshold.
    """
    order = relevance.sort_values(ascending=False, kind="stable").index
    kept = []
    for feature in order:
        if not kept or corr.loc[feature, kept].max() <= threshold:
            kept.append(feature)
    # Back to the original column order, which is easier to read and compare.
    return [c for c in corr.columns if c in set(kept)]


def correlated_pairs(corr, threshold):
    """List the feature pairs with |r| above ``threshold``, strongest first."""
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack().rename("abs_r").reset_index()
    pairs.columns = ["feature_a", "feature_b", "abs_r"]
    pairs = pairs[pairs["abs_r"] > threshold]
    return pairs.sort_values(["abs_r", "feature_a", "feature_b"],
                             ascending=[False, True, True]).reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. Chi-square
# --------------------------------------------------------------------------
def chi2_scores(X, y, verify=True):
    """Chi-square statistic of every feature against the label.

    Why discretize: the chi-square test compares counts in a table of
    categories (feature value x class). The features here are continuous
    (durations, rates, lengths), and sklearn's chi2 would treat their raw
    values as if they were counts, which makes the score depend on the unit of
    measure. So each feature is first cut into quantile bins (same number of
    rows per bin where ties allow it), with bin edges learned on train.

    Each bin becomes a 0/1 column; sklearn chi2 scores every bin column, and
    the scores of the bins of one feature add up to the classic chi-square
    statistic of the (bin x class) table of that feature.

    ``verify`` re-computes every statistic with scipy as a cross-check. It is
    slow, so stage 5 turns it off inside the cross-validation folds.
    """
    binner = KBinsDiscretizer(
        n_bins=CHI2_N_BINS, encode="onehot", strategy="quantile",
        quantile_method="averaged_inverted_cdf",
        # No subsampling: bin edges use every training row and the result
        # does not depend on a random seed.
        subsample=None)
    with warnings.catch_warnings():
        # Features with many repeated values end up with fewer bins; sklearn
        # warns about it. The real number of bins is reported per feature.
        warnings.simplefilter("ignore")
        onehot = binner.fit_transform(X)
    bin_scores, _ = chi2(onehot, y)
    bin_scores = np.nan_to_num(bin_scores, nan=0.0)

    n_bins = binner.n_bins_
    starts = np.concatenate([[0], np.cumsum(n_bins)[:-1]])
    statistic = np.add.reduceat(bin_scores, starts)
    n_classes = y.nunique()
    dof = (n_bins - 1) * (n_classes - 1)
    # Cramer's V rescales the statistic to [0, 1] removing the effect of the
    # sample size and of the number of bins, which differs between features.
    denom = len(X) * np.minimum(n_bins - 1, n_classes - 1)
    cramers_v = np.sqrt(np.divide(statistic, denom, out=np.zeros(len(denom)),
                                  where=denom > 0))
    out = pd.DataFrame({
        "chi2": statistic,
        "chi2_bins": n_bins,
        "chi2_dof": dof,
        "chi2_p_value": stats.chi2.sf(statistic, np.maximum(dof, 1)),
        "cramers_v": cramers_v,
    }, index=X.columns)
    # A feature with a single bin (constant) has no table to test.
    out.loc[out["chi2_bins"] <= 1, ["chi2", "chi2_p_value"]] = [0.0, 1.0]
    out["chi2_rank"] = out["chi2"].rank(ascending=False, method="first").astype(int)

    if verify:
        check_against_scipy(X, y, binner, out)
    return out


def check_against_scipy(X, y, binner, scores):
    """Verify the summed sklearn scores equal scipy's contingency chi-square.

    This is the evidence for the claim in ``chi2_scores`` that adding the bin
    scores gives the classic test statistic.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        codes = KBinsDiscretizer(
            n_bins=CHI2_N_BINS, encode="ordinal", strategy="quantile",
            quantile_method="averaged_inverted_cdf", subsample=None
        ).fit_transform(X)
    for i, col in enumerate(X.columns):
        if scores.loc[col, "chi2_bins"] <= 1:
            continue
        table = pd.crosstab(codes[:, i], y.to_numpy())
        expected = stats.chi2_contingency(table, correction=False)[0]
        assert np.isclose(scores.loc[col, "chi2"], expected, rtol=1e-6), col


def chi2_subsets(scores):
    """Return {(method, k): the k features with the highest chi-square}."""
    ranked = scores.sort_values("chi2_rank").index
    ordered = list(scores.index)
    return {("chi2_top_k", k): [c for c in ordered if c in set(ranked[:k])]
            for k in CHI2_TOP_K}


# --------------------------------------------------------------------------
# Feature subsets evaluated in stage 5
# --------------------------------------------------------------------------
def top_k(ranking, k, column_order):
    """Return the ``k`` best features of ``ranking`` in original column order.

    ``ranking`` is a Series of scores (higher is better). Ties keep the
    original column order, so the result is deterministic.
    """
    best = set(ranking.sort_values(ascending=False, kind="stable").index[:k])
    return [c for c in column_order if c in best]


def evaluation_subsets(X, y, verify=False):
    """Fit every filter on (X, y) and return {config name: (family, features)}.

    This is the single definition of the configurations compared in stage 5.
    It must be called with training rows only: once per cross-validation fold
    (with that fold's training part) and once with the full training set for
    the final test evaluation.

    Families:
      baseline   all features
      variance   Low Variance, quasi-constant rule
      pearson    Pearson at each threshold
      chi2       top k by the chi-square statistic
      cramers_v  top k by Cramer's V (chi-square corrected for sample size and
                 number of bins)
      combined   Low Variance, then Pearson on the survivors, then the top k
                 survivors by chi-square
    """
    columns = list(X.columns)
    var = variance_scores(X, y)
    relevance = relevance_scores(X, y)
    corr = correlation_matrix(X)
    chi = chi2_scores(X, y, verify=verify)

    subsets = {"baseline": ("baseline", columns)}

    after_variance = list(
        var.index[var["top_value_share"] < EVAL_QUASI_CONSTANT_THRESHOLD])
    subsets[f"variance_qc_{EVAL_QUASI_CONSTANT_THRESHOLD}"] = (
        "variance", after_variance)

    for thr in PEARSON_THRESHOLDS:
        subsets[f"pearson_{thr:.2f}"] = (
            "pearson", pearson_keep(corr, relevance, thr))

    for k in CHI2_TOP_K:
        subsets[f"chi2_top_{k}"] = ("chi2", top_k(chi["chi2"], k, columns))
        subsets[f"cramers_v_top_{k}"] = (
            "cramers_v", top_k(chi["cramers_v"], k, columns))

    # Combined pipeline. Each step only sees the survivors of the previous
    # one. Chi-square is univariate, so the scores computed above are the same
    # ones the survivors would get if scored again.
    for thr in PEARSON_THRESHOLDS:
        after_pearson = pearson_keep(
            corr.loc[after_variance, after_variance],
            relevance[after_variance], thr)
        for k in COMBINED_TOP_K:
            subsets[f"combined_p{thr:.2f}_k{k}"] = (
                "combined", top_k(chi.loc[after_pearson, "chi2"], k, columns))
    return subsets
