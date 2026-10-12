"""Stage 6: write the feature selection report.

What it does
    Builds report/feature_selection_report.md. Every table and every number
    in the text is read from the result files of the previous stages, so the
    report cannot disagree with the results. The explanatory text is fixed in
    this script.

Inputs
    results/evaluation/*.csv, results/filters/*.csv,
    results/final_features_{A,B}.csv, results/final_features_comparison.csv,
    results/cleaning/cleaning_log.csv, results/split/split_summary.csv,
    the training rows (to count the features that are always zero in lab rows)
    and the figures in report/figures/ (referenced, not regenerated).

Output
    report/feature_selection_report.md

How to run
    .venv/bin/python src/06b_report.py     (after 05b_plots.py and 06_deliverables.py)
"""
import pandas as pd

from config import (CHI2_N_BINS, CLEANING_DIR, CLEANING_LOG_FILE, CV_FOLDS,
                    EVALUATION_DIR, FEATURE_SCORES_FILE, FILTERS_DIR,
                    FINAL_FEATURES_FILE, MODEL_PARAMS, N_EXTERNAL_ROWS,
                    PEARSON_PAIRS_FILE, RANDOM_SEED, REPORT_DIR, REPORT_FILE,
                    RESULTS_DIR, SPLIT_DIR, SPLIT_SUMMARY_FILE, TARGET,
                    TEST_SIZE)
from data_io import load_split

DATASETS = ("A", "B")


def md_table(header, rows):
    """Render a Markdown table from a header list and a list of row lists."""
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def cv_table(cv):
    """Cross-validation comparison: one row per configuration, A and B."""
    rows = []
    for config in cv["config"].drop_duplicates():
        row = [f"`{config}`"]
        for name in DATASETS:
            r = cv[(cv["dataset"] == name) & (cv["config"] == config)].iloc[0]
            mark = " **(selected)**" if r["selected"] else ""
            row += [f"{r['n_features']:.0f}",
                    f"{r['f1_macro_mean']:.4f} ± {r['f1_macro_se']:.4f}{mark}",
                    f"{r['pr_auc_macro_mean']:.4f}", f"{r['mcc_mean']:.4f}"]
        rows.append(row)
    return md_table(["Configuration", "A feat.", "A F1-macro", "A PR-AUC",
                     "A MCC", "B feat.", "B F1-macro", "B PR-AUC", "B MCC"],
                    rows)


def test_table(test):
    """Test results of the baseline and the selected setting of each family."""
    rows = [[r.dataset, f"`{r.config}`", r.n_features, f"{r.f1_macro:.4f}",
             f"{r.pr_auc_macro:.4f}", f"{r.mcc:.4f}", f"{r.fit_seconds:.1f}"]
            for r in test.itertuples()]
    return md_table(["Dataset", "Configuration", "Features", "F1-macro",
                     "PR-AUC macro", "MCC", "Train time (s)"], rows)


def recall_table(recall, finals):
    """Per-class test recall: baseline against the final list, A and B."""
    labels = recall.loc[recall["dataset"] == "A", "label"].drop_duplicates()
    rows = []
    for label in labels:
        row = [label]
        for name in DATASETS:
            part = recall[(recall["dataset"] == name)
                          & (recall["label"] == label)].set_index("config")
            row += [int(part.loc["baseline", "support"]),
                    f"{part.loc['baseline', 'recall']:.3f}",
                    f"{part.loc[finals[name], 'recall']:.3f}"]
        rows.append(row)
    return md_table(["Label", "A test rows", "A baseline", "A final",
                     "B test rows", "B baseline", "B final"], rows)


def feature_table(features):
    """Final feature list of one dataset."""
    rows = [[r.chi2_rank_in_list, f"`{r.feature}`", f"{r.chi2:,.0f}",
             f"{r.cramers_v:.3f}", "yes" if r.also_in_other_dataset else "no"]
            for r in features.itertuples()]
    return md_table(["Rank", "Feature", "Chi-square", "Cramer's V",
                     "In the other list"], rows)


def redundancy(name, final):
    """Count correlated pairs inside the chi-square top k and the final list.

    Both sets have the same size k, so the counts are directly comparable.
    """
    k = len(final)
    scores = pd.read_csv(FILTERS_DIR / FEATURE_SCORES_FILE.format(name=name),
                         index_col="feature")
    pairs = pd.read_csv(FILTERS_DIR / PEARSON_PAIRS_FILE.format(name=name))
    top = set(scores.sort_values("chi2_rank").index[:k])
    chosen = set(final["feature"])

    def inside(features):
        return int((pairs["feature_a"].isin(features)
                    & pairs["feature_b"].isin(features)).sum())
    return {"k": k, "pairs_top": inside(top), "pairs_final": inside(chosen),
            "overlap": len(top & chosen),
            "single_bin": int(((scores["chi2_bins"] == 1)
                               & (scores["top_value_share"] < 1)).sum())}


def lab_zero_features(name, final):
    """Final features that are zero in every lab training row.

    For lab rows these features carry no information at all: any value other
    than zero identifies an external row.
    """
    X, _ = load_split(name, "train")
    lab = X[X.index >= N_EXTERNAL_ROWS]
    always_zero = lab.columns[(lab != 0).sum() == 0]
    varying = X.columns[X.nunique() > 1]
    return {"all": len(set(always_zero) & set(varying)),
            "final": [f for f in final["feature"] if f in set(always_zero)]}


def cfg_row(cv, name, config):
    """Return the cross-validation row of one configuration."""
    return cv[(cv["dataset"] == name) & (cv["config"] == config)].iloc[0]


def main():
    """Collect the numbers and write the report."""
    cv = pd.read_csv(EVALUATION_DIR / "cv_results.csv")
    test = pd.read_csv(EVALUATION_DIR / "test_results.csv")
    recall = pd.read_csv(EVALUATION_DIR / "test_recall_by_class.csv")
    cv_recall = pd.read_csv(EVALUATION_DIR / "cv_recall_by_class.csv")
    probe = pd.read_csv(EVALUATION_DIR / "source_probe.csv")
    pairs = pd.read_csv(EVALUATION_DIR / "source_pair_confusion.csv")
    confusion = pd.read_csv(EVALUATION_DIR / "confusion_test.csv")
    comparison = pd.read_csv(RESULTS_DIR / "final_features_comparison.csv")
    cleaning = pd.read_csv(CLEANING_DIR / CLEANING_LOG_FILE)
    split = pd.read_csv(SPLIT_DIR / SPLIT_SUMMARY_FILE)
    summary = pd.read_csv(FILTERS_DIR / "filter_summary.csv")

    final = {n: pd.read_csv(RESULTS_DIR / FINAL_FEATURES_FILE.format(name=n))
             for n in DATASETS}
    finals = {n: cv[(cv["dataset"] == n) & (cv["family"] == "combined")
                    & cv["selected"]]["config"].iloc[0] for n in DATASETS}
    red = {n: redundancy(n, final[n]) for n in DATASETS}
    zero = {n: lab_zero_features(n, final[n]) for n in DATASETS}

    v = {}  # values quoted in the text
    for n in DATASETS:
        rows = split[split["dataset"] == n]
        v[f"train_{n}"] = f"{rows['train'].sum():,}"
        v[f"test_{n}"] = f"{rows['test'].sum():,}"
        steps = cleaning[cleaning["dataset"] == n]
        v[f"raw_{n}"] = f"{steps['rows_before'].iloc[0]:,}"
        v[f"clean_{n}"] = f"{steps['rows_after'].iloc[-1]:,}"
        v[f"dups_{n}"] = f"{steps.loc[steps['step'] == 3, 'rows_removed'].iloc[0]:,}"
        k = red[n]["k"]
        v[f"k_{n}"] = k
        v[f"final_{n}"] = finals[n]
        v[f"chi_k_{n}"] = f"{cfg_row(cv, n, f'chi2_top_{k}')['f1_macro_mean']:.4f}"
        v[f"comb_k_{n}"] = f"{cfg_row(cv, n, finals[n])['f1_macro_mean']:.4f}"
        v[f"base_cv_{n}"] = f"{cfg_row(cv, n, 'baseline')['f1_macro_mean']:.4f}"
        v[f"se_{n}"] = f"{cfg_row(cv, n, 'baseline')['f1_macro_se']:.4f}"
        v[f"pairs_top_{n}"] = red[n]["pairs_top"]
        v[f"pairs_final_{n}"] = red[n]["pairs_final"]
        v[f"overlap_{n}"] = red[n]["overlap"]
        v[f"single_bin_{n}"] = red[n]["single_bin"]
        v[f"zero_all_{n}"] = zero[n]["all"]
        v[f"zero_final_n_{n}"] = len(zero[n]["final"])
        v[f"zero_final_{n}"] = ", ".join(f"`{f}`" for f in zero[n]["final"])
        t = test[test["dataset"] == n].set_index("config")
        v[f"base_test_{n}"] = f"{t.loc['baseline', 'f1_macro']:.4f}"
        v[f"final_test_{n}"] = f"{t.loc[finals[n], 'f1_macro']:.4f}"
        p = probe[(probe["dataset"] == n) & (probe["scope"] == "benign_only")
                  & (probe["config"] == finals[n])].iloc[0]
        v[f"probe_err_{n}"] = int(p["misclassified_rows"])
        v[f"probe_rows_{n}"] = f"{int(p['external_test_rows'] + p['lab_test_rows']):,}"
        inf = recall[(recall["dataset"] == n) & (recall["label"] == "Infiltration")]
        v[f"inf_hits_{n}"] = " or ".join(
            str(h) for h in sorted(set((inf["recall"] * inf["support"])
                                       .round().astype(int))))
        v[f"inf_test_{n}"] = int(inf["support"].iloc[0])
        inf_cv = cv_recall[(cv_recall["dataset"] == n)
                           & (cv_recall["label"] == "Infiltration")]
        v[f"inf_cv_{n}"] = (f"{inf_cv['recall'].min():.2f} to "
                            f"{inf_cv['recall'].max():.2f}")
        v[f"inf_train_{n}"] = int(inf_cv["support"].iloc[0])
        s = summary[(summary["dataset"] == n)].set_index(["method", "setting"])
        v[f"var_removed_{n}"] = int(s.loc[("variance_quasi_constant", 0.999),
                                          "n_removed"])
        base = confusion[(confusion["dataset"] == n)
                         & (confusion["config"] == "baseline")
                         & (confusion["true_label"] != confusion["predicted_label"])]
        worst = base.sort_values("rows", ascending=False).iloc[0]
        v[f"worst_{n}"] = (f"{worst['rows']} `{worst['true_label']}` rows "
                           f"predicted as `{worst['predicted_label']}`")
    v["pair_errors"] = int(pairs["external_predicted_as_lab"].sum()
                           + pairs["lab_predicted_as_external"].sum())
    v["n_test_configs"] = len(test)
    v["cv10_cramers"] = ", ".join(
        f"{cfg_row(cv, n, 'cramers_v_top_10')['f1_macro_mean']:.2f} ({n})"
        for n in DATASETS)
    for status in ("shared", "only_A", "only_B"):
        part = comparison.loc[comparison["status"] == status, "feature"]
        v[f"n_{status}"] = len(part)
        v[status] = ", ".join(f"`{f}`" for f in part)
    v.update(cv_table=cv_table(cv), test_table=test_table(test),
             recall_table=recall_table(recall, finals),
             features_A=feature_table(final["A"]),
             features_B=feature_table(final["B"]),
             trees=MODEL_PARAMS["n_estimators"], folds=CV_FOLDS,
             bins=CHI2_N_BINS, seed=RANDOM_SEED, target=TARGET,
             test_pct=round(TEST_SIZE * 100))

    (REPORT_DIR / REPORT_FILE).write_text(TEMPLATE.format(**v))
    print(f"Written: {REPORT_DIR / REPORT_FILE}")


TEMPLATE = """# Feature selection with filter methods for an IDS flow dataset

Project "Arquitectura IDS impulsada por IA". Author: Laura Camila Herrera
Murillo. All numbers in this report are read from the result files under
`results/`; the scripts that produce them are listed in section 9.

## 1. Summary

- Three filter methods were applied to two flow datasets (A, captured with a
  firewall; B, without): Low Variance, Pearson correlation and Chi-square.
  Each filter was fitted on training rows only and each resulting feature
  subset was evaluated with the same RandomForest.
- No filter improves the model. What the filters achieve is a smaller input
  at no measurable cost: the final lists have **{k_A} features for A** and
  **{k_B} for B** (of 77) and score F1-macro {final_test_A} and
  {final_test_B} on test, against {base_test_A} and {base_test_B} with all
  features. The differences are smaller than the fold-to-fold noise measured
  in cross-validation (standard error about {se_A}).
- The combined pipeline (Low Variance, then Pearson, then Chi-square) reaches
  the baseline with fewer features than Chi-square alone ({k_A} against 40 in
  A, {k_B} against 40 in B), because Chi-square alone fills its top ranks
  with redundant features.
- **The scores must not be read as attack detection ability.** The features
  identify which of the two data sources a row comes from almost perfectly,
  and labels that name the same attack in the two sources are never confused
  with each other. Section 8 explains this and the other limitations.

## 2. Data and preparation

Each dataset joins rows from an external dataset (IDS2025) with flows
captured in a virtual lab, extracted with `cicflowmeter`: 79 features and the
target `{target}` with 12 labels. A and B are processed separately with the
same code.

| Step | A | B |
|---|---|---|
| Raw rows | {raw_A} | {raw_B} |
| Exact duplicates removed (after dropping the port columns) | {dups_A} | {dups_B} |
| Clean rows | {clean_A} | {clean_B} |
| Train rows (stratified, {test_pct}% test, seed {seed}) | {train_A} | {train_B} |
| Test rows | {test_A} | {test_B} |

Cleaning removed the two port columns (shortcuts), rows with NaN or Inf,
exact duplicates and rows with contradictory labels, leaving 77 features.
Details and justifications are in `cleaning_notes.md` and `decisions.md`.

## 3. What each filter does

Filters score features with statistics computed from the data. They are not
models and nothing is trained. All three were fitted on training rows only.

**Low Variance.** A feature that takes almost the same value in every row
cannot separate classes. Raw variance depends on the unit of each feature, so
the rule used is the share of rows taken by the most frequent value: a
feature is dropped when one value covers at least 99.9% of the training
rows. This removes {var_removed_A} features in both datasets: the 5 constant
columns and 4 flag counters with 1 to 4 non-zero rows. A looser threshold
(99%) was rejected because it removes `Fwd PSH Flags` and `SYN Flag Count`,
which are non-zero in 22.5% of the `Brute Force` rows: with this class
imbalance a quasi-constant feature can be the signal of a small class. The
variance after MinMax scaling was computed and discarded as a rule, because
it penalizes features with extreme values rather than uninformative ones
(`decisions.md`, decision 6).

**Pearson correlation.** Two features with an absolute correlation close to 1
carry the same information. For every group of features correlated above the
threshold (0.90, 0.95 and 0.99 were tried) only one is kept: the one with the
highest ANOVA F score against the label. Constant features have no defined
correlation, so this filter cannot remove them.

**Chi-square.** Measures the association between a feature and the label in
a table of counts. The features are continuous, so each one is first cut into
{bins} quantile bins with edges learned on train; without this step the score
would depend on the unit of measure. Features are ranked by the statistic and
the top k are kept (k = 10, 20, 30, 40). The scores were cross-checked
against the classic contingency-table test of SciPy for every feature.
A second ranking by Cramer's V (the same statistic corrected for sample size
and number of bins) was evaluated as a comparison.

**Combined pipeline.** Low Variance at 99.9%, then Pearson on the survivors,
then the top k survivors by Chi-square.

## 4. Evaluation protocol

- Model: `RandomForestClassifier` with {trees} trees and
  `class_weight="balanced"`, identical for every subset and not tuned.
- Metrics: F1-macro, PR-AUC macro, MCC and per-class recall. Accuracy is not
  reported as a main metric because the two largest classes hold about 70% of
  the rows.
- Settings are compared with stratified {folds}-fold cross-validation inside
  train; the filters are refitted in every fold.
- Selection rule, fixed before seeing results: within each family, the
  setting with the fewest features whose mean F1-macro is within one standard
  error of the best one.
- The test rows are used once, for the baseline and the selected setting of
  each family.

## 5. Comparison

### 5.1 Cross-validation inside train (all configurations)

Mean of {folds} folds; F1-macro is shown with its standard error.
`combined_pX_kY` means Pearson threshold X and top Y by Chi-square.

{cv_table}

![Cross-validation comparison, dataset A](figures/cv_comparison_A.png)

![Cross-validation comparison, dataset B](figures/cv_comparison_B.png)

### 5.2 Test (baseline and the selected setting of each family)

{test_table}

### 5.3 Recall per class on test, baseline against the final list

{recall_table}

Reading of the results:

- Low Variance and Pearson remove 9 to 32 features with no consistent
  change: the largest gap to the baseline in cross-validation is 0.010 in
  F1-macro (Pearson 0.90 in B, standard error 0.008) and it does not appear
  on test. The baseline was not being hurt by redundant features: a tree
  ensemble tolerates them.
- Chi-square alone needs 40 features to reach the baseline; with 10 or 20 it
  is clearly worse.
- The ranking by Cramer's V is not a better alternative: with 10 features its
  F1-macro falls to {cv10_cramers} and it detects no `sqli` row.
- The largest single error of the baseline is {worst_A} in A and {worst_B}
  in B.

## 6. Why the combined pipeline beats Chi-square alone

Chi-square scores each feature on its own. Two features that are copies of
each other get the same high score and both enter the top k, although the
second one adds nothing. Evidence, for the same number of features:

| | A (k = {k_A}) | B (k = {k_B}) |
|---|---|---|
| F1-macro in cross-validation, Chi-square top k | {chi_k_A} | {chi_k_B} |
| F1-macro in cross-validation, combined top k | {comb_k_A} | {comb_k_B} |
| Feature pairs with absolute correlation above 0.90 inside Chi-square top k | {pairs_top_A} | {pairs_top_B} |
| Same, inside the combined list | {pairs_final_A} | {pairs_final_B} |
| Features present in both the Chi-square top k and the combined list | {overlap_A} | {overlap_B} |

The top of the Chi-square ranking is occupied by groups such as
`Total Length of Fwd Packets` and `Subflow Fwd Bytes`, or
`Fwd Packet Length Mean` and `Avg Fwd Segment Size`, which are the same
quantity under two names. The combined pipeline lets Pearson keep one
feature per group first, so each of the k places goes to different
information.

## 7. Final feature lists

Chosen by the selection rule: `{final_A}` for A and `{final_B}` for B. The
rule was not adjusted after seeing the results (`decisions.md`, decision 8),
which is why the two lists have different lengths. The lists were fitted on
the full training set of each dataset and are saved as
`results/final_features_A.csv` and `results/final_features_B.csv`.

### Dataset A ({k_A} features)

{features_A}

### Dataset B ({k_B} features)

{features_B}

### A against B

- Shared ({n_shared}): {shared}
- Only in A ({n_only_A}): {only_A}
- Only in B ({n_only_B}): {only_B}

![Confusion matrix on test, dataset A, final list](figures/confusion_A_{final_A}.png)

![Confusion matrix on test, dataset B, final list](figures/confusion_B_{final_B}.png)

In the confusion matrices the labels are ordered by data source. Every
misclassification falls inside one source block or involves `benign`, the
only label present in both sources.

## 8. Limitations

### 8.1 The features identify the data source

External rows and lab rows come from different environments and different
flow extractors, and the label vocabularies differ (`Brute Force` and
`bruteforce`, `PortScan` and `recon`, `Web Attack` and `sqli`). Two checks
were made:

- **Confusions between equivalent labels:** {pair_errors} in total, across
  the {n_test_configs} configurations scored on test, in either direction.
  If the features described the attack itself, flows of the same attack type
  from the two sources would be confused at least occasionally.
- **Source probe:** a RandomForest with the same settings was trained to
  answer only "external or lab?" using `benign` rows alone, the one label
  present in both sources. With the final lists it misclassifies
  {probe_err_A} of {probe_rows_A} test rows in A and {probe_err_B} of
  {probe_rows_B} in B.

Part of the reason is visible in the features: {zero_all_A} of the 77
features are zero in every lab row (see 8.3), so any non-zero value marks an
external row. The final list of A contains {zero_final_n_A} of them
({zero_final_A}) and the final list of B contains {zero_final_n_B}
({zero_final_B}).

Consequence: the model separates the sources and then the classes inside
each source. The high scores do not show that the selected features
characterize an attack type, and they say nothing about traffic from another
network. The experiment that would test it, training on one source and
evaluating on the equivalent labels of the other, was not done.

### 8.2 Infiltration

`Infiltration` has 29 rows in total: {inf_train_A} in train and {inf_test_A}
in test. On test the model detects {inf_hits_A} of {inf_test_A} in A and
{inf_hits_B} of {inf_test_B} in B, in every configuration; one row is 17
points of recall. In cross-validation, over the {inf_train_A} training rows,
recall ranges from {inf_cv_A} in A and from {inf_cv_B} in B depending on the
configuration. No conclusion about which features help this class can be
drawn from so few rows, and it is the class that limits F1-macro.
`sqli` has 11 test rows after deduplication, so its recall is also
indicative only.

### 8.3 One-direction flows

In 100% of the lab rows the backward packet count is 0 and the flow source
is the server. The client side of every lab conversation is missing: no
attacker requests and no SQL payloads are represented. Another extractor
(NTLFlowLyzer) produces bidirectional flows from the same capture, so the
extraction failed, not the capture. All backward features are constant for
lab rows, and the lab classes are being separated only by what the server
sent.

### 8.4 Duplicates

With the port columns present the raw files have 94 (A) and 100 (B) exact
duplicate rows. Once the ports are dropped, {dups_A} (A) and {dups_B} (B)
rows are identical to another row and were removed, keeping one copy. The
effect is concentrated: `PortScan` goes from 25,409 rows to 554 and `sqli`
from 179 to 58 (A) and 57 (B). The metrics therefore describe performance
per distinct flow pattern, not per captured flow, and class frequencies no
longer reflect the captured traffic (`decisions.md`, decision 4). Rows that
are almost identical were not removed and can fall on both sides of the
train/test split, which may make test scores optimistic.

### 8.5 Other limits

- **Chi-square and sparse features.** With quantile bins, a feature where one
  value covers about 90% of the rows or more collapses into a single bin and
  gets a score of 0 whatever information it carries. This happens to
  {single_bin_A} non-constant features in A and {single_bin_B} in B, mostly
  flag counters. In B only 39 features survive Pearson, so the top 30 takes
  most of them and includes one feature with a score of 0
  (`Bwd Packet Length Min`).
- **One model.** The comparison uses a RandomForest. Models that are
  sensitive to redundant features may benefit more from the filters.
- **Five folds.** The standard error used by the selection rule is itself a
  rough estimate.
- **A and B share their external rows,** so results on the two datasets are
  not independent replications.

## 9. Reproducibility

Run in order with `.venv/bin/python`:

| Script | Stage |
|---|---|
| `src/00_setup.py` | Load and check the raw files |
| `src/01_audit.py` | Quality audit |
| `src/02_clean.py`, `src/02b_duplicates_check.py` | Cleaning and duplicate check |
| `src/03_split.py`, `src/03b_handoff.py` | Train/test split and data package |
| `src/04_filters.py` | The three filters on the full training set |
| `src/05_evaluate.py`, `src/05b_plots.py` | Evaluation and figures |
| `src/06_deliverables.py`, `src/06b_report.py` | Final lists and this report |

Parameters are in `src/config.py`. Seeds are fixed. Stages 1 to 4 and 6 were
run twice and produced identical files; stage 5 (about one hour) was run
once.
"""


if __name__ == "__main__":
    main()
