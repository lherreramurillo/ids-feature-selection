"""Stage 6: final feature lists and the handoff package v2.

What it does
    1. Writes the final feature list of each dataset: the features of the
       combined pipeline setting that the selection rule chose in stage 5.
       These lists were fitted on the full training set of each dataset.
    2. Compares the lists of A and B (shared and dataset-specific features).
    3. Builds the handoff package v2 for the team that trains the models: the
       two lists, the updated notes and a README that explains how to use the
       lists and what the data-source finding means for their models.
       v2 is an add-on to v1: it does not repeat the data files.

Inputs
    results/evaluation/cv_results.csv        which setting the rule selected
    results/evaluation/selected_features.csv features of that setting
    results/evaluation/test_results.csv, source_probe.csv,
    source_pair_confusion.csv                numbers quoted in the README
    results/filters/feature_scores_{A,B}.csv scores of every feature
    report/decisions.md

Outputs
    results/final_features_A.csv, results/final_features_B.csv
        Columns: feature, chi2_rank_in_list, chi2, cramers_v, f_classif,
        also_in_other_dataset. Ordered by chi-square, best first.
    results/final_features_comparison.csv
        One row per feature that is in at least one list.
    data/handoff/v2/ and data/handoff/feature_selection_handoff_v2.zip

How to run
    .venv/bin/python src/06_deliverables.py

Deterministic: reads fixed tables, and the zip stores fixed timestamps.
"""
import shutil
import zipfile

import pandas as pd

from config import (CLEANING_NOTES_FILE, DECISIONS_FILE, EVALUATION_DIR, FEATURE_SCORES_FILE,
                    FILTERS_DIR, FINAL_FAMILY, FINAL_FEATURES_FILE,
                    HANDOFF_DIR, HANDOFF_NAME, HANDOFF_V2_NAME, REPORT_DIR,
                    RESULTS_DIR, TARGET)

DATASETS = ("A", "B")

README = """# Feature selection handoff, v2

Add-on to `{v1_name}.zip`. The data files and the train/test row ids of v1 are
unchanged and are still the ones to use. This package adds the selected
features and one finding you need to know before interpreting your metrics.

## Files

| File | What it is |
|---|---|
| `final_features_A.csv` | The {n_a} features selected for dataset A, best first. |
| `final_features_B.csv` | The {n_b} features selected for dataset B, best first. |
| `decisions.md` | Updated decision log (replaces the copy in v1). |
| `cleaning_notes.md` | Corrected cleaning notes (replaces the copy in v1). |
| `README.md` | This file. |

Columns of the feature files: `chi2_rank_in_list` (1 = highest chi-square
score), `feature` (column name in the parquet files), `chi2`, `cramers_v`,
`f_classif` (scores against the label, computed on the training rows) and
`also_in_other_dataset`.

A and B have different lists ({n_shared} features in common). Use each list
with its own dataset.

## How the lists were obtained

Three filters applied in sequence, fitted on the training rows only:
Low Variance (drop features where one value covers at least 99.9% of the
rows), Pearson (from every group of features with absolute correlation above
0.90 keep the one most related to the label), chi-square (keep the top k of
the survivors). k was chosen by cross-validation inside train: {k_a} for A,
{k_b} for B.

Test results with a RandomForest (200 trees, balanced class weights):

| Dataset | Features | F1-macro | PR-AUC macro | MCC |
|---|---|---|---|---|
{test_rows}

## Loading example

Needs `pandas` and `pyarrow`. Paths assume the v1 files are in the same
folder.

```python
import pandas as pd

TARGET = "{target}"

df = pd.read_parquet("A_clean.parquet")
features = pd.read_csv("final_features_A.csv")["feature"].tolist()
train = df.loc[pd.read_csv("A_train_idx.csv")["row_id"]]
test = df.loc[pd.read_csv("A_test_idx.csv")["row_id"]]

X_train, y_train = train[features], train[TARGET]
X_test, y_test = test[features], test[TARGET]

print(X_train.shape, X_test.shape)
```

The rules of v1 still apply: use these row ids, do not split again, fit every
transformation on train only.

## Important: high metrics do not mean the model recognizes the attack

Each dataset mixes rows from two sources: an external dataset (IDS2025) and
our lab captures. They were produced in different environments and with
different flow extractors. We tested whether the features give away the
source of a row, and they do:

- **A model can tell the source almost without error.** Using only `benign`
  rows (the one label present in both sources, so the traffic type is the
  same), a RandomForest trained to answer "external or lab?" misclassified
  {probe_a_err} of {probe_a_rows:,} test rows in A and {probe_b_err} of
  {probe_b_rows:,} in B, with the final feature lists.
- **Labels that name the same attack are never confused.** Across all the
  configurations scored on test there were {pair_errors} confusions between
  `Brute Force` and `bruteforce`, between `PortScan` and `recon`, and between
  `Web Attack` and `sqli`, in either direction.

What this means for your models:

1. A model first separates the two sources and then separates classes inside
   each source. Part of the 12-class problem is therefore solved by
   recognizing the environment, not the attack.
2. A high F1 or accuracy on this test set is not evidence that the model
   would detect the same attack type in traffic from another network. Do not
   claim generalization from these numbers.
3. Do not read feature importance as "what characterizes an attack". A
   feature can be important because it separates lab traffic from external
   traffic.
4. If you want evidence of attack detection, the test that would give it is
   cross-source: train on one source and evaluate on the equivalent labels of
   the other. It has not been done.

Other limits that still apply: `Infiltration` has 6 test rows (4 of 6
detected by the reference model, in every configuration), `sqli` has 11, and
in all lab rows only one direction of the flow was extracted.
"""


def final_config(cv_results, name):
    """Return the configuration the selection rule chose for the final list."""
    rows = cv_results[(cv_results["dataset"] == name)
                      & (cv_results["family"] == FINAL_FAMILY)
                      & cv_results["selected"]]
    # Exactly one setting per family is flagged by the rule; anything else
    # means the evaluation tables are not the ones this script expects.
    assert len(rows) == 1, f"{name}: expected one selected {FINAL_FAMILY} row"
    return rows["config"].iloc[0]


def final_features(name, config, selected_features):
    """Build the final feature table of one dataset, best chi-square first."""
    chosen = selected_features[(selected_features["dataset"] == name)
                               & (selected_features["config"] == config)]
    scores = pd.read_csv(FILTERS_DIR / FEATURE_SCORES_FILE.format(name=name),
                         index_col="feature")
    table = scores.loc[chosen["feature"], ["chi2", "cramers_v", "f_classif"]]
    table = table.sort_values("chi2", ascending=False, kind="stable")
    table.insert(0, "chi2_rank_in_list", range(1, len(table) + 1))
    return table.reset_index()


def compare_lists(lists):
    """Return one row per feature present in at least one final list."""
    a, b = (set(lists[name]["feature"]) for name in DATASETS)
    rows = [{"feature": f, "in_A": f in a, "in_B": f in b}
            for f in sorted(a | b)]
    table = pd.DataFrame(rows)
    table["status"] = table.apply(
        lambda r: "shared" if r["in_A"] and r["in_B"]
        else ("only_A" if r["in_A"] else "only_B"), axis=1)
    return table.sort_values(["status", "feature"]).reset_index(drop=True)


def build_readme(lists, configs):
    """Fill the handoff README with the numbers of the current results."""
    test = pd.read_csv(EVALUATION_DIR / "test_results.csv")
    probe = pd.read_csv(EVALUATION_DIR / "source_probe.csv")
    pairs = pd.read_csv(EVALUATION_DIR / "source_pair_confusion.csv")

    test_rows, probe_numbers = [], {}
    for name in DATASETS:
        for config, label in (("baseline", "baseline, all features"),
                              (configs[name], "final list")):
            row = test[(test["dataset"] == name)
                       & (test["config"] == config)].iloc[0]
            test_rows.append(
                f"| {name} | {row['n_features']}, {label} | "
                f"{row['f1_macro']:.4f} | {row['pr_auc_macro']:.4f} | "
                f"{row['mcc']:.4f} |")
        row = probe[(probe["dataset"] == name)
                    & (probe["scope"] == "benign_only")
                    & (probe["config"] == configs[name])].iloc[0]
        probe_numbers[name] = (int(row["misclassified_rows"]),
                               int(row["external_test_rows"]
                                   + row["lab_test_rows"]))
    shared = set(lists["A"]["feature"]) & set(lists["B"]["feature"])
    return README.format(
        v1_name=HANDOFF_NAME, target=TARGET,
        n_a=len(lists["A"]), n_b=len(lists["B"]), n_shared=len(shared),
        k_a=len(lists["A"]), k_b=len(lists["B"]),
        test_rows="\n".join(test_rows),
        probe_a_err=probe_numbers["A"][0], probe_a_rows=probe_numbers["A"][1],
        probe_b_err=probe_numbers["B"][0], probe_b_rows=probe_numbers["B"][1],
        pair_errors=int(pairs["external_predicted_as_lab"].sum()
                        + pairs["lab_predicted_as_external"].sum()),
    )


def make_zip(files, zip_path):
    """Zip ``files`` inside one folder with fixed timestamps (identical reruns)."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(files, key=lambda p: p.name):
            info = zipfile.ZipInfo(f"{HANDOFF_V2_NAME}/{path.name}",
                                   date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, path.read_bytes())


def main():
    """Write the final lists, their comparison and the handoff package v2."""
    cv_results = pd.read_csv(EVALUATION_DIR / "cv_results.csv")
    selected_features = pd.read_csv(EVALUATION_DIR / "selected_features.csv")

    configs = {name: final_config(cv_results, name) for name in DATASETS}
    lists = {name: final_features(name, configs[name], selected_features)
             for name in DATASETS}
    comparison = compare_lists(lists)
    comparison.to_csv(RESULTS_DIR / "final_features_comparison.csv", index=False)

    package_dir = HANDOFF_DIR / "v2"
    package_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for name in DATASETS:
        other = DATASETS[1 - DATASETS.index(name)]
        lists[name]["also_in_other_dataset"] = lists[name]["feature"].isin(
            set(lists[other]["feature"]))
        path = RESULTS_DIR / FINAL_FEATURES_FILE.format(name=name)
        lists[name].to_csv(path, index=False)
        files.append(shutil.copyfile(path, package_dir / path.name))
        print(f"\n=== Dataset {name}: {configs[name]}, "
              f"{len(lists[name])} features ===")
        print(lists[name].to_string(index=False,
                                    float_format=lambda v: f"{v:.4g}"))

    # Both notes replace the copies shipped in v1, which were written before
    # the later decisions and corrections.
    for note in (DECISIONS_FILE, CLEANING_NOTES_FILE):
        files.append(shutil.copyfile(REPORT_DIR / note, package_dir / note))
    readme = package_dir / "README.md"
    readme.write_text(build_readme(lists, configs))
    files.append(readme)
    zip_path = HANDOFF_DIR / f"{HANDOFF_V2_NAME}.zip"
    make_zip(files, zip_path)

    print("\n=== A vs B ===")
    for status, part in comparison.groupby("status"):
        print(f"{status} ({len(part)}): {', '.join(part['feature'])}")
    print(f"\nZip: {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
