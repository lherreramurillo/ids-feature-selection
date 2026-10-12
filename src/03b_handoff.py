"""Build the handoff package for the team that trains the models.

What it does
    Copies the clean datasets, the train/test row ids and the two notes into
    data/handoff/, writes a README for readers who do not know this project,
    and compresses everything into one zip file. Nothing is recomputed: the
    package is a snapshot of the outputs of stages 2 and 3.

Inputs
    data/interim/{A,B}_clean.parquet
    data/interim/{A,B}_train_idx.csv, data/interim/{A,B}_test_idx.csv
    report/decisions.md, report/cleaning_notes.md
    results/split/split_summary.csv (row counts quoted in the README)

Outputs (data/handoff/)
    A copy of every input file above except the split summary, README.md and
    feature_selection_handoff_v1.zip (all files inside one folder).

How to run
    .venv/bin/python src/03b_handoff.py

Re-running it gives a byte-identical zip: file order and the timestamps
stored inside the zip are fixed.
"""
import shutil
import zipfile

import pandas as pd

from config import (CLEAN_FILE, CLEANING_NOTES_FILE, DECISIONS_FILE,
                    HANDOFF_DIR, HANDOFF_NAME, INTERIM_DIR, RANDOM_SEED,
                    REPORT_DIR, ROW_ID, SPLIT_DIR,
                    SPLIT_SUMMARY_FILE, TARGET, TEST_IDX_FILE, TEST_SIZE,
                    TRAIN_IDX_FILE)

DATASETS = ("A", "B")

README = """# Feature selection handoff, v1

Clean datasets and a fixed train/test split for the project "Arquitectura IDS
impulsada por IA" (network intrusion detection). Train your models on these
files so that your results are comparable with the feature selection results.

## Background in five lines

- Traffic was captured in a virtual lab (attacker, vulnerable web/API target,
  benign host, firewall) and converted to flow features with `cicflowmeter`.
- Dataset **A** = capture with the firewall. Dataset **B** = capture without
  it. They are two separate datasets: never mix their rows.
- Each one also contains rows from an external dataset (IDS2025), which is why
  there are 12 labels with two naming styles (`Brute Force` and `bruteforce`).
- The raw files were cleaned (stage 2) and split into train and test (stage 3).
- This package does not contain the selected feature list yet. It will be
  delivered later as `final_features_A.csv` and `final_features_B.csv`.

## 1. Files

| File | What it is |
|---|---|
| `A_clean.parquet` | Dataset A after cleaning: {a_rows:,} rows, {n_features} numeric features + target `{target}`. |
| `B_clean.parquet` | Dataset B after cleaning: {b_rows:,} rows, same columns as A. |
| `A_train_idx.csv`, `A_test_idx.csv` | Row ids of the train ({a_train:,}) and test ({a_test:,}) rows of A. One column, `{row_id}`. |
| `B_train_idx.csv`, `B_test_idx.csv` | Row ids of the train ({b_train:,}) and test ({b_test:,}) rows of B. One column, `{row_id}`. |
| `decisions.md` | Why ports were dropped, why the 12 labels were kept, how the split was made, why duplicates were removed. |
| `cleaning_notes.md` | What cleaning removed, rule by rule and class by class, and the known limitations. |
| `README.md` | This file. |

About the parquet files:

- The DataFrame index is `{row_id}`: the position of the row in the original
  raw CSV. It is not a feature. The idx files refer to this index.
- `Source Port` and `Destination Port` were removed. Rows with NaN or Inf,
  exact duplicate rows and rows with contradictory labels were removed.
- Nothing is scaled or encoded. All features are numeric, with no missing
  values.
- Five constant columns are still present on purpose (the Low Variance filter
  removes them later).

## 2. Loading example

Needs `pandas` and `pyarrow` (`pip install pandas pyarrow`).

```python
import pandas as pd

TARGET = "{target}"

df = pd.read_parquet("A_clean.parquet")
train_ids = pd.read_csv("A_train_idx.csv")["{row_id}"]
test_ids = pd.read_csv("A_test_idx.csv")["{row_id}"]

train = df.loc[train_ids]
test = df.loc[test_ids]

X_train, y_train = train.drop(columns=TARGET), train[TARGET]
X_test, y_test = test.drop(columns=TARGET), test[TARGET]

print(X_train.shape, X_test.shape)  # ({a_train}, {n_features}) ({a_test}, {n_features})
```

For dataset B replace `A_` with `B_` in the three file names.

## 3. Mandatory rules

1. **Use exactly these row ids.** Train rows come from `*_train_idx.csv` and
   test rows from `*_test_idx.csv`. The split is stratified by label,
   {test_pct}% test, seed {seed}.
2. **Do not split the data again.** No new `train_test_split` on the full
   dataset, no reshuffling between train and test. If you need a validation
   set or cross-validation, take it from the train rows only.
3. **Fit every transformation on train only.** Scalers, encoders, imputers,
   discretizers, feature selectors and resamplers (SMOTE and similar) are
   fitted on the train rows and then applied to the test rows. Fitting them on
   all rows leaks test information into the model.
4. The test rows are used once, for the final evaluation. Do not tune
   hyperparameters against them.

## 4. Warnings

- **`sqli` has {sqli_test} rows in test** and **`Infiltration` has
  {inf_test} rows in test** (in A and in B). One misclassified row changes
  the recall of `Infiltration` by about 17 points and the recall of `sqli` by
  about 9 points. Report these per-class metrics as indicative only.
- **Accuracy is not a valid main metric.** The two largest classes hold more
  than 60% of the rows. Use F1-macro, per-class recall, MCC or PR-AUC.
- **One-direction flows.** In 100% of the lab rows the backward packet count
  is 0 and the flow source is the server. The client side of every lab
  conversation is missing (no attacker requests, no SQL payloads), because the
  flow extraction failed, not the capture. All `Bwd` features carry no
  information for lab rows.
- **Two data sources are mixed.** External and lab rows come from different
  extractors. A model can score well by telling the sources apart instead of
  detecting the attack. `benign` contains rows from both sources.
- **Class sizes changed after removing duplicates.** `PortScan` went from
  25,409 rows to 554, and `sqli` from 179 to 58 (A) and 57 (B). See
  `decisions.md`, decision 4.

## Rows per label

{label_table}
"""


def label_table(summary):
    """Render train/test rows per label, A and B side by side, as Markdown."""
    wide = summary.pivot(index=TARGET, columns="dataset",
                         values=["train", "test"])
    wide = wide.sort_values(("train", "A"), ascending=False)
    lines = ["| Label | A train | A test | B train | B test |",
             "|---|---|---|---|---|"]
    for label, row in wide.iterrows():
        lines.append(f"| {label} | {row[('train', 'A')]:,} | "
                     f"{row[('test', 'A')]:,} | {row[('train', 'B')]:,} | "
                     f"{row[('test', 'B')]:,} |")
    return "\n".join(lines)


def build_readme(summary, n_features):
    """Fill the README template with the real counts of the current split."""
    totals = summary.groupby("dataset")[["train", "test"]].sum()
    test_rows = summary.set_index(["dataset", TARGET])["test"]
    # The warnings quote one number for A and B; stop if they ever differ so
    # the README cannot silently state something false.
    for label in ("sqli", "Infiltration"):
        assert test_rows["A", label] == test_rows["B", label], \
            f"{label}: test rows differ in A and B"
    return README.format(
        target=TARGET, row_id=ROW_ID, n_features=n_features,
        a_rows=totals.loc["A"].sum(), b_rows=totals.loc["B"].sum(),
        a_train=totals.loc["A", "train"], a_test=totals.loc["A", "test"],
        b_train=totals.loc["B", "train"], b_test=totals.loc["B", "test"],
        test_pct=round(TEST_SIZE * 100), seed=RANDOM_SEED,
        sqli_test=test_rows["A", "sqli"],
        inf_test=test_rows["A", "Infiltration"],
        label_table=label_table(summary),
    )


def make_zip(files, zip_path):
    """Zip ``files`` inside one folder, with fixed timestamps.

    The fixed timestamp makes the zip byte-identical between runs, so its
    checksum can be used to confirm that everyone has the same package.
    """
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(files, key=lambda p: p.name):
            info = zipfile.ZipInfo(f"{HANDOFF_NAME}/{path.name}",
                                   date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, path.read_bytes())


def main():
    """Copy the files, write the README and build the zip."""
    HANDOFF_DIR.mkdir(parents=True, exist_ok=True)

    sources = [REPORT_DIR / DECISIONS_FILE, REPORT_DIR / CLEANING_NOTES_FILE]
    for name in DATASETS:
        sources += [INTERIM_DIR / CLEAN_FILE.format(name=name),
                    INTERIM_DIR / TRAIN_IDX_FILE.format(name=name),
                    INTERIM_DIR / TEST_IDX_FILE.format(name=name)]
    files = [shutil.copyfile(src, HANDOFF_DIR / src.name) for src in sources]

    # The package must be self-consistent: every id in the idx files has to
    # exist in the parquet shipped next to it.
    n_features = None
    for name in DATASETS:
        df = pd.read_parquet(HANDOFF_DIR / CLEAN_FILE.format(name=name))
        ids = pd.concat([
            pd.read_csv(HANDOFF_DIR / TRAIN_IDX_FILE.format(name=name))[ROW_ID],
            pd.read_csv(HANDOFF_DIR / TEST_IDX_FILE.format(name=name))[ROW_ID]])
        assert ids.is_unique and set(ids) == set(df.index), \
            f"{name}: split ids do not match the clean file"
        n_features = df.shape[1] - 1

    summary = pd.read_csv(SPLIT_DIR / SPLIT_SUMMARY_FILE)
    readme = HANDOFF_DIR / "README.md"
    readme.write_text(build_readme(summary, n_features))
    files.append(readme)

    zip_path = HANDOFF_DIR / f"{HANDOFF_NAME}.zip"
    make_zip(files, zip_path)

    print("Files in the package:")
    for path in sorted(files, key=lambda p: p.name):
        print(f"  {path.name:<22} {path.stat().st_size / 1024**2:8.2f} MB")
    print(f"\nZip: {zip_path} ({zip_path.stat().st_size / 1024**2:.2f} MB)")


if __name__ == "__main__":
    main()
