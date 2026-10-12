"""Stage 3: stratified train/test split of the clean datasets.

What it does
    Splits each clean dataset (A and B, separately) into train and test with
    one stratified hold-out split by label, and saves which rows went where.
    No feature values are changed and no data is copied: only row ids are
    written, so model training can load the clean file and select exactly the
    same rows.

Inputs
    data/interim/A_clean.parquet, data/interim/B_clean.parquet
    TEST_SIZE and RANDOM_SEED from src/config.py

Outputs
    data/interim/A_train_idx.csv, A_test_idx.csv, B_train_idx.csv, B_test_idx.csv
        One column, "row_id": the index of the clean parquet file, which is
        the row position in the raw CSV. Usage:
            df = pd.read_parquet("data/interim/A_clean.parquet")
            train = df.loc[pd.read_csv("data/interim/A_train_idx.csv")["row_id"]]
    results/split/split_summary.csv
        Rows per dataset and label in train and test.

How to run
    .venv/bin/python src/03_split.py

The split is reproducible: fixed seed, fixed input row order, sorted outputs.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (CLEAN_FILE, INTERIM_DIR, RANDOM_SEED, ROW_ID, SPLIT_DIR,
                    SPLIT_SUMMARY_FILE, TARGET, TEST_IDX_FILE, TEST_SIZE,
                    TRAIN_IDX_FILE)


def split_dataset(df):
    """Return (train_ids, test_ids) for one clean dataset.

    Stratifying by label keeps every class in both partitions with the same
    share it has in the full data. Without it, a class with 29 rows could end
    up with almost none in test and its recall could not be measured.
    """
    train_ids, test_ids = train_test_split(
        df.index.to_numpy(),
        test_size=TEST_SIZE,
        stratify=df[TARGET],
        random_state=RANDOM_SEED,
    )
    # Sorted so the files do not depend on the shuffle order, only on which
    # rows were chosen.
    return sorted(train_ids), sorted(test_ids)


def check_split(df, train_ids, test_ids):
    """Fail loudly if the split is not a clean partition of the dataset."""
    train_set, test_set = set(train_ids), set(test_ids)
    assert not train_set & test_set, "a row is in both train and test"
    assert train_set | test_set == set(df.index), "some rows were left out"
    # Stage 2 removed exact duplicates, so no feature vector can sit on both
    # sides. Checked here because the whole evaluation relies on it.
    features = df.columns.drop(TARGET)
    assert not df.duplicated(subset=features).any(), "duplicates across split"


def summarize(name, df, train_ids, test_ids):
    """Build the per-label table of train and test rows for one dataset."""
    table = pd.DataFrame({
        "train": df.loc[train_ids, TARGET].value_counts(),
        "test": df.loc[test_ids, TARGET].value_counts(),
    }).fillna(0).astype(int)
    table["total"] = table["train"] + table["test"]
    table["test_pct"] = (table["test"] / table["total"] * 100).round(2)
    table = table.sort_values("total", ascending=False)
    table = table.rename_axis(TARGET).reset_index()
    table.insert(0, "dataset", name)
    return table


def main():
    """Split A and B with the same parameters and write ids and summary."""
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name in ("A", "B"):
        df = pd.read_parquet(INTERIM_DIR / CLEAN_FILE.format(name=name))
        train_ids, test_ids = split_dataset(df)
        check_split(df, train_ids, test_ids)

        pd.Series(train_ids, name=ROW_ID).to_csv(
            INTERIM_DIR / TRAIN_IDX_FILE.format(name=name), index=False)
        pd.Series(test_ids, name=ROW_ID).to_csv(
            INTERIM_DIR / TEST_IDX_FILE.format(name=name), index=False)

        table = summarize(name, df, train_ids, test_ids)
        summaries.append(table)
        print(f"\n=== Dataset {name}: {len(df)} rows -> "
              f"train {len(train_ids)}, test {len(test_ids)} "
              f"(test_size={TEST_SIZE}, seed={RANDOM_SEED}) ===")
        print(table.drop(columns="dataset").to_string(index=False))

    pd.concat(summaries, ignore_index=True).to_csv(
        SPLIT_DIR / SPLIT_SUMMARY_FILE, index=False)
    print(f"\nWritten: train/test ids in {INTERIM_DIR}, "
          f"summary in {SPLIT_DIR / SPLIT_SUMMARY_FILE}")


if __name__ == "__main__":
    main()
