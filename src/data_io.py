"""Loading helpers shared by the stages that work on the train/test split."""
import pandas as pd

from config import (CLEAN_FILE, INTERIM_DIR, ROW_ID, TARGET, TEST_IDX_FILE,
                    TRAIN_IDX_FILE)


def load_split(name, part):
    """Return (X, y) for the ``part`` ("train" or "test") rows of dataset ``name``.

    Rows are selected with the ids saved by stage 3, so every stage works on
    exactly the same partition.
    """
    idx_file = {"train": TRAIN_IDX_FILE, "test": TEST_IDX_FILE}[part]
    df = pd.read_parquet(INTERIM_DIR / CLEAN_FILE.format(name=name))
    ids = pd.read_csv(INTERIM_DIR / idx_file.format(name=name))[ROW_ID]
    rows = df.loc[ids]
    return rows.drop(columns=TARGET), rows[TARGET]
