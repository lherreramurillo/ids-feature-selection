"""Stage 1: read-only quality audit of the two merged CSVs.

What it does
    Measures the data quality problems that stage 2 will have to clean, without
    changing anything: missing and infinite values, constant columns, exact
    duplicate rows and conflicting duplicates (same features, different label).
    Datasets A and B go through exactly the same logic.

Inputs
    data/raw/ids2025_pcaps_A_fw_merged.csv
    data/raw/ids2025_pcaps_B_no_fw_merged.csv
    (paths and file names come from src/config.py)

Outputs (results/audit/)
    audit_summary_A.csv, audit_summary_B.csv
        One row per column: dtype, NaN count, Inf count, unique values, share
        of the most frequent value, min and max.
    label_counts.csv
        Rows per label in A and in B.
    duplicates_report.csv
        Section "exact_duplicates_by_label": extra copies of fully identical
        rows, per label. Section "conflict_group": one row per group of rows
        that share all features but not the label.
    nan_inf_rows_by_label.csv
        Rows containing NaN or Inf, per label.

How to run
    .venv/bin/python src/01_audit.py

The script is deterministic: every output is sorted, so re-running it produces
identical files.
"""
import numpy as np
import pandas as pd

from config import (AUDIT_DIR, AUDIT_SUMMARY_FILE, DUPLICATES_REPORT_FILE,
                    LABEL_COUNTS_FILE, NAN_INF_BY_LABEL_FILE, RARE_LABELS,
                    RAW_FILES, TARGET)


def load_raw(path):
    """Load one raw CSV fully in memory, exactly as it is on disk."""
    return pd.read_csv(path, low_memory=False)


def column_summary(df):
    """Build the per-column audit table.

    Returns a DataFrame with one row per column of ``df`` (target included).
    """
    rows = []
    for col in df.columns:
        s = df[col]
        numeric = pd.api.types.is_numeric_dtype(s)
        # Only floats can hold Inf; NaN and Inf are counted apart because they
        # come from different causes (0/0 vs x/0 in the rate features).
        n_inf = int(np.isinf(s).sum()) if numeric else 0
        # NaN counts as a value here: a column that is 99% NaN is as useless
        # as one that is 99% zeros.
        top_share = s.value_counts(dropna=False).iloc[0] / len(s) * 100
        # Min and max over finite values only. With Inf included the max would
        # just say "inf" and hide the real range of the feature.
        finite = s[np.isfinite(s)] if numeric else None
        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "n_nan": int(s.isna().sum()),
            "n_inf": n_inf,
            "n_unique": int(s.nunique(dropna=True)),
            "top_value_share_pct": round(top_share, 4),
            "min_finite": finite.min() if numeric else np.nan,
            "max_finite": finite.max() if numeric else np.nan,
        })
    return pd.DataFrame(rows)


def nan_inf_by_label(df, features):
    """Count, per label, the rows that contain at least one NaN or Inf.

    Stage 2 drops these rows, so we need to know beforehand which classes pay
    for it (losing a handful of rows matters for the rare classes).
    """
    values = df[features]
    has_nan = values.isna().any(axis=1)
    has_inf = np.isinf(values.select_dtypes("number")).any(axis=1)
    flags = pd.DataFrame({
        TARGET: df[TARGET],
        "rows_total": 1,
        "rows_with_nan": has_nan,
        "rows_with_inf": has_inf,
        "rows_with_nan_or_inf": has_nan | has_inf,
    })
    out = flags.groupby(TARGET).sum().astype(int).reset_index()
    out["affected_pct_of_label"] = (
        out["rows_with_nan_or_inf"] / out["rows_total"] * 100).round(4)
    return out.sort_values(TARGET).reset_index(drop=True)


def exact_duplicates_by_label(df):
    """Count extra copies of fully identical rows (features AND label), per label.

    "Extra copies" means every occurrence after the first one, which is what
    drop_duplicates(keep="first") would remove.
    """
    extra = df.duplicated(keep="first")
    return df.loc[extra, TARGET].value_counts().sort_index()


def conflict_groups(df, features):
    """Find groups of rows with identical features but more than one label.

    These rows are a labelling contradiction: the model receives the same
    input with two different answers. Returns one row per group with its
    labels (and how many rows carry each) and the total rows in the group.
    """
    # Narrow down to rows whose feature vector appears more than once before
    # grouping: grouping all rows by 79 columns would be slow and pointless.
    repeated = df[df.duplicated(subset=features, keep=False)]
    if repeated.empty:
        return pd.DataFrame(columns=["labels", "n_rows", "first_row_index"])
    # dropna=False so rows with NaN in a feature are still grouped together.
    group_id = repeated.groupby(features, dropna=False, sort=False).ngroup()
    groups = []
    for _, part in repeated.groupby(group_id):
        counts = part[TARGET].value_counts().sort_index()
        if len(counts) > 1:
            groups.append({
                "labels": " | ".join(f"{k} ({v})" for k, v in counts.items()),
                "n_rows": len(part),
                # Row position in the raw CSV, to be able to inspect the group.
                "first_row_index": int(part.index.min()),
            })
    out = pd.DataFrame(groups, columns=["labels", "n_rows", "first_row_index"])
    return out.sort_values("first_row_index").reset_index(drop=True)


def audit_dataset(name, path):
    """Run every audit on one dataset and return its report pieces."""
    df = load_raw(path)
    features = [c for c in df.columns if c != TARGET]

    summary = column_summary(df)
    summary.to_csv(AUDIT_DIR / AUDIT_SUMMARY_FILE.format(name=name), index=False)

    nan_inf = nan_inf_by_label(df, features)
    nan_inf.insert(0, "dataset", name)

    dup = exact_duplicates_by_label(df)
    conflicts = conflict_groups(df, features)
    dup_report = pd.concat([
        pd.DataFrame({
            "dataset": name,
            "section": "exact_duplicates_by_label",
            "group_id": "",
            "labels": dup.index,
            "n_rows": dup.values,
            "first_row_index": "",
        }),
        pd.DataFrame({
            "dataset": name,
            "section": "conflict_group",
            "group_id": [f"{name}-{i + 1}" for i in range(len(conflicts))],
            "labels": conflicts["labels"],
            "n_rows": conflicts["n_rows"],
            "first_row_index": conflicts["first_row_index"],
        }),
    ], ignore_index=True)

    feat = summary[summary["column"] != TARGET]
    print(f"\n=== Dataset {name}: {path.name} ===")
    print(f"  shape: {df.shape}")
    print(f"  NaN cells: {feat['n_nan'].sum()} in "
          f"{feat.loc[feat['n_nan'] > 0, 'column'].tolist()}")
    print(f"  Inf cells: {feat['n_inf'].sum()} in "
          f"{feat.loc[feat['n_inf'] > 0, ['column', 'n_inf']].values.tolist()}")
    print(f"  rows with NaN or Inf: {nan_inf['rows_with_nan_or_inf'].sum()}")
    constant = feat.loc[feat["n_unique"] <= 1, "column"].tolist()
    print(f"  constant columns ({len(constant)}): {constant}")
    quasi = feat[(feat["top_value_share_pct"] >= 99) & (feat["n_unique"] > 1)]
    print(f"  quasi-constant columns, top value >= 99% ({len(quasi)}): "
          f"{quasi['column'].tolist()}")
    print(f"  exact duplicate rows (extra copies): {dup.sum()}")
    print(f"  rows sharing features with another row: "
          f"{df.duplicated(subset=features, keep=False).sum()}")
    print(f"  conflict groups: {len(conflicts)}, "
          f"rows involved: {conflicts['n_rows'].sum()}")

    affected = nan_inf[nan_inf["rows_with_nan_or_inf"] > 0]
    print("\n  rows with NaN or Inf, by label:")
    print("  " + affected.drop(columns="dataset").to_string(index=False)
          .replace("\n", "\n  "))
    rare_hit = affected[affected[TARGET].isin(RARE_LABELS)]
    print(f"  rare classes affected: "
          f"{rare_hit[TARGET].tolist() if len(rare_hit) else 'none'}")

    print("\n  exact duplicates by label:")
    print("  " + dup.to_string().replace("\n", "\n  "))
    print("\n  conflict groups:")
    print("  " + (conflicts.to_string(index=False).replace("\n", "\n  ")
                  if len(conflicts) else "none"))

    return df[TARGET].value_counts().rename(name), nan_inf, dup_report


def main():
    """Audit A and B with the same logic and write the combined reports."""
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    label_counts, nan_inf_parts, dup_parts = [], [], []
    for name, path in RAW_FILES.items():
        counts, nan_inf, dup_report = audit_dataset(name, path)
        label_counts.append(counts)
        nan_inf_parts.append(nan_inf)
        dup_parts.append(dup_report)

    labels = pd.concat(label_counts, axis=1).fillna(0).astype(int)
    labels = labels.sort_index().rename_axis(TARGET).reset_index()
    labels.to_csv(AUDIT_DIR / LABEL_COUNTS_FILE, index=False)
    pd.concat(nan_inf_parts, ignore_index=True).to_csv(
        AUDIT_DIR / NAN_INF_BY_LABEL_FILE, index=False)
    pd.concat(dup_parts, ignore_index=True).to_csv(
        AUDIT_DIR / DUPLICATES_REPORT_FILE, index=False)

    print(f"\nReports written to {AUDIT_DIR}")


if __name__ == "__main__":
    main()
