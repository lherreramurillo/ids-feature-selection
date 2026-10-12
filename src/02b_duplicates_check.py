"""Stage 2 check: how dropping the port columns changes the duplicate counts.

What it does
    Read-only verification behind decision 4 (report/decisions.md). The audit
    found 94 (A) and 100 (B) exact duplicates in the raw files, but cleaning
    removed about 35,000. This script counts duplicates per label with and
    without the port columns, and describes how concentrated each class is in
    a few repeated feature vectors.

Inputs
    The two raw CSVs (paths from src/config.py). Nothing is modified.

Outputs
    results/audit/duplicates_ports_effect.csv, one row per dataset and label:
        rows                        rows in the raw file
        dup_with_ports              extra copies of identical rows, all columns
        dup_without_ports           same, ignoring the port columns
        dup_only_without_ports      duplicates that appear only without ports
        unique_vectors              distinct rows once ports are ignored
        largest_group_rows          rows of the most repeated vector
        top5_groups_rows            rows of the five most repeated vectors
        single_packet_rows          rows with exactly one forward packet
        distinct_src_ports, distinct_dst_ports

How to run
    .venv/bin/python src/02b_duplicates_check.py
"""
import pandas as pd

from config import (AUDIT_DIR, DUPLICATES_PORTS_FILE, PORT_COLUMNS, RAW_FILES,
                    TARGET)


def duplicates_by_label(df, columns):
    """Count extra copies (all occurrences after the first) per label.

    ``columns`` are the columns compared; the label is always one of them, so
    two rows only count as duplicates when they also agree on the label.
    """
    extra = df.duplicated(subset=columns, keep="first")
    return df.loc[extra, TARGET].value_counts()


def ports_effect(df):
    """Build the per-label table comparing duplicates with and without ports."""
    no_ports = [c for c in df.columns if c not in PORT_COLUMNS]
    table = pd.DataFrame({
        "rows": df[TARGET].value_counts(),
        "dup_with_ports": duplicates_by_label(df, list(df.columns)),
        "dup_without_ports": duplicates_by_label(df, no_ports),
    }).fillna(0).astype(int)
    table["dup_only_without_ports"] = (
        table["dup_without_ports"] - table["dup_with_ports"])
    table["unique_vectors"] = table["rows"] - table["dup_without_ports"]

    # Size of each group of identical rows (ports ignored). A class where a
    # few groups hold most rows has far fewer independent examples than its
    # row count suggests.
    group_id = df.groupby(no_ports, dropna=False, sort=False).ngroup()
    sizes = df.groupby([df[TARGET], group_id]).size()
    table["largest_group_rows"] = sizes.groupby(level=0).max()
    table["top5_groups_rows"] = sizes.groupby(level=0).apply(
        lambda s: s.nlargest(5).sum())
    # One-packet flows carry almost no statistics (no inter-arrival times, a
    # single length), so they are the rows most likely to be identical.
    table["single_packet_rows"] = (
        df["Total Fwd Packets"].eq(1).groupby(df[TARGET]).sum())
    table["distinct_src_ports"] = df.groupby(TARGET)["Source Port"].nunique()
    table["distinct_dst_ports"] = df.groupby(TARGET)["Destination Port"].nunique()
    return table.sort_index().rename_axis(TARGET).reset_index()


def main():
    """Run the check on A and B and write the combined table."""
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    for name, path in RAW_FILES.items():
        df = pd.read_csv(path, low_memory=False)
        table = ports_effect(df)
        table.insert(0, "dataset", name)
        parts.append(table)
        print(f"\n=== Dataset {name}: duplicates with ports "
              f"{table['dup_with_ports'].sum()}, without ports "
              f"{table['dup_without_ports'].sum()} ===")
        print(table.drop(columns="dataset").to_string(index=False))
    pd.concat(parts, ignore_index=True).to_csv(
        AUDIT_DIR / DUPLICATES_PORTS_FILE, index=False)
    print(f"\nWritten: {AUDIT_DIR / DUPLICATES_PORTS_FILE}")


if __name__ == "__main__":
    main()
