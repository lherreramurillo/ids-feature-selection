"""Stage 2: clean the two merged CSVs and save one clean copy of each.

What it does
    Applies the same four rules, in this order, to datasets A and B:
      1. Drop the port columns (only if DROP_PORTS is True).
      2. Turn +Inf / -Inf into NaN and drop the rows that contain NaN.
      3. Drop exact duplicate rows, keeping the first occurrence.
      4. Drop every row of a conflict group (same features, different label).
    It deliberately does NOT drop constant columns (the Low Variance filter of
    stage 4 must be the one that catches them), does NOT scale anything
    (scalers are fitted later, on training rows only) and does NOT rename or
    merge labels (see report/decisions.md).

Inputs
    data/raw/ids2025_pcaps_A_fw_merged.csv
    data/raw/ids2025_pcaps_B_no_fw_merged.csv
    (never modified; paths and parameters come from src/config.py)

Outputs
    data/interim/A_clean.parquet, data/interim/B_clean.parquet
        Clean rows. The index "row_id" is the row position in the raw CSV.
    results/cleaning/cleaning_log.csv
        One row per rule and dataset: rows before, rows after and rows removed
        per label.
    report/cleaning_notes.md
        Human-readable summary, generated from the real counts of this run.

How to run
    .venv/bin/python src/02_clean.py

The script is deterministic (no randomness, stable row order), so re-running
it produces identical files.
"""
import numpy as np
import pandas as pd

from config import (CLEAN_FILE, CLEANING_DIR, CLEANING_LOG_FILE,
                    CLEANING_NOTES_FILE, DROP_PORTS, INTERIM_DIR, PORT_COLUMNS,
                    RARE_LABELS, RAW_FILES, REPORT_DIR, ROW_ID, TARGET)


def log_step(log, name, step, rule, before, after, labels):
    """Append one cleaning step to ``log``, with the rows removed per label.

    ``before`` and ``after`` are the DataFrames around the step; ``labels`` is
    the full list of labels so every log row has the same columns.
    """
    removed = (before[TARGET].value_counts()
               .sub(after[TARGET].value_counts(), fill_value=0))
    entry = {
        "dataset": name,
        "step": step,
        "rule": rule,
        "rows_before": len(before),
        "rows_after": len(after),
        "rows_removed": len(before) - len(after),
        "cols_before": before.shape[1],
        "cols_after": after.shape[1],
    }
    entry.update({f"removed_{lab}": int(removed.get(lab, 0)) for lab in labels})
    log.append(entry)


def drop_ports(df):
    """Rule 1: remove the port columns.

    Ports identify the lab setup, not the attack: Source Port is the server
    port (8888 in more than 97% of the lab rows) and Destination Port is the
    client's ephemeral port. Keeping them
    would let a model separate lab rows from external rows for free.
    This must happen BEFORE the duplicate rules, because rows that differ only
    in the port are the same flow as far as the remaining features can tell.
    """
    return df.drop(columns=PORT_COLUMNS)


def drop_non_finite(df, name):
    """Rule 2: convert +/-Inf to NaN and drop the rows that contain NaN.

    The non-finite values come from rate features divided by a zero flow
    duration, so they are undefined, not "very large". Dropping is preferred
    to imputing because the affected rows are a tiny share of large classes.
    The run stops if a rare class would lose rows, since that needs a human
    decision instead of a silent drop.
    """
    features = df.columns.drop(TARGET)
    clean = df.replace([np.inf, -np.inf], np.nan)
    bad = clean[features].isna().any(axis=1)
    rare_hit = df.loc[bad & df[TARGET].isin(RARE_LABELS), TARGET].value_counts()
    if len(rare_hit):
        raise SystemExit(
            f"STOP ({name}): rows with NaN/Inf belong to rare classes "
            f"{rare_hit.to_dict()}. Nothing was written. Decide how to handle "
            f"them before continuing.")
    return clean[~bad]


def drop_exact_duplicates(df):
    """Rule 3: drop fully identical rows (features and label), keep the first.

    Repeated rows add no information and, after a random split, the same row
    could land in both train and test and inflate the test metrics.
    """
    return df.drop_duplicates(keep="first")


def find_conflicts(df):
    """Return the rows that share all features with a row of another label.

    Called after rule 3, so any two rows with equal features necessarily have
    different labels: every repeated feature vector is a conflict.
    """
    features = df.columns.drop(TARGET)
    return df[df.duplicated(subset=features, keep=False)]


def drop_conflicts(df, conflicts):
    """Rule 4: drop every row of every conflict group.

    With identical inputs and different labels there is no way to know which
    label is right, so keeping any copy would teach the model a coin flip.
    """
    return df.drop(index=conflicts.index)


def clean_dataset(name, path, log):
    """Apply the four rules to one dataset; return (raw, clean, conflicts)."""
    raw = pd.read_csv(path, low_memory=False)
    raw.index.name = ROW_ID
    labels = sorted(raw[TARGET].unique())
    print(f"\n=== Dataset {name}: {path.name} ===")

    df = raw
    if DROP_PORTS:
        step = drop_ports(df)
        log_step(log, name, 1, "drop port columns", df, step, labels)
        df = step

    step = drop_non_finite(df, name)
    log_step(log, name, 2, "drop rows with NaN or Inf", df, step, labels)
    df = step

    step = drop_exact_duplicates(df)
    log_step(log, name, 3, "drop exact duplicates (keep first)", df, step, labels)
    df = step

    conflicts = find_conflicts(df)
    print(f"  conflict rows to remove: {len(conflicts)}, by label: "
          f"{conflicts[TARGET].value_counts().sort_index().to_dict()}")
    step = drop_conflicts(df, conflicts)
    log_step(log, name, 4, "drop conflict groups (all copies)", df, step, labels)
    df = step

    for entry in log[-4:] if DROP_PORTS else log[-3:]:
        print(f"  rule {entry['step']} {entry['rule']}: "
              f"{entry['rows_before']} -> {entry['rows_after']} rows "
              f"(-{entry['rows_removed']}), {entry['cols_after']} cols")

    # Sanity checks: the clean file must be what the rules promise.
    features = df.columns.drop(TARGET)
    assert np.isfinite(df[features]).all().all(), "non-finite values remain"
    assert not df.duplicated(subset=features).any(), "duplicates remain"

    df.to_parquet(INTERIM_DIR / CLEAN_FILE.format(name=name), engine="pyarrow")
    return raw, df, conflicts


def before_after(raw, clean):
    """Build the per-label table of rows before and after cleaning."""
    table = pd.DataFrame({
        "before": raw[TARGET].value_counts(),
        "after": clean[TARGET].value_counts(),
    }).fillna(0).astype(int)
    table["removed"] = table["before"] - table["after"]
    table["removed_pct"] = (table["removed"] / table["before"] * 100).round(2)
    return table.sort_values("before", ascending=False)


def md_table(df, index_name):
    """Render a DataFrame as a Markdown table (avoids the tabulate dependency)."""
    cols = [index_name] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    # itertuples keeps each column's own type (iterrows would turn the integer
    # counts into floats when a row also holds a percentage).
    for idx, *row in df.itertuples():
        cells = [f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)
                 for v in row]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def write_notes(tables, log, conflicts):
    """Write report/cleaning_notes.md from the counts of this run."""
    log_df = pd.DataFrame(log)
    out = [
        "# Cleaning notes (stage 2)",
        "",
        "Generated by `src/02_clean.py`. All numbers come from the run that "
        "wrote this file. Datasets A (with firewall) and B (without firewall) "
        "are cleaned separately with identical rules. The raw CSVs are never "
        "modified.",
        "",
        "## Rules, in order",
        "",
        f"1. **Drop port columns** ({', '.join(PORT_COLUMNS)}; "
        f"applied: {DROP_PORTS}). `Source Port` is 8888 in 99.13% (A) and "
        "97.64% (B) of the lab rows and in no external row, so it separates "
        "lab rows from external rows without describing any attack behaviour "
        "(measurements in `decisions.md`, decision 1). `Destination Port` in the lab rows is the "
        "client's ephemeral port. Done first so that the duplicate rules see "
        "the same columns the models will see.",
        "2. **Drop rows with NaN or Inf.** `Flow Bytess` and `Flow Packetss` "
        "are rates; they are undefined when `Flow Duration` is 0. Inf is "
        "converted to NaN and the row is dropped. No row of "
        f"{', '.join(RARE_LABELS)} is affected (the script stops if one is).",
        "3. **Drop exact duplicates, keep the first.** Identical rows add no "
        "information and could land on both sides of a random train/test "
        "split.",
        "4. **Drop conflict groups.** Rows with identical features and "
        "different labels are removed entirely (all copies), because their "
        "label is ambiguous.",
        "",
        "Not done on purpose: constant columns are kept (the Low Variance "
        "filter in stage 4 removes them), nothing is scaled (scalers are "
        "fitted on training rows only), and the 12 labels are kept as they "
        "are.",
        "",
        "## Rows per rule",
        "",
        md_table(log_df.set_index(log_df["dataset"] + " / " +
                                  log_df["step"].astype(str))[
            ["rule", "rows_before", "rows_after", "rows_removed"]],
            "dataset / rule"),
        "",
        "The per-label breakdown of every rule is in "
        "`results/cleaning/cleaning_log.csv`.",
        "",
    ]
    for name, table in tables.items():
        conf = conflicts[name][TARGET].value_counts().sort_index().to_dict()
        out += [
            f"## Dataset {name}: rows per label, before and after",
            "",
            md_table(table, TARGET),
            "",
            f"Total: {table['before'].sum():,} -> {table['after'].sum():,} rows. "
            f"Conflict rows removed by rule 4: {sum(conf.values())} ({conf}).",
            "",
        ]
    out += [
        "## Why so many duplicates appear",
        "",
        "With the port columns present, the raw files contain only 94 (A) and "
        "100 (B) exact duplicate rows (see `results/audit/`). Once the ports "
        "are dropped, many rows that differed only in a port number become "
        "identical, which is why rule 3 removes far more rows than the audit "
        "reported. This is expected and is the reason ports are dropped "
        "before deduplication: those rows are indistinguishable to any model "
        "trained without ports.",
        "",
        "## Limitations",
        "",
        "- **One-direction flows.** In 100% of the lab rows the backward "
        "packet count is 0 and the flow source is the server "
        "(172.16.30.10). The client side of every conversation is "
        "missing: no attacker requests and no SQL payloads are represented. "
        "NTLFlowLyzer extracts bidirectional flows from the same PCAP, so the "
        "flow extraction failed, not the capture. Cleaning cannot repair "
        "this; all `Bwd` features carry no information for lab rows.",
        "- **Two sources mixed.** External rows (IDS2025) and lab rows come "
        "from different extractors and use different label vocabularies. "
        "Features may separate the data source instead of the attack.",
        "- **Small classes after deduplication.** See the tables above; "
        "classes with few remaining rows give unstable per-class metrics.",
        "- **Near duplicates remain.** Only exact duplicates are removed. "
        "Rows that differ in a single timing value are kept and may still "
        "fall on both sides of the train/test split.",
        "",
    ]
    (REPORT_DIR / CLEANING_NOTES_FILE).write_text("\n".join(out))


def main():
    """Clean A and B with the same rules and write all outputs."""
    for folder in (INTERIM_DIR, CLEANING_DIR, REPORT_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    log, tables, conflicts = [], {}, {}
    for name, path in RAW_FILES.items():
        raw, clean, conf = clean_dataset(name, path, log)
        tables[name] = before_after(raw, clean)
        conflicts[name] = conf
        print(f"\n  before / after by label ({name}):")
        print("  " + tables[name].to_string().replace("\n", "\n  "))
        print(f"  final shape: {clean.shape}")

    pd.DataFrame(log).to_csv(CLEANING_DIR / CLEANING_LOG_FILE, index=False)
    write_notes(tables, log, conflicts)
    print(f"\nWritten: {INTERIM_DIR}, {CLEANING_DIR}, "
          f"{REPORT_DIR / CLEANING_NOTES_FILE}")


if __name__ == "__main__":
    main()
