"""Stage 0: load both merged CSVs and confirm they match the documented numbers.

Nothing is cleaned or modified here. Run with:
    .venv/bin/python src/00_setup.py
"""
import pandas as pd

from config import (EXPECTED_LABELS, EXPECTED_SHAPE, INTERIM_DIR, RAW_FILES,
                    REPORT_DIR, RESULTS_DIR, TARGET)


def check(label, ok):
    print(f"  [{'OK' if ok else 'MISMATCH'}] {label}")
    return ok


def main():
    for folder in (INTERIM_DIR, RESULTS_DIR, REPORT_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    frames = {}
    all_ok = True
    for name, path in RAW_FILES.items():
        df = pd.read_csv(path, low_memory=False)
        frames[name] = df
        mem_mb = df.memory_usage(deep=True).sum() / 1024**2

        print(f"\n=== Dataset {name}: {path.name} ===")
        print(f"  shape: {df.shape}, memory: {mem_mb:.0f} MB")
        print(f"  dtypes: {df.dtypes.astype(str).value_counts().to_dict()}")
        non_numeric = [c for c in df.columns
                       if not pd.api.types.is_numeric_dtype(df[c])]
        print(f"  non-numeric columns: {non_numeric}")

        all_ok &= check(f"shape == {EXPECTED_SHAPE[name]}",
                        df.shape == EXPECTED_SHAPE[name])
        all_ok &= check(f"target '{TARGET}' present", TARGET in df.columns)
        all_ok &= check("column names are unique", df.columns.is_unique)

        counts = df[TARGET].value_counts(dropna=False)
        expected = pd.Series(EXPECTED_LABELS[name])
        table = pd.DataFrame({"found": counts, "expected": expected})
        table["share_%"] = (table["found"] / len(df) * 100).round(3)
        table = table.sort_values("found", ascending=False)
        print("\n  label distribution:")
        print(table.to_string().replace("\n", "\n  "))
        all_ok &= check("label counts match",
                        table["found"].equals(table["expected"])
                        or (table["found"] == table["expected"]).all())

    a, b = frames["A"], frames["B"]
    print("\n=== A vs B ===")
    all_ok &= check("same columns in the same order",
                    list(a.columns) == list(b.columns))

    print(f"\nColumns ({a.shape[1]}):")
    print(", ".join(a.columns))
    print(f"\nRESULT: {'all checks passed' if all_ok else 'SOME CHECKS FAILED'}")


if __name__ == "__main__":
    main()
