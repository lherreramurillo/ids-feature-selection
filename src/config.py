"""Shared paths and constants for every stage."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
RESULTS_DIR = ROOT / "results"
REPORT_DIR = ROOT / "report"

TARGET = "newLabel"

# Stage 1 (audit) outputs. The audit is read-only: it only writes reports here.
AUDIT_DIR = RESULTS_DIR / "audit"
AUDIT_SUMMARY_FILE = "audit_summary_{name}.csv"  # one per dataset (A, B)
LABEL_COUNTS_FILE = "label_counts.csv"
DUPLICATES_REPORT_FILE = "duplicates_report.csv"
NAN_INF_BY_LABEL_FILE = "nan_inf_rows_by_label.csv"

# Stage 2 (cleaning) parameters and outputs.
# Ports are dropped because they are shortcuts, not behaviour: Source Port is
# 8888 in more than 97% of the lab rows and in no external row. See
# report/decisions.md, decision 1.
DROP_PORTS = True
PORT_COLUMNS = ["Source Port", "Destination Port"]
# Name of the index kept in the clean files: the row position in the raw CSV,
# so any clean row can be traced back to its original line.
ROW_ID = "row_id"
CLEAN_FILE = "{name}_clean.parquet"  # written to INTERIM_DIR, one per dataset
CLEANING_DIR = RESULTS_DIR / "cleaning"
CLEANING_LOG_FILE = "cleaning_log.csv"
CLEANING_NOTES_FILE = "cleaning_notes.md"  # written to REPORT_DIR

# Check of how dropping the ports changes the duplicate counts (written to
# AUDIT_DIR). Supports decision 4 in report/decisions.md.
DUPLICATES_PORTS_FILE = "duplicates_ports_effect.csv"

# Stage 3 (train/test split). See report/decisions.md, decision 3.
TEST_SIZE = 0.20
RANDOM_SEED = 42
TRAIN_IDX_FILE = "{name}_train_idx.csv"  # written to INTERIM_DIR
TEST_IDX_FILE = "{name}_test_idx.csv"
SPLIT_DIR = RESULTS_DIR / "split"
SPLIT_SUMMARY_FILE = "split_summary.csv"

# Handoff package for the team that trains the models (clean data, split ids
# and notes). Lives under data/, which git ignores.
HANDOFF_DIR = ROOT / "data" / "handoff"
HANDOFF_NAME = "feature_selection_handoff_v1"  # zip name and folder inside it
DECISIONS_FILE = "decisions.md"  # in REPORT_DIR

# Stage 4 (filters). Every filter is fitted on the training rows only.
FILTERS_DIR = RESULTS_DIR / "filters"
FEATURE_SCORES_FILE = "feature_scores_{name}.csv"    # one row per feature
PEARSON_PAIRS_FILE = "pearson_pairs_{name}.csv"      # pairs above the lowest threshold
FEATURE_SUBSETS_FILE = "feature_subsets_{name}.csv"  # kept features per filter setting
FILTER_SUMMARY_FILE = "filter_summary.csv"
# Low Variance, rule 1: drop a feature when its most frequent value covers at
# least this share of the training rows (1.0 = strictly constant).
QUASI_CONSTANT_THRESHOLDS = [1.0, 0.999, 0.99, 0.95]
# Low Variance, rule 2: drop a feature when its variance after MinMax scaling
# to [0, 1] is at most this value (sklearn VarianceThreshold).
SCALED_VARIANCE_THRESHOLDS = [0.0, 1e-4, 1e-3, 1e-2]
# Pearson: drop one feature of every pair with |r| above the threshold.
PEARSON_THRESHOLDS = [0.90, 0.95, 0.99]
# Chi-square: quantile bins per feature and sizes of the top-k subsets.
CHI2_N_BINS = 10
CHI2_TOP_K = [10, 20, 30, 40]

# Stage 5 (evaluation). See report/decisions.md, decisions 5 to 7.
EVALUATION_DIR = RESULTS_DIR / "evaluation"
# Settings of each filter that are evaluated with the classifier.
EVAL_QUASI_CONSTANT_THRESHOLD = 0.999
COMBINED_TOP_K = [10, 20, 30]  # the combined pipeline leaves about 40 features
# Same model and hyperparameters for every feature subset.
MODEL_PARAMS = {
    "n_estimators": 200,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}
CV_FOLDS = 5
FIGURES_DIR = REPORT_DIR / "figures"
# Labels that exist only in the external dataset; used to order the confusion
# matrix by data source (benign exists in both sources).
EXTERNAL_LABELS = ["Botnet ARES", "Brute Force", "Dos/DDos", "Infiltration",
                   "PortScan", "Web Attack"]
LAB_LABELS = ["bruteforce", "dos_httpflood", "dos_slowhttptest", "recon",
              "sqli"]
# Rows with row_id below this value come from the external IDS2025 dataset;
# the rest are lab rows (the merge script wrote the external rows first).
N_EXTERNAL_ROWS = 91_830
# Label pairs that name the same attack type in the external and lab data.
EQUIVALENT_LABELS = [("Brute Force", "bruteforce"), ("PortScan", "recon"),
                     ("Web Attack", "sqli")]

# Stage 6 (deliverables). The delivered list is the setting that the selection
# rule chose within this family; see report/decisions.md, decision 8.
FINAL_FAMILY = "combined"
FINAL_FEATURES_FILE = "final_features_{name}.csv"  # written to RESULTS_DIR
HANDOFF_V2_NAME = "feature_selection_handoff_v2"
REPORT_FILE = "feature_selection_report.md"  # written to REPORT_DIR
REPORT_PDF_FILE = "feature_selection_report.pdf"
# Date shown in the header of the report.
REPORT_DATE = "2026-10-14"
# The NTLFlowLyzer comparison (bidirectional flows) was made on this capture
# only, not on the 12 lab captures.
NTLFLOWLYZER_CHECKED_PCAP = "fw_benign_20260828_01"
# Spanish version of the report, generated from the same values.
REPORT_FILE_ES = "feature_selection_report_es.md"
REPORT_PDF_FILE_ES = "feature_selection_report_es.pdf"
REPORT_DOCX_FILE = "feature_selection_report.docx"
REPORT_DOCX_FILE_ES = "feature_selection_report_es.docx"

# Classes with so few rows that losing any of them must be noticed.
RARE_LABELS = ["sqli", "Infiltration", "dos_slowhttptest"]

# Condition A = with firewall, condition B = without firewall.
RAW_FILES = {
    "A": RAW_DIR / "ids2025_pcaps_A_fw_merged.csv",
    "B": RAW_DIR / "ids2025_pcaps_B_no_fw_merged.csv",
}

# Numbers recorded in CLAUDE.md by the previous session; stage 0 re-checks them.
EXPECTED_SHAPE = {"A": (370_545, 80), "B": (345_192, 80)}
EXPECTED_LABELS = {
    "A": {
        "dos_httpflood": 127_047, "benign": 105_306, "bruteforce": 66_723,
        "recon": 4_822, "dos_slowhttptest": 823, "sqli": 179,
        "Dos/DDos": 26_066, "PortScan": 25_409, "Brute Force": 10_201,
        "Web Attack": 2_067, "Botnet ARES": 1_873, "Infiltration": 29,
    },
    "B": {
        "dos_httpflood": 123_945, "benign": 103_943, "bruteforce": 42_113,
        "recon": 8_545, "dos_slowhttptest": 822, "sqli": 179,
        "Dos/DDos": 26_066, "PortScan": 25_409, "Brute Force": 10_201,
        "Web Attack": 2_067, "Botnet ARES": 1_873, "Infiltration": 29,
    },
}
