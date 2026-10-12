# IDS Feature Selection (Laura's assignment)

Context handoff from a previous Claude session (updated 2026-10-12). Read fully before acting.

## Who and how

- User: Laura. Speak to her in Spanish (Latin American, "tu" register, informal).
- Everything written to disk goes in English: file names, code, comments, variable names, notebook markdown, chart titles, axis labels, reports.
- She has never done an ML feature selection pipeline. Explain each stage briefly in plain Spanish before running it: what it does, why, how to read the result. She learns by doing, so build section by section, not one giant script.
- One stage at a time. Show the real output, interpret it, then move on.
- Never use the em dash character anywhere.
- Project rules: no claim without real output or a cited source. If something was not verified, say so. No pleasing answers: if something is wrong, say it directly. Statistical rigor calibrated to the real class imbalance, not generic formulas.
- Deadline: around 2026-10-14.

## Project in one paragraph

Team project "Arquitectura IDS impulsada por IA". A fully virtual GNS3 lab (Kali attacker, crAPI vulnerable web/API target at 172.16.30.10:8888, benign traffic host 203.0.113.20, pfSense firewall) captured traffic with tcpdump. Two conditions: with firewall (A, `fw`) and without firewall (B, `nofw`). Six lab classes: benign, recon, bruteforce, sqli, dos_slowhttptest, dos_httpflood. A teammate (Pato) trains the models; Laura's selected features feed his models, so her features MUST exist in the dataset he trains on.

## Dataset to use (team decision): Albeiro's two merged CSVs

The team decided to use Albeiro's files, A and B treated as separate datasets:
- `data/raw/ids2025_pcaps_A_fw_merged.csv`: 370,545 rows x 80 cols
- `data/raw/ids2025_pcaps_B_no_fw_merged.csv`: 345,192 rows x 80 cols
- 79 features + target `newLabel`. Small enough to load fully in RAM.

Run the whole pipeline separately on A and on B.

### Verified facts about these files (previous session)
Built with the Python `cicflowmeter` and a merge script (`merge_ids2025.py`). Each file = 91,830 rows from an external dataset `IDS2025.xlsx` (first rows, identical in A and B) + the lab rows of that condition.

Label distribution, A (B in parentheses where it differs):
- dos_httpflood 127,047 (123,945), benign 105,306 (103,943), bruteforce 66,723 (42,113), recon 4,822 (8,545), dos_slowhttptest 823 (822), sqli 179
- External only: Dos/DDos 26,066, PortScan 25,409, Brute Force 10,201, Web Attack 2,067, Botnet ARES 1,873, Infiltration 29
- benign = 26,185 external + lab benign.

Quality, each file:
- NaN: 77 cells, all in `Flow Bytess`. Inf: 177 cells in `Flow Bytess` and `Flow Packetss`.
- Constant columns (5): Bwd PSH Flags, Bwd URG Flags, Bwd Avg Bytes Bulk, Bwd Avg Packets Bulk, Bwd Avg Bulk Rate.
- Exact duplicate rows: 94 in A, 100 in B (0.03%). Conflicting duplicates (same features, different label): 5 in A, 4 in B.
- No timestamp, no IPs, no run_id (dropped by the merge script). Ports were kept.

### Known limitations (must appear in the final report)
1. In 100% of the lab rows (all 12 per-PCAP CSVs) backward packets = 0 and the source is the server 172.16.30.10:8888. The client side of every conversation is missing (no attacker requests, no SQL payloads). NTLFlowLyzer output of the same PCAP has bidirectional flows, so the extraction failed, not the capture.
2. Two label vocabularies and two extractors are mixed (`Brute Force` vs `bruteforce`, `PortScan` vs `recon`, `Web Attack` vs `sqli`). Distributions differ by source (benign median `Init_Win_bytes_forward` 130 external vs 65160 lab).
3. `Source Port` = 8888 in 99.13% (A) and 97.64% (B) of the lab rows (re-measured 2026-10-12; the rest are `recon` rows from server ports 8443, 8025, 5500) and in 0 external rows: a near-perfect shortcut that separates lab from external rows.
Consequence: features ranked as important may be detecting the data source, not the attack. Say this explicitly when interpreting results.

### Decisions to confirm with Pato BEFORE stage 3 (features must match his models)
- Labels: keep the 12 `newLabel` values as they are, or harmonize? Use whatever he trains with.
- Drop `Source Port` and `Destination Port`? Recommended (shortcut). Must be the same in his pipeline.
- Train/test split: which method, test size and random seed he uses. Feature selection must be fitted on HIS training rows only, or it leaks test information into the selection.

## What the professor asked

Apply three filter methods: Low Variance, Pearson Correlation, Chi-square. They are not models and are not "trained". Each method produces a feature subset; the SAME model is trained on each subset and compared against a baseline with all features.

## Plan (run for A, then for B)

Stage 0. Setup: create folders, install packages, load both files, confirm shapes match the numbers above.

Stage 1. Audit: re-verify the quality numbers above. Show results before cleaning.

Stage 2. Hygiene
- Drop ports if confirmed. Handle NaN and Inf explicitly (report how many rows each rule touched).
- Remove exact duplicates; report conflicting duplicates.

Stage 3. Train/test split matching Pato's. No timestamp or run_id exists, so a stratified split by `newLabel` is the likely option; state the near-duplicate leakage risk as a limitation. Every selector below is fitted on TRAIN only.

Stage 4. Filters
- Low Variance: raw variance depends on each feature's scale, so use a quasi-constant rule (share of the most frequent value) or variance after MinMax scaling. Try a few thresholds; report how many features each removes. The 5 constant columns must be caught.
- Pearson: absolute pairwise correlation; for each pair above threshold drop one (keep the one more related to the target). Try 0.90, 0.95, 0.99.
- Chi-square: sklearn `chi2` needs non-negative, count-like input. Discretize with quantile bins (`KBinsDiscretizer`, strategy quantile) fitted on train, then score. Select top k for several k. Document why discretization is needed.

Stage 5. Evaluation
- Same classifier and hyperparameters for every subset (RandomForest or LightGBM, class weights for minority classes).
- Metrics: F1-macro, PR-AUC macro, MCC, per-class recall (especially sqli, Infiltration, dos_slowhttptest), number of features, training time. Accuracy is not a valid primary metric with this imbalance.
- Compare: baseline (all features) vs each filter vs combined pipeline (variance then Pearson then chi-square).

Stage 6. Deliverables
- `results/final_features_A.csv` and `results/final_features_B.csv` for Pato.
- Results tables and a few charts.
- Short report in English: methods, thresholds, results, choice, limitations.

## Layout

```
ids-feature-selection/
  CLAUDE.md
  data/raw/          the two merged CSVs (do not modify)
  data/interim/      cleaned copies
  src/               one script or notebook per stage
  results/
  report/
```

Python packages: pandas, numpy, scikit-learn, scipy, matplotlib, seaborn (lightgbm optional).

## Environment

- OS: Linux, user shell is fish.
- Always run Python through the project venv: .venv/bin/python (pandas 3.0.6, scikit-learn 1.9.1, scipy 1.18.1 verified installed there).
