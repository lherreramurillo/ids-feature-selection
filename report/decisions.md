# Decision log

Methodological decisions of the feature selection pipeline. Pato (model
training) delegated decisions 1 to 3 to this pipeline on 2026-10-12, so his
models must use the files produced here (`data/interim/`) to stay consistent.

Numbers quoted below come from `results/audit/` and
`results/cleaning/cleaning_log.csv`.

## Decision 1: drop `Source Port` and `Destination Port`

- **Date:** 2026-10-12
- **Status:** applied in stage 2 (`DROP_PORTS = True` in `src/config.py`)
- **Decision:** remove both port columns before any other cleaning rule.
- **Justification:**
  - `Source Port` identifies the lab setup. In the raw files it is 8888 in
    99.13% of the lab rows of A and 97.64% of B; the remaining lab rows are
    `recon` flows from three other server ports (8443, 8025, 5500). No
    external (IDS2025) row has source port 8888, and only 1 of the 91,830
    external rows uses any of those four ports. The column therefore
    separates lab rows from external rows almost perfectly without describing
    attack behaviour, and any filter or model would reward it.
  - In the lab rows the flow source is the server, so `Destination Port` is
    the client's ephemeral port: it is 32768 or higher in 100% of the lab
    rows, against 7.70% of the external rows, and below 1024 in 0% of the lab
    rows against 68.66% of the external rows. It gives away the source in the
    same way.
  - Supporting source for the source port: Engelen, Rimmer and Joosen,
    "Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case
    Study", IEEE Security and Privacy Workshops 2021
    (DOI 10.1109/SPW53761.2021.00009). Their published preprocessing code
    removes flow id, source IP, destination IP, source port and timestamp as
    features (`MakeDataNumpyFriendly.py` in
    https://github.com/GintsEngelen/WTMC2021-Code). That code keeps the
    destination port, so dropping `Destination Port` here is not backed by
    that source: it rests only on the measurements above.
- **Alternatives rejected:**
  - Keep both ports: keeps the shortcut.
  - Drop only `Source Port`: `Destination Port` still encodes the source
    (ephemeral ports in the lab vs service ports in the external data).
- **Cost:**
  - In real traffic the destination port carries signal (service type); that
    signal is lost. This is the part of the decision that departs from the
    cited work, which keeps the destination port.
  - Rows that differed only in a port become identical. Exact duplicates rise
    from 94 to 35,062 in A and from 100 to 36,079 in B. After cleaning,
    `PortScan` keeps 554 of 25,409 rows and `sqli` keeps 58 (A) and 57 (B) of
    179.

## Decision 2: keep the 12 `newLabel` values, no harmonization

- **Date:** 2026-10-12
- **Status:** applied (no label is renamed or merged anywhere in the pipeline)
- **Decision:** keep the 12 labels exactly as they come in the merged files.
- **Justification:**
  - The candidate pairs (`Brute Force`/`bruteforce`, `PortScan`/`recon`,
    `Web Attack`/`sqli`) are not the same phenomenon measured the same way:
    they come from different environments and different flow extractors, and
    the lab rows contain only one direction of each flow.
  - `Web Attack` in the external data is broader than SQL injection.
  - Merging would assert an equivalence the data does not support and would
    hide the rare lab classes inside larger external ones.
- **Alternatives rejected:**
  - Merge the three pairs into 9 classes: simpler task, but mixes two
    distributions under one name.
  - Drop the external rows and keep the 6 lab classes: outside the delegated
    scope, changes the team's dataset decision.
- **Cost:**
  - Very small classes remain (`Infiltration` 29 rows, `sqli` 58/57).
  - A model may separate paired labels by data source rather than by attack.
  - `benign` already mixes external and lab rows and cannot be separated
    (no source column). This is a limitation for the final report.

## Decision 3: stratified hold-out split, 80/20, seed 42

- **Date:** 2026-10-12
- **Status:** applied in stage 3 (`src/03_split.py`, `TEST_SIZE = 0.20`,
  `RANDOM_SEED = 42` in `src/config.py`)
- **Decision:** one stratified split by `newLabel` with scikit-learn
  `train_test_split(test_size=0.20, stratify=y, random_state=42)`, done on the
  cleaned data, separately for A and B. The `row_id` values of the train and
  test rows are saved in `data/interim/` so that model training uses exactly
  the same rows. Every feature selector, scaler and discretizer is fitted on
  the training rows only.
- **Justification:**
  - Stratification keeps every class in both partitions with its original
    share. A plain random split could leave `Infiltration` (29 rows) with
    almost no test rows.
  - Group or time based splits are the stronger protection against leakage,
    but the files have no timestamp, IP or run identifier to build groups.
  - Splitting after deduplication prevents the same row from appearing in
    both train and test.
  - 80/20 with a fixed seed is the conventional, reproducible choice.
- **Alternatives rejected:**
  - Random split without stratification: unstable class shares for rare
    classes.
  - 70/30: slightly more test rows for rare classes (about 9 instead of 6 for
    `Infiltration`), not enough to make their metrics stable, and less
    training data for them.
  - Group or temporal split: not possible with the available columns.
- **Cost:**
  - The test set holds 6 `Infiltration` rows and 11 `sqli` rows (A and B),
    so per-class metrics for them move in large steps (one row is 1/6 of the
    `Infiltration` recall) and must be reported with that caveat.
  - Near duplicates (rows that differ slightly) can still fall on both sides
    of the split; test metrics may be optimistic.

## Decision 4: remove exact duplicates instead of keeping them with a group split

- **Date:** 2026-10-12
- **Status:** applied (confirms rule 3 of stage 2; no change to the clean files)
- **Question:** the audit found 94 (A) and 100 (B) exact duplicates, but
  stage 2 removed 35,062 and 36,079. Is removing them the right choice?
- **Verification** (`src/02b_duplicates_check.py`,
  `results/audit/duplicates_ports_effect.csv`):
  - With the port columns, exact duplicates are 94 in A and 100 in B.
  - Without the port columns they are 35,160 in A and 36,177 in B. Removing
    first the 127 rows with NaN or Inf leaves 35,062 and 36,079, which is what
    rule 3 removed.
  - So 35,066 (A) and 36,077 (B) duplicates exist only because the ports were
    the single difference between otherwise identical rows.
  - `PortScan`: 25,409 rows, 573 distinct feature vectors. 25,153 rows are
    flows with one forward packet. The scan probes 999 distinct destination
    ports with the same tiny flow, so the port was the only thing that
    changed. The five most repeated vectors hold 4,583 rows.
  - `sqli`: 179 rows, 59 distinct feature vectors. 123 rows are flows with one
    forward packet, zero duration and zero backward packets, and they collapse
    into three vectors (42, 42 and 39 rows; forward length 74, 66 and 60
    bytes). The other 56 rows are all different from each other.
- **Decision:** keep option (a), remove exact duplicates and keep the first
  copy, then split with a plain stratified split.
- **Justification:**
  - Both options contain the same information: the number of distinct
    feature vectors per class is identical (573 for `PortScan`, 59 for
    `sqli`). Option (b) adds repeated rows, not new examples.
  - Duplicates bias learning and evaluation toward the most frequent records.
    This is the reason NSL-KDD was built from KDD CUP 99 by removing redundant
    records from the train and test sets (Tavallaee, Bagheri, Lu and Ghorbani,
    "A Detailed Analysis of the KDD CUP 99 Data Set", IEEE CISDA 2009,
    https://www.ee.torontomu.ca/~bagheri/papers/cisda.pdf).
  - The filters of stage 4 depend on row counts. The chi-square statistic
    grows with the number of rows, so 24,836 repeated `PortScan` rows would
    raise the score of whatever identifies a one-packet probe.
  - A group split cannot be stratified well here: in `sqli`, three groups
    hold 123 of 179 rows, so the class share in test would depend on where
    those three groups fall, and a single prediction on one repeated vector
    would move the recall by up to 42 rows at once.
- **Alternative rejected:** option (b), keep the duplicates and use a group
  split so that all copies of a row fall on the same side. It prevents the
  same leakage and keeps the original class frequencies, which is valid when
  the goal is to estimate performance per flow in the original traffic mix.
  Rejected for the reasons above and because it also needs a rule for groups
  that contain more than one label.
- **Impact per class** (rows before cleaning -> rows after all cleaning rules;
  full detail in `results/cleaning/cleaning_log.csv`):

| newLabel | A before | A after | B before | B after |
|---|---|---|---|---|
| dos_httpflood | 127,047 | 127,015 | 123,945 | 123,529 |
| benign | 105,306 | 104,171 | 103,943 | 102,803 |
| bruteforce | 66,723 | 66,722 | 42,113 | 42,112 |
| Dos/DDos | 26,066 | 21,364 | 26,066 | 21,364 |
| PortScan | 25,409 | 554 | 25,409 | 554 |
| Brute Force | 10,201 | 6,888 | 10,201 | 6,888 |
| recon | 4,822 | 4,286 | 8,545 | 7,378 |
| Web Attack | 2,067 | 2,031 | 2,067 | 2,031 |
| Botnet ARES | 1,873 | 1,373 | 1,873 | 1,373 |
| dos_slowhttptest | 823 | 823 | 822 | 822 |
| sqli | 179 | 58 | 179 | 57 |
| Infiltration | 29 | 29 | 29 | 29 |

- **Cost:**
  - Class frequencies no longer reflect the captured traffic: `PortScan` goes
    from 6.9% of the rows in A to 0.17%. Metrics describe performance per
    distinct flow pattern, not per captured flow.
  - `sqli` keeps 58 (A) and 57 (B) rows, 11 of them in test.
  - Most of the removed `sqli` rows are single-packet flows from the server
    side with no payload statistics. They describe the one-direction
    extraction problem more than SQL injection behaviour (interpretation, not
    verified against the PCAPs).

## Decision 5: RandomForest as the single evaluation model

- **Date:** 2026-10-12
- **Status:** applied in stage 5 (`MODEL_PARAMS` in `src/config.py`)
- **Decision:** every feature subset is evaluated with the same
  `RandomForestClassifier(n_estimators=200, class_weight="balanced",
  random_state=42)`, all other hyperparameters at the scikit-learn defaults.
- **Justification:**
  - The comparison is between feature subsets, so the model must be a fixed
    instrument. RandomForest gives stable results with default settings, which
    avoids tuning hyperparameters per subset (tuning would mix the effect of
    the features with the effect of the tuning).
  - It does not need scaled inputs, consistent with the choice of not scaling
    the data.
  - `class_weight="balanced"` weights each class inversely to its frequency,
    which is needed with classes of 23 and 47 training rows next to classes
    of about 100,000.
  - Measured cost on this machine (dataset A, full train, 77 features, 16
    cores): 21.5 s per fit, against 74.8 s for LightGBM with 200 rounds. The
    whole experiment (22 configurations, 5 folds, 2 datasets) fits in about
    70 minutes.
- **Alternatives rejected:**
  - LightGBM: 3.5 times slower here, and with balanced weights on classes of
    23 rows it is sensitive to settings (learning rate, leaf size) that would
    need tuning.
  - 100 trees instead of 200: 11.0 s per fit, but noisier estimates for the
    rare classes; the time budget did not require it.
- **Cost:** the ranking of feature subsets is only shown for this model.
  Models that are sensitive to redundant or unscaled features (linear models,
  k-NN, neural networks) may benefit more from the filters than a tree
  ensemble does.

## Decision 6: leave the MinMax-scaled variance threshold out of the evaluation

- **Date:** 2026-10-12
- **Status:** applied in stage 5 (the variant is still computed and reported
  in stage 4)
- **Decision:** Low Variance is evaluated only with the quasi-constant rule at
  0.999 (drop a feature when one value covers at least 99.9% of the training
  rows). The variance-after-MinMax thresholds are not evaluated.
- **Justification** (`results/filters/`):
  - MinMax scaling maps the largest value of a feature to 1. Features with
    extreme values (durations, byte counts) end up with almost all rows near
    0, so their scaled variance is tiny even though they separate classes.
  - With a scaled-variance threshold of 0.001 the filter removes 9 of the 20
    features ranked best by chi-square in A and in B; with 0.01 it removes 16
    (A) and 13 (B). It penalizes skewed features, not uninformative ones.
  - The quasi-constant rule at 0.999 removes 9 features in both datasets: the
    5 constant columns and 4 flag counts with 1 to 4 non-zero training rows.
  - The 0.99 threshold was also discarded: it removes `Fwd PSH Flags` and
    `SYN Flag Count`, which are non-zero in 22.5% of the `Brute Force` rows.
    With this class imbalance a quasi-constant feature can be the signal of a
    small class.
- **Alternative rejected:** keeping the scaled-variance variants in the
  comparison. It would add configurations already known to discard useful
  features for a reason unrelated to their information.

## Decision 7: evaluation protocol

- **Date:** 2026-10-12
- **Status:** applied in stage 5 (`src/05_evaluate.py`)
- **Decision:**
  - Settings are compared with stratified 5-fold cross-validation inside the
    training rows. The filters are refitted on the training part of every
    fold.
  - Within each filter family, the selected setting is the one with the
    fewest features whose mean F1-macro is within one standard error of the
    best setting of that family (one-standard-error rule).
  - The test rows are used once: for the baseline and for the selected
    setting of each family.
  - The chi-square ranking requested for the assignment is evaluated next to
    a ranking by Cramer's V, as two separate families.
- **Justification:**
  - Choosing thresholds by looking at test scores would leak test information
    into the selection, the same leak the fixed split is meant to prevent.
  - Refitting the filters per fold keeps the held-out rows out of the feature
    selection, so the cross-validation score is an honest estimate.
  - The one-standard-error rule avoids keeping more features for a difference
    smaller than the fold-to-fold noise.
  - The chi-square statistic grows with the number of bins, and 66 of the 77
    features have fewer than 10 quantile bins because of repeated values. The
    two rankings share only 7 of their top 20 features, so the choice of
    ranking is evaluated instead of assumed.
- **Alternatives rejected:**
  - Evaluating all 22 configurations on test and picking the best: leaks.
  - Fitting the filters once on the full training set and reusing them in
    every fold: faster, slightly optimistic.
- **Cost:** with 5 folds the standard error is itself a rough estimate.
  Per-class results for `Infiltration` (23 training rows) and `sqli` (47/46)
  stay unstable under any protocol.

## Decision 8: deliver the settings chosen by the selection rule, unchanged

- **Date:** 2026-10-12
- **Status:** applied in stage 6 (`results/final_features_A.csv`,
  `results/final_features_B.csv`)
- **Decision:** the final feature lists are the combined pipeline (Low
  Variance at 0.999, then Pearson at 0.90, then chi-square top k) with the k
  chosen by the one-standard-error rule in cross-validation: **k = 20 for A**
  and **k = 30 for B**. The rule was fixed in decision 7 before any result was
  seen and is not adjusted afterwards.
- **Justification:**
  - Changing the rule, or overriding its choice, after looking at the results
    is a form of selection on the outcome. Even when only cross-validation
    scores are used, picking the setting that "looks better" after the fact
    makes the reported scores optimistic and the procedure impossible to
    reproduce by someone who has not seen the results.
  - In A the rule's choice (k = 20, F1-macro 0.9613 in cross-validation) is
    within one standard error (0.0077) of the best combined setting (k = 30,
    0.9662). The difference is not distinguishable from fold-to-fold noise,
    which is exactly the case the rule was designed for.
  - The test score of the delivered A setting (F1-macro 0.9678 with 20
    features, baseline 0.9713 with 77) was measured once, for the setting the
    rule chose. Switching to k = 30 would require a second look at the test
    rows for a setting chosen after seeing results.
- **Alternative rejected:** using k = 30 in both datasets for consistency.
  It is a reasonable configuration and gives the same list length to both
  datasets, but it would replace a pre-declared rule with a judgment made
  after seeing the scores.
- **Cost:** A and B are delivered with lists of different length (20 and 30
  features), so models trained on the two datasets do not share the same
  input columns.
