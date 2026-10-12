"""Stage 5: evaluate every feature subset with the same classifier.

What it does
    For datasets A and B separately:
    1. Cross-validation inside train. In every fold the filters are refitted
       on the training part of that fold, each feature subset is used to train
       the same RandomForest, and it is scored on the held-out part. The test
       rows are not touched in this step.
    2. Selection. Within each filter family the setting chosen is the one with
       the fewest features whose mean F1-macro is within one standard error of
       the best setting of that family.
    3. Final test. The filters are fitted on the full training set, and only
       the baseline and the chosen setting of each family are trained on train
       and scored once on test (metrics, per-class recall, confusion matrix).
    4. Data source checks. Counts the confusions between labels that name the
       same attack in the two data sources, and trains a probe that tries to
       tell external rows from lab rows using the same features.

Inputs
    data/interim/{A,B}_clean.parquet and the train/test ids of stage 3
    Configurations from filters.evaluation_subsets; MODEL_PARAMS, CV_FOLDS and
    the other parameters from src/config.py

Outputs (results/evaluation/)
    cv_folds.csv              one row per dataset, configuration and fold
    cv_results.csv            mean and std over folds, plus the "selected" flag
    cv_recall_by_class.csv    recall per class from the pooled held-out predictions
    test_results.csv          test metrics of the baseline and selected settings
    test_recall_by_class.csv  recall per class on test
    confusion_test.csv        test confusion matrices, long format
    selected_features.csv     features of every configuration scored on test
    source_pair_confusion.csv confusions between equivalent external/lab labels
    source_probe.csv          how well the features separate external from lab

How to run
    .venv/bin/python src/05_evaluate.py          (about 70 minutes)
    .venv/bin/python src/05_evaluate.py A        (one dataset only)

Reproducibility: folds, filters and the model use fixed seeds, so all metrics
are identical between runs. Only the timing columns change.
"""
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, matthews_corrcoef, recall_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize

from config import (CV_FOLDS, EQUIVALENT_LABELS, EVALUATION_DIR, MODEL_PARAMS,
                    N_EXTERNAL_ROWS, RANDOM_SEED)
from data_io import load_split
from filters import evaluation_subsets


def log(message):
    """Print a progress line immediately (the run takes about an hour)."""
    print(message, flush=True)


def fit_predict(X_train, y_train, X_eval, features):
    """Train the fixed model on ``features`` and predict ``X_eval``.

    Returns (classes, class probabilities, training seconds). The model and
    its hyperparameters are the same for every subset, so differences in the
    scores come from the features and not from tuning.
    """
    model = RandomForestClassifier(**MODEL_PARAMS)
    start = time.perf_counter()
    model.fit(X_train[features], y_train)
    seconds = time.perf_counter() - start
    return model.classes_, model.predict_proba(X_eval[features]), seconds


def score(y_true, classes, proba):
    """Compute the metrics that stay meaningful under class imbalance.

    Accuracy is left out on purpose: the two largest classes hold about 70%
    of the rows, so it would look high even if the rare attacks were missed.
    F1-macro and PR-AUC macro give every class the same weight; MCC uses the
    whole confusion matrix.
    """
    predicted = classes[proba.argmax(axis=1)]
    onehot = label_binarize(y_true, classes=classes)
    return predicted, {
        "f1_macro": f1_score(y_true, predicted, average="macro"),
        "pr_auc_macro": average_precision_score(onehot, proba, average="macro"),
        "mcc": matthews_corrcoef(y_true, predicted),
    }


def recall_by_class(y_true, predicted, classes):
    """Return a table with the support and recall of every class."""
    recalls = recall_score(y_true, predicted, labels=classes, average=None,
                           zero_division=0)
    support = pd.Series(y_true).value_counts().reindex(classes).to_numpy()
    return pd.DataFrame({"label": classes, "support": support,
                         "recall": recalls})


def cross_validate(name, X, y):
    """Run the cross-validation on the training rows of one dataset.

    Returns (per-fold rows, pooled per-class recall rows). The filters are
    refitted in every fold: selecting features with rows that are later used
    to score the model would make every subset look better than it is.
    """
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                            random_state=RANDOM_SEED)
    fold_rows, held_out = [], {}
    for fold, (train_pos, val_pos) in enumerate(folds.split(X, y), start=1):
        X_tr, y_tr = X.iloc[train_pos], y.iloc[train_pos]
        X_val, y_val = X.iloc[val_pos], y.iloc[val_pos]
        start = time.perf_counter()
        subsets = evaluation_subsets(X_tr, y_tr)
        log(f"  [{name}] fold {fold}/{CV_FOLDS}: filters fitted in "
            f"{time.perf_counter() - start:.0f}s")
        for config, (family, features) in subsets.items():
            classes, proba, seconds = fit_predict(X_tr, y_tr, X_val, features)
            predicted, metrics = score(y_val, classes, proba)
            fold_rows.append({"dataset": name, "config": config,
                              "family": family, "fold": fold,
                              "n_features": len(features), **metrics,
                              "fit_seconds": seconds})
            # Keep the held-out predictions: pooled over the folds they give
            # per-class recall on all training rows of a class (23 rows for
            # Infiltration instead of 4 or 5 per fold).
            held_out.setdefault(config, pd.Series(index=y.index, dtype=object))
            held_out[config].iloc[val_pos] = predicted
        best = max(fold_rows[-len(subsets):], key=lambda r: r["f1_macro"])
        log(f"  [{name}] fold {fold}/{CV_FOLDS}: {len(subsets)} models "
            f"trained, best F1-macro {best['f1_macro']:.4f} ({best['config']})")

    classes = np.array(sorted(y.unique()))
    recall_rows = []
    for config, predicted in held_out.items():
        table = recall_by_class(y, predicted, classes)
        table.insert(0, "config", config)
        table.insert(0, "dataset", name)
        recall_rows.append(table)
    return pd.DataFrame(fold_rows), pd.concat(recall_rows, ignore_index=True)


def summarize_cv(folds):
    """Average the folds and flag the selected setting of every family.

    Selection rule (one-standard-error rule): among the settings of a family,
    take those whose mean F1-macro is within one standard error of the best
    one, and choose the one with the fewest features. Differences smaller
    than the fold-to-fold noise are not a reason to keep more features.
    """
    grouped = folds.groupby(["dataset", "config", "family"], sort=False)
    summary = grouped.agg(
        n_features=("n_features", "mean"),
        f1_macro_mean=("f1_macro", "mean"), f1_macro_std=("f1_macro", "std"),
        pr_auc_macro_mean=("pr_auc_macro", "mean"),
        pr_auc_macro_std=("pr_auc_macro", "std"),
        mcc_mean=("mcc", "mean"), mcc_std=("mcc", "std"),
        fit_seconds_mean=("fit_seconds", "mean"),
    ).reset_index()
    summary["f1_macro_se"] = summary["f1_macro_std"] / np.sqrt(CV_FOLDS)
    summary["selected"] = False
    for _, family in summary.groupby(["dataset", "family"]):
        best = family.loc[family["f1_macro_mean"].idxmax()]
        close = family[family["f1_macro_mean"]
                       >= best["f1_macro_mean"] - best["f1_macro_se"]]
        chosen = close.sort_values(["n_features", "f1_macro_mean"],
                                   ascending=[True, False]).index[0]
        summary.loc[chosen, "selected"] = True
    return summary


def evaluate_on_test(name, selected, X_train, y_train, X_test, y_test):
    """Train the selected configurations on train and score them once on test.

    Returns (metrics, per-class recall, confusion matrices, feature lists,
    subsets), the first four as DataFrames.
    """
    # verify=True repeats the scipy cross-check of the chi-square scores on
    # the exact rows used for the final feature lists.
    subsets = evaluation_subsets(X_train, y_train, verify=True)
    metric_rows, recall_rows, confusion_rows, feature_rows = [], [], [], []
    for config in selected:
        family, features = subsets[config]
        classes, proba, seconds = fit_predict(X_train, y_train, X_test, features)
        predicted, metrics = score(y_test, classes, proba)
        metric_rows.append({"dataset": name, "config": config,
                            "family": family, "n_features": len(features),
                            **metrics, "fit_seconds": seconds})
        table = recall_by_class(y_test, predicted, classes)
        table.insert(0, "config", config)
        table.insert(0, "dataset", name)
        recall_rows.append(table)
        matrix = confusion_matrix(y_test, predicted, labels=classes)
        for i, true_label in enumerate(classes):
            for j, predicted_label in enumerate(classes):
                confusion_rows.append({
                    "dataset": name, "config": config,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "rows": int(matrix[i, j])})
        feature_rows += [{"dataset": name, "config": config, "feature": f}
                         for f in features]
        log(f"  [{name}] test {config}: {len(features)} features, "
            f"F1-macro {metrics['f1_macro']:.4f}, MCC {metrics['mcc']:.4f}")
    return (pd.DataFrame(metric_rows), pd.concat(recall_rows, ignore_index=True),
            pd.DataFrame(confusion_rows), pd.DataFrame(feature_rows), subsets)


def equivalent_pair_confusion(confusion):
    """Count test confusions between labels that name the same attack.

    If the features described the attack, an external "Brute Force" flow and
    a lab "bruteforce" flow should look alike and be confused with each other
    at least sometimes. No confusion at all between them means the model can
    always tell which source a flow comes from.
    """
    rows = []
    for (dataset, config), part in confusion.groupby(["dataset", "config"],
                                                     sort=False):
        cell = part.set_index(["true_label", "predicted_label"])["rows"]
        support = part.groupby("true_label")["rows"].sum()
        for external, lab in EQUIVALENT_LABELS:
            rows.append({
                "dataset": dataset, "config": config,
                "external_label": external, "lab_label": lab,
                "external_test_rows": int(support[external]),
                "lab_test_rows": int(support[lab]),
                "external_predicted_as_lab": int(cell[external, lab]),
                "lab_predicted_as_external": int(cell[lab, external]),
            })
    return pd.DataFrame(rows)


def source_probe(name, selected, subsets, X_train, X_test, y_train, y_test):
    """Test directly whether the features give away the data source.

    A second model, with the same hyperparameters, is trained to answer only
    "external or lab?". The source of a row is known from its row_id: the
    merge script wrote the external rows first. Two versions are run:
      all_rows     every row. Easy by construction, since most labels exist
                   in one source only.
      benign_only  only benign rows, the one label present in both sources.
                   Here the traffic type is the same, so whatever separates
                   the rows is the source (lab setup and extractor), not an
                   attack.
    """
    rows = []
    for scope in ("all_rows", "benign_only"):
        keep_train = y_train == "benign" if scope == "benign_only" else y_train == y_train
        keep_test = y_test == "benign" if scope == "benign_only" else y_test == y_test
        X_tr, X_te = X_train[keep_train], X_test[keep_test]
        source_tr = pd.Series(np.where(X_tr.index < N_EXTERNAL_ROWS,
                                       "external", "lab"), index=X_tr.index)
        source_te = pd.Series(np.where(X_te.index < N_EXTERNAL_ROWS,
                                       "external", "lab"), index=X_te.index)
        for config in selected:
            features = subsets[config][1]
            classes, proba, _ = fit_predict(X_tr, source_tr, X_te, features)
            predicted = classes[proba.argmax(axis=1)]
            recalls = recall_score(source_te, predicted, average=None,
                                   labels=["external", "lab"])
            rows.append({
                "dataset": name, "scope": scope, "config": config,
                "n_features": len(features),
                "external_test_rows": int((source_te == "external").sum()),
                "lab_test_rows": int((source_te == "lab").sum()),
                "external_recall": recalls[0], "lab_recall": recalls[1],
                "misclassified_rows": int((predicted != source_te).sum()),
                "mcc": matthews_corrcoef(source_te, predicted),
            })
        log(f"  [{name}] source probe {scope}: done")
    return pd.DataFrame(rows)


def run_dataset(name):
    """Run cross-validation, selection, test and source checks for one dataset."""
    start = time.perf_counter()
    X_train, y_train = load_split(name, "train")
    X_test, y_test = load_split(name, "test")
    log(f"\n=== Dataset {name}: train {len(X_train)}, test {len(X_test)} ===")

    folds, cv_recall = cross_validate(name, X_train, y_train)
    summary = summarize_cv(folds)
    selected = summary.loc[summary["selected"], "config"].tolist()
    log(f"  [{name}] selected for test: {selected}")

    test, test_recall, confusion, features, subsets = evaluate_on_test(
        name, selected, X_train, y_train, X_test, y_test)
    probe = source_probe(name, selected, subsets, X_train, X_test,
                         y_train, y_test)
    log(f"  [{name}] finished in {(time.perf_counter() - start) / 60:.1f} min")
    return {
        "cv_folds": folds, "cv_results": summary,
        "cv_recall_by_class": cv_recall, "test_results": test,
        "test_recall_by_class": test_recall, "confusion_test": confusion,
        "selected_features": features,
        "source_pair_confusion": equivalent_pair_confusion(confusion),
        "source_probe": probe,
    }


def write_outputs(name, tables):
    """Write the tables of one dataset, replacing its rows in the shared files.

    Each file holds A and B. Replacing only the rows of the dataset that was
    just computed lets one dataset be re-run without losing the other.
    """
    for key, table in tables.items():
        path = EVALUATION_DIR / f"{key}.csv"
        if path.exists():
            previous = pd.read_csv(path)
            table = pd.concat([previous[previous["dataset"] != name], table],
                              ignore_index=True)
        table.sort_values("dataset", kind="stable").to_csv(path, index=False)


def main():
    """Evaluate the datasets named on the command line (default: A and B)."""
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[1:] or ["A", "B"]:
        write_outputs(name, run_dataset(name))
    log(f"\nWritten: {EVALUATION_DIR}")


if __name__ == "__main__":
    main()
