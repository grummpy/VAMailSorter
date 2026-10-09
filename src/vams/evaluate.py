"""Score the held-out test split once, with a threshold already chosen.

``evaluate`` does not refit models and does not call ``select_threshold``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    precision_recall_fscore_support,
)

from vams.data import as_text, content_fingerprints, decode_label, encode_labels
from vams.explain import explain_message, format_hits, format_terms
from vams.metrics import expected_calibration_error
from vams.rules import suspicious_score


def _class_report(y_true, p_action, threshold: float) -> dict:
    y = np.asarray(y_true, dtype=int)
    pred = (np.asarray(p_action) >= threshold).astype(int)
    precision, recall, fscore, support = precision_recall_fscore_support(
        y, pred, labels=[1, 0], zero_division=0
    )
    f2 = float(fbeta_score(y, pred, beta=2, pos_label=1, zero_division=0))
    matrix = confusion_matrix(y, pred, labels=[1, 0])
    return {
        "threshold": float(threshold),
        "precision_action_needed": float(precision[0]),
        "recall_action_needed": float(recall[0]),
        "f1_action_needed": float(fscore[0]),
        "f2_action_needed": float(f2),
        "support_action_needed": int(support[0]),
        "precision_informational": float(precision[1]),
        "recall_informational": float(recall[1]),
        "f1_informational": float(fscore[1]),
        "support_informational": int(support[1]),
        "confusion_matrix_labels": ["ACTION_NEEDED", "INFORMATIONAL"],
        "confusion_matrix": matrix.tolist(),
        "brier": float(brier_score_loss(y, p_action)),
        "ece_10": float(expected_calibration_error(y, p_action, n_bins=10)),
    }


def _suspicious_report(y_true, scores, threshold: float) -> dict:
    y = np.asarray(y_true, dtype=int)
    pred = (np.asarray(scores) >= threshold).astype(int)
    precision, recall, fscore, support = precision_recall_fscore_support(
        y, pred, labels=[1, 0], zero_division=0
    )
    return {
        "threshold": float(threshold),
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "f1": float(fscore[0]),
        "support_suspicious": int(support[0]),
        "support_benign": int(support[1]),
    }


def _note(row, predicted: int, y_true: int, detected: bool) -> str:
    if int(row["suspicious"]) == 1 and not detected:
        return "Suspicious cues were below the validation cutoff."
    if int(row["suspicious"]) == 0 and detected:
        return "Benign mail matched a suspicious rule."
    if row["hard_case"] == "urgent_newsletter" and predicted == 1:
        return "Urgent newsletter wording was scored as action needed."
    if row["hard_case"] == "polite_deadline" and predicted == 0:
        return "Polite deadline was scored as informational."
    if predicted != y_true:
        return "Linear term weights favored the other class."
    return "Held-out hard case scored as labeled."


def _error_table(bundle, frame: pd.DataFrame) -> pd.DataFrame:
    texts = as_text(frame)
    y = np.asarray(encode_labels(frame["label"]), dtype=int)
    proba = bundle.predict_action_proba(texts)
    pred = (proba >= bundle.threshold).astype(int)
    rows = []
    for index in range(len(frame)):
        record = frame.iloc[index]
        detected = suspicious_score(texts[index])[0] >= bundle.suspicious_threshold
        wrong = int(pred[index]) != int(y[index])
        hard = record["hard_case"] not in {"", "none"}
        sus_mismatch = bool(detected) != bool(int(record["suspicious"]))
        if not (wrong or hard or sus_mismatch):
            continue
        explained = explain_message(bundle, texts[index])
        rows.append(
            {
                "id": record["id"],
                "subject": record["subject"],
                "notice_type": record["notice_type"],
                "hard_case": record["hard_case"],
                "suspicious": int(record["suspicious"]),
                "y_true": decode_label(y[index]),
                "y_pred": decode_label(pred[index]),
                "triage_label": explained["triage_label"],
                "p_action_needed": round(float(proba[index]), 4),
                "class_correct": int(not wrong),
                "suspicious_detected": int(detected),
                "top_terms": format_terms(explained["top_terms"]),
                "rule_hits": format_hits(explained["rule_hits"]),
                "notes": _note(record, int(pred[index]), int(y[index]), detected),
            }
        )
    return pd.DataFrame(rows)


def _save_confusion(matrix, labels, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=18, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Held-out test confusion matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color="#142033")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_reliability(y, p, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#8a8175", label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, marker="o", color="#1f4e79", label="Logistic model")
    ax.set_xlabel("Mean calibrated P(ACTION_NEEDED)")
    ax.set_ylabel("Fraction that are ACTION_NEEDED")
    ax.set_title("Reliability diagram (held-out test)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _model_card(metrics: dict, test_frame: pd.DataFrame) -> str:
    primary = metrics["held_out"]["tfidf_logreg"]
    matrix = primary["confusion_matrix"]
    comparisons = metrics["held_out"]["comparisons"]
    sus = metrics["held_out"]["suspicious"]
    cv_lines = [
        "| Model | F2 mean | F2 std | Precision mean | Recall mean |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics["cv_train"]:
        cv_lines.append(
            f"| {row['model']} | {_fmt(row['f2_mean'])} | {_fmt(row['f2_std'])} | "
            f"{_fmt(row['precision_mean'])} | {_fmt(row['recall_mean'])} |"
        )
    comp_lines = [
        "| Model | Precision ACTION | Recall ACTION | F2 ACTION | Brier |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in comparisons.items():
        comp_lines.append(
            f"| {name} | {_fmt(row['precision_action_needed'])} | "
            f"{_fmt(row['recall_action_needed'])} | {_fmt(row['f2_action_needed'])} | "
            f"{row['brier']:.6f} |"
        )
    mistakes = metrics["error_analysis_mistakes"]
    if mistakes:
        mistake_lines = [
            "| Id | True | Predicted | P(action) | Note |",
            "| --- | --- | --- | ---: | --- |",
        ]
        for row in mistakes[:15]:
            mistake_lines.append(
                f"| {row['id']} | {row['y_true']} | {row['y_pred']} | "
                f"{row['p_action_needed']:.3f} | {row['notes']} |"
            )
    else:
        mistake_lines = ["No class mismatches on the held-out test split."]
    alarms = metrics["suspicious_false_alarms"]
    if alarms:
        alarm_lines = [
            "| Id | Notice | Subject | Note |",
            "| --- | --- | --- | --- |",
        ]
        for row in alarms:
            alarm_lines.append(
                f"| {row['id']} | {row['notice_type']} | {row['subject']} | {row['notes']} |"
            )
    else:
        alarm_lines = ["No suspicious false alarms on the held-out test split."]
    n_test = int(len(test_frame))
    return f"""# Model card: VA Mail Sorter

## Model details

- **Name:** VA Mail Sorter triage classifier
- **Type:** TF-IDF (word unigrams and bigrams) + L2 logistic regression, sigmoid-calibrated with `CalibratedClassifierCV` (`method="sigmoid"`, `ensemble=False`, 5-fold on the training split)
- **Deployed C:** {metrics["best_C"]} (chosen by training-split stratified CV, maximizing F2 of ACTION_NEEDED, then recall)
- **Decision threshold:** {primary["threshold"]:.2f} on calibrated P(ACTION_NEEDED)
- **Threshold policy:** {metrics["threshold_policy"]}, selected on the **validation** split only
- **Seed:** {metrics["seed"]}
- **Comparisons (not deployed):** keyword rules, calibrated linear SVM (C={metrics["best_svm_C"]}), calibrated multinomial naive Bayes (alpha={metrics["best_nb_alpha"]})
- **Suspicious override:** rule score, cutoff {sus["threshold"]} chosen on validation ({metrics["suspicious_policy"]}). When it fires, the triage label is SUSPICIOUS and replaces the class label in the report.

Probabilities used at inference come from the sigmoid calibrator. Term explanations come from the single logistic regression fit on the full training split (the `ensemble=False` estimator).

## Intended use

A local triage aid for a veteran sorting mail that *resembles* VA, DFAS, and VGLI (Prudential/Securian-administered life insurance) notices into ACTION_NEEDED or INFORMATIONAL, with a calibrated probability and the top reasons.

The tool never replies, deletes, or moves mail.

## Out of scope

This is not legal advice, benefits advice, or an official VA, DFAS, Prudential, or Securian system. It does not decide claims, debts, premiums, or appeals. A score is not a determination. Verify the contents of a real letter on VA.gov, myPay, or the insurer's own site. Do not click links in mail that looks suspicious.

## Data

- Public corpus: synthetic subjects and bodies only (`data/synthetic/emails.csv`)
- Rows: {metrics["n_train"]} train / {metrics["n_val"]} validation / {n_test} test
- Labels: ACTION_NEEDED and INFORMATIONAL, plus a SUSPICIOUS flag that overrides the class at triage time
- Split: whole synthetic template families are assigned before rendering; families do not cross partitions (seed {metrics["seed"]})
- The test split is not used to choose C, the class threshold, or the suspicious cutoff
- Labeling rules: `docs/labeling_guide.md`

## Evaluation procedure

1. Family-aware stratified 5-fold CV on the training split compares the keyword baseline, logistic regression, linear SVM, and multinomial naive Bayes. A synthetic template family is kept wholly on one side of every fold. CV predictions use each model's default decision rule (probability 0.5, or the SVM margin).
2. The deployed model is logistic regression, because its coefficients are the explanation. Its C is the CV winner inside the logistic grid only.
3. `CalibratedClassifierCV` fits the sigmoid map from out-of-fold training scores.
4. The class threshold and the suspicious cutoff are chosen on validation only.
5. The figures below are a single pass over the held-out test split.

Recall of ACTION_NEEDED is the priority: a missed deadline costs more than an extra review. The validation rule keeps the highest recall that still holds precision of at least 0.70. Ties break toward the threshold closest to 0.5. If no threshold clears the floor, the rule falls back to F2.

## Held-out test metrics (deployed model)

| | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| ACTION_NEEDED | {_fmt(primary["precision_action_needed"])} | {_fmt(primary["recall_action_needed"])} | {_fmt(primary["f1_action_needed"])} | {primary["support_action_needed"]} |
| INFORMATIONAL | {_fmt(primary["precision_informational"])} | {_fmt(primary["recall_informational"])} | {_fmt(primary["f1_informational"])} | {primary["support_informational"]} |

- F2 (ACTION_NEEDED): {_fmt(primary["f2_action_needed"])}
- Brier score: {primary["brier"]:.6f}
- Expected calibration error (10 equal-width bins): {primary["ece_10"]:.3f}
- Confusion matrix (rows true, columns predicted; order ACTION_NEEDED, INFORMATIONAL): {matrix}
- Validation operating point at this same threshold: precision {_fmt(metrics["threshold_val_precision"])}, recall {_fmt(metrics["threshold_val_recall"])}, F2 {_fmt(metrics["threshold_val_f2"])}

![Held-out confusion matrix](../reports/confusion_matrix.png)

![Held-out reliability diagram](../reports/calibration_curve.png)

### Suspicious override on the held-out split

| Precision | Recall | F1 | Suspicious support |
| ---: | ---: | ---: | ---: |
| {_fmt(sus["precision"])} | {_fmt(sus["recall"])} | {_fmt(sus["f1"])} | {sus["support_suspicious"]} |

## Training-split cross-validation

Default decision rule, not the validation threshold. Used to pick logistic C and to report baselines.

{chr(10).join(cv_lines)}

## Held-out comparison at each model's own validation threshold

The deployed model stays logistic regression even if another row is higher. Test numbers were not used to choose it.

{chr(10).join(comp_lines)}

## Error analysis

Class mismatches on the held-out split, up to 15 rows. The full table, including hard cases that were scored correctly, is `reports/error_analysis.csv`.

{chr(10).join(mistake_lines)}

Suspicious false alarms (benign mail at or above the validation cutoff). The cue list matches words such as "password" even when the sentence says the office will not ask for one.

{chr(10).join(alarm_lines)}

## Explanations

Each prediction lists up to eight terms by contribution (TF-IDF value times the logistic coefficient) in the direction of the predicted class, and any rule hits for a deadline date, a dollar amount, or the words debt, due, action required, and decision.

## Ethical considerations

- A false informational label can hide a deadline. The threshold policy spends precision to avoid that, and still will miss some polite notices.
- A false action label creates review work. That is the cheaper error for this aid, not a reason to treat the score as a demand.
- Phishing look-alikes are a separate flag. The flag is a heuristic. It is not a determination that a real message is safe.
- The corpus has no real veterans, claim numbers, Social Security numbers, or account numbers. Do not point this training code at a real mailbox export and then publish the result.

## Limitations

- Every public email is synthetic. Performance on real VA, DFAS, or VGLI mail is unmeasured.
- The test split is small, so a single letter moves precision and recall.
- Calibration is estimated on that same small test split; the reliability diagram is noisy.
- Linear TF-IDF explanations miss negation scope and document layout that a scanned letter would have.
- The suspicious rules look for a fixed cue list (passwords, payment scams, suspension threats, verify-your-account language, look-alike links). A phish that avoids those words will not be flagged.
- Optional Gmail access is read-only, off by default, and not part of the reported metrics.

## Privacy

`data/real/` and `secrets/` are gitignored. Training and the numbers in this card use `data/synthetic/` only.
"""


def evaluate(
    bundle, test_df, out_dir: Path, model_card_path: Path, n_train: int, n_val: int
) -> dict:
    if "split" in test_df.columns and not (test_df["split"] == "test").all():
        raise ValueError("evaluate expects the held-out test split only")
    lineage_attributes = (
        "train_ids",
        "validation_ids",
        "train_template_families",
        "validation_template_families",
        "train_content_fingerprints",
        "validation_content_fingerprints",
    )
    if getattr(bundle, "lineage_version", 0) < 2 or any(
        not hasattr(bundle, attribute) for attribute in lineage_attributes
    ):
        raise ValueError("bundle lacks required split lineage; retrain before held-out evaluation")
    known_ids = set(bundle.train_ids) | set(bundle.validation_ids)
    overlap = set(test_df["id"]) & known_ids
    if overlap:
        raise ValueError(f"test ids overlap stored training identities: {sorted(overlap)[:5]}")
    if "template_family" not in test_df.columns:
        raise ValueError("evaluate needs template_family lineage")
    known_families = set(bundle.train_template_families) | set(bundle.validation_template_families)
    family_overlap = set(test_df["template_family"].astype(str)) & known_families
    if family_overlap:
        raise ValueError(
            f"test template families overlap stored training lineage: {sorted(family_overlap)[:5]}"
        )
    known_content = set(bundle.train_content_fingerprints) | set(bundle.validation_content_fingerprints)
    if set(content_fingerprints(test_df)) & known_content:
        raise ValueError("test content overlaps stored training lineage")

    out_dir.mkdir(parents=True, exist_ok=True)
    texts = as_text(test_df)
    y = np.asarray(encode_labels(test_df["label"]), dtype=int)
    primary_p = bundle.predict_action_proba(texts)
    primary = _class_report(y, primary_p, bundle.threshold)

    comparisons = {}
    for name, threshold in bundle.comparison_thresholds.items():
        comparisons[name] = _class_report(y, bundle.comparison_proba(name, texts), threshold)
    comparisons["tfidf_logreg"] = primary

    sus = _suspicious_report(
        test_df["suspicious"].to_numpy(),
        [suspicious_score(text)[0] for text in texts],
        bundle.suspicious_threshold,
    )
    errors = _error_table(bundle, test_df)
    errors.to_csv(out_dir / "error_analysis.csv", index=False)
    _save_confusion(
        primary["confusion_matrix"],
        primary["confusion_matrix_labels"],
        out_dir / "confusion_matrix.png",
    )
    _save_reliability(y, primary_p, out_dir / "calibration_curve.png")
    pd.DataFrame(bundle.cv_results).to_csv(out_dir / "cv_results.csv", index=False)

    mistakes = []
    alarms = []
    if not errors.empty:
        wrong = errors.loc[errors["class_correct"] == 0]
        mistakes = wrong.to_dict(orient="records")
        alarms = errors.loc[
            (errors["suspicious"] == 0) & (errors["suspicious_detected"] == 1)
        ].to_dict(orient="records")

    metrics = {
        "seed": bundle.seed,
        "fitted_on": bundle.fitted_on,
        "threshold_selected_on": bundle.threshold_selected_on,
        "test_used_for_tuning": False,
        "threshold_policy": bundle.threshold_policy,
        "threshold_val_precision": bundle.threshold_val_precision,
        "threshold_val_recall": bundle.threshold_val_recall,
        "threshold_val_f2": bundle.threshold_val_f2,
        "suspicious_policy": bundle.suspicious_policy,
        "best_C": bundle.best_C,
        "best_svm_C": bundle.best_svm_C,
        "best_nb_alpha": bundle.best_nb_alpha,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": int(len(test_df)),
        "cv_train": bundle.cv_results,
        "held_out": {
            "tfidf_logreg": primary,
            "comparisons": {
                "keyword_rules": comparisons["keyword_rules"],
                "tfidf_linear_svc": comparisons["tfidf_linear_svc"],
                "tfidf_multinomial_nb": comparisons["tfidf_multinomial_nb"],
                "tfidf_logreg": primary,
            },
            "suspicious": sus,
        },
        "error_analysis_mistakes": mistakes,
        "suspicious_false_alarms": alarms,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    model_card_path.parent.mkdir(parents=True, exist_ok=True)
    model_card_path.write_text(_model_card(metrics, test_df), encoding="utf-8")
    return metrics
