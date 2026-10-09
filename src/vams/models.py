"""Train the calibrated logistic model and the comparison baselines.

Fitting uses the training split only. The decision threshold uses the
validation split only. This module does not read the held-out test split.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from vams.data import as_text, content_fingerprints, encode_labels
from vams.metrics import select_suspicious_threshold, select_threshold
from vams.rules import KeywordRuleClassifier, suspicious_score

LR_C_GRID = (0.25, 1.0, 4.0)
SVM_C_GRID = (0.5, 1.0, 4.0)
NB_ALPHA_GRID = (0.1, 0.5, 1.0)


def make_vectorizer() -> TfidfVectorizer:
    # No English stop list: "no" and "not" carry the difference between
    # "action required" and "no action required".
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )


def make_logreg(C: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", make_vectorizer()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=seed,
                    max_iter=2000,
                ),
            ),
        ]
    )


def make_svm(C: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", make_vectorizer()),
            (
                "clf",
                LinearSVC(
                    C=C,
                    class_weight="balanced",
                    dual="auto",
                    random_state=seed,
                    max_iter=5000,
                ),
            ),
        ]
    )


def make_nb(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", make_vectorizer()),
            ("clf", MultinomialNB(alpha=alpha)),
        ]
    )


def make_cv(y: np.ndarray, groups, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return deterministic, template-family-disjoint folds.

    A rendered synthetic message is not an independent sample: messages from
    the same authoring family share paragraphs.  Grouped folds keep every
    family on one side of each internal model-selection and calibration split.
    """
    labels = np.asarray(y, dtype=int)
    families = np.asarray(groups, dtype=str)
    if len(labels) != len(families):
        raise ValueError("CV labels and template families must have the same length")
    unique_families = np.unique(families)
    if len(unique_families) < 2:
        raise ValueError("family-aware CV needs at least two template families")
    family_counts = [len(np.unique(families[labels == label])) for label in np.unique(labels)]
    if min(family_counts) < 2:
            raise ValueError("each class needs at least two template families for grouped CV")
    n_splits = min(5, min(family_counts))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(labels)), labels, families))
    for train_index, test_index in folds:
        if set(families[train_index]) & set(families[test_index]):
            raise AssertionError("template family crossed an internal CV fold")
        if set(labels[train_index]) != set(labels) or set(labels[test_index]) != set(labels):
            raise AssertionError("an internal CV fold lost a class")
    return folds


def _cv_row(name: str, estimator, texts, y, cv) -> dict:
    scoring = {
        "f2": _f2_scorer,
        "precision": _precision_scorer,
        "recall": _recall_scorer,
    }
    result = cross_validate(estimator, texts, y, cv=cv, scoring=scoring, n_jobs=1)
    return {
        "model": name,
        "f2_mean": float(result["test_f2"].mean()),
        "f2_std": float(result["test_f2"].std(ddof=0)),
        "precision_mean": float(result["test_precision"].mean()),
        "precision_std": float(result["test_precision"].std(ddof=0)),
        "recall_mean": float(result["test_recall"].mean()),
        "recall_std": float(result["test_recall"].std(ddof=0)),
    }


def _f2_scorer(estimator, X, y):
    pred = estimator.predict(X)
    return fbeta_score(y, pred, beta=2, pos_label=1, zero_division=0)


def _precision_scorer(estimator, X, y):
    pred = estimator.predict(X)
    return precision_score(y, pred, pos_label=1, zero_division=0)


def _recall_scorer(estimator, X, y):
    pred = estimator.predict(X)
    return recall_score(y, pred, pos_label=1, zero_division=0)


def _calibrate(estimator, texts, y, cv) -> CalibratedClassifierCV:
    # ensemble=False fits one model on all training rows and uses family-aware
    # out-of-fold scores only to fit the sigmoid calibrator. Term weights then
    # belong to the same model that is scored at inference time.
    calibrated = CalibratedClassifierCV(
        estimator=estimator,
        method="sigmoid",
        cv=cv,
        ensemble=False,
    )
    calibrated.fit(texts, y)
    return calibrated


def _action_proba(calibrated: CalibratedClassifierCV, texts) -> np.ndarray:
    proba = calibrated.predict_proba(list(texts))
    classes = [int(label) for label in calibrated.classes_]
    return proba[:, classes.index(1)]


def _guard_frames(train_df, val_df) -> None:
    required = {"id", "template_family"}
    if not required <= set(train_df.columns) or not required <= set(val_df.columns):
        raise ValueError("train and validation frames need id and template_family columns")
    overlap = set(train_df["id"]) & set(val_df["id"])
    if overlap:
        raise ValueError(f"train/validation id overlap: {sorted(overlap)[:5]}")
    family_overlap = set(train_df["template_family"]) & set(val_df["template_family"])
    if family_overlap:
        raise ValueError(f"train/validation template overlap: {sorted(family_overlap)[:5]}")
    content_overlap = set(content_fingerprints(train_df)) & set(content_fingerprints(val_df))
    if content_overlap:
        raise ValueError("train/validation content overlap")
    if "split" in train_df.columns and not (train_df["split"] == "train").all():
        raise ValueError("training frame contains rows that are not in the train split")
    if "split" in val_df.columns and not (val_df["split"] == "val").all():
        raise ValueError("validation frame contains rows that are not in the val split")


@dataclass
class ModelBundle:
    seed: int
    threshold: float
    threshold_policy: str
    threshold_selected_on: str
    threshold_val_precision: float
    threshold_val_recall: float
    threshold_val_f2: float
    suspicious_threshold: float
    suspicious_policy: str
    best_C: float
    best_svm_C: float
    best_nb_alpha: float
    cv_results: list[dict]
    cv_n_splits: int
    calibrated: CalibratedClassifierCV
    explain_estimator: Pipeline
    # Version 2 includes family and content lineage required for held-out evaluation.
    lineage_version: int = 0
    comparisons: dict = field(default_factory=dict)
    comparison_thresholds: dict = field(default_factory=dict)
    fitted_on: str = "train"
    train_ids: tuple[str, ...] = ()
    validation_ids: tuple[str, ...] = ()
    train_template_families: tuple[str, ...] = ()
    validation_template_families: tuple[str, ...] = ()
    train_content_fingerprints: tuple[str, ...] = ()
    validation_content_fingerprints: tuple[str, ...] = ()

    def predict_action_proba(self, texts) -> np.ndarray:
        return _action_proba(self.calibrated, texts)

    def comparison_proba(self, name: str, texts) -> np.ndarray:
        model = self.comparisons[name]
        if name == "keyword_rules":
            proba = model.predict_proba(list(texts))
            classes = [int(label) for label in model.classes_]
            return proba[:, classes.index(1)]
        return _action_proba(model, texts)


def train(train_df, val_df, seed: int = 42) -> ModelBundle:
    """Fit on train, choose thresholds on validation, return a bundle."""
    _guard_frames(train_df, val_df)
    texts = as_text(train_df)
    y = np.asarray(encode_labels(train_df["label"]), dtype=int)
    families = train_df["template_family"].astype(str).to_numpy()
    cv = make_cv(y, families, seed)
    val_texts = as_text(val_df)
    y_val = np.asarray(encode_labels(val_df["label"]), dtype=int)

    with threadpool_limits(limits=1):
        cv_results: list[dict] = []
        cv_results.append(
            _cv_row("keyword_rules", KeywordRuleClassifier(), texts, y, cv)
        )

        best_C = LR_C_GRID[0]
        best_key = (-1.0, -1.0, 0.0)
        for C in LR_C_GRID:
            row = _cv_row(f"tfidf_logreg_C{C}", make_logreg(C, seed), texts, y, cv)
            cv_results.append(row)
            key = (row["f2_mean"], row["recall_mean"], -float(C))
            if key > best_key:
                best_key = key
                best_C = float(C)

        best_svm_C = SVM_C_GRID[0]
        best_svm_key = (-1.0, -1.0, 0.0)
        for C in SVM_C_GRID:
            row = _cv_row(f"tfidf_linear_svc_C{C}", make_svm(C, seed), texts, y, cv)
            cv_results.append(row)
            key = (row["f2_mean"], row["recall_mean"], -float(C))
            if key > best_svm_key:
                best_svm_key = key
                best_svm_C = float(C)

        best_alpha = NB_ALPHA_GRID[0]
        best_nb_key = (-1.0, -1.0, 0.0)
        for alpha in NB_ALPHA_GRID:
            row = _cv_row(
                f"tfidf_multinomial_nb_alpha{alpha}",
                make_nb(alpha),
                texts,
                y,
                cv,
            )
            cv_results.append(row)
            key = (row["f2_mean"], row["recall_mean"], -float(alpha))
            if key > best_nb_key:
                best_nb_key = key
                best_alpha = float(alpha)

        calibrated = _calibrate(make_logreg(best_C, seed), texts, y, cv)
        explain_estimator = calibrated.calibrated_classifiers_[0].estimator
        comparisons = {
            "keyword_rules": KeywordRuleClassifier().fit(texts, y),
            "tfidf_linear_svc": _calibrate(make_svm(best_svm_C, seed), texts, y, cv),
            "tfidf_multinomial_nb": _calibrate(make_nb(best_alpha), texts, y, cv),
        }

        bundle_partial_threshold_model = calibrated
        p_val = _action_proba(bundle_partial_threshold_model, val_texts)
        choice = select_threshold(y_val, p_val)

        comparison_thresholds = {}
        keyword = comparisons["keyword_rules"]
        keyword_p = keyword.predict_proba(val_texts)[:, list(keyword.classes_).index(1)]
        keyword_choice = select_threshold(y_val, keyword_p)
        comparison_thresholds["keyword_rules"] = keyword_choice.threshold
        for name in ("tfidf_linear_svc", "tfidf_multinomial_nb"):
            comp_choice = select_threshold(y_val, _action_proba(comparisons[name], val_texts))
            comparison_thresholds[name] = comp_choice.threshold

        sus_scores = [suspicious_score(text)[0] for text in val_texts]
        sus_threshold, sus_policy, _, _ = select_suspicious_threshold(
            val_df["suspicious"].to_numpy(),
            sus_scores,
        )

    return ModelBundle(
        lineage_version=2,
        seed=seed,
        threshold=choice.threshold,
        threshold_policy=choice.policy,
        threshold_selected_on="validation",
        threshold_val_precision=choice.precision,
        threshold_val_recall=choice.recall,
        threshold_val_f2=choice.f2,
        suspicious_threshold=sus_threshold,
        suspicious_policy=sus_policy,
        best_C=best_C,
        best_svm_C=best_svm_C,
        best_nb_alpha=best_alpha,
        cv_results=cv_results,
        cv_n_splits=len(cv),
        calibrated=calibrated,
        explain_estimator=explain_estimator,
        comparisons=comparisons,
        comparison_thresholds=comparison_thresholds,
        train_ids=tuple(sorted(str(value) for value in train_df["id"])),
        validation_ids=tuple(sorted(str(value) for value in val_df["id"])),
        train_template_families=tuple(sorted(set(train_df["template_family"].astype(str)))),
        validation_template_families=tuple(sorted(set(val_df["template_family"].astype(str)))),
        train_content_fingerprints=tuple(sorted(set(content_fingerprints(train_df)))),
        validation_content_fingerprints=tuple(sorted(set(content_fingerprints(val_df)))),
    )


def save_bundle(bundle: ModelBundle, path) -> None:
    path = str(path)
    joblib.dump(bundle, path)


def load_bundle(path) -> ModelBundle:
    bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle):
        raise TypeError(f"Expected a ModelBundle at {path}")
    return bundle
