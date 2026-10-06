"""Per-message reasons: linear term contributions plus rule hits."""

from __future__ import annotations

import numpy as np

from vams.data import decode_label
from vams.rules import rule_hits, suspicious_score


def top_weighted_terms(pipeline, text: str, predicted_label: str, k: int = 8) -> list[dict]:
    """Terms whose TF-IDF weight most supports the predicted class.

    Coefficients come from the logistic regression fit on the full training
    split. A positive weight pushes toward ACTION_NEEDED.
    """
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]
    row = vectorizer.transform([text])
    coef = classifier.coef_.ravel()
    contribution = np.asarray(row.multiply(coef).todense()).ravel()
    names = np.asarray(vectorizer.get_feature_names_out())
    if predicted_label == "ACTION_NEEDED":
        order = np.argsort(-contribution)
    else:
        order = np.argsort(contribution)
    chosen: list[dict] = []
    for index in order:
        weight = float(contribution[index])
        if weight == 0.0:
            continue
        chosen.append({"term": str(names[index]), "weight": weight})
        if len(chosen) >= k:
            break
    return chosen


def explain_message(bundle, text: str, k: int = 8) -> dict:
    """Calibrated probability, class decision, term weights, and rule hits."""
    proba = bundle.predict_action_proba([text])[0]
    predicted = int(proba >= bundle.threshold)
    predicted_label = decode_label(predicted)
    score, cues = suspicious_score(text)
    suspicious = bool(score >= bundle.suspicious_threshold)
    triage = "SUSPICIOUS" if suspicious else predicted_label
    confidence = float(proba if predicted == 1 else 1.0 - proba)
    terms = top_weighted_terms(bundle.explain_estimator, text, predicted_label, k=k)
    return {
        "predicted_class": predicted_label,
        "p_action_needed": float(proba),
        "confidence": confidence,
        "threshold": float(bundle.threshold),
        "triage_label": triage,
        "suspicious": suspicious,
        "suspicious_score": float(score),
        "suspicious_cues": cues,
        "top_terms": terms,
        "rule_hits": rule_hits(text),
    }


def format_terms(terms: list[dict]) -> str:
    return "; ".join(f"{item['term']} ({item['weight']:+.3f})" for item in terms)


def format_hits(hits: list[dict]) -> str:
    return "; ".join(f"{item['rule']}={item['match']}" for item in hits)
