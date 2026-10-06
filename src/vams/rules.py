"""Keyword baseline, explanation rules, and the suspicious-mail override.

Rule hits are evidence shown next to a prediction. They do not send, delete,
or move mail. A suspicious score at or above the validation cutoff overrides
the action/informational class in the triage label.
"""

from __future__ import annotations

import math
import re

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

# Longer phrases are matched first. A negative phrase masks its span so a
# positive phrase inside it (for example "action required" inside "no action
# required") is not also counted.
POSITIVE_PHRASES: tuple[tuple[str, float], ...] = (
    ("action required", 3.0),
    ("action needed", 2.6),
    ("past due", 2.4),
    ("amount due", 2.2),
    ("balance due", 2.2),
    ("premium due", 2.3),
    ("overpayment", 2.4),
    ("direct deposit was returned", 2.1),
    ("update your direct deposit", 1.9),
    ("payment was rejected", 1.8),
    ("verify your address", 2.0),
    ("address verification", 2.0),
    ("higher-level review", 1.6),
    ("notice of disagreement", 1.6),
    ("response window", 1.7),
    ("must submit", 1.7),
    ("respond by", 1.7),
    ("reply by", 1.7),
    ("no later than", 1.5),
    ("please confirm your appointment", 1.6),
    ("you must", 1.5),
    ("grace period", 1.4),
    ("will end on", 1.5),
    ("evidence by", 1.4),
    ("debt", 2.0),
    ("waiver", 1.2),
    ("appeal", 1.2),
    ("lapse", 1.7),
    ("deadline", 1.4),
    ("reschedule", 1.1),
)
NEGATIVE_PHRASES: tuple[tuple[str, float], ...] = (
    ("no action is required", 3.2),
    ("no action required", 3.2),
    ("no response is needed", 2.6),
    ("no response needed", 2.5),
    ("for your information", 2.0),
    ("optional survey", 2.0),
    ("thank you for your payment", 1.7),
    ("payment was received", 1.6),
    ("payment posted", 1.5),
    ("still processing", 1.4),
    ("we have received", 1.2),
    ("holiday hours", 1.5),
    ("newsletter", 2.2),
    ("unsubscribe", 1.8),
    ("your feedback", 1.2),
    ("now available", 0.8),
)

_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
DEADLINE_RE = re.compile(
    rf"\b(?:{_MONTHS})\s+\d{{1,2}},\s+\d{{4}}\b|\b\d{{1,2}}/\d{{1,2}}/\d{{4}}\b",
    re.IGNORECASE,
)
DOLLAR_RE = re.compile(r"\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?")
KEYWORD_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("keyword_debt", re.compile(r"\b(?:debt|overpayment)\b", re.IGNORECASE)),
    ("keyword_due", re.compile(r"\b(?:due|overdue)\b", re.IGNORECASE)),
    ("keyword_action_required", re.compile(r"action\s+required", re.IGNORECASE)),
    ("keyword_decision", re.compile(r"\bdecision\b", re.IGNORECASE)),
)

# Fixed cues. The cutoff that turns a score into a flag is chosen on validation.
SUSPICIOUS_PATTERNS: tuple[tuple[str, float, re.Pattern[str]], ...] = (
    (
        "credential_request",
        2.0,
        re.compile(
            r"\b(?:password|passcode|one-time code|social security|ssn)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "payment_scam",
        2.0,
        re.compile(r"\b(?:gift cards?|wire transfer|bitcoin|crypto wallet)\b", re.IGNORECASE),
    ),
    (
        "suspension_threat",
        1.5,
        re.compile(
            r"\b(?:suspended|account will be deleted|benefits will be stopped)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "verify_account",
        1.5,
        re.compile(r"verify your (?:account|identity|login)", re.IGNORECASE),
    ),
    (
        "lookalike_url",
        1.5,
        re.compile(r"https?://\S*(?:login|verify-now|secure-update)\S*", re.IGNORECASE),
    ),
)

RULE_TEMPERATURE = 1.6


def _mask_phrases(text_lower: str, phrases: tuple[tuple[str, float], ...], occupied: np.ndarray):
    hits: list[tuple[str, float]] = []
    ordered = sorted(phrases, key=lambda item: len(item[0]), reverse=True)
    for phrase, weight in ordered:
        start = 0
        while True:
            index = text_lower.find(phrase, start)
            if index < 0:
                break
            end = index + len(phrase)
            if not occupied[index:end].any():
                occupied[index:end] = True
                hits.append((phrase, weight))
            start = index + 1
    return hits


def rule_components(text: str) -> tuple[float, float, list[str]]:
    """Return positive weight, negative weight, and matched phrases."""
    lowered = text.lower()
    occupied = np.zeros(len(lowered), dtype=bool)
    negative = _mask_phrases(lowered, NEGATIVE_PHRASES, occupied)
    positive = _mask_phrases(lowered, POSITIVE_PHRASES, occupied)
    pos = sum(weight for _, weight in positive)
    neg = sum(weight for _, weight in negative)
    if DEADLINE_RE.search(text):
        pos += 0.35
    if DOLLAR_RE.search(text):
        pos += 0.35
    matched = [phrase for phrase, _ in positive + negative]
    return pos, neg, matched


def action_probability(text: str) -> float:
    pos, neg, _ = rule_components(text)
    raw = pos - neg
    return 1.0 / (1.0 + math.exp(-raw / RULE_TEMPERATURE))


def rule_hits(text: str) -> list[dict[str, str]]:
    """Spans a reviewer can check: dates, dollars, and a few decision words."""
    hits: list[dict[str, str]] = []
    for match in DEADLINE_RE.finditer(text):
        hits.append({"rule": "deadline_date", "match": match.group(0)})
    for match in DOLLAR_RE.finditer(text):
        hits.append({"rule": "dollar_amount", "match": match.group(0)})
    for name, pattern in KEYWORD_RULES:
        for match in pattern.finditer(text):
            hits.append({"rule": name, "match": match.group(0)})
    return hits


def suspicious_score(text: str) -> tuple[float, list[str]]:
    score = 0.0
    cues: list[str] = []
    for name, weight, pattern in SUSPICIOUS_PATTERNS:
        match = pattern.search(text)
        if match:
            score += weight
            cues.append(f"{name}:{match.group(0)}")
    return score, cues


class KeywordRuleClassifier(ClassifierMixin, BaseEstimator):
    """Unfitted phrase weights squashed into P(ACTION_NEEDED).

    ``predict`` uses 0.5 so cross-validation is comparable. The shipped
    operating point is a separate threshold chosen on the validation split.
    """

    def fit(self, X, y):
        del X
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        positive = np.array([action_probability(text) for text in X], dtype=float)
        proba = np.zeros((len(positive), len(self.classes_)), dtype=float)
        for col, label in enumerate(self.classes_):
            proba[:, col] = positive if int(label) == 1 else 1.0 - positive
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        action_col = list(self.classes_).index(1)
        return (proba[:, action_col] >= 0.5).astype(int)
