"""Load the synthetic mailbox and compare texts for leakage."""

from __future__ import annotations

import re
from hashlib import sha256
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = (
    "id",
    "split",
    "template_family",
    "source",
    "notice_type",
    "hard_case",
    "label",
    "suspicious",
    "from_addr",
    "to_addr",
    "date",
    "subject",
    "body",
)
LABELS = ("ACTION_NEEDED", "INFORMATIONAL")
SPLITS = ("train", "val", "test")
NEAR_DUPLICATE_JACCARD = 0.80


def document_text(subject: str, body: str) -> str:
    return f"{subject or ''}\n\n{body or ''}".strip()


def as_text(frame: pd.DataFrame) -> list[str]:
    subjects = frame["subject"].fillna("").astype(str)
    bodies = frame["body"].fillna("").astype(str)
    return [document_text(s, b) for s, b in zip(subjects, bodies, strict=True)]


def content_fingerprints(frame: pd.DataFrame) -> tuple[str, ...]:
    """Stable exact-content identities for split-lineage guards."""
    return tuple(
        sha256(normalize_text(text).encode("utf-8")).hexdigest() for text in as_text(frame)
    )


def load_emails(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    frame["suspicious"] = frame["suspicious"].astype(int)
    return frame


def require_split(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}")
    part = frame.loc[frame["split"] == split].copy()
    if part.empty:
        raise ValueError(f"No rows with split={split}")
    return part


def encode_labels(labels) -> list[int]:
    mapping = {"INFORMATIONAL": 0, "ACTION_NEEDED": 1}
    encoded = []
    for label in labels:
        if label not in mapping:
            raise ValueError(f"Unexpected label {label!r}")
        encoded.append(mapping[label])
    return encoded


def decode_label(value: int) -> str:
    return "ACTION_NEEDED" if int(value) == 1 else "INFORMATIONAL"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def word_trigrams(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    if len(words) < 3:
        return {tuple(words)} if words else set()
    return {(words[i], words[i + 1], words[i + 2]) for i in range(len(words) - 2)}


def jaccard(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float:
    if not left and not right:
        return 1.0
    union = len(left | right)
    if union == 0:
        return 0.0
    return len(left & right) / union
