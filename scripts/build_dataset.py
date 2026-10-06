"""Build the deterministic synthetic mailbox and the 60/20/20 split.

Run from the repository root after installing the package:

    python scripts/build_dataset.py

The held-out test split is written once and is not used to author rules.
"""

from __future__ import annotations

import itertools
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from banks import CATEGORIES, DISCLAIMER  # noqa: E402
from vams.data import as_text, jaccard, normalize_text, word_trigrams  # noqa: E402
from vams.privacy import scan_text  # noqa: E402

TO_ADDRS = (
    "Veteran <veteran@mailbox.example.com>",
    "Veteran <inbox@mailbox.example.com>",
    "Household <household@mailbox.example.com>",
)
TARGET_JACCARD = 0.65


def pack_triples(n: int = 4) -> list[tuple[int, int, int]]:
    """n^2 triples that share at most one coordinate.

    For each opener and middle there is one closing, ``(-a-b) mod n``.
    Two different triples therefore cannot match on two positions.
    """
    triples = []
    for opener, middle in itertools.product(range(n), repeat=2):
        closing = (-opener - middle) % n
        triples.append((opener, middle, closing))
    return triples


def build_rows() -> list[dict]:
    triples = pack_triples(4)
    rows: list[dict] = []
    serial = 1
    for category in CATEGORIES:
        slots = ("openers", "middles", "closings", "subjects")
        for slot in slots:
            if len(category[slot]) != 4:
                raise SystemExit(
                    f"{category['notice_type']} {category['hard_case']} needs 4 {slot}"
                )
        sources = category["sources"] if "sources" in category else [category["source"]]
        from_addrs = category["from_addrs"]
        for index, (a, b, c) in enumerate(triples):
            subject = category["subjects"][a]
            if "subject_tails" in category:
                subject = f"{subject}: {category['subject_tails'][b]}"
            body = "\n\n".join(
                (
                    category["openers"][a],
                    category["middles"][b],
                    category["closings"][c],
                    DISCLAIMER,
                )
            )
            rows.append(
                {
                    "id": f"vams-{serial:04d}",
                    "source": sources[index % len(sources)],
                    "notice_type": category["notice_type"],
                    "hard_case": category["hard_case"],
                    "label": category["label"],
                    "suspicious": int(category["suspicious"]),
                    "from_addr": from_addrs[index % len(from_addrs)],
                    "to_addr": TO_ADDRS[index % len(TO_ADDRS)],
                    "date": (date(2026, 1, 5) + timedelta(days=serial % 180)).isoformat(),
                    "subject": subject,
                    "body": body,
                }
            )
            serial += 1
    return rows


def assign_splits(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["stratum"] = frame["label"] + "|" + frame["suspicious"].astype(str)
    index = frame.index.to_numpy()
    train_idx, temp_idx = train_test_split(
        index, test_size=0.40, random_state=42, stratify=frame.loc[index, "stratum"]
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=42,
        stratify=frame.loc[temp_idx, "stratum"],
    )
    frame["split"] = ""
    frame.loc[train_idx, "split"] = "train"
    frame.loc[val_idx, "split"] = "val"
    frame.loc[test_idx, "split"] = "test"
    if (frame["split"] == "").any():
        raise SystemExit("Split assignment left rows unlabeled")
    return frame.drop(columns=["stratum"])


def assert_privacy(frame: pd.DataFrame) -> None:
    problems: list[str] = []
    for row in frame.itertuples(index=False):
        blob = "\n".join(
            str(getattr(row, col)) for col in ("from_addr", "to_addr", "subject", "body", "source")
        )
        problems.extend(scan_text(blob, where=row.id))
    if problems:
        raise SystemExit("Privacy scan failed:\n" + "\n".join(problems[:30]))


def assert_diversity(frame: pd.DataFrame) -> None:
    texts = as_text(frame)
    normalized = [normalize_text(text) for text in texts]
    if len(normalized) != len(set(normalized)):
        raise SystemExit("Exact duplicate document text")
    shingles = [word_trigrams(text) for text in texts]
    splits = frame["split"].tolist()
    worst = 0.0
    worst_pair = ("", "")
    ids = frame["id"].tolist()
    for i in range(len(shingles)):
        for j in range(i + 1, len(shingles)):
            score = jaccard(shingles[i], shingles[j])
            if score > worst:
                worst = score
                worst_pair = (ids[i], ids[j])
            if splits[i] != splits[j] and score > TARGET_JACCARD:
                raise SystemExit(
                    f"Near-duplicate across splits {ids[i]} vs {ids[j]} jaccard={score:.3f}"
                )
            if score > 0.80:
                raise SystemExit(f"Near-duplicate {ids[i]} vs {ids[j]} jaccard={score:.3f}")
    print(f"Worst trigram Jaccard {worst:.3f} between {worst_pair[0]} and {worst_pair[1]}")


def main() -> None:
    frame = pd.DataFrame(build_rows())
    frame = assign_splits(frame)
    column_order = [
        "id",
        "split",
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
    ]
    frame = frame[column_order].sort_values("id")
    assert_privacy(frame)
    assert_diversity(frame)
    out = ROOT / "data" / "synthetic" / "emails.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    split_dir = ROOT / "data" / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        frame.loc[frame["split"] == split].to_csv(split_dir / f"{split}.csv", index=False)
    counts = frame.groupby(["split", "label", "suspicious"]).size()
    print(f"Wrote {len(frame)} emails to {out}")
    print(counts.to_string())


if __name__ == "__main__":
    main()
