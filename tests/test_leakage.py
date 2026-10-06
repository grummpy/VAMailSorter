from pathlib import Path

from vams.data import (
    NEAR_DUPLICATE_JACCARD,
    as_text,
    jaccard,
    load_emails,
    normalize_text,
    word_trigrams,
)

ROOT = Path(__file__).resolve().parents[1]


def test_no_duplicate_or_near_duplicate_text_across_splits():
    frame = load_emails(ROOT / "data" / "synthetic" / "emails.csv")
    texts = as_text(frame)
    normalized = [normalize_text(text) for text in texts]
    assert len(normalized) == len(set(normalized))

    ids = frame["id"].tolist()
    splits = frame["split"].tolist()
    shingles = [word_trigrams(text) for text in texts]
    seen = {}
    for index, text in enumerate(normalized):
        assert text not in seen, f"duplicate text {ids[index]} and {seen[text]}"
        seen[text] = ids[index]

    for i in range(len(shingles)):
        for j in range(i + 1, len(shingles)):
            if splits[i] == splits[j]:
                continue
            score = jaccard(shingles[i], shingles[j])
            assert score <= NEAR_DUPLICATE_JACCARD, (
                f"{ids[i]} ({splits[i]}) and {ids[j]} ({splits[j]}) jaccard={score:.3f}"
            )

    id_sets = {
        split: set(frame.loc[frame["split"] == split, "id"]) for split in ("train", "val", "test")
    }
    assert id_sets["train"].isdisjoint(id_sets["val"])
    assert id_sets["train"].isdisjoint(id_sets["test"])
    assert id_sets["val"].isdisjoint(id_sets["test"])
