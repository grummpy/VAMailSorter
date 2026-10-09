from pathlib import Path

import pandas as pd
import pytest

from vams.data import LABELS, REQUIRED_COLUMNS, SPLITS, load_emails
from vams.privacy import scan_text

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic" / "emails.csv"


@pytest.fixture(scope="module")
def emails() -> pd.DataFrame:
    return load_emails(DATA)


def test_dataset_size_and_schema(emails: pd.DataFrame):
    assert 200 <= len(emails) <= 400
    assert list(emails.columns) == list(REQUIRED_COLUMNS)
    assert set(emails["label"]) <= set(LABELS)
    assert set(emails["split"]) == set(SPLITS)
    assert set(emails["suspicious"]) <= {0, 1}
    assert emails["subject"].str.strip().ne("").all()
    assert emails["body"].str.strip().ne("").all()
    assert emails["id"].is_unique


def test_public_corpus_has_no_secrets(emails: pd.DataFrame):
    problems = []
    raw = DATA.read_text(encoding="utf-8")
    problems.extend(scan_text(raw, where="emails.csv"))
    for split in SPLITS:
        path = ROOT / "data" / "splits" / f"{split}.csv"
        problems.extend(scan_text(path.read_text(encoding="utf-8"), where=path.name))
    assert problems == []


def test_scanner_rejects_ssn_and_foreign_email():
    problems = scan_text("ssn 123-45-6789 mailed to veteran@gmail.com", where="probe")
    assert any("SSN" in item for item in problems)
    assert any("gmail.com" in item for item in problems)
    assert scan_text("Balance $1,240.00 due March 2, 2026 at desk@va.example.com") == []
    assert any("8+" in item for item in scan_text("account 123456789", where="probe"))


def test_labeling_invariants(emails: pd.DataFrame):
    phishing = emails["hard_case"] == "phishing"
    assert phishing.sum() > 0
    assert (emails.loc[phishing, "suspicious"] == 1).all()
    assert (emails.loc[~phishing, "suspicious"] == 0).all()
    urgent = emails["hard_case"] == "urgent_newsletter"
    assert (emails.loc[urgent, "label"] == "INFORMATIONAL").all()
    polite = emails["hard_case"] == "polite_deadline"
    assert (emails.loc[polite, "label"] == "ACTION_NEEDED").all()
    assert (
        emails.loc[emails["notice_type"] == "debt_overpayment", "label"] == "ACTION_NEEDED"
    ).all()
    assert (emails.loc[emails["notice_type"] == "newsletter", "label"] == "INFORMATIONAL").all()
    for hard in ("phishing", "urgent_newsletter", "polite_deadline"):
        assert (emails["hard_case"] == hard).any()


def test_splits_match_files_and_cover_flags(emails: pd.DataFrame):
    expected_sizes = {"train": 192, "val": 80, "test": 32}
    for split in SPLITS:
        part = emails.loc[emails["split"] == split]
        on_disk = load_emails(ROOT / "data" / "splits" / f"{split}.csv")
        assert set(part["id"]) == set(on_disk["id"])
        assert len(part) == expected_sizes[split]
        assert set(part["label"]) == set(LABELS)
    # Whole-template assignment means the two synthetic phishing families are
    # isolated in validation and test rather than copied into every partition.
    assert emails.loc[emails["split"] == "train", "suspicious"].sum() == 0
    assert emails.loc[emails["split"] == "val", "suspicious"].sum() == 16
    assert emails.loc[emails["split"] == "test", "suspicious"].sum() == 16


def test_gitignore_blocks_real_mail_and_secrets():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/real/" in text
    assert "secrets/" in text
    assert "gmail_client.json" in text
    assert "gmail_token.json" in text
