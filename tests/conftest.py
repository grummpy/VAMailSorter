from pathlib import Path

import pytest

from vams.data import load_emails, require_split
from vams.models import train

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def trained_bundle():
    frame = load_emails(ROOT / "data" / "synthetic" / "emails.csv")
    return train(require_split(frame, "train"), require_split(frame, "val"), seed=42)
