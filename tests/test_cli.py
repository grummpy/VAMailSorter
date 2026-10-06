import csv
import subprocess
import sys
from pathlib import Path

from vams.mailio import load_path
from vams.models import save_bundle

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def test_mail_loader_reads_eml_and_mbox():
    messages = load_path(FIXTURES)
    subjects = {message["subject"] for message in messages}
    assert "Overpayment balance due" in subjects
    assert "Monthly example newsletter" in subjects
    assert "Verify your account now" in subjects
    assert "Please confirm your appointment" in subjects
    assert "Example 1099-R is now available" in subjects
    assert len(messages) == 5


def test_classify_writes_csv_and_html(trained_bundle, tmp_path: Path):
    model = tmp_path / "model.joblib"
    save_bundle(trained_bundle, model)
    csv_path = tmp_path / "report.csv"
    html_path = tmp_path / "report.html"
    before = {path: path.read_bytes() for path in FIXTURES.iterdir() if path.is_file()}
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "vams",
            "classify",
            str(FIXTURES),
            "--model",
            str(model),
            "--csv",
            str(csv_path),
            "--html",
            str(html_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    after = {path: path.read_bytes() for path in FIXTURES.iterdir() if path.is_file()}
    assert before == after
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    by_subject = {row["subject"]: row for row in rows}
    assert "p_action_needed" in rows[0]
    assert "top_terms" in rows[0]
    assert "rule_hits" in rows[0]
    suspicious = by_subject["Verify your account now"]
    assert suspicious["triage_label"] == "SUSPICIOUS"
    assert "keyword_debt" in by_subject["Overpayment balance due"]["rule_hits"]
    html = html_path.read_text(encoding="utf-8")
    assert "Overpayment balance due" in html
    assert "not legal or benefits advice" in html
    assert "SUSPICIOUS" in html


def test_classify_requires_a_model(tmp_path: Path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "vams",
            "classify",
            str(FIXTURES),
            "--model",
            str(tmp_path / "missing.joblib"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "vams train" in completed.stderr


def test_evaluate_does_not_move_the_threshold(trained_bundle, tmp_path: Path):
    from vams.data import load_emails, require_split
    from vams.evaluate import evaluate

    frame = load_emails(ROOT / "data" / "synthetic" / "emails.csv")
    threshold = trained_bundle.threshold
    metrics = evaluate(
        trained_bundle,
        require_split(frame, "test"),
        out_dir=tmp_path,
        model_card_path=tmp_path / "model_card.md",
        n_train=int((frame["split"] == "train").sum()),
        n_val=int((frame["split"] == "val").sum()),
    )
    assert trained_bundle.threshold == threshold
    assert metrics["test_used_for_tuning"] is False
    assert metrics["threshold_selected_on"] == "validation"
    assert (tmp_path / "confusion_matrix.png").exists()
    assert (tmp_path / "calibration_curve.png").exists()
    assert "Held-out test metrics" in (tmp_path / "model_card.md").read_text(encoding="utf-8")
