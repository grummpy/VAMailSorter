import csv
import subprocess
import sys
from pathlib import Path

import pytest

from vams.mailio import load_path
from vams.models import save_bundle
from vams.report import write_csv

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


def test_csv_formula_safety_preserves_numbers_and_raw_evidence(tmp_path: Path):
    output = tmp_path / "report.csv"
    write_csv(
        [{"source_file": "=danger", "subject": "+formula", "p_action_needed": 0.25,
          "confidence": 0.75, "threshold": 0.5, "suspicious": False,
          "message_id": "@id", "from": "-sender", "date": "2026-01-01",
          "triage_label": "ACTION_NEEDED", "predicted_class": "ACTION_NEEDED",
          "suspicious_cues": "", "top_terms": "", "rule_hits": "raw $1,240.00"}],
        output,
    )
    with output.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["source_file"] == "'=danger"
    assert row["subject"] == "'+formula"
    assert row["message_id"] == "'@id"
    assert row["p_action_needed"] == "0.25"
    assert row["confidence"] == "0.75"
    assert row["rule_hits"] == "raw $1,240.00"


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
    assert metrics["cv_n_splits"] == 5


def test_model_card_reports_the_actual_grouped_fold_count(trained_bundle, tmp_path: Path):
    from vams.data import load_emails, require_split
    from vams.evaluate import _model_card, evaluate

    frame = load_emails(ROOT / "data" / "synthetic" / "emails.csv")
    metrics = evaluate(
        trained_bundle,
        require_split(frame, "test"),
        out_dir=tmp_path,
        model_card_path=tmp_path / "model_card.md",
        n_train=192,
        n_val=80,
    )
    metrics["cv_n_splits"] = 4
    card = _model_card(metrics, require_split(frame, "test"))
    assert "4-fold on the training split" in card
    assert "Family-aware stratified 4-fold CV" in card


def test_evaluate_rejects_stored_lineage_overlap_and_legacy_bundle(trained_bundle, tmp_path: Path):
    from vams.data import load_emails, require_split
    from vams.evaluate import evaluate

    frame = load_emails(ROOT / "data" / "synthetic" / "emails.csv")
    test = require_split(frame, "test")
    kwargs = {
        "out_dir": tmp_path,
        "model_card_path": tmp_path / "card.md",
        "n_train": 192,
        "n_val": 80,
    }
    by_id = test.copy()
    by_id.loc[by_id.index[0], "id"] = trained_bundle.train_ids[0]
    with pytest.raises(ValueError, match="ids overlap"):
        evaluate(trained_bundle, by_id, **kwargs)
    by_family = test.copy()
    by_family.loc[by_family.index[0], "template_family"] = trained_bundle.train_template_families[0]
    with pytest.raises(ValueError, match="template families overlap"):
        evaluate(trained_bundle, by_family, **kwargs)
    by_content = test.copy()
    source = require_split(frame, "train").iloc[0]
    by_content.loc[by_content.index[0], ["subject", "body"]] = [source.subject, source.body]
    with pytest.raises(ValueError, match="content overlaps"):
        evaluate(trained_bundle, by_content, **kwargs)
    trained_bundle.lineage_version = 1
    with pytest.raises(ValueError, match="lacks required split lineage"):
        evaluate(trained_bundle, test, **kwargs)


def test_evaluate_rejects_pre_fold_count_bundle(trained_bundle, tmp_path: Path):
    from vams.data import load_emails, require_split
    from vams.evaluate import evaluate

    frame = load_emails(ROOT / "data" / "synthetic" / "emails.csv")
    del trained_bundle.cv_n_splits
    with pytest.raises(ValueError, match="lacks required split lineage"):
        evaluate(
            trained_bundle,
            require_split(frame, "test"),
            out_dir=tmp_path,
            model_card_path=tmp_path / "card.md",
            n_train=192,
            n_val=80,
        )
