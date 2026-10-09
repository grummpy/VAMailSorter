from pathlib import Path

from vams.explain import explain_message, top_weighted_terms
from vams.rules import action_probability, rule_hits, suspicious_score

GMAIL_SOURCE = Path(__file__).resolve().parents[1] / "src" / "vams" / "gmail_fetch.py"


def test_rule_hits_cover_dates_dollars_and_decision_words():
    text = (
        "Subject: Debt notice\n\n"
        "Your debt of $125.00 is due by March 3, 2026. "
        "Action required after the decision."
    )
    names = {hit["rule"] for hit in rule_hits(text)}
    assert "deadline_date" in names
    assert "dollar_amount" in names
    assert "keyword_debt" in names
    assert "keyword_due" in names
    assert "keyword_action_required" in names
    assert "keyword_decision" in names


def test_dollar_rule_handles_valid_formats_without_truncating_malformed_values():
    hits = rule_hits("Amounts: $1240.00, $1,240.00, and malformed $12,34.")
    amounts = [hit["match"] for hit in hits if hit["rule"] == "dollar_amount"]
    assert amounts == ["$1240.00", "$1,240.00"]


def test_negation_is_not_counted_as_action_required():
    forced = action_probability(
        "Action required. The debt is due by March 3, 2026. Amount due $40.00."
    )
    negated = action_probability(
        "This is a newsletter. No action is required. No response is needed. Unsubscribe."
    )
    assert forced > negated
    assert negated < 0.5


def test_explanation_uses_terms_and_suspicious_override(trained_bundle):
    bundle = trained_bundle
    debt = (
        "Overpayment notice\n\nThe debt balance of $180.00 is due by March 2, 2026. "
        "Action required: pay or request a waiver."
    )
    explained = explain_message(bundle, debt)
    assert explained["top_terms"]
    assert all("term" in item and "weight" in item for item in explained["top_terms"])
    assert any(hit["rule"] == "keyword_debt" for hit in explained["rule_hits"])
    terms = top_weighted_terms(bundle.explain_estimator, debt, explained["predicted_class"])
    assert terms[0]["term"]
    if explained["predicted_class"] == "ACTION_NEEDED":
        assert terms[0]["weight"] > 0

    lure = (
        "Statement ready\n\nVerify your identity and type your password at "
        "http://va-benefits.login.example.com/verify or the account will be deleted."
    )
    flagged = explain_message(bundle, lure)
    assert flagged["triage_label"] == "SUSPICIOUS"
    assert flagged["suspicious"] is True
    assert suspicious_score(lure)[0] >= bundle.suspicious_threshold


def test_gmail_module_is_readonly():
    from vams.gmail_fetch import GMAIL_SCOPES

    source = GMAIL_SOURCE.read_text(encoding="utf-8")
    assert GMAIL_SCOPES == ["https://www.googleapis.com/auth/gmail.readonly"]
    for banned in (
        "gmail.modify",
        "gmail.send",
        "messages().send",
        "messages().delete",
        "messages().modify",
        "messages().trash",
    ):
        assert banned not in source
