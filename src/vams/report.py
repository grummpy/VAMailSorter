"""CSV and standalone HTML triage reports. Reports do not alter mail."""

from __future__ import annotations

import csv
import html
from pathlib import Path

from vams.explain import explain_message, format_hits, format_terms

DISCLAIMER = (
    "Triage aid only. This is not legal or benefits advice and it is not an official "
    "VA, DFAS, or insurer tool. Verify anything important on VA.gov, myPay, or the "
    "insurer's own site. Do not click links in suspicious mail. This program never "
    "replies, deletes, or moves messages."
)


def classify_messages(bundle, messages: list[dict]) -> list[dict]:
    rows = []
    for message in messages:
        text = f"{message.get('subject', '')}\n\n{message.get('body', '')}".strip()
        explained = explain_message(bundle, text)
        rows.append(
            {
                "source_file": message.get("source_file", ""),
                "message_id": message.get("message_id", ""),
                "subject": message.get("subject", ""),
                "from": message.get("from", ""),
                "date": message.get("date", ""),
                "triage_label": explained["triage_label"],
                "predicted_class": explained["predicted_class"],
                "p_action_needed": f"{explained['p_action_needed']:.4f}",
                "confidence": f"{explained['confidence']:.4f}",
                "threshold": f"{explained['threshold']:.2f}",
                "suspicious": int(explained["suspicious"]),
                "suspicious_cues": "; ".join(explained["suspicious_cues"]),
                "top_terms": format_terms(explained["top_terms"]),
                "rule_hits": format_hits(explained["rule_hits"]),
            }
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_file",
        "message_id",
        "subject",
        "from",
        "date",
        "triage_label",
        "predicted_class",
        "p_action_needed",
        "confidence",
        "threshold",
        "suspicious",
        "suspicious_cues",
        "top_terms",
        "rule_hits",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _badge(label: str) -> str:
    css = {
        "ACTION_NEEDED": "badge action",
        "INFORMATIONAL": "badge info",
        "SUSPICIOUS": "badge suspicious",
    }.get(label, "badge")
    return f'<span class="{css}">{html.escape(label)}</span>'


def write_html(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body_rows = []
    for row in rows:
        body_rows.append(
            "<tr>"
            f"<td>{html.escape(row['message_id'])}</td>"
            f"<td>{html.escape(row['from'])}</td>"
            f"<td>{html.escape(row['subject'])}</td>"
            f"<td>{_badge(row['triage_label'])}</td>"
            f"<td>{html.escape(row['predicted_class'])}</td>"
            f"<td>{html.escape(row['p_action_needed'])}</td>"
            f"<td>{html.escape(row['confidence'])}</td>"
            f"<td>{html.escape(row['suspicious_cues'])}</td>"
            f"<td>{html.escape(row['top_terms'])}</td>"
            f"<td>{html.escape(row['rule_hits'])}</td>"
            "</tr>"
        )
    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>VA Mail Sorter report</title>
<style>
body {{ font-family: Georgia, serif; margin: 2rem; color: #142033; background: #f7f4ef; }}
h1 {{ font-family: "Segoe UI", sans-serif; letter-spacing: 0.02em; }}
.disclaimer {{ background: #fff6e8; border-left: 4px solid #b7791f; padding: 0.8rem 1rem; }}
table {{ border-collapse: collapse; width: 100%; background: white; font-size: 0.92rem; }}
th, td {{ border-bottom: 1px solid #e4ddd2; text-align: left; vertical-align: top; }}
th, td {{ padding: 0.45rem; }}
th {{ font-family: "Segoe UI", sans-serif; background: #10243f; color: white; }}
.badge {{ font-family: "Segoe UI", sans-serif; font-size: 0.78rem; padding: 0.15rem 0.4rem; }}
.action {{ background: #f3d2a4; }}
.info {{ background: #d5e4f5; }}
.suspicious {{ background: #f3c1c1; }}
</style>
</head>
<body>
<h1>VA Mail Sorter</h1>
<p class="disclaimer">{html.escape(DISCLAIMER)}</p>
<p>{len(rows)} message{"s" if len(rows) != 1 else ""}.</p>
<table>
<thead>
<tr>
<th>Id</th><th>From</th><th>Subject</th><th>Triage</th><th>Class</th>
<th>P(action)</th><th>Confidence</th><th>Suspicious cues</th><th>Top terms</th><th>Rule hits</th>
</tr>
</thead>
<tbody>
{"".join(body_rows)}
</tbody>
</table>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")
