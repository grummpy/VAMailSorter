"""Privacy checks for the public synthetic corpus.

The public dataset may contain example.com, example.org, and example.net
addresses only. Social-security-like numbers and long account-like digit
strings are rejected.
"""

from __future__ import annotations

import re

ALLOWED_HOST = re.compile(r"^(?:[a-z0-9-]+\.)*example\.(?:com|org|net)$", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@([A-Za-z0-9.\-]+\.[A-Za-z]{2,})\b")
URL_RE = re.compile(r"\bhttps?://([^/\s\"'<>]+)", re.IGNORECASE)
SSN_DASH_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
SSN_SPACE_RE = re.compile(r"\b\d{3}\s+\d{2}\s+\d{4}\b")
LONG_DIGIT_RE = re.compile(r"\b\d{8,}\b")


def _host_ok(host: str) -> bool:
    host = host.strip().lower().rstrip(".")
    if "@" in host:
        return False
    if ":" in host:
        host = host.split(":", 1)[0]
    return bool(ALLOWED_HOST.match(host))


def scan_text(text: str, where: str = "text") -> list[str]:
    """Return human-readable problems found in ``text``."""
    problems: list[str] = []
    if SSN_DASH_RE.search(text) or SSN_SPACE_RE.search(text):
        problems.append(f"{where}: SSN-like number pattern")
    if LONG_DIGIT_RE.search(text):
        problems.append(f"{where}: 8+ digit sequence (account-like)")
    for match in EMAIL_RE.finditer(text):
        domain = match.group(1)
        if not _host_ok(domain):
            problems.append(f"{where}: email outside example.com/org/net ({match.group(0)})")
    for match in URL_RE.finditer(text):
        if not _host_ok(match.group(1)):
            problems.append(f"{where}: URL host outside example.com/org/net ({match.group(0)})")
    return problems
