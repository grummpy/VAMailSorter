"""Read .eml files and mbox exports. Nothing here writes to a mailbox."""

from __future__ import annotations

import email
import email.policy
import mailbox
import re
from email.message import EmailMessage
from html import unescape
from pathlib import Path

HTML_TAG_RE = re.compile(r"(?is)<(script|style)\b.*?>.*?</\1>|<[^>]+>")
WS_RE = re.compile(r"\s+")


def strip_html(html: str) -> str:
    text = HTML_TAG_RE.sub(" ", html)
    text = unescape(text)
    return WS_RE.sub(" ", text).strip()


def _body(msg: EmailMessage) -> str:
    if msg.is_multipart():
        plain: list[str] = []
        html: list[str] = []
        for part in msg.walk():
            disposition = str(part.get("Content-Disposition") or "")
            if "attachment" in disposition.lower():
                continue
            content_type = part.get_content_type()
            if content_type not in {"text/plain", "text/html"}:
                continue
            try:
                content = part.get_content()
            except Exception:
                payload = part.get_payload(decode=True) or b""
                content = payload.decode("utf-8", errors="replace")
            if not isinstance(content, str):
                continue
            if content_type == "text/plain":
                plain.append(content)
            else:
                html.append(content)
        if plain:
            return "\n".join(plain).strip()
        if html:
            return strip_html("\n".join(html))
        return ""
    try:
        content = msg.get_content()
    except Exception:
        payload = msg.get_payload(decode=True) or b""
        content = payload.decode("utf-8", errors="replace")
    if not isinstance(content, str):
        return ""
    if msg.get_content_type() == "text/html":
        return strip_html(content)
    return content.strip()


def parse_bytes(data: bytes, source_file: str, message_id: str) -> dict:
    msg = email.message_from_bytes(data, policy=email.policy.default)
    return {
        "source_file": source_file,
        "message_id": message_id,
        "subject": str(msg.get("subject") or ""),
        "from": str(msg.get("from") or ""),
        "to": str(msg.get("to") or ""),
        "date": str(msg.get("date") or ""),
        "body": _body(msg),
    }


def parse_eml(path: Path) -> dict:
    return parse_bytes(path.read_bytes(), source_file=str(path), message_id=path.name)


def _looks_like_mbox(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.readline().startswith(b"From ")
    except OSError:
        return False


def parse_mbox(path: Path) -> list[dict]:
    box = mailbox.mbox(path)
    messages = []
    try:
        for index, message in enumerate(box):
            messages.append(
                parse_bytes(
                    message.as_bytes(),
                    source_file=str(path),
                    message_id=f"{path.name}#{index}",
                )
            )
    finally:
        box.close()
    return messages


def load_path(path: Path) -> list[dict]:
    """Load one .eml, one mbox, or every mail file under a directory."""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        files = sorted(
            item
            for item in path.rglob("*")
            if item.is_file()
            and not item.name.startswith(".")
            and item.suffix.lower() in {".eml", ".mbox", ".mbx"}
        )
        messages: list[dict] = []
        for item in files:
            if item.suffix.lower() == ".eml":
                messages.append(parse_eml(item))
            else:
                messages.extend(parse_mbox(item))
        return messages
    suffix = path.suffix.lower()
    if suffix == ".eml":
        return [parse_eml(path)]
    if suffix in {".mbox", ".mbx"} or _looks_like_mbox(path):
        return parse_mbox(path)
    raise ValueError(f"Unsupported mail file {path}. Use a .eml file, an mbox, or a directory.")
