"""Optional read-only Gmail fetch. Off unless ``vams classify --gmail`` is passed.

The OAuth client file is the user's own and must live in a gitignored path.
The only scope requested is ``gmail.readonly``. This module lists and downloads
messages. It has no send, delete, modify, trash, or label call.
"""

from __future__ import annotations

import base64
from pathlib import Path

from vams.mailio import parse_bytes

GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
DEFAULT_CLIENT = Path("secrets/gmail_client.json")
DEFAULT_TOKEN = Path("secrets/gmail_token.json")


def fetch_messages(
    client_secrets: Path,
    token_path: Path,
    query: str | None = None,
    max_results: int = 20,
) -> list[dict]:
    """Download recent messages as parsed RFC822. Requires the optional extra."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "Gmail fetch is optional and off by default. "
            'Install it with pip install -e ".[gmail]" '
            "and put your OAuth client JSON at secrets/gmail_client.json "
            "(that path is gitignored)."
        ) from exc

    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), GMAIL_SCOPES)
    if creds is None or not creds.valid:
        if creds is not None and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not client_secrets.exists():
                raise FileNotFoundError(
                    f"OAuth client file not found at {client_secrets}. "
                    "Create a desktop client in your own Google Cloud project and do not commit it."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets), GMAIL_SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json(), encoding="utf-8")

    service = build("gmail", "v1", credentials=creds, cache_discovery=False)
    listed = (
        service.users()
        .messages()
        .list(userId="me", q=query or None, maxResults=max_results)
        .execute()
    )
    messages: list[dict] = []
    for item in listed.get("messages", []):
        raw = service.users().messages().get(userId="me", id=item["id"], format="raw").execute()
        data = base64.urlsafe_b64decode(raw["raw"])
        messages.append(parse_bytes(data, source_file="gmail", message_id=item["id"]))
    return messages
