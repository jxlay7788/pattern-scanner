"""Minimal Telegram Bot API client — no SDK dependency, just HTTP."""

import requests

API_BASE = "https://api.telegram.org/bot{token}/sendMessage"
MAX_MESSAGE_LEN = 4000  # Telegram's hard cap is 4096; leave headroom for HTML tags


def _split_message(text: str, limit: int = MAX_MESSAGE_LEN) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


def send_message(token: str, chat_id: str, text: str, parse_mode: str = "HTML") -> list[dict]:
    """Sends text to a Telegram chat, splitting into multiple messages if needed.

    Returns the list of API responses (one per chunk).
    """
    url = API_BASE.format(token=token)
    responses = []
    for chunk in _split_message(text):
        resp = requests.post(
            url,
            data={"chat_id": chat_id, "text": chunk, "parse_mode": parse_mode},
            timeout=15,
        )
        resp.raise_for_status()
        body = resp.json()
        if not body.get("ok"):
            raise RuntimeError(f"Telegram API error: {body}")
        responses.append(body)
    return responses
