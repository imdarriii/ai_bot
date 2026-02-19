"""Google Calendar API client — create events with Meet links."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import httpx

from .config import settings

logger = logging.getLogger(__name__)

_TOKEN_FILE = Path("google_tokens.json")
_SCOPES = "https://www.googleapis.com/auth/calendar.events"
_CALENDAR_API = "https://www.googleapis.com/calendar/v3"


def get_auth_url(redirect_uri: str) -> str:
    """Build Google OAuth authorization URL."""
    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": _SCOPES,
        "access_type": "offline",
        "prompt": "consent",
    }
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    return f"https://accounts.google.com/o/oauth2/v2/auth?{qs}"


async def exchange_code(code: str, redirect_uri: str) -> dict:
    """Exchange authorization code for tokens and save them."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        resp.raise_for_status()
        tokens = resp.json()

    # Full replace — clear old account tokens
    _TOKEN_FILE.write_text(json.dumps(tokens))
    logger.info("Google Calendar connected successfully (tokens replaced)")
    return tokens


def _save_tokens(tokens: dict) -> None:
    """Save tokens to file (merge — used for token refresh only)."""
    existing = _load_tokens()
    existing.update(tokens)
    _TOKEN_FILE.write_text(json.dumps(existing))


def _load_tokens() -> dict:
    """Load tokens from file."""
    if _TOKEN_FILE.exists():
        return json.loads(_TOKEN_FILE.read_text())
    return {}


def is_connected() -> bool:
    """Check if Google Calendar is connected (has refresh token)."""
    tokens = _load_tokens()
    return bool(tokens.get("refresh_token"))


async def _get_access_token() -> str:
    """Get a valid access token, refreshing if needed."""
    tokens = _load_tokens()
    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        raise RuntimeError("Google Calendar not connected")

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "refresh_token": refresh_token,
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "grant_type": "refresh_token",
            },
        )
        resp.raise_for_status()
        new_tokens = resp.json()

    _save_tokens(new_tokens)
    return new_tokens["access_token"]


async def create_event(
    title: str,
    date_str: str,
    time_str: str,
    duration_minutes: int = 60,
    description: str = "",
    timezone: str = "Asia/Bishkek",
) -> dict:
    """Create a Google Calendar event with a Google Meet link.

    Args:
        title: Event title
        date_str: Date in YYYY-MM-DD format
        time_str: Time in HH:MM format (24h)
        duration_minutes: Duration in minutes (default 60)
        description: Optional description
        timezone: Timezone (default Asia/Bishkek)

    Returns:
        Dict with event details (link, id, etc.)
    """
    access_token = await _get_access_token()

    start_dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    end_dt = start_dt + timedelta(minutes=duration_minutes)

    event_body = {
        "summary": title,
        "description": description,
        "start": {
            "dateTime": start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "timeZone": timezone,
        },
        "end": {
            "dateTime": end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "timeZone": timezone,
        },
        "conferenceData": {
            "createRequest": {
                "requestId": f"alex-bot-{int(start_dt.timestamp())}",
                "conferenceSolutionKey": {"type": "hangoutsMeet"},
            },
        },
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_CALENDAR_API}/calendars/primary/events",
            params={"conferenceDataVersion": 1},
            headers={"Authorization": f"Bearer {access_token}"},
            json=event_body,
        )
        resp.raise_for_status()
        event = resp.json()

    meet_link = ""
    conf = event.get("conferenceData", {})
    for ep in conf.get("entryPoints", []):
        if ep.get("entryPointType") == "video":
            meet_link = ep.get("uri", "")
            break

    result = {
        "event_id": event.get("id", ""),
        "title": title,
        "start": f"{date_str} {time_str}",
        "duration": duration_minutes,
        "meet_link": meet_link,
        "calendar_link": event.get("htmlLink", ""),
    }

    logger.info("Calendar event created: %s at %s %s (Meet: %s)",
                title, date_str, time_str, meet_link)
    return result
