"""Async HTTP client wrapping the Recall AI REST API."""

from __future__ import annotations

import asyncio
import logging

import httpx

from ..config import settings
from .models import (
    BotResponse,
    CreateBotRequest,
    OutputAudioRequest,
    SendChatMessageRequest,
)

logger = logging.getLogger(__name__)

_RETRY_STATUS = {507}
_MAX_RETRIES = 5
_RETRY_DELAY = 30


class RecallClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self._api_key = api_key or settings.recall_api_key
        self._base_url = base_url or settings.recall_base_url
        self._client: httpx.AsyncClient | None = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={"Authorization": f"Token {self._api_key}"},
                timeout=httpx.Timeout(30.0),
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _request(self, method: str, path: str, *, json: dict | None = None) -> httpx.Response:
        client = await self._ensure_client()
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = await client.request(method, path, json=json)
                if resp.status_code in _RETRY_STATUS:
                    logger.warning("Recall 507, retry %d/%d", attempt, _MAX_RETRIES)
                    await asyncio.sleep(_RETRY_DELAY)
                    continue
                resp.raise_for_status()
                return resp
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in _RETRY_STATUS:
                    last_exc = exc
                    await asyncio.sleep(_RETRY_DELAY)
                    continue
                logger.error("Recall API %d: %s", exc.response.status_code, exc.response.text[:500])
                raise
            except httpx.HTTPError as exc:
                last_exc = exc
                await asyncio.sleep(3)
        raise RuntimeError(f"Recall API failed after {_MAX_RETRIES} retries") from last_exc

    async def create_bot(self, request: CreateBotRequest) -> BotResponse:
        resp = await self._request("POST", "/api/v1/bot/", json=request.model_dump(exclude_none=True))
        return BotResponse.model_validate(resp.json())

    async def get_bot(self, bot_id: str) -> BotResponse:
        resp = await self._request("GET", f"/api/v1/bot/{bot_id}/")
        return BotResponse.model_validate(resp.json())

    async def send_audio(self, bot_id: str, audio_b64: str) -> None:
        payload = OutputAudioRequest(b64_data=audio_b64)
        await self._request("POST", f"/api/v1/bot/{bot_id}/output_audio/", json=payload.model_dump())

    async def send_chat_message(self, bot_id: str, message: str) -> None:
        payload = SendChatMessageRequest(message=message)
        await self._request("POST", f"/api/v1/bot/{bot_id}/send_chat_message/", json=payload.model_dump())

    async def pause_recording(self, bot_id: str) -> None:
        await self._request("POST", f"/api/v1/bot/{bot_id}/pause_recording/")

    async def get_chat_messages(self, bot_id: str) -> list[dict]:
        resp = await self._request("GET", f"/api/v1/bot/{bot_id}/chat_messages/")
        return resp.json()

    async def leave_call(self, bot_id: str) -> None:
        await self._request("POST", f"/api/v1/bot/{bot_id}/leave_call/")
