"""Pydantic models matching Recall AI API schemas."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class BotStatus(StrEnum):
    JOINING_CALL = "joining_call"
    IN_WAITING_ROOM = "in_waiting_room"
    IN_CALL_NOT_RECORDING = "in_call_not_recording"
    RECORDING_PERMISSION_ALLOWED = "recording_permission_allowed"
    RECORDING_PERMISSION_DENIED = "recording_permission_denied"
    IN_CALL_RECORDING = "in_call_recording"
    CALL_ENDED = "call_ended"
    DONE = "done"
    FATAL = "fatal"


class RealTimeEndpointType(StrEnum):
    WEBHOOK = "webhook"
    WEBSOCKET = "websocket"


class AudioData(BaseModel):
    kind: str = "mp3"
    b64_data: str


class InCallRecording(BaseModel):
    data: AudioData


class AutomaticAudioOutput(BaseModel):
    in_call_recording: InCallRecording | None = None


class RealTimeEndpoint(BaseModel):
    type: RealTimeEndpointType
    url: str
    events: list[str] = Field(default_factory=list)


class RecordingConfig(BaseModel):
    video_mixed_mp4: dict[str, Any] | None = None
    audio_mixed_raw: dict[str, Any] | None = None
    audio_separate_raw: dict[str, Any] | None = None
    transcript: dict[str, Any] | None = None
    realtime_endpoints: list[RealTimeEndpoint] = Field(default_factory=list)


class TranscriptionOptions(BaseModel):
    provider: str = "default"


class CreateBotRequest(BaseModel):
    meeting_url: str
    bot_name: str = "AI Assistant"
    automatic_audio_output: AutomaticAudioOutput | None = None
    recording_config: RecordingConfig = Field(default_factory=RecordingConfig)
    transcription_options: TranscriptionOptions | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class StatusChange(BaseModel):
    code: str
    message: str | None = None
    created_at: datetime | None = None


class BotResponse(BaseModel):
    id: str
    meeting_url: dict | str | None = None
    bot_name: str | None = None
    status_changes: list[StatusChange] = Field(default_factory=list)
    recordings: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def latest_status(self) -> str | None:
        if self.status_changes:
            return self.status_changes[-1].code
        return None


class OutputAudioRequest(BaseModel):
    kind: str = "mp3"
    b64_data: str


class SendChatMessageRequest(BaseModel):
    message: str
