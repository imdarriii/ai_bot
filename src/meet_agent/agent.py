"""Main agent: Recall (WebSocket audio in) → Pipeline (Deepgram+GPT+11Labs) → Recall REST API audio out.

Same business logic as v1 but with modular pipeline:
- Deepgram Nova-2 for STT (faster, more accurate than Whisper)
- GPT-4o-mini for LLM (Chat API instead of Realtime)
- ElevenLabs for TTS (natural voice instead of OpenAI alloy)
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import subprocess
import sys
import time
import httpx
import uvicorn
from fastapi import FastAPI, WebSocket, Request

from .config import settings
from .pipeline import PipecatPipeline, should_leave
from .transport.client import RecallClient
from .transport.models import (
    AudioData,
    AutomaticAudioOutput,
    CreateBotRequest,
    InCallRecording,
    RealTimeEndpoint,
    RealTimeEndpointType,
    RecordingConfig,
)

# Silent MP3 (0.1s) — required for output_audio endpoint
_SILENT_MP3_B64 = "SUQzBAAAAAAAIlRTU0UAAAAOAAADTGF2ZjYxLjcuMTAwAAAAAAAAAAAAAAD/80TEAAAAA0gAAAAATEFNRTMuMTAwVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVMQU1FMy7/80TEUwAAA0gAAAAAMTAwVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVMQU1FMy7/80TEpgAAA0gAAAAAMTAwVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVMQU1FMy7/80TErAAAA0gAAAAAMTAwVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVMQU1FMy7/80TErAAAA0gAAAAAMTAwVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVX/80TErAAAA0gAAAAAVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVX/80TErAAAA0gAAAAAVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVU="

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agent")

try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG_PATH = "ffmpeg"

# ── Filler filter (same as v1) ────────────────────────────────

_FILLER_EXACT = [
    "got it", "no worries", "take your time", "i'm here",
    "happy to help", "sure thing", "alright", "okay",
    "no problem", "understood", "noted", "sounds good",
    "of course", "sure", "right", "absolutely",
]

_FILLER_STARTS = [
    "i will remain silent", "i'll remain silent", "i'll stay silent",
    "i'm here whenever", "i'm here if", "i'm here to help",
    "i'm here—", "i'm here -", "i'm here,", "i'm still here",
    "take your time", "no worries",
    "sure, just let me know", "sure, let me know",
    "sure. please share", "sure. let me know",
    "whenever you're ready", "feel free to",
    "let me know if there's anything", "let me know if you need",
    "let me know if anything", "let me know how",
    "let me know what you", "just let me know",
    "don't hesitate",
    "hi there", "hello!", "hey there", "hey!",
    "hi! please let me know", "hello. what do you need",
    "how can i help", "how can i assist",
    "if you need any", "if you have any",
    "if there's anything", "if anything comes up",
    "got it, let me know",
    "recording started", "recording stopped",
    "understood. let's", "noted. proceed",
    "understood. i'll stay", "understood. i'll remain",
    "i'm ready to proceed", "i'll stay silent", "i'll remain silent",
]

_FILLER_CONTAINS = [
    "let me know if you need", "let me know if there's anything",
    "let me know what you", "let me know how i can",
    "feel free to ask", "i'm here to help", "i'm here whenever",
    "i'm still here", "don't hesitate to ask",
    "how can i help", "how can i assist",
    "if you need anything", "if you need any more",
    "whenever you're ready", "take your time",
    "just let me know",
]


_CONFIRMATIONS = {
    "i'm here", "i'm here, listening", "i'm back, listening",
    "okay, i'm muted", "recording stopped", "yes, i am here",
    "i'm here, listening.", "i'm back, listening.",
}


def _is_filler(text: str) -> bool:
    lower = text.lower().strip().rstrip("!.")
    # Never filter bot confirmations
    if lower in _CONFIRMATIONS:
        return False
    if "sent" in lower and "chat" in lower:
        return False
    if "check the chat" in lower:
        return False
    if lower in _FILLER_EXACT:
        return True
    for pattern in _FILLER_STARTS:
        if lower.startswith(pattern):
            return True
    if len(lower) < 100:
        for pattern in _FILLER_CONTAINS:
            if pattern in lower:
                return True
    return False


def _pcm_to_mp3_b64(pcm: bytes, sample_rate: int = 24000) -> str:
    proc = subprocess.run(
        [
            FFMPEG_PATH,
            "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
            "-codec:a", "libmp3lame", "-b:a", "48k", "-f", "mp3", "pipe:1",
        ],
        input=pcm,
        capture_output=True,
        timeout=15,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {proc.stderr.decode()[:200]}")
    return base64.b64encode(proc.stdout).decode("ascii")


# Phantom transcripts — short noise
_PHANTOM_WORDS = {
    "you", "oh", "ah", "um", "hmm", "hm", "uh", "eh",
    "thank you", "thanks", "okay", "ok", "yes", "no",
    "bye", "hi", "hello", "hey", "right", "sure",
    "good evening", "good morning", "good night",
    "yeah", "yep", "nope", "wow", "huh", "mhm",
    "so", "well", "like", "just", "but", "and",
    "please", "sorry", "excuse me", "pardon",
}


class MeetAgent:
    """AI agent that joins Google Meet, listens via audio, and responds."""

    def __init__(self, meet_url: str) -> None:
        self.meet_url = meet_url
        self.recall = RecallClient()
        self.pipeline = PipecatPipeline()

        self.bot_id: str | None = None
        self._leave_event = asyncio.Event()
        self._active_ws_id: int = 0

        # Audio output buffer — accumulate PCM chunks from TTS
        self._audio_buffer = bytearray()

        self._is_talking = False
        self._response_queue: asyncio.Queue = asyncio.Queue()
        self._response_worker_task: asyncio.Task | None = None

        # Wake word tracking (bot name: Alex)
        self._alex_active = False
        self._last_alex_time: float = 0.0

        # Mute
        self._muted = False
        self._mute_after_response = False

        # Recording state
        self._recording = True

        # Command lock: blocks LLM responses, only confirmations pass
        self._confirmation_only = False

        self._admitted = False
        self._leaving = False

        # Chat
        self._seen_chat_ids: set[str] = set()
        self._chat_history: list[dict] = []

        # Speaker name mapping: Deepgram speaker_id → real name from Recall
        self._participants: dict[str, str] = {}  # participant_id → name
        self._recent_recall_transcripts: list[tuple[float, str, str]] = []  # (time, name, text)

        # Meeting transcript for summary
        self._meeting_transcript: list[dict] = []  # {"speaker": ..., "text": ..., "time": ...}
        self._meeting_start_time: float = 0.0

        self._app = FastAPI(title="Meet Agent v2")
        self._setup_routes()

    def _setup_routes(self) -> None:
        @self._app.websocket("/ws/audio")
        async def audio_ws(websocket: WebSocket):
            await self._handle_audio_ws(websocket)

        @self._app.post("/webhook/chat")
        async def chat_webhook(request: Request):
            body = await request.json()
            logger.info(">>> Chat webhook: %s", str(body)[:300])
            asyncio.create_task(self._handle_chat_webhook(body))
            return {"status": "ok"}

        @self._app.get("/health")
        async def health():
            return {"status": "ok"}

    # ── WebSocket: audio IN from Recall ──────────────────────

    async def _handle_audio_ws(self, websocket: WebSocket) -> None:
        await websocket.accept()
        ws_id = id(websocket)
        self._active_ws_id = ws_id
        logger.info(">>> Recall audio stream connected (ws=%d)", ws_id)

        chunks = 0
        total_bytes = 0

        try:
            while True:
                if self._active_ws_id != ws_id:
                    break

                data = await websocket.receive()

                if "text" in data and data["text"]:
                    try:
                        msg = json.loads(data["text"])
                        event = msg.get("event", "")

                        if event == "audio_mixed_raw.data":
                            b64_audio = msg["data"]["data"]["buffer"]
                            pcm = base64.b64decode(b64_audio)
                            chunks += 1
                            total_bytes += len(pcm)

                            # Send audio directly to Deepgram (16kHz, no resampling needed)
                            if self._admitted:
                                await self.pipeline.send_audio(pcm)

                            if chunks % 100 == 1:
                                logger.info(
                                    "Audio IN: %d chunks, %d KB → Deepgram",
                                    chunks, total_bytes // 1024,
                                )
                        else:
                            logger.info("WS event: %s", event or data["text"][:200])

                    except (json.JSONDecodeError, KeyError):
                        logger.info("WS text: %s", data["text"][:200])

                elif "bytes" in data and data["bytes"]:
                    chunks += 1
                    total_bytes += len(data["bytes"])
                    await self.pipeline.send_audio(data["bytes"])

        except Exception as e:
            logger.info("Audio IN WebSocket closed: %s", e)
        finally:
            logger.info("Audio IN ended (ws=%d): %d chunks, %d KB", ws_id, chunks, total_bytes // 1024)

    # ── Pipeline callbacks ───────────────────────────────────

    async def _on_audio_chunk(self, pcm_data: bytes) -> None:
        """Accumulate PCM audio chunks from ElevenLabs TTS."""
        if self._leaving:
            return
        self._audio_buffer.extend(pcm_data)

    async def _on_interrupted(self) -> None:
        """User started speaking — discard accumulated audio."""
        self._audio_buffer.clear()
        logger.info("Interrupted — audio buffer cleared")

    async def _on_turn_text(self, text: str) -> None:
        """Response complete — queue for sequential processing."""
        if self._leaving:
            self._audio_buffer.clear()
            return

        # Command lock: block LLM responses, only allow short confirmations
        if self._confirmation_only:
            if text and len(text) > 40:
                logger.info("Blocked (command active, not confirmation): %s", text[:80])
                self._audio_buffer.clear()
                return
            # This is the confirmation — let it through, reset flag
            self._confirmation_only = False

        pcm = bytes(self._audio_buffer)
        self._audio_buffer.clear()

        # Filter filler
        if text and _is_filler(text):
            logger.info("Filtered filler: %s", text[:80])
            return

        await self._response_queue.put((text, pcm))

    async def _response_worker(self) -> None:
        """Process responses one at a time."""
        while not self._leaving:
            try:
                text, pcm = await asyncio.wait_for(self._response_queue.get(), timeout=1)
            except asyncio.TimeoutError:
                continue

            if self._leaving:
                break

            self._is_talking = True
            try:
                if self.bot_id and pcm:
                    try:
                        # ElevenLabs already outputs MP3 — just base64 encode
                        mp3_b64 = base64.b64encode(pcm).decode("ascii")
                        tasks = [self.recall.send_audio(self.bot_id, mp3_b64)]
                        if text:
                            tasks.append(self.recall.send_chat_message(self.bot_id, text))
                        await asyncio.gather(*tasks)
                        # Wait for audio to finish playing before sending next
                        # MP3 at 64kbps: duration = size_bytes * 8 / 64000
                        duration = len(pcm) * 8 / 64000
                        await asyncio.sleep(duration)
                        logger.info("Audio sent: %d KB MP3 (%.1fs)", len(pcm) // 1024, duration)
                        if text:
                            logger.info("Chat sent: %s", text[:80])
                            self._chat_history.append({"sender": "Alex (bot)", "text": text})
                            # Add bot response to meeting transcript for summary
                            elapsed = time.time() - self._meeting_start_time if self._meeting_start_time else 0
                            self._meeting_transcript.append({
                                "speaker": "Alex (bot)",
                                "text": text,
                                "time": f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}",
                            })
                    except Exception:
                        logger.exception("Failed to send audio/chat")
                elif self.bot_id and text:
                    try:
                        await self.recall.send_chat_message(self.bot_id, text)
                        logger.info("Chat sent: %s", text[:80])
                    except Exception:
                        logger.exception("Failed to send chat message")
            finally:
                self._is_talking = False
                if self._mute_after_response:
                    self._mute_after_response = False
                    self._muted = True
                    self._alex_active = False
                    logger.info(">>> Bot MUTED (after confirmation sent)")
                else:
                    logger.info("Unmuted — listening again")

    # ── Chat handling ─────────────────────────────────────────

    async def _handle_chat_webhook(self, body: dict) -> None:
        try:
            event = body.get("event", "")

            # Track participant join/leave
            if event == "participant_events.join":
                p = body.get("data", {}).get("data", {}).get("participant", {})
                pid = str(p.get("id", ""))
                name = p.get("name", "")
                if pid and name:
                    self._participants[pid] = name
                    logger.info("Participant joined: %s (id=%s)", name, pid)
                return

            if event == "participant_events.leave":
                p = body.get("data", {}).get("data", {}).get("participant", {})
                pid = str(p.get("id", ""))
                name = self._participants.pop(pid, "Unknown")
                logger.info("Participant left: %s (id=%s)", name, pid)
                return

            # Recall transcript — use for speaker name mapping
            if event == "transcript.data":
                t_data = body.get("data", {}).get("data", {})
                speaker = t_data.get("speaker", {})
                speaker_name = speaker.get("name", "")
                text = t_data.get("words", "")
                if isinstance(text, list):
                    text = " ".join(w.get("text", "") for w in text)
                text = text.strip()
                if speaker_name and text:
                    self._recent_recall_transcripts.append((time.time(), speaker_name, text))
                    # Keep only last 20, max 30 seconds old
                    now = time.time()
                    self._recent_recall_transcripts = [
                        (t, n, tx) for t, n, tx in self._recent_recall_transcripts
                        if now - t < 30
                    ][-20:]
                    logger.info("Recall transcript [%s]: %s", speaker_name, text[:80])
                return

            if event != "participant_events.chat_message":
                return
            outer_data = body.get("data", {})
            inner_data = outer_data.get("data", {})
            participant = inner_data.get("participant", {})
            sender_name = participant.get("name", "Unknown")
            chat_content = inner_data.get("data", {})
            text = chat_content.get("text", "")
            if not text:
                return
            await self._reply_to_chat(sender_name, text)
        except Exception:
            logger.exception("Failed to handle chat webhook")

    async def _reply_to_chat(self, sender_name: str, text: str) -> None:
        if self._leaving:
            return
        dedup_key = f"{sender_name}:{text}"
        if dedup_key in self._seen_chat_ids:
            return
        self._seen_chat_ids.add(dedup_key)

        logger.info("Chat from %s: %s", sender_name, text[:100])
        self._chat_history.append({"sender": sender_name, "text": text})

        try:
            reply = await self.pipeline.chat_reply(text)
            logger.info("Chat reply: %s", reply[:100])
            if self.bot_id and reply:
                formatted = f"Re: {text[:80]} — {reply}"
                await self.recall.send_chat_message(self.bot_id, formatted)
                self._chat_history.append({"sender": "Alex (bot)", "text": reply})
        except Exception:
            logger.exception("Failed to reply to chat")

        await self.pipeline.inject_chat_context(self._chat_history)

    async def _poll_chat_messages(self) -> None:
        logger.info("Chat polling started (every 3s)")
        first_poll = True
        while not self._leaving and self.bot_id:
            try:
                raw = await self.recall.get_chat_messages(self.bot_id)
                messages = raw.get("results", []) if isinstance(raw, dict) else raw

                if first_poll:
                    for msg in messages:
                        msg_id = str(msg.get("id", id(msg)))
                        self._seen_chat_ids.add(msg_id)
                        sender = msg.get("sender", msg.get("participant", {}))
                        sender_name = sender.get("name", "Unknown")
                        text = msg.get("text", msg.get("message", ""))
                        if text:
                            self._seen_chat_ids.add(f"{sender_name}:{text}")
                    first_poll = False
                else:
                    for msg in messages:
                        msg_id = str(msg.get("id", id(msg)))
                        if msg_id in self._seen_chat_ids:
                            continue
                        self._seen_chat_ids.add(msg_id)
                        sender = msg.get("sender", msg.get("participant", {}))
                        sender_name = sender.get("name", "Unknown")
                        text = msg.get("text", msg.get("message", ""))
                        is_bot = sender.get("is_bot", False)
                        if is_bot or not text:
                            continue
                        self._chat_history.append({"sender": sender_name, "text": text})
                        asyncio.create_task(self._reply_to_chat(sender_name, text))
            except Exception:
                if first_poll:
                    logger.warning("Chat poll failed — disabled")
                    return

            await asyncio.sleep(3)

    # ── Meeting summary ─────────────────────────────────────

    _SUMMARY_PROMPT = """\
You are a meeting assistant. Analyze the meeting transcript below and create a structured summary.

IMPORTANT RULES:
- Write in ENGLISH
- Be concise and specific
- Only include sections that have actual content (skip empty sections)
- For Q&A, include the actual question AND the answer given
- For action items, include WHO is responsible if mentioned

Format:

📋 MEETING SUMMARY

🕐 Duration: {duration}
👥 Participants: {participants}

📌 Topics Discussed:
• [topic 1]
• [topic 2]

❓ Questions & Answers:
• Q (speaker name): [question]
  A: [answer given]

✅ Decisions Made:
• [decision]

📝 Action Items:
• [who]: [what they need to do]

💡 Key Takeaways:
• [important point]

---
TRANSCRIPT:
{transcript}
"""

    async def _generate_and_send_summary(self) -> None:
        """Generate meeting summary from collected transcripts and send to chat."""
        if not self._meeting_transcript:
            if self.bot_id:
                await self.recall.send_chat_message(self.bot_id, "No transcript data to summarize yet.")
            return

        # Build transcript text
        lines = []
        for entry in self._meeting_transcript:
            lines.append(f"[{entry['time']}] {entry['speaker']}: {entry['text']}")
        transcript_text = "\n".join(lines)

        # Calculate duration
        elapsed = time.time() - self._meeting_start_time if self._meeting_start_time else 0
        duration = f"{int(elapsed // 60)} min {int(elapsed % 60)} sec"

        # Get participant names
        speakers = list(set(e["speaker"] for e in self._meeting_transcript))
        participants = ", ".join(speakers) if speakers else "Unknown"

        prompt = self._SUMMARY_PROMPT.format(
            duration=duration,
            participants=participants,
            transcript=transcript_text,
        )

        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=settings.openai_api_key)
            resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional meeting summarizer."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2000,
            )
            summary = resp.choices[0].message.content

            if self.bot_id and summary:
                # Send to chat (split if too long for one message)
                if len(summary) > 4000:
                    parts = [summary[i:i+3900] for i in range(0, len(summary), 3900)]
                    for i, part in enumerate(parts):
                        await self.recall.send_chat_message(self.bot_id, part)
                        if i < len(parts) - 1:
                            await asyncio.sleep(0.5)
                else:
                    await self.recall.send_chat_message(self.bot_id, summary)

                logger.info("Meeting summary sent to chat (%d chars)", len(summary))

                # Also speak a short version
                await self.pipeline.send_confirmation(
                    "I've posted the meeting summary in the chat. Check it out!"
                )
        except Exception:
            logger.exception("Failed to generate summary")
            if self.bot_id:
                await self.recall.send_chat_message(self.bot_id, "Sorry, failed to generate summary.")

    async def _on_search_result(self, results: str) -> None:
        if self._leaving:
            return
        if self.bot_id and results:
            try:
                await self.recall.send_chat_message(self.bot_id, results)
                self._chat_history.append({"sender": "Alex (bot)", "text": f"[Search results]\n{results}"})
            except Exception:
                logger.exception("Failed to send search results")

    # ── Speaker name resolution ──────────────────────────────

    def _resolve_speaker_name(self, dg_speaker: str, text: str) -> str:
        """Try to match Deepgram speaker label to a real name using Recall transcripts."""
        # If pipeline already has a real name mapped, use it
        if dg_speaker and not dg_speaker.startswith("Speaker "):
            return dg_speaker

        # Try to match by text similarity with recent Recall transcripts
        text_lower = text.lower().strip()
        text_words = set(text_lower.split())
        if len(text_words) < 2:
            return dg_speaker

        best_name = None
        best_overlap = 0.0
        now = time.time()

        for t, name, recall_text in self._recent_recall_transcripts:
            if now - t > 15:  # only match within 15 seconds
                continue
            r_words = set(recall_text.lower().strip().split())
            if not r_words:
                continue
            overlap = len(text_words & r_words) / min(len(text_words), len(r_words))
            if overlap > best_overlap and overlap > 0.5:
                best_overlap = overlap
                best_name = name

        if best_name:
            # Save mapping in pipeline for future use
            logger.info("Speaker resolved: %s → %s (%.0f%% match)", dg_speaker, best_name, best_overlap * 100)
            # Extract speaker number and update pipeline mapping
            try:
                num = int(dg_speaker.replace("Speaker ", "")) - 1
                self.pipeline.set_speaker_name(num, best_name)
            except (ValueError, IndexError):
                pass
            self._backfill_speaker_name(dg_speaker, best_name)
            return best_name

        # Fallback: if only 1 human participant, use their name
        non_bot_names = [n for n in self._participants.values() if n.lower() != settings.bot_name.lower()]
        if len(non_bot_names) == 1:
            name = non_bot_names[0]
            try:
                num = int(dg_speaker.replace("Speaker ", "")) - 1
                self.pipeline.set_speaker_name(num, name)
            except (ValueError, IndexError):
                pass
            self._backfill_speaker_name(dg_speaker, name)
            return name

        return dg_speaker

    def _backfill_speaker_name(self, old_label: str, real_name: str) -> None:
        """Update old transcript entries with the resolved real name."""
        for entry in self._meeting_transcript:
            if entry["speaker"] == old_label:
                entry["speaker"] = real_name

    # ── Voice command processing ──────────────────────────────

    async def _on_heard(self, text: str, speaker: str = "Unknown") -> None:
        """Called when Deepgram transcribes what someone said.

        Key difference from v1: here we DECIDE whether to generate a response
        (in v1, OpenAI Realtime generated responses automatically and we filtered output).
        Speaker comes from Deepgram diarization — identifies WHO is speaking.
        """
        speaker = self._resolve_speaker_name(speaker, text)
        logger.info("[%s]: %s", speaker, text)
        lower = text.lower().strip().rstrip(".!,")

        # Ignore phantom/noise transcripts
        if lower in _PHANTOM_WORDS:
            logger.info("Phantom ignored: %s", text)
            return

        # ── Collect transcript for meeting summary ──
        elapsed = time.time() - self._meeting_start_time if self._meeting_start_time else 0
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        self._meeting_transcript.append({
            "speaker": speaker,
            "text": text,
            "time": f"{minutes:02d}:{seconds:02d}",
        })

        # ── SYSTEM COMMANDS — always work, even when muted ──

        # Summarize command
        _summary_phrases = [
            "summarize", "summary", "summarise", "sum up",
            "give me a summary", "meeting summary",
            "саммари", "подведи итоги", "итоги встречи",
            "подытожь", "резюме", "резюмируй",
        ]
        if any(p in lower for p in _summary_phrases):
            logger.info("Summary requested!")
            await self.pipeline.cancel_response()
            self._confirmation_only = True
            self._alex_active = True
            self._last_alex_time = time.time()
            await self.pipeline.send_confirmation("Let me prepare the meeting summary.")
            asyncio.create_task(self._generate_and_send_summary())
            return

        # Leave the call
        if should_leave(text):
            logger.info("Leave command detected!")
            await self.pipeline.cancel_response()
            self._audio_buffer.clear()
            self._is_talking = False
            # Send farewell to chat before leaving
            if self.bot_id:
                try:
                    await self.recall.send_chat_message(
                        self.bot_id,
                        "Okay, I'm leaving. Recording is stopped. Goodbye!",
                    )
                except Exception:
                    pass
            self._leaving = True
            self._leave_event.set()
            return


        # ── MUTE/UNMUTE ──────────────────────────────────────

        _unmute_phrases = ["unmute", "un mute", "on mute", "размьють", "разммьють",
                           "включи микрофон", "unmute yourself", "un-mute",
                           "alex unmute", "алекс размьють", "you mute", "unm ute"]
        if any(p in lower for p in _unmute_phrases):
            if self._muted:
                self._muted = False
                self._mute_after_response = False
                self._alex_active = True
                self._last_alex_time = time.time()
                logger.info(">>> Bot UNMUTED + active")
                await self.pipeline.cancel_response()
                self._confirmation_only = True  # block any in-flight LLM response
                await self.pipeline.send_confirmation("I'm back, listening.")
            return

        _mute_phrases = ["mute", "замьють", "замьютьс", "выключи микрофон",
                         "mute yourself", "alex mute", "алекс замьють"]
        if any(p in lower for p in _mute_phrases):
            await self.pipeline.cancel_response()
            self._audio_buffer.clear()  # discard any in-flight LLM audio
            while not self._response_queue.empty():
                try:
                    self._response_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            self._alex_active = True
            self._last_alex_time = time.time()
            self._confirmation_only = True  # block any in-flight LLM response
            self._mute_after_response = True
            logger.info(">>> Bot MUTING — confirmation first")
            await self.pipeline.send_confirmation("Okay, I'm muted.")
            return

        # If muted — but "Alex" was said, auto-unmute
        if self._muted:
            _wake_words_muted = ["alex", "алекс", "aleks", "аlex"]
            if any(w in lower for w in _wake_words_muted):
                self._muted = False
                self._alex_active = True
                self._last_alex_time = time.time()
                # Check if there's a question — answer it, not just "I'm here"
                clean = lower
                for w in _wake_words_muted:
                    clean = clean.replace(w, "").strip()
                clean = clean.strip(",.!? ")
                if clean and len(clean) >= 3:
                    logger.info(">>> Alex called while muted — UNMUTED + answering: %s", text[:80])
                    asyncio.create_task(self.pipeline.generate_response(text, speaker=speaker))
                else:
                    logger.info(">>> Alex called while muted — AUTO-UNMUTED")
                    await self.pipeline.send_confirmation("I'm here.")
            else:
                logger.info("Muted — ignored: %s", text[:80])
            return

        # ── WAKE WORD LOGIC ──────────────────────────────────

        _wake_words = [
            "alex", "алекс", "aleks", "аlex", "alex,",
            "alec", "alexa", "aleksa", "al ex", "al-ex",
            "alix", "alex's", "alexs", "oleks",
        ]
        if any(w in lower for w in _wake_words):
            self._alex_active = True
            self._last_alex_time = time.time()
            logger.info(">>> ALEX called — bot is active")

        # Deactivate after 2 minutes
        if self._alex_active and (time.time() - self._last_alex_time > 120):
            self._alex_active = False
            logger.info("Alex timeout (2 min) — bot goes silent")

        # "That's all" / "thanks Alex" = done
        if self._alex_active and any(p in lower for p in [
            "that's all", "that is all", "thanks alex", "thank you alex",
            "спасибо алекс", "это всё", "всё спасибо",
        ]):
            self._alex_active = False
            logger.info(">>> Conversation ended — bot goes silent")
            return

        # ── GENERATE RESPONSE ─────────────────────────────────
        has_wake = any(w in lower for w in _wake_words)

        if has_wake:
            self._alex_active = True
            self._last_alex_time = time.time()
            # Strip wake word to get actual content
            clean = lower
            for w in _wake_words:
                clean = clean.replace(w, "").strip()
            clean = clean.strip(",.!? ")
            if not clean or len(clean) < 3:
                # Just called by name — confirm presence
                logger.info("Wake word only — confirming presence")
                await self.pipeline.send_confirmation("I'm here, listening.")
                return
            # Directly addressed — always respond
            logger.info("Generating response for [%s]: %s", speaker, text[:80])
            asyncio.create_task(self.pipeline.generate_response(text, speaker=speaker))

        elif self._alex_active:
            # Follow-up window — only respond to questions, not statements
            is_followup = (
                "?" in text
                or len(lower.strip()) < 30  # short answers like "6PM", "yes", "at 5"
                or lower.lstrip().startswith((
                    "what", "who", "where", "when", "why", "how",
                    "is it", "is there", "is that", "are there", "are you",
                    "do you", "does", "can you", "could you",
                    "will ", "would ", "should ",
                    "tell me", "explain", "describe",
                    "search", "find", "look up",
                    "and what", "and how", "and where", "and who",
                    "at ", "for ", "on ", "schedule", "create",
                ))
            )
            if is_followup:
                self._last_alex_time = time.time()
                logger.info("Follow-up question for [%s]: %s", speaker, text[:80])
                asyncio.create_task(self.pipeline.generate_response(text, speaker=speaker))
            else:
                logger.info("Statement during active window — silent: %s", text[:80])

    # ── External control ─────────────────────────────────────

    @property
    def status(self) -> str:
        if not self.bot_id:
            return "idle"
        if self._leaving:
            return "leaving"
        if self._admitted:
            return "in_call"
        return "joining"

    async def leave(self) -> None:
        if not self._leaving and self.bot_id:
            self._leaving = True
            self._audio_buffer.clear()
            self._is_talking = False
            await self.pipeline.cancel_response()
            self._leave_event.set()

    # ── Main lifecycle ────────────────────────────────────────

    async def run(self, *, start_server: bool = True) -> None:
        tunnel = None
        server_task = None
        try:
            # 1. Public URLs
            _use_ngrok = not settings.webhook_base_url or settings.webhook_base_url == "https://localhost"
            if _use_ngrok:
                from pyngrok import ngrok
                logger.info("Starting ngrok on port %d ...", settings.webhook_port)
                tunnel = ngrok.connect(settings.webhook_port, "http")
                public_url = tunnel.public_url.replace("http://", "https://")
                logger.info("ngrok: %s", public_url)
            else:
                public_url = settings.webhook_base_url.rstrip("/")
                logger.info("Using preset URL: %s", public_url)
            ws_url = public_url.replace("https://", "wss://")

            # 2. Start HTTP/WS server
            if start_server:
                server = uvicorn.Server(uvicorn.Config(
                    self._app, host="0.0.0.0", port=settings.webhook_port, log_level="warning",
                ))
                server_task = asyncio.create_task(server.serve())
                await asyncio.sleep(1)

            # 3. Set pipeline callbacks (Deepgram connects later, after admission)
            self.pipeline.on_audio_chunk = self._on_audio_chunk
            self.pipeline.on_turn_text = self._on_turn_text
            self.pipeline.on_transcript = self._on_heard
            self.pipeline.on_interrupted = self._on_interrupted
            self.pipeline.on_search_result = self._on_search_result

            # 4. Create Recall bot
            request = CreateBotRequest(
                meeting_url=self.meet_url,
                bot_name=settings.bot_name,
                automatic_audio_output=AutomaticAudioOutput(
                    in_call_recording=InCallRecording(
                        data=AudioData(kind="mp3", b64_data=_SILENT_MP3_B64),
                    ),
                ),
                recording_config=RecordingConfig(
                    video_mixed_mp4={},
                    audio_mixed_raw={},
                    realtime_endpoints=[
                        RealTimeEndpoint(
                            type=RealTimeEndpointType.WEBSOCKET,
                            url=f"{ws_url}/ws/audio",
                            events=["audio_mixed_raw.data"],
                        ),
                        RealTimeEndpoint(
                            type=RealTimeEndpointType.WEBHOOK,
                            url=f"{public_url}/webhook/chat",
                            events=[
                                "participant_events.chat_message",
                                "participant_events.join",
                                "participant_events.leave",
                            ],
                        ),
                    ],
                ),
            )
            logger.info("Creating bot for %s ...", self.meet_url)
            bot = await self.recall.create_bot(request)
            self.bot_id = bot.id
            logger.info("Bot created: %s", bot.id)

            print()
            print("=" * 50)
            print("  AI AGENT V2 (ALEX) IS JOINING THE MEETING")
            print("  Deepgram STT + GPT-4o-mini + ElevenLabs TTS")
            print("  1. Go to Meet and ADMIT the bot")
            print("  2. Say 'Alex' to activate the bot")
            print("  Say 'leave the call' to end the session.")
            print("=" * 50)
            print()

            # 4b. Wait for admit (max 3 min, then give up)
            logger.info("Waiting for bot to be admitted...")
            admitted = False
            for _ in range(180):
                try:
                    bot_info = await self.recall.get_bot(self.bot_id)
                    status = bot_info.latest_status or ""
                    if "in_call" in status.lower():
                        self._admitted = True
                        admitted = True
                        logger.info("Bot admitted (status=%s)", status)
                        break
                    if any(s in status.lower() for s in ["fatal", "done", "error"]):
                        logger.warning("Bot failed (status=%s) — aborting", status)
                        break
                except Exception:
                    pass
                await asyncio.sleep(1)

            if not admitted:
                logger.warning("Bot was NOT admitted after 3 min — cleaning up")
                try:
                    await self.recall.leave_call(self.bot_id)
                except Exception:
                    pass
                return

            # 4c. Connect Deepgram NOW (after admission — no idle timeouts)
            await self.pipeline.connect()

            # 4d. Start workers
            chat_poll_task = asyncio.create_task(self._poll_chat_messages())
            self._response_worker_task = asyncio.create_task(self._response_worker())

            # 4e. Greeting
            self._is_talking = True
            self._alex_active = True
            self._last_alex_time = time.time()
            self._meeting_start_time = time.time()
            await asyncio.sleep(3)
            await self.pipeline.send_greeting()
            await asyncio.sleep(8)
            self._is_talking = False
            self._alex_active = False
            self._last_alex_time = 0.0
            logger.info("Greeting done — bot is SILENT until 'Alex' is called")

            # 5. Wait to leave
            await self._leave_event.wait()

            # 6. Leave — auto-summary before leaving
            logger.info("Leaving meeting ...")

            # Generate and send summary if there's transcript data
            if self._meeting_transcript and self.bot_id:
                try:
                    logger.info("Generating auto-summary before leaving...")
                    await self._generate_and_send_summary()
                except Exception:
                    logger.exception("Auto-summary failed")

            if self._recording and self.bot_id:
                try:
                    await self.recall.pause_recording(self.bot_id)
                    self._recording = False
                except Exception:
                    pass

            try:
                await self.recall.send_chat_message(self.bot_id, "Stopping the recording and leaving. Goodbye!")
                await asyncio.sleep(2)
            except Exception:
                pass

            chat_poll_task.cancel()
            if self._response_worker_task:
                self._response_worker_task.cancel()
            try:
                await self.recall.leave_call(self.bot_id)
            except Exception:
                pass
            if server_task:
                server_task.cancel()

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            if self.bot_id and not self._leaving:
                try:
                    await self.recall.leave_call(self.bot_id)
                except Exception:
                    pass
            await self.pipeline.close()
            await self.recall.close()
            if tunnel:
                from pyngrok import ngrok
                ngrok.disconnect(tunnel.public_url)
            logger.info("Agent stopped.")
