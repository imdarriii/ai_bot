"""Pipeline: Deepgram STT → GPT-4o-mini → ElevenLabs TTS.

Modular voice pipeline using separate best-in-class services:
- Deepgram Nova-2 for streaming speech-to-text (direct WebSocket, no SDK)
- OpenAI GPT-4o-mini for language model
- ElevenLabs for natural text-to-speech

Audio input:  PCM S16LE 16kHz mono (from Recall)
Audio output: PCM S16LE 24kHz mono (from ElevenLabs, streamed)
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from datetime import date, timedelta
from typing import Callable, Awaitable
from urllib.parse import urlencode

import httpx
import websockets
from openai import AsyncOpenAI
from ddgs import DDGS

from .config import settings

logger = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are Alex — a professional AI assistant in a Google Meet call. You are serious, precise, and efficient. ALWAYS respond in ENGLISH.

You ONLY speak when addressed by name "Alex". If no one called you — stay silent. After being called, answer follow-up questions without requiring your name again.

Messages from participants are prefixed with their speaker label, e.g. "[Dariga]: question". Use this to address people BY NAME in your responses. For example: "Sure, Dariga, the capital of France is Paris." or "Good question, Dariga." Always include the speaker's name at least once in your response — it makes the conversation feel personal and natural.

When you answer:
- Be direct and concise. 1-3 sentences max. Give the answer and stop.
- Always use the speaker's name naturally in your response (e.g. "Sure, John, ..." or "John, the answer is..."). NEVER prefix your response with "[Alex]:" or any brackets.
- NEVER add pleasantries, offers of help, or filler. No "let me know", "I'm here", "feel free to ask", "happy to help". Just answer.
- NEVER ask clarifying questions. Do your best with what you heard.
- Do NOT use web_search for simple factual questions you already know (capitals, math, definitions, history, science). Just answer directly.
- ONLY use web_search when the user explicitly asks to "search", "find", "look up", or needs real-time data (news, weather, stock prices). You MUST call the tool — never say "I sent links" without calling it.
- After web_search returns, say "I sent the links to the chat, take a look."
- For scheduling/meeting requests — you MUST call the create_calendar_event function. NEVER say "meeting scheduled" or "done" without actually calling the tool first. Use ONLY dates from this table:
{date_table}
  If title is missing, use "Meeting". If time is NOT specified, ask "What time?" and wait — do NOT use a default time, do NOT call the tool yet. After the tool returns, briefly confirm the date and time.
- IMPORTANT: You MUST use the create_calendar_event tool for ANY request about scheduling, creating, or setting up meetings. Do NOT generate a response about scheduling without calling the tool. This is critical.
- You have FULL access to the meeting chat. Messages from chat are injected into context as [CHAT HISTORY UPDATE]. When someone asks about what was said in chat, who wrote what, or asks to repeat/explain something from chat — answer using that data.
- NEVER say "Recording stopped", "Recording started", or anything about recording status. That is handled by the system, not you.
- Leave only when explicitly told "leave", "goodbye", or "you can go".
"""

CHAT_SYSTEM_PROMPT = """\
Your name is Alex. You are an AI assistant in a Google Meet call.
Someone wrote you a message in the chat. Reply helpfully and concisely in ENGLISH.
If asked to search for something, use the information you have.
ALWAYS respond in ENGLISH.
Keep replies short — 2-3 sentences max.
"""

LEAVE_PHRASES = [
    "leave the call", "leave the meeting", "leave call", "leave meeting",
    "exit the call", "exit the meeting",
    "please leave the call", "please leave the meeting",
    "alex leave the call", "alex leave the meeting",
    "alex you can leave", "alex please leave",
    "alex leave", "you can leave", "please leave",
    "leave now", "go away", "disconnect",
    "алекс покинь звонок", "алекс выйди из звонка",
    "покинь звонок", "выйди из звонка", "уйди из звонка",
]

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the internet ONLY when the user explicitly asks to search/find/look up something, or when the question requires real-time data (news, stock prices, weather). Do NOT use for simple factual questions you already know.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
            },
            "required": ["query"],
        },
    },
}

CALENDAR_TOOL = {
    "type": "function",
    "function": {
        "name": "create_calendar_event",
        "description": "Create a Google Calendar event with a Google Meet link. Use when someone asks to schedule, create, or set up a meeting/event/call.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Event title (e.g. 'Team standup')"},
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                "time": {"type": "string", "description": "Start time in HH:MM 24h format (e.g. '17:00')"},
                "duration_minutes": {"type": "integer", "description": "Duration in minutes (default 60)"},
                "description": {"type": "string", "description": "Optional event description"},
            },
            "required": ["title", "date", "time"],
        },
    },
}


def _do_web_search(query: str, max_results: int = 5) -> str:
    try:
        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found."
        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            url = r.get("href", "")
            if url:
                lines.append(f"{i}. {title}\n{url}")
            else:
                lines.append(f"{i}. {title}")
        return "\n\n".join(lines)
    except Exception as e:
        logger.exception("Search error")
        return f"Search failed: {e}"


def should_leave(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in LEAVE_PHRASES)


# ── Callback types ────────────────────────────────────────────

AudioChunkCallback = Callable[[bytes], Awaitable[None]]
TurnTextCallback = Callable[[str], Awaitable[None]]
TranscriptCallback = Callable[[str, str], Awaitable[None]]  # (text, speaker)
InterruptCallback = Callable[[], Awaitable[None]]
SearchResultCallback = Callable[[str], Awaitable[None]]


class PipecatPipeline:
    """Streaming voice pipeline: Deepgram STT → GPT-4o-mini → ElevenLabs TTS."""

    @staticmethod
    def _build_date_table(today: date) -> str:
        """Build a 14-day date reference table for the LLM."""
        lines = []
        for i in range(14):
            d = today + timedelta(days=i)
            day_name = d.strftime("%A")
            if i == 0:
                lines.append(f"  TODAY ({day_name}): {d.isoformat()}")
            elif i == 1:
                lines.append(f"  TOMORROW ({day_name}): {d.isoformat()}")
            else:
                lines.append(f"  {day_name}: {d.isoformat()}")
        return "\n".join(lines)

    def __init__(self) -> None:
        # Services
        self._openai = AsyncOpenAI(api_key=settings.openai_api_key)

        self._elevenlabs_key = settings.elevenlabs_api_key
        self._voice_id = settings.elevenlabs_voice_id

        # Deepgram direct WebSocket (no SDK)
        self._dg_ws = None
        self._dg_recv_task: asyncio.Task | None = None
        self._dg_reconnect_lock = asyncio.Lock()
        self._dg_reconnecting = False

        # Conversation history for LLM
        today = date.today()
        weekday = today.strftime("%A")  # e.g. "Thursday"
        self._messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(
                today=today.isoformat(),
                weekday=weekday,
                date_table=self._build_date_table(today),
            )}
        ]

        # Generation state
        self._generating = False
        self._cancel_event = asyncio.Event()
        self._closed = False
        self._generation_started_at: float = 0.0
        self._tts_playing = False  # True when TTS audio is being sent
        self._ready_for_input = False  # True after greeting is done

        # Echo detection: track recent bot outputs
        self._recent_bot_outputs: list[tuple[float, str]] = []

        # Speaker diarization: map Deepgram speaker IDs to names
        self._speakers: dict[int, str] = {}
        self._speaker_count = 0

        # Callbacks
        self._audio_chunk_cb: AudioChunkCallback | None = None
        self._turn_text_cb: TurnTextCallback | None = None
        self._transcript_cb: TranscriptCallback | None = None
        self._interrupt_cb: InterruptCallback | None = None
        self._search_result_cb: SearchResultCallback | None = None

    # ── Callback properties ───────────────────────────────────

    @property
    def on_audio_chunk(self) -> AudioChunkCallback | None:
        return self._audio_chunk_cb

    @on_audio_chunk.setter
    def on_audio_chunk(self, cb: AudioChunkCallback | None) -> None:
        self._audio_chunk_cb = cb

    @property
    def on_turn_text(self) -> TurnTextCallback | None:
        return self._turn_text_cb

    @on_turn_text.setter
    def on_turn_text(self, cb: TurnTextCallback | None) -> None:
        self._turn_text_cb = cb

    @property
    def on_transcript(self) -> TranscriptCallback | None:
        return self._transcript_cb

    @on_transcript.setter
    def on_transcript(self, cb: TranscriptCallback | None) -> None:
        self._transcript_cb = cb

    @property
    def on_interrupted(self) -> InterruptCallback | None:
        return self._interrupt_cb

    @on_interrupted.setter
    def on_interrupted(self, cb: InterruptCallback | None) -> None:
        self._interrupt_cb = cb

    @property
    def on_search_result(self) -> SearchResultCallback | None:
        return self._search_result_cb

    @on_search_result.setter
    def on_search_result(self, cb: SearchResultCallback | None) -> None:
        self._search_result_cb = cb

    # ── Connect / Close ───────────────────────────────────────

    async def connect(self) -> None:
        """Start Deepgram live transcription via direct WebSocket."""
        self._closed = False
        await self._start_deepgram()

    async def _start_deepgram(self) -> None:
        """Connect to Deepgram WebSocket API directly (no SDK)."""
        # Cancel old receive task first
        if self._dg_recv_task and not self._dg_recv_task.done():
            self._dg_recv_task.cancel()
            try:
                await self._dg_recv_task
            except (asyncio.CancelledError, Exception):
                pass
            self._dg_recv_task = None

        # Close old WebSocket
        if self._dg_ws:
            try:
                await self._dg_ws.close()
            except Exception:
                pass
            self._dg_ws = None

        params = {
            "model": "nova-2",
            "language": "en",
            "smart_format": "true",
            "encoding": "linear16",
            "sample_rate": "16000",
            "channels": "1",
            "interim_results": "false",
            "vad_events": "true",
            "endpointing": "500",
            "diarize": "true",
            "keywords": "Alex:5,mute:3,unmute:3,leave:3,summarize:3",
            "punctuate": "true",
        }
        url = f"wss://api.deepgram.com/v1/listen?{urlencode(params)}"
        headers = {"Authorization": f"Token {settings.deepgram_api_key}"}

        try:
            self._dg_ws = await websockets.connect(
                url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10,
            )
            logger.info("Deepgram STT connected (nova-2, 16kHz, direct WebSocket)")
            # Start background receiver
            self._dg_recv_task = asyncio.create_task(self._dg_receive_loop())
        except Exception:
            logger.exception("Deepgram WebSocket connection failed")
            self._dg_ws = None

    async def _dg_receive_loop(self) -> None:
        """Background task: receive and process Deepgram messages."""
        try:
            async for raw_msg in self._dg_ws:
                if self._closed:
                    break
                try:
                    msg = json.loads(raw_msg)
                    msg_type = msg.get("type", "")

                    if msg_type == "Results":
                        await self._handle_dg_result(msg)
                    elif msg_type == "SpeechStarted":
                        await self._on_dg_speech_started()
                    elif msg_type == "Error":
                        logger.error("Deepgram error: %s", msg)
                except Exception:
                    logger.exception("Error processing Deepgram message")
        except asyncio.CancelledError:
            logger.debug("Deepgram receive loop cancelled")
            return
        except websockets.ConnectionClosed:
            logger.warning("Deepgram WebSocket closed")
        except Exception:
            logger.exception("Deepgram receive loop error")

        # Trigger reconnect (but don't do it inline — use the lock-protected method)
        if not self._closed:
            asyncio.create_task(self._reconnect_deepgram())

    async def _handle_dg_result(self, msg: dict) -> None:
        """Process a Deepgram transcription result."""
        try:
            channel = msg.get("channel", {})
            alternatives = channel.get("alternatives", [])
            if not alternatives:
                return
            alt = alternatives[0]
            transcript = alt.get("transcript", "").strip()
            if not transcript:
                return

            # Only final results
            is_final = msg.get("is_final", False)
            if not is_final:
                return

            # Extract speaker from diarization
            speaker_name = "Unknown"
            words = alt.get("words", [])
            if words and "speaker" in words[0] and words[0]["speaker"] is not None:
                speaker_id = words[0]["speaker"]
                if speaker_id not in self._speakers:
                    self._speaker_count += 1
                    self._speakers[speaker_id] = f"Speaker {self._speaker_count}"
                speaker_name = self._speakers[speaker_id]

            # Echo detection
            if self._is_echo(transcript):
                logger.info("Echo suppressed: %s", transcript[:80])
                return

            logger.info("Heard [%s]: %s", speaker_name, transcript)
            if self._transcript_cb:
                asyncio.create_task(self._transcript_cb(transcript, speaker_name))
        except Exception:
            logger.exception("Error handling Deepgram transcript")

    async def _on_dg_speech_started(self) -> None:
        """User started speaking — interrupt current generation if TTS is playing."""
        if not self._generating:
            return

        elapsed = time.time() - self._generation_started_at
        # During LLM generation phase (no audio playing yet), ignore speech events
        # — these are likely leftover speech or background noise
        if not self._tts_playing and elapsed < 4.0:
            logger.debug("Speech during LLM wait (%.1fs) — ignored", elapsed)
            return

        logger.info("Speech detected — interrupting generation (%.1fs, tts=%s)", elapsed, self._tts_playing)
        self._cancel_event.set()
        if self._interrupt_cb:
            asyncio.create_task(self._interrupt_cb())

    async def close(self) -> None:
        """Shut down all connections."""
        self._closed = True
        self._cancel_event.set()
        if self._dg_recv_task:
            self._dg_recv_task.cancel()
        if self._dg_ws:
            try:
                await self._dg_ws.close()
            except Exception:
                pass
        logger.info("Pipeline closed")

    # ── Audio input ───────────────────────────────────────────

    async def send_audio(self, pcm_bytes: bytes) -> None:
        """Forward PCM 16kHz audio to Deepgram for transcription."""
        if not self._ready_for_input:
            return
        if self._dg_reconnecting or self._closed:
            return  # Drop audio during reconnect — it's transient
        if self._dg_ws:
            try:
                await self._dg_ws.send(pcm_bytes)
            except Exception:
                logger.warning("Deepgram send failed — scheduling reconnect")
                asyncio.create_task(self._reconnect_deepgram())

    async def _reconnect_deepgram(self) -> None:
        """Reconnect Deepgram with lock to prevent concurrent reconnects."""
        if self._closed:
            return
        if self._dg_reconnecting:
            return  # Another reconnect is already in progress
        async with self._dg_reconnect_lock:
            if self._closed:
                return
            self._dg_reconnecting = True
            try:
                logger.info("Deepgram reconnecting...")
                await asyncio.sleep(1)
                await self._start_deepgram()
                logger.info("Deepgram reconnected successfully")
            except Exception:
                logger.exception("Deepgram reconnect failed")
            finally:
                self._dg_reconnecting = False

    # ── LLM + TTS generation ─────────────────────────────────

    def set_speaker_name(self, speaker_id: int, name: str) -> None:
        """Map a Deepgram speaker ID to a real name."""
        self._speakers[speaker_id] = name
        logger.info("Speaker %d → %s", speaker_id, name)

    async def generate_response(self, user_text: str, speaker: str = "Unknown") -> None:
        """Generate LLM response and stream TTS audio.

        Flow: user_text → GPT-4o-mini (streaming) → sentence buffer →
              ElevenLabs TTS (parallel per sentence) → audio chunks sent immediately.

        Each sentence is TTS'd and sent to Recall independently for low latency.
        The first sentence starts playing while LLM is still generating the rest.
        """
        if self._generating:
            # Cancel current generation to handle the new question
            logger.info("New question while generating — cancelling previous response")
            self._cancel_event.set()
            for _ in range(30):  # wait up to 3s for cleanup
                await asyncio.sleep(0.1)
                if not self._generating:
                    break
            if self._generating:
                logger.warning("Previous generation stuck — forcing reset")
                self._generating = False
                self._tts_playing = False

        # Include speaker context so LLM knows WHO is asking
        if speaker and speaker != "Unknown":
            content = f"[{speaker}]: {user_text}"
        else:
            content = user_text
        self._messages.append({"role": "user", "content": content})
        self._cancel_event.clear()
        self._generating = True
        self._generation_started_at = time.time()
        self._tts_playing = False

        try:
            # Stream from LLM
            stream = await self._openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=self._messages,
                stream=True,
                tools=[WEB_SEARCH_TOOL, CALENDAR_TOOL],
            )

            full_text = ""
            sentence_buffer = ""
            tool_calls: list[dict] = []
            sentence_num = 0
            tts_queue: asyncio.Queue[str | None] = asyncio.Queue()

            async def tts_worker():
                while True:
                    sentence = await tts_queue.get()
                    if sentence is None:
                        break
                    if self._cancel_event.is_set():
                        continue
                    await self._tts_generate(sentence)
                    if self._turn_text_cb and not self._cancel_event.is_set():
                        await self._turn_text_cb(sentence)

            worker_task = asyncio.create_task(tts_worker())

            async for chunk in stream:
                if self._cancel_event.is_set():
                    logger.info("Generation cancelled")
                    break

                choice = chunk.choices[0]
                delta = choice.delta

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        while tc.index >= len(tool_calls):
                            tool_calls.append({"id": "", "name": "", "arguments": ""})
                        if tc.id:
                            tool_calls[tc.index]["id"] = tc.id
                        if tc.function and tc.function.name:
                            tool_calls[tc.index]["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            tool_calls[tc.index]["arguments"] += tc.function.arguments
                    continue

                delta_content = delta.content or ""
                if not delta_content:
                    continue

                full_text += delta_content
                sentence_buffer += delta_content

                if any(sentence_buffer.rstrip().endswith(p) for p in ".!?\n"):
                    text_to_speak = sentence_buffer.strip()
                    sentence_buffer = ""
                    if text_to_speak:
                        sentence_num += 1
                        logger.info("Sentence %d → TTS: %s", sentence_num, text_to_speak[:60])
                        await tts_queue.put(text_to_speak)

            # Handle tool calls (web search, calendar)
            if tool_calls and not self._cancel_event.is_set():
                logger.info("Tool calls detected: %s", [tc["name"] for tc in tool_calls])
                await tts_queue.put(None)
                await worker_task
                # Say "searching" before the actual search so user isn't left in silence
                await self._tts_generate("Let me look that up.")
                if self._turn_text_cb:
                    await self._turn_text_cb("Let me look that up.")
                await self._handle_tool_calls(tool_calls, full_text)
                return
            elif not tool_calls:
                logger.info("No tool calls in response (text only): %s", full_text[:100])

            # Flush remaining text
            if sentence_buffer.strip() and not self._cancel_event.is_set():
                await tts_queue.put(sentence_buffer.strip())

            await tts_queue.put(None)
            await worker_task

            if full_text and not self._cancel_event.is_set():
                self._messages.append({"role": "assistant", "content": full_text})
                self._recent_bot_outputs.append((time.time(), full_text))
                if len(self._recent_bot_outputs) > 10:
                    self._recent_bot_outputs = self._recent_bot_outputs[-10:]

            if len(self._messages) > 42:
                self._messages = [self._messages[0]] + self._messages[-40:]

        except Exception:
            logger.exception("Generation error")
        finally:
            self._generating = False
            self._tts_playing = False

    async def _handle_tool_calls(self, tool_calls: list[dict], partial_text: str) -> None:
        """Execute tool calls and continue generation with results."""
        tc_messages = []
        for tc in tool_calls:
            tc_messages.append({
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments"]},
            })

        self._messages.append({
            "role": "assistant",
            "content": partial_text or None,
            "tool_calls": tc_messages,
        })

        for tc in tool_calls:
            if tc["name"] == "web_search":
                try:
                    args = json.loads(tc["arguments"])
                    query = args.get("query", "")
                    logger.info("Searching: %s", query)
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, _do_web_search, query
                    )
                    if self._search_result_cb and result:
                        asyncio.create_task(self._search_result_cb(result))
                except Exception:
                    result = "Search failed"
                    logger.exception("Search error")
            elif tc["name"] == "create_calendar_event":
                try:
                    from . import google_calendar
                    args = json.loads(tc["arguments"])
                    logger.info("Creating calendar event: %s", args)
                    if not google_calendar.is_connected():
                        result = "Google Calendar is not connected. Ask the user to connect it on the web interface first."
                    else:
                        event = await google_calendar.create_event(
                            title=args.get("title", "Meeting"),
                            date_str=args.get("date", ""),
                            time_str=args.get("time", ""),
                            duration_minutes=args.get("duration_minutes", 60),
                            description=args.get("description", ""),
                        )
                        meet_link = event.get("meet_link", "")
                        cal_link = event.get("calendar_link", "")
                        result = f"Event created successfully. Title: {event['title']}, Start: {event['start']}, Duration: {event['duration']} min."
                        if meet_link:
                            result += f" Meet link: {meet_link}"
                        # Send event details to chat
                        if self._search_result_cb:
                            chat_msg = f"Meeting scheduled: {event['title']} on {event['start']}"
                            if meet_link:
                                chat_msg += f"\nMeet: {meet_link}"
                            if cal_link:
                                chat_msg += f"\nCalendar: {cal_link}"
                            asyncio.create_task(self._search_result_cb(chat_msg))
                except Exception:
                    result = "Failed to create calendar event"
                    logger.exception("Calendar event error")
            else:
                result = f"Unknown function: {tc['name']}"

            self._messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

        if not self._cancel_event.is_set():
            stream = await self._openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=self._messages,
                stream=True,
            )

            full_text = ""
            sentence_buffer = ""

            async for chunk in stream:
                if self._cancel_event.is_set():
                    break
                content = chunk.choices[0].delta.content or ""
                if content:
                    full_text += content
                    sentence_buffer += content
                    if any(sentence_buffer.rstrip().endswith(p) for p in ".!?\n"):
                        await self._tts_generate(sentence_buffer.strip())
                        sentence_buffer = ""

            if sentence_buffer.strip() and not self._cancel_event.is_set():
                await self._tts_generate(sentence_buffer.strip())

            if full_text and not self._cancel_event.is_set():
                self._messages.append({"role": "assistant", "content": full_text})
                self._recent_bot_outputs.append((time.time(), full_text))
                if self._turn_text_cb:
                    await self._turn_text_cb(full_text)

    async def send_confirmation(self, text: str) -> None:
        """Send a short confirmation directly to TTS (skips LLM for speed).

        Confirmations are NOT interruptible — they always play fully.
        """
        if self._generating:
            self._cancel_event.set()
            await asyncio.sleep(0.1)

        self._cancel_event.clear()
        self._generating = True
        self._generation_started_at = time.time()  # Reset cooldown so interrupts are blocked
        self._tts_playing = False
        try:
            await self._tts_generate(text)
            self._recent_bot_outputs.append((time.time(), text))
            if self._turn_text_cb:
                await self._turn_text_cb(text)
        except Exception:
            logger.exception("Confirmation TTS error")
        finally:
            self._generating = False
            self._tts_playing = False

    async def start_listening(self) -> None:
        """Connect Deepgram (call AFTER bot admitted, before greeting)."""
        if not self._dg_ws:
            await self._start_deepgram()

    async def send_greeting(self) -> None:
        """Greet the meeting. No audio is processed until greeting finishes."""
        greeting = "Just a heads up, this meeting is being recorded. Hi everyone, I'm Alex!"
        await self.send_confirmation(greeting)
        self._ready_for_input = True
        logger.info("Greeting done — now accepting audio input")

    async def cancel_response(self) -> None:
        """Cancel current generation."""
        self._cancel_event.set()

    # ── ElevenLabs TTS ────────────────────────────────────────

    async def _tts_generate(self, text: str) -> None:
        """Stream TTS audio from ElevenLabs REST API."""
        if not text.strip() or self._cancel_event.is_set():
            return

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self._voice_id}/stream"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream(
                    "POST",
                    url,
                    headers={
                        "xi-api-key": self._elevenlabs_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "text": text,
                        "model_id": "eleven_turbo_v2_5",
                        "voice_settings": {
                            "stability": 0.5,
                            "similarity_boost": 0.8,
                        },
                    },
                    params={"output_format": "mp3_44100_64"},
                ) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        logger.error("ElevenLabs error %d: %s", response.status_code, body[:200])
                        return

                    async for chunk in response.aiter_bytes(chunk_size=4096):
                        if self._cancel_event.is_set():
                            break
                        if self._audio_chunk_cb and chunk:
                            self._tts_playing = True
                            await self._audio_chunk_cb(chunk)
        except Exception:
            logger.exception("ElevenLabs TTS error")

    # ── Echo detection ────────────────────────────────────────

    # Stop words excluded from echo comparison to avoid false positives
    _STOP_WORDS = frozenset(
        "the a an is are was were of in on at to for and or but i you it he she "
        "they we my your his her that this what about how do does did can could "
        "would should will shall".split()
    )

    def _is_echo(self, transcript: str) -> bool:
        """Check if transcript matches a recent bot output (echo).

        Uses content-word comparison (stop words removed) to avoid false
        positives like "The capital of France" matching "Tokyo is the capital
        of Japan" due to shared function words.
        """
        now = time.time()
        # Shorter window: 15s (was 30s) — echo happens within seconds
        self._recent_bot_outputs = [
            (t, text) for t, text in self._recent_bot_outputs if now - t < 15
        ]
        if not self._recent_bot_outputs:
            return False

        t_lower = transcript.lower().strip()
        if len(t_lower) < 5:
            return False
        t_words = set(t_lower.split()) - self._STOP_WORDS
        if not t_words:
            return False

        for _, bot_text in self._recent_bot_outputs:
            b_words = set(bot_text.lower().strip().split()) - self._STOP_WORDS
            if not b_words:
                continue
            overlap = len(t_words & b_words)
            smaller = min(len(t_words), len(b_words))
            if smaller > 0 and overlap / smaller > 0.5:
                return True
        return False

    # ── Chat reply (text only) ────────────────────────────────

    async def chat_reply(self, message: str) -> str:
        """Text-only reply via OpenAI Chat API."""
        try:
            resp = await self._openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                    {"role": "user", "content": message},
                ],
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.error("Chat API error: %s", e)
            return "Sorry, I couldn't process that message."

    async def inject_chat_context(self, chat_history: list[dict]) -> None:
        """Inject chat history into LLM context."""
        if not chat_history:
            return
        lines = []
        for msg in chat_history[-30:]:
            sender = msg.get("sender", "Unknown")
            text = msg.get("text", "")
            lines.append(f"{sender}: {text}")
        summary = "\n".join(lines)
        self._messages.append({
            "role": "user",
            "content": f"[CHAT HISTORY UPDATE]\n{summary}",
        })
        logger.info("Chat context injected (%d messages)", len(chat_history))
