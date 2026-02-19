"""Always-running web server for Fly.io deployment.

Entry point: python -m meet_agent.server
"""

from __future__ import annotations

import asyncio
import logging

import uvicorn
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .agent import MeetAgent
from .config import settings
from . import google_calendar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("server")

app = FastAPI(title="Meet Agent v2 Server")

_agent: MeetAgent | None = None
_agent_task: asyncio.Task | None = None


class JoinRequest(BaseModel):
    meeting_url: str


_HTML = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Alex v2 — Meet Agent</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:system-ui,sans-serif;background:#0f0f0f;color:#e0e0e0;
       display:flex;justify-content:center;align-items:center;min-height:100vh}
  .card{background:#1a1a1a;border-radius:16px;padding:40px;width:100%;max-width:480px;
        box-shadow:0 8px 32px rgba(0,0,0,.4)}
  h1{font-size:24px;margin-bottom:4px}
  .badge{display:inline-block;font-size:11px;background:#2a5db8;color:#fff;
         padding:2px 8px;border-radius:6px;margin-left:8px;vertical-align:middle}
  .sub{color:#888;font-size:14px;margin-bottom:28px}
  input{width:100%;padding:14px 16px;border-radius:10px;border:1px solid #333;
        background:#111;color:#fff;font-size:15px;margin-bottom:16px;outline:none}
  input:focus{border-color:#4f8cff}
  .btn{width:100%;padding:14px;border:none;border-radius:10px;font-size:16px;
       font-weight:600;cursor:pointer;transition:.2s;display:flex;align-items:center;justify-content:center;gap:8px}
  .join{background:#4f8cff;color:#fff}.join:hover{background:#3a7ae8}
  .join:disabled{background:#333;color:#666;cursor:not-allowed}
  .join.loading{background:#2a5db8;color:#fff;cursor:wait}
  .leave{background:#ff4f4f;color:#fff;margin-top:10px}.leave:hover{background:#e83a3a}
  .status{margin-top:20px;padding:14px;border-radius:10px;background:#111;
          text-align:center;font-size:14px}
  .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px;vertical-align:middle}
  .dot.idle{background:#666}.dot.joining{background:#f0ad4e;animation:pulse 1s infinite}
  .dot.in_call{background:#5cb85c}.dot.leaving{background:#ff4f4f}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
  @keyframes spin{to{transform:rotate(360deg)}}
  .circle-spinner{width:22px;height:22px;border:3px solid rgba(255,255,255,.2);
       border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite}
  .stack{color:#555;font-size:12px;margin-top:16px;text-align:center}
  .gcal{width:100%;padding:12px;border:1px solid #333;border-radius:10px;font-size:14px;
        cursor:pointer;transition:.2s;display:flex;align-items:center;justify-content:center;gap:8px;
        margin-top:12px;background:transparent;color:#aaa}
  .gcal:hover{border-color:#4f8cff;color:#fff}
  .gcal.connected{border-color:#5cb85c;color:#5cb85c;cursor:default}
</style></head>
<body><div class="card">
  <h1>Alex <span class="badge">v2</span></h1>
  <p class="sub">Paste a Google Meet link and Alex will join as AI assistant</p>
  <input id="url" type="text" placeholder="https://meet.google.com/xxx-yyyy-zzz">
  <button class="btn join" id="joinBtn" onclick="doJoin()">Join Meeting</button>
  <button class="btn leave" id="leaveBtn" onclick="doLeave()" style="display:none">Leave Meeting</button>
  <div class="status" id="st"><span class="dot idle"></span> Idle — waiting for a link</div>
  <button class="gcal" id="gcalBtn" onclick="connectCal()">Connect Google Calendar</button>
  <div class="stack">Deepgram STT + GPT-4o-mini + ElevenLabs TTS</div>
</div>
<script>
const $ = id => document.getElementById(id);
async function doJoin(){
  const url = $('url').value.trim();
  if(!url){alert('Paste a Meet link first');return}
  $('joinBtn').disabled=true;
  $('joinBtn').classList.add('loading');
  $('joinBtn').innerHTML='<div class="circle-spinner"></div>';
  $('st').innerHTML = '<span class="dot joining"></span> Sending bot to the meeting...';
  try{
    const r = await fetch('/join',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({meeting_url:url})});
    const d = await r.json();
    if(d.error){alert(d.error);resetJoinBtn();return;}
    $('joinBtn').innerHTML='<div class="circle-spinner"></div> Waiting for admit...';
    $('leaveBtn').style.display='flex';
    poll();
  }catch(e){alert('Failed to connect');resetJoinBtn();}
}
function resetJoinBtn(){
  $('joinBtn').disabled=false;
  $('joinBtn').classList.remove('loading');
  $('joinBtn').innerHTML='Join Meeting';
}
async function doLeave(){
  $('leaveBtn').innerHTML='<div class="circle-spinner"></div> Leaving...';
  $('leaveBtn').disabled=true;
  await fetch('/leave',{method:'POST'});
  poll();
}
async function poll(){
  const r = await fetch('/status');
  const d = await r.json();
  const s = d.status||'idle';
  const labels = {
    idle:'Idle — waiting for a link',
    joining:'Bot is joining... admit it in Google Meet',
    in_call:'In the call — Alex is active',
    leaving:'Leaving the meeting...'
  };
  $('st').innerHTML = '<span class="dot '+s+'"></span> '+(labels[s]||s);
  if(s==='idle'){
    resetJoinBtn();
    $('leaveBtn').style.display='none';
    $('leaveBtn').innerHTML='Leave Meeting';
    $('leaveBtn').disabled=false;
  } else if(s==='joining'){
    $('joinBtn').disabled=true;
    $('joinBtn').classList.add('loading');
    $('joinBtn').innerHTML='<div class="circle-spinner"></div> Waiting for admit...';
    setTimeout(poll, 2000);
  } else if(s==='in_call'){
    $('joinBtn').disabled=true;
    $('joinBtn').classList.remove('loading');
    $('joinBtn').innerHTML='Alex is in the call';
    $('leaveBtn').innerHTML='Leave Meeting';
    $('leaveBtn').disabled=false;
    setTimeout(poll, 3000);
  } else {
    setTimeout(poll, 2000);
  }
}
function connectCal(){
  if($('gcalBtn').classList.contains('connected'))return;
  window.open('/auth/google','_blank','width=500,height=600');
  // Poll for connection status
  const ci=setInterval(async()=>{
    const r=await fetch('/auth/google/status');
    const d=await r.json();
    if(d.connected){
      clearInterval(ci);
      $('gcalBtn').classList.add('connected');
      $('gcalBtn').innerHTML='Google Calendar connected';
    }
  },2000);
}
// Check calendar status on load
fetch('/auth/google/status').then(r=>r.json()).then(d=>{
  if(d.connected){
    $('gcalBtn').classList.add('connected');
    $('gcalBtn').innerHTML='Google Calendar connected';
  }
});
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return _HTML


@app.get("/health")
async def health():
    return {"status": "ok"}


_join_lock = asyncio.Lock()


@app.post("/join")
async def join(req: JoinRequest):
    global _agent, _agent_task

    if _join_lock.locked():
        return {"error": "A join is already in progress", "status": "joining"}

    async with _join_lock:
        if _agent is not None:
            # If agent task is still running, don't replace it
            if _agent_task and not _agent_task.done():
                return {"error": "Bot is already starting or in a session", "status": "joining"}
            # Agent exists but task is done — clean up and proceed
            _agent = None
            _agent_task = None

        _agent = MeetAgent(req.meeting_url)
        _agent_task = asyncio.create_task(_run_agent())

    return {"status": "joining", "meeting_url": req.meeting_url}


@app.get("/status")
async def status():
    if _agent is None:
        return {"status": "idle"}
    return {"status": _agent.status, "bot_id": _agent.bot_id}


@app.post("/leave")
async def leave():
    if _agent is None:
        return {"status": "idle"}
    await _agent.leave()
    return {"status": "leaving"}


@app.websocket("/ws/audio")
async def audio_ws(websocket: WebSocket):
    if _agent is None:
        await websocket.accept()
        await websocket.close(code=1013)
        return
    await _agent._handle_audio_ws(websocket)


@app.post("/webhook/chat")
async def chat_webhook(request: Request):
    if _agent is None:
        return {"status": "no_agent"}
    body = await request.json()
    logger.info(">>> Chat webhook received: %s", str(body)[:300])
    asyncio.create_task(_agent._handle_chat_webhook(body))
    return {"status": "ok"}


@app.get("/auth/google")
async def google_auth(request: Request):
    """Redirect user to Google OAuth consent screen."""
    redirect_uri = str(request.base_url).rstrip("/") + "/auth/google/callback"
    url = google_calendar.get_auth_url(redirect_uri)
    from starlette.responses import RedirectResponse
    return RedirectResponse(url)


@app.get("/auth/google/callback")
async def google_callback(request: Request, code: str = ""):
    """Handle OAuth callback from Google."""
    if not code:
        return HTMLResponse("<h2>Error: no code received</h2>")
    redirect_uri = str(request.base_url).rstrip("/") + "/auth/google/callback"
    try:
        await google_calendar.exchange_code(code, redirect_uri)
        return HTMLResponse(
            "<h2>Google Calendar connected!</h2>"
            "<p>You can close this tab and go back to Alex.</p>"
            "<script>setTimeout(()=>window.close(),2000)</script>"
        )
    except Exception as e:
        logger.exception("Google OAuth failed")
        return HTMLResponse(f"<h2>Error: {e}</h2>")


@app.get("/auth/google/status")
async def google_auth_status():
    """Check if Google Calendar is connected."""
    return {"connected": google_calendar.is_connected()}


async def _run_agent():
    global _agent, _agent_task
    try:
        await _agent.run(start_server=False)
    except Exception:
        logger.exception("Agent session failed")
    finally:
        _agent = None
        _agent_task = None
        logger.info("Agent session ended — server continues running")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.webhook_port, log_level="info")
