"""Entry point: python -m meet_agent.server"""
import uvicorn
from .config import settings

if __name__ == "__main__":
    uvicorn.run("meet_agent.server:app", host="0.0.0.0", port=settings.webhook_port, log_level="info")
