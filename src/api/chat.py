from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


from src.config.settings import settings
from src.core.fallback import chat_with_fallback
from config.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

class ChatRequest(BaseModel):
    message: str

@router.post("/stream")
async def stream_chat(request: ChatRequest):
    primary = settings.GROQ_MODEL or "groq/llama-3.3-70b-versatile"
    if primary and not (primary.startswith("groq/") or "/" in primary):
        primary = f"groq/{primary}"
        
    fallback = settings.GEMINI_MODEL or "gemini/gemini-1.5-flash"
    if fallback and not (fallback.startswith("gemini/") or "/" in fallback):
        fallback = f"gemini/{fallback}"
    
    logger.info(f"Received stream chat request. Primary: {primary}, Fallback: {fallback}")
    
    async def event_generator():
        try:
            async for chunk in chat_with_fallback(primary, fallback, request.message):
                yield chunk
        except Exception as e:
            logger.error(f"Error in stream_chat event_generator: {e}")
            yield f"\n[ERROR: {str(e)}]"

    return StreamingResponse(event_generator(), media_type="text/plain")
