from typing import AsyncGenerator
from src.core.retry import chat_retry
from config.logger import get_logger
from src.core.exceptions import AIError

logger = get_logger(__name__)

async def chat_with_fallback(primary_model: str, fallback_model: str, user: str) -> AsyncGenerator[str, None]:
    try:
        logger.info(f"Attempting primary model: {primary_model}")
        async for chunk in chat_retry(primary_model, user):
            yield chunk
        return
    except Exception as e:
        logger.warning(f"Primary model {primary_model} failed. Falling back to {fallback_model}. Error: {e}")
        try:
            async for chunk in chat_retry(fallback_model, user):
                yield chunk
        except Exception as fallback_err:
            logger.error(f"Fallback model {fallback_model} failed as well. Error: {fallback_err}")
            raise AIError(f"Both primary model ({primary_model}) and fallback model ({fallback_model}) failed.", details=str(fallback_err))
