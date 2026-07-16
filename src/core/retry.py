import asyncio
from typing import AsyncGenerator
from src.services.ai.llm_service import chat
from src.core.exceptions import AIError
from config.logger import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 2

async def chat_retry(model: str, user: str) -> AsyncGenerator[str, None]:
    last_exception = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"Attempt {attempt} using {model}")
            # We must iterate inside the try block to catch streaming exceptions.
            async for chunk in chat(model, user):
                yield chunk
            # Successfully finished stream, exit the retry loop.
            return
        except Exception as e:
            last_exception = e
            logger.error(f"Attempt {attempt} using {model} failed: {e}")

            if attempt < MAX_RETRIES:
                logger.info(f"Retrying after {RETRY_DELAY} seconds...")
                await asyncio.sleep(RETRY_DELAY)

    if last_exception:
        raise AIError(f"Model {model} failed after {MAX_RETRIES} retries.", details=str(last_exception))