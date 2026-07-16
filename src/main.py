import os
# Force override HF_ENDPOINT if it is pointing to the unreachable mirror hf-mirror.com
# This must happen at the very first line of import to prevent huggingface_hub from caching it.
if "hf-mirror.com" in os.environ.get("HF_ENDPOINT", ""):
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"

from contextlib import asynccontextmanager

from fastapi import FastAPI
from core.response import JSONResponse

from core.exceptions import EXCEPTION_HANDLERS
from config.logger import setup_logging
from config.logger import get_logger

import litellm
from src.services.pinecone.pinecone_service import pinecone_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    # Configure litellm to drop unsupported parameters (like response_format for gemini)
    litellm.drop_params = True
    try:
        pinecone_service.reset_database()
    except Exception as e:
        # We catch exceptions so that startup doesn't crash completely if pinecone keys are missing or invalid
        # or we can let it fail if the user prefers, but standard practice is logging and warning.
        # Let's log it.
        
        get_logger(__name__).error(f"Failed to reset Pinecone database: {e}")
    yield


from src.api.chat import router as chat_router
from src.api.upload_pdf import router as pdf_router

app = FastAPI(lifespan=lifespan,
              title= "JOB SEARCH AI")

for exc_type_or_code, handler in EXCEPTION_HANDLERS.items():
    app.add_exception_handler(exc_type_or_code, handler)



@app.get('/health')
def check_health():
    return JSONResponse(status_code=200,content={"message":"running success!!"})


app.include_router(chat_router, prefix="/api/v1")
app.include_router(pdf_router, prefix="/api/v1")
