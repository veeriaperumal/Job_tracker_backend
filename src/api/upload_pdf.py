from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from typing import Dict, List

from src.services.validation.size_validator import validate_file_size
from src.services.validation.extension_validator import validate_extension
from src.services.validation.mime_validator import validate_mime_type
from src.services.parser.parser_router import parse_file
from src.services.chunking.semantic_chunker import chunk_resume
from src.services.uploads.storage_service import save_uploaded_file
from src.services.pinecone.pinecone_service import pinecone_service
from src.core.response import JSONResponse
from config.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/pdf", tags=["pdf"])


class ChunkResponse(BaseModel):
    filename: str
    saved_path: str
    size_bytes: int
    chunks: Dict[str, List[str]]

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    logger.info(f"Received upload request for file: {file.filename}")
    
    # 1. Read bytes and validate size
    contents = await file.read()
    size_bytes = len(contents)
    
    # Run validations
    validate_file_size(size_bytes)
    validate_extension(file.filename)
    validate_mime_type(file.content_type)
    
    # Save file to uploads directory
    saved_path = save_uploaded_file(contents, file.filename)
    logger.info(f"Saved uploaded PDF to: {saved_path}")
    
    # 2. Parse PDF content
    logger.info(f"Parsing PDF text for: {file.filename}")
    text = parse_file(contents, file.filename)
    
    # 3. Perform Section-aware Semantic Chunking
    logger.info(f"Chunking resume content for: {file.filename}")
    chunks = await chunk_resume(text)
    
    # 4. Embed and store chunks in Pinecone
    try:
        pinecone_service.upsert_chunks(file.filename, chunks)
    except Exception as e:
        logger.error(f"Failed to upsert chunks to Pinecone: {e}")
    
    return ChunkResponse(
        filename=file.filename,
        saved_path=saved_path,
        size_bytes=size_bytes,
        chunks=chunks
    )
