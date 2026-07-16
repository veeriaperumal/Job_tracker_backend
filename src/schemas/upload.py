# app/schemas/upload.py

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class UploadStatus(str, Enum):
    UPLOADING = "UPLOADING"
    VALIDATING = "VALIDATING"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class UploadResponse(BaseModel):
    """
    Response returned immediately after a successful upload.
    """

    upload_id: UUID
    filename: str
    status: UploadStatus
    message: str

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "upload_id": "5cb70db7-c66b-48f5-a624-cbc4ec09b1ef",
                "filename": "resume.pdf",
                "status": "QUEUED",
                "message": "Resume uploaded successfully."
            }
        },
    )


class UploadStatusResponse(BaseModel):
    """
    Response for upload status endpoint.
    """

    upload_id: UUID
    filename: str
    status: UploadStatus
    created_at: datetime
    updated_at: datetime

    error: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class UploadErrorResponse(BaseModel):
    """
    Standard API error response.
    """

    detail: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detail": "Unsupported file type."
            }
        }
    )