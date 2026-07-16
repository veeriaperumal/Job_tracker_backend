import os
from core.exceptions import ValidationError

def validate_extension(filename: str) -> None:
    """Validate that the file extension is strictly .pdf."""
    _, ext = os.path.splitext(filename)
    if ext.lower() != ".pdf":
        raise ValidationError(f"Unsupported file extension: {ext}. Only .pdf files are allowed.")
