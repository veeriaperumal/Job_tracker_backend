from core.exceptions import ValidationError

def validate_mime_type(content_type: str) -> None:
    """Validate that the MIME type is strictly application/pdf."""
    if content_type.lower() != "application/pdf":
        raise ValidationError(f"Unsupported MIME type: {content_type}. Only PDF files are allowed.")
