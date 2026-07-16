from core.exceptions import ValidationError

def validate_file_size(size_bytes: int, max_size_mb: float = 5.0) -> None:
    """Validate that the file size is less than max_size_mb (default 5MB)."""
    max_bytes = int(max_size_mb * 1024 * 1024)
    if size_bytes > max_bytes:
        raise ValidationError(f"File size of {size_bytes / (1024 * 1024):.2f}MB exceeds the maximum allowed limit of {max_size_mb}MB.")
