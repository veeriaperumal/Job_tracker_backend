import os
import uuid

UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")

def save_uploaded_file(file_bytes: bytes, filename: str) -> str:
    """
    Save the file bytes to the uploads directory.
    Generates a unique name to prevent naming collisions.
    Returns the absolute path to the saved file.
    """
    # Ensure uploads directory exists
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
    # Generate unique filename
    unique_filename = f"{uuid.uuid4()}_{filename}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    with open(file_path, "wb") as f:
        f.write(file_bytes)
        
    return file_path
