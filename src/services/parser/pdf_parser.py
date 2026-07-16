import fitz
from core.exceptions import ValidationError

def parse_pdf_to_text(file_bytes: bytes) -> str:
    """Extract all text content from PDF bytes using PyMuPDF (fitz)."""
    try:
        text = ""
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            if doc.page_count == 0:
                raise ValidationError("The uploaded PDF has no pages.")
            for page in doc:
                text += page.get_text() + "\n"
        
        if not text.strip():
            raise ValidationError("Could not extract any text from the PDF. It may be scanned or empty.")
            
        return text
    except Exception as e:
        if isinstance(e, ValidationError):
            raise e
        raise ValidationError(f"Error parsing PDF file: {str(e)}")
