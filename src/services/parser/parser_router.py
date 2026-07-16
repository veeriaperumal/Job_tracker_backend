from src.services.parser.pdf_parser import parse_pdf_to_text

def parse_file(file_bytes: bytes, filename: str) -> str:
    """Route parsing based on file. Currently strictly PDF."""
    return parse_pdf_to_text(file_bytes)
