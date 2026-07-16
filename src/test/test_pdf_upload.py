import pytest
import fitz
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from src.main import app
from src.core.exceptions import ValidationError
from src.services.validation.size_validator import validate_file_size
from src.services.validation.extension_validator import validate_extension
from src.services.validation.mime_validator import validate_mime_type
from src.services.parser.pdf_parser import parse_pdf_to_text
from src.services.chunking.semantic_chunker import heuristic_chunker, chunk_resume

client = TestClient(app, raise_server_exceptions=False)

def create_mock_pdf(text: str) -> bytes:
    """Helper to generate PDF bytes containing specified text using PyMuPDF."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    return doc.tobytes()

# --- Validation Tests ---

def test_validate_file_size():
    # 5MB limit check (5 * 1024 * 1024 = 5242880 bytes)
    validate_file_size(5000)  # Ok
    with pytest.raises(ValidationError):
        validate_file_size(6 * 1024 * 1024)

def test_validate_extension():
    validate_extension("resume.pdf")
    validate_extension("RESUME.PDF")
    with pytest.raises(ValidationError):
        validate_extension("resume.txt")

def test_validate_mime_type():
    validate_mime_type("application/pdf")
    validate_mime_type("APPLICATION/PDF")
    with pytest.raises(ValidationError):
        validate_mime_type("text/plain")

# --- Parser Tests ---

def test_pdf_parser():
    sample_text = "John Doe\nSoftware Engineer"
    pdf_bytes = create_mock_pdf(sample_text)
    extracted = parse_pdf_to_text(pdf_bytes)
    assert "John Doe" in extracted
    assert "Software Engineer" in extracted

def test_pdf_parser_empty():
    doc = fitz.open()
    doc.new_page() # Empty page
    pdf_bytes = doc.tobytes()
    with pytest.raises(ValidationError):
        parse_pdf_to_text(pdf_bytes)

# --- Chunker Tests ---

def test_heuristic_chunker():
    sample_resume = """
    John Doe
    Email: john@example.com
    
    P R O F E S S I O N A L  S U M M A R Y
    Experienced python developer with a focus on web APIs.
    
    P R O F E S S I O N A L  E X P E R I E N C E
    - Senior Engineer at Apple, 2022-present
    - Software Engineer at Google, 2020-2022
    
    E D U C A T I O N
    BS in Computer Science, Stanford University
    
    T E C H N I C A L  S K I L L S
    Python, FastAPI, Docker, Kubernetes
    """
    chunks = heuristic_chunker(sample_resume)
    
    # Assert correct routing to sections
    assert len(chunks["Summary"]) > 0
    assert any("Apple" in c or "Google" in c for c in chunks["Experience"])
    assert any("Stanford" in c for c in chunks["Education"])
    assert any("Python" in c for c in chunks["Skills"])

@pytest.mark.asyncio
@patch("src.services.chunking.semantic_chunker.acompletion")
async def test_chunk_resume_llm_success(mock_acompletion):
    # Mock LLM success response
    mock_response = AsyncMock()
    mock_response.choices = [
        AsyncMock(message=AsyncMock(content='{"Summary": ["Experienced dev"], "Experience": ["Worked at Google"]}', role="assistant"))
    ]
    mock_acompletion.return_value = mock_response

    result = await chunk_resume("Dummy text")
    assert result["Summary"] == ["Experienced dev"]
    assert result["Experience"] == ["Worked at Google"]
    assert result["Skills"] == []  # Defaults to empty list if absent

@pytest.mark.asyncio
@patch("src.services.chunking.semantic_chunker.acompletion", side_effect=Exception("API Error"))
async def test_chunk_resume_llm_failure_fallback(mock_acompletion):
    # If LLM fails, it should seamlessly fallback to heuristic
    sample_resume = """
    Summary
    Passionate developer.
    """
    result = await chunk_resume(sample_resume)
    assert "Passionate developer" in result["Summary"][0]

# --- API Integration Tests ---

@patch("src.api.upload_pdf.save_uploaded_file")
@patch("src.services.chunking.semantic_chunker.chunk_resume", new_callable=AsyncMock)
def test_api_upload_success(mock_chunk_resume, mock_save_uploaded_file):
    mock_save_uploaded_file.return_value = "uploads/mocked_resume.pdf"
    mock_chunk_resume.return_value = {
        "Summary": ["Passionate developer"],
        "Skills": ["Python"],
        "Experience": [],
        "Education": [],
        "Projects": [],
        "Certificates": [],
        "Achievements": [],
        "Languages": [],
        "Contact": ["john@example.com"],
        "Links": []
    }
    
    pdf_bytes = create_mock_pdf("Summary\nPassionate developer\nSkills\nPython\nContact\njohn@example.com")
    
    response = client.post(
        "/api/v1/pdf/upload",
        files={"file": ("resume.pdf", pdf_bytes, "application/pdf")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "resume.pdf"
    assert data["saved_path"] == "uploads/mocked_resume.pdf"
    assert data["chunks"]["Summary"] == ["Passionate developer"]
    assert "Python" in data["chunks"]["Skills"]

def test_api_upload_invalid_mime():
    response = client.post(
        "/api/v1/pdf/upload",
        files={"file": ("resume.pdf", b"plain text content", "text/plain")}
    )
    assert response.status_code == 422
    assert "MIME type" in response.json()["message"]

def test_api_upload_too_large():
    # 6MB dummy content
    large_content = b"a" * (6 * 1024 * 1024)
    response = client.post(
        "/api/v1/pdf/upload",
        files={"file": ("large.pdf", large_content, "application/pdf")}
    )
    assert response.status_code == 422
    assert "exceeds" in response.json()["message"]
