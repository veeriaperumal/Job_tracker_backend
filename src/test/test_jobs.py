from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from src.main import app
from src.services.jobs.job_service import (
    generate_queries,
    normalize_results,
    remove_duplicates
)

client = TestClient(app, raise_server_exceptions=False)

def test_generate_queries_without_profile():
    queries = generate_queries("Python Developer", None)
    assert "site:linkedin.com/jobs" in queries["linkedin"]
    assert '"Python Developer"' in queries["linkedin"]

def test_generate_queries_with_profile():
    profile = {
        "skills": ["Django", "FastAPI"],
        "experience_level": "Senior",
        "technologies": ["PostgreSQL"],
        "role_keywords": ["Backend Engineer"]
    }
    queries = generate_queries("Python Developer", profile)
    assert "site:linkedin.com/jobs" in queries["linkedin"]
    assert '"Python Developer"' in queries["linkedin"]

def test_normalize_results():
    raw = {
        "linkedin": [
            {"title": "Python Developer at Google", "link": "https://linkedin.com/jobs/1", "snippet": "Write code.", "source": "linkedin"},
            {"title": "Software Engineer", "company": "Apple", "link": "https://linkedin.com/jobs/2", "snippet": "Design systems.", "source": "linkedin"}
        ]
    }
    normalized = normalize_results(raw)
    assert len(normalized) == 2
    assert normalized[0]["company"] == "Google"
    assert normalized[0]["title"] == "Python Developer"
    assert normalized[1]["company"] == "Apple"

def test_remove_duplicates():
    jobs = [
        {"title": "Python Developer", "company": "Google", "link": "https://linkedin.com/jobs/1", "snippet": "snippet"},
        # Exact duplicate link
        {"title": "Python Developer", "company": "Google", "link": "https://linkedin.com/jobs/1?ref=some", "snippet": "snippet"},
        # Duplicate title and company
        {"title": "Python Developer", "company": "Google", "link": "https://indeed.com/jobs/1", "snippet": "snippet"},
        # Unique
        {"title": "Senior Python Developer", "company": "Google", "link": "https://linkedin.com/jobs/2", "snippet": "snippet"}
    ]
    unique = remove_duplicates(jobs)
    assert len(unique) == 2
    assert unique[0]["link"] == "https://linkedin.com/jobs/1"
    assert unique[1]["title"] == "Senior Python Developer"

@patch("src.api.jobs.get_latest_resume")
@patch("src.api.jobs.extract_resume_profile")
@patch("src.api.jobs.search_yahoo")
@patch("src.api.jobs.rank_jobs")
async def test_api_jobs_search(mock_rank, mock_search, mock_extract, mock_get_resume):
    # Setup mocks
    mock_get_resume.return_value = ("my_resume.pdf", "Mock resume text")
    mock_extract.return_value = {
        "skills": ["Python", "FastAPI"],
        "experience_level": "Mid",
        "technologies": [],
        "role_keywords": []
    }
    
    # 5 platforms, return mock result for each
    mock_search.side_effect = lambda q, platform: [
        {"title": f"Dev at {platform.capitalize()}", "link": f"https://{platform}.com/job", "snippet": "Write Python code", "source": platform}
    ]
    
    mock_rank.side_effect = lambda jobs, kw, prof: [
        {**job, "match_score": 90, "match_reason": "Matches your resume profile."} for job in jobs
    ]
    
    response = client.post(
        "/api/v1/jobs/search",
        json={"keyword": "Python Developer", "use_resume": True}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["keyword"] == "Python Developer"
    assert data["resume_used"] == "my_resume.pdf"
    assert len(data["jobs"]) == 5
    assert data["jobs"][0]["match_score"] == 90
