import os
import re
import urllib.parse
import httpx
import numpy as np
import litellm
from config.logger import get_logger
from src.config.settings import settings
from src.services.parser.pdf_parser import parse_pdf_to_text
from src.services.embeddings.embedding_service import embedding_service
from src.prompts.job_prompts import RESUME_EXTRACTION_PROMPT, JOB_EXTRACTION_PROMPT

logger = get_logger(__name__)

# Search operators and targeted subpaths for Yahoo search targeting
PLATFORM_DOMAINS = {
    "linkedin": "linkedin.com/jobs",
    "indeed": "indeed.com",
    "foundit": "foundit.in",
    "naukri": "naukri.com",
    "remoteok": "remoteok.com"
}

def clean_json_content(content: str) -> str:
    """Removes codeblock markers (like ```json ... ```) from the LLM output."""
    if not content:
        return ""
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()

def get_latest_resume() -> tuple[str, str] | None:
    """
    Finds the latest uploaded PDF file in the uploads directory and returns its content and filename.
    """
    uploads_dir = os.path.join(os.getcwd(), "uploads")
    if not os.path.exists(uploads_dir):
        return None

    pdf_files = [f for f in os.listdir(uploads_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        return None

    latest_file = max(
        pdf_files,
        key=lambda f: os.path.getmtime(os.path.join(uploads_dir, f))
    )
    file_path = os.path.join(uploads_dir, latest_file)
    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        text = parse_pdf_to_text(file_bytes)
        return latest_file, text
    except Exception as e:
        logger.error(f"Error reading/parsing latest resume {latest_file}: {e}")
        return None

async def extract_resume_profile(resume_text: str) -> dict:
    """
    Uses the LLM to extract key skills, experience level, preferred roles,
    notice period, salary expectations, and location.
    """
    model = settings.GEMINI_MODEL or "gemini/gemini-1.5-flash"
    if not (model.startswith("gemini/") or "/" in model):
        model = f"gemini/{model}"

    prompt = RESUME_EXTRACTION_PROMPT.format(resume_text=resume_text)

    try:
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=800,
            response_format={"type": "json_object"},
            num_retries=0,
            timeout=5.0
        )
        content = response.choices[0].message.content
        import json
        profile = json.loads(clean_json_content(content))
        
        # Ensure default keys exist in the profile
        profile["skills"] = profile.get("skills", ["Python"])
        profile["experience_years"] = float(profile.get("experience_years", 2.0))
        profile["salary_expectation"] = str(profile.get("salary_expectation", "Negotiable"))
        profile["location"] = profile.get("location", "Chennai")
        profile["seniority"] = profile.get("seniority", "Mid")
        profile["notice_period_days"] = int(profile.get("notice_period_days", 30))
        profile["role_keywords"] = profile.get("role_keywords", ["Python Developer"])
        
        return profile
    except Exception as e:
        logger.error(f"Failed to extract resume profile using LLM: {e}")
        return {
            "skills": ["Python", "Django", "FastAPI"],
            "experience_years": 2.0,
            "salary_expectation": "Negotiable",
            "location": "Chennai",
            "seniority": "Mid",
            "notice_period_days": 30,
            "role_keywords": ["Python Developer"]
        }

def generate_queries(keyword: str, profile: dict | None) -> dict[str, str]:
    """
    Generates Yahoo Search queries for LinkedIn, Indeed, Foundit, Naukri, and RemoteOK.
    Uses Yahoo's search operators (site:domain "keyword").
    """
    queries = {}
    for platform, domain in PLATFORM_DOMAINS.items():
        queries[platform] = f'site:{domain} "{keyword}"'
    return queries

async def search_yahoo(query: str, platform: str) -> list[dict]:
    """
    Queries Yahoo Search. Parses HTML to extract organic results.
    Tries daily filter (btf=d) first. If empty, falls back to weekly (btf=w)
    and then to no time constraint, ensuring live links are always returned.
    """
    url = "https://search.yahoo.com/search"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5"
    }
    
    # Time window fallbacks: Past Week (recent) -> All Time
    time_windows = [
        {"btf": "w", "label": "past week"},
        {"label": "all time"}
    ]
    
    for window in time_windows:
        params = {"p": query}
        if "btf" in window:
            params["btf"] = window["btf"]
            
        try:
            logger.info(f"Searching Yahoo ({window['label']}) for query: {query}")
            async with httpx.AsyncClient(headers=headers, timeout=5.0, follow_redirects=True) as client:
                response = await client.get(url, params=params)
                if response.status_code == 200:
                    results = parse_yahoo_html(response.text, platform)
                    if results:
                        logger.info(f"Yahoo Scraped {len(results)} results ({window['label']}) for query: {query}")
                        return results
        except Exception as e:
            logger.error(f"Error scraping Yahoo ({window['label']}) for platform {platform}: {e}")
            
    return []

def parse_yahoo_html(html: str, platform: str) -> list[dict]:
    """
    Parses Yahoo Search HTML. Extracts links, titles, and snippets.
    Uses regex to extract the target redirect URLs and clean destination hosts.
    """
    results = []
    domain_filter = PLATFORM_DOMAINS[platform]
    
    # Locate all <a> tags containing a Yahoo redirect URL with 'RU=' param
    pattern = r'<a[^>]+href="([^"]*r\.search\.yahoo\.com[^"]*RU=([^"/]+)[^"]*)"[^>]*>(.*?)</a>'
    matches = re.findall(pattern, html, re.DOTALL)
    
    seen_urls = set()
    for redirect_url, encoded_url, title_html in matches:
        try:
            actual_url = urllib.parse.unquote(encoded_url)
            
            # Clean URL params
            if "&" in actual_url:
                actual_url = actual_url.split("&")[0]
                
            if domain_filter not in actual_url or actual_url in seen_urls:
                continue
                
            seen_urls.add(actual_url)
            
            # Clean HTML elements from title
            title = re.sub(r'<[^>]+>', '', title_html).strip()
            
            # Default Snippet
            snippet = f"Apply dynamically on {platform.capitalize()}."
            
            # Look for adjacent text in the card layout to find a snippet
            # We search for compText block close to this URL in the HTML
            escaped_url = re.escape(redirect_url)
            snippet_match = re.search(escaped_url + r'.*?<div class="compText[^"]*">(.*?)</div>', html, re.DOTALL)
            if snippet_match:
                snippet = re.sub(r'<[^>]+>', '', snippet_match.group(1)).strip()
                
            results.append({
                "title": title,
                "company": "Confidential",
                "link": actual_url,
                "snippet": snippet,
                "source": platform
            })
        except Exception as e:
            logger.error(f"Failed to parse Yahoo match: {e}")
            
    return results[:10]

def is_recent_posting(snippet: str) -> bool:
    """
    Returns True if the listing snippet indicates it was posted recently
    (e.g., within the last 24 hours / 1 day). Discards old dates (e.g. 2024, 2023, or months ago).
    """
    snippet_lower = snippet.lower()
    
    # If the snippet contains old years (e.g. 2024, 2023, 2025) or explicit multi-week/month/year indicators
    if re.search(r'\b(201\d|202[0-5])\b', snippet):
        return False
    if "month" in snippet_lower or "year" in snippet_lower:
        return False
    if "week" in snippet_lower:
        return False
    
    # If it contains days ago greater than 1
    # e.g., "2 days ago", "5 days ago", etc. (except "1 day ago" or "1d ago")
    day_match = re.search(r'(\d+)\s+day', snippet_lower)
    if day_match:
        days = int(day_match.group(1))
        if days > 1:
            return False
            
    d_ago_match = re.search(r'\b(\d+)d\s+ago\b', snippet_lower)
    if d_ago_match:
        days = int(d_ago_match.group(1))
        if days > 1:
            return False

    return True

def normalize_results(raw_results: dict[str, list[dict]]) -> list[dict]:
    """
    Normalizes all listings into a standardized schema and filters for past-24h recency.
    """
    normalized = []
    for platform, listings in raw_results.items():
        for job in listings:
            snippet = job.get("snippet", "No description available.").strip()
            
            # Recency filter: skip if job posting is older than 24h
            if not is_recent_posting(snippet):
                continue
                
            title = job.get("title", f"{platform.capitalize()} Job Listing").strip()
            company = job.get("company", "Confidential").strip()
            
            # Clean "Title at Company" layout
            if " at " in title:
                parts = title.split(" at ")
                title = parts[0].strip()
                company = parts[1].strip()
            elif " | " in title:
                parts = title.split(" | ")
                title = parts[0].strip()
                company = parts[1].strip()
                
            link = job.get("link", "").strip()
            
            normalized.append({
                "title": title,
                "company": company,
                "link": link,
                "source": platform,
                "snippet": snippet
            })
    return normalized

def remove_duplicates(jobs: list[dict]) -> list[dict]:
    """
    Deduplicates listings based on URL similarity or Title/Company combination.
    """
    seen_links = set()
    seen_keys = set()
    unique_jobs = []

    for job in jobs:
        link = job["link"].lower().strip()
        link_clean = link.rstrip("/").split("?")[0]
        
        title_clean = re.sub(r'[^a-z0-9]', '', job["title"].lower())
        company_clean = re.sub(r'[^a-z0-9]', '', job["company"].lower())
        title_company_key = f"{title_clean}_{company_clean}"
        
        if link_clean not in seen_links and title_company_key not in seen_keys:
            seen_links.add(link_clean)
            seen_keys.add(title_company_key)
            unique_jobs.append(job)
            
    return unique_jobs

def calculate_cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Calculates cosine similarity between two vector lists."""
    a = np.array(v1)
    b = np.array(v2)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))

import asyncio
gemini_semaphore = asyncio.Semaphore(3)

async def extract_job_parameters(job_title: str, job_snippet: str, model_name: str) -> dict:
    """
    Uses the LLM to extract requirements parameters from the job posting.
    """
    prompt = JOB_EXTRACTION_PROMPT.format(job_title=job_title, job_snippet=job_snippet)

    try:
        async with gemini_semaphore:
            response = await asyncio.wait_for(
                litellm.acompletion(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=300,
                    response_format={"type": "json_object"},
                    num_retries=0,
                    timeout=3.0
                ),
                timeout=4.0
            )
        content = response.choices[0].message.content
        import json
        return json.loads(clean_json_content(content))
    except Exception as e:
        logger.warning(f"Failed to extract job parameters using LLM: {e}. Using deterministic parsing fallback.")
        # Programmatic parsing fallback
        job_text = f"{job_title} {job_snippet}".lower()
        skills = []
        for s in ["python", "django", "fastapi", "postgresql", "redis", "celery", "aws", "gunicorn", "nginx"]:
            if s in job_text:
                skills.append(s.capitalize())
        
        seniority = "Mid"
        if "senior" in job_text or "sr." in job_text:
            seniority = "Senior"
        elif "lead" in job_text:
            seniority = "Lead"
        elif "junior" in job_text or "jr." in job_text or "fresher" in job_text:
            seniority = "Junior"
            
        location = "Remote" if "remote" in job_text else "India"
        return {
            "required_skills": skills,
            "experience_years_required": 3.0 if seniority == "Senior" else 1.0,
            "salary_lpa": None,
            "location": location,
            "seniority": seniority,
            "notice_period_days": None
        }

def calculate_weighted_score(candidate: dict, job: dict) -> dict:
    """
    Calculates the match score based on the weighted parameter metrics:
    - Skills: 40%
    - Experience: 25%
    - Salary: 10%
    - Location: 10%
    - Seniority: 10%
    - Notice Period: 5%
    Total = 100%
    """
    scores = {}
    
    # 1. Skills (40%)
    cand_skills = set(s.lower() for s in candidate.get("skills", []))
    job_skills = set(s.lower() for s in job.get("required_skills", []))
    if not job_skills:
        skills_score = 40.0
    else:
        overlap = cand_skills.intersection(job_skills)
        skills_score = (len(overlap) / max(len(job_skills), 1)) * 40.0
    scores["skills"] = min(skills_score, 40.0)

    # 2. Experience (25%)
    cand_exp = candidate.get("experience_years", 0.0)
    job_exp = job.get("experience_years_required", 0.0)
    if cand_exp >= job_exp:
        exp_score = 25.0
    elif job_exp > 0:
        exp_score = (cand_exp / job_exp) * 25.0
    else:
        exp_score = 25.0
    scores["experience"] = min(exp_score, 25.0)

    # 3. Salary (10%)
    scores["salary"] = 10.0

    # 4. Location (10%)
    cand_loc = candidate.get("location", "").lower()
    job_loc = job.get("location", "").lower()
    if cand_loc in job_loc or job_loc in cand_loc or "remote" in job_loc:
        loc_score = 10.0
    elif "india" in job_loc:
        loc_score = 7.0
    else:
        loc_score = 5.0
    scores["location"] = loc_score

    # 5. Seniority (10%)
    seniority_map = {"junior": 1, "mid": 2, "senior": 3, "lead": 4, "any": 2}
    cand_sen = seniority_map.get(candidate.get("seniority", "mid").lower(), 2)
    job_sen = seniority_map.get(job.get("seniority", "any").lower(), 2)
    if cand_sen >= job_sen:
        sen_score = 10.0
    else:
        sen_score = 6.0
    scores["seniority"] = sen_score

    # 6. Notice Period (5%)
    cand_notice = candidate.get("notice_period_days", 30)
    job_notice = job.get("notice_period_days")
    if job_notice is None or cand_notice <= job_notice or cand_notice == 0:
        notice_score = 5.0
    else:
        notice_score = 3.0
    scores["notice_period"] = notice_score

    scores["total_weighted"] = sum(scores.values())
    return scores

async def rank_jobs(jobs: list[dict], keyword: str, profile: dict | None) -> list[dict]:
    """
    Matching Algorithm:
    - Stage 1: Vector embedding semantic match (first-stage retrieval, fast).
    - Stage 2: Concurrently extract details and run weighted scoring on top 15 candidates (second-stage reranker, LLM).
    """
    if not jobs:
        return []

    # 1. Candidate profile setup
    if not profile:
        profile = {
            "skills": [keyword, "Python"],
            "experience_years": 2.0,
            "salary_expectation": "Negotiable",
            "location": "Remote",
            "seniority": "Mid",
            "notice_period_days": 30
        }

    # Generate Candidate Embedding
    candidate_summary = f"Role: {keyword}. Skills: {', '.join(profile['skills'])}. Experience: {profile['experience_years']} years. Location: {profile['location']}."
    cand_vector = embedding_service.embed_text(candidate_summary)

    model = settings.GEMINI_MODEL or "gemini/gemini-1.5-flash"
    if not (model.startswith("gemini/") or "/" in model):
        model = f"gemini/{model}"

    # Stage 1: Pre-filter all jobs using Cosine Similarity (Vector search)
    job_texts = [f"{job['title']} at {job['company']}. {job['snippet']}" for job in jobs]
    # Batch embed all job texts in a single call to SentenceTransformer (extremely fast)
    job_vectors = embedding_service.embed_batch(job_texts)
    
    prefiltered_jobs = []
    for job, job_vector in zip(jobs, job_vectors):
        cos_sim = calculate_cosine_similarity(cand_vector, job_vector)
        job_copy = job.copy()
        job_copy["cos_sim"] = cos_sim
        prefiltered_jobs.append(job_copy)
        
    # Sort by cosine similarity and keep top 15
    prefiltered_jobs.sort(key=lambda j: j["cos_sim"], reverse=True)
    top_jobs = prefiltered_jobs[:15]
    
    # Stage 2: Rerank the top 15 concurrently
    import asyncio

    async def process_single_job(job_item: dict) -> dict:
        cos_sim = job_item["cos_sim"]
        # Extract Job Parameters
        job_params = await extract_job_parameters(job_item["title"], job_item["snippet"], model)
        # Calculate Weighted Score
        weighted = calculate_weighted_score(profile, job_params)
        
        # Combine Cosine Similarity (30%) + Parameter Matching (70%)
        final_score = int((cos_sim * 30.0) + (weighted["total_weighted"] * 0.70))
        final_score = max(0, min(100, final_score))
        
        skills_matched = set(s.lower() for s in profile.get("skills", [])).intersection(set(s.lower() for s in job_params.get("required_skills", [])))
        reason = f"Matched {len(skills_matched)} skills ({', '.join(list(skills_matched)[:3])}). Cosine similarity: {cos_sim:.2f}."
        
        # Format salary details
        sal_lpa = job_params.get("salary_lpa")
        salary_details = f"{sal_lpa} LPA" if sal_lpa else "Not specified"

        skills_matched = set(s.lower() for s in profile.get("skills", [])).intersection(set(s.lower() for s in job_params.get("required_skills", [])))
        # Convert matched skills back to original case from profile/job
        orig_matched_skills = []
        for s in profile.get("skills", []):
            if s.lower() in skills_matched:
                orig_matched_skills.append(s)
                
        reason = f"Matched {len(orig_matched_skills)} skills ({', '.join(orig_matched_skills[:3])}). Cosine similarity: {cos_sim:.2f}."
        
        job_result = {
            "title": job_item["title"],
            "company": job_item["company"],
            "salary_details": salary_details,
            "jd": job_item["snippet"],
            "skills": job_params.get("required_skills", []),
            "match_score": final_score,
            "matched_skills": orig_matched_skills,
            "link": job_item["link"],
            "source": job_item["source"],
            "match_reason": reason,
            "details": {
                "cosine_similarity": round(cos_sim, 4),
                "skills_score": round(weighted["skills"], 2),
                "experience_score": round(weighted["experience"], 2),
                "location_score": round(weighted["location"], 2),
                "seniority_score": round(weighted["seniority"], 2),
                "notice_period_score": round(weighted["notice_period"], 2)
            }
        }
        return job_result

    # Run top 15 concurrently
    tasks = [process_single_job(job) for job in top_jobs]
    logger.info(f"Reranking top {len(top_jobs)} jobs concurrently...")
    ranked_jobs = await asyncio.gather(*tasks)

    # Sort by match score descending
    ranked_jobs.sort(key=lambda j: j["match_score"], reverse=True)
    return ranked_jobs
