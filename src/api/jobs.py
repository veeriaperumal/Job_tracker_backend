from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from src.services.jobs.job_service import (
    get_latest_resume,
    extract_resume_profile,
    generate_queries,
    search_yahoo,
    normalize_results,
    remove_duplicates,
    rank_jobs
)
from config.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])

class JobSearchRequest(BaseModel):
    keyword: str
    use_resume: Optional[bool] = True

class JobSearchResponse(BaseModel):
    keyword: str
    resume_used: Optional[str] = None
    queries: Dict[str, str]
    jobs: List[Dict[str, Any]]

@router.post("/search", response_model=JobSearchResponse)
async def search_jobs(request: JobSearchRequest):
    logger.info(f"Received job search request for: {request.keyword}")
    
    resume_name = None
    profile = None
    
    if request.use_resume:
        logger.info("Attempting to parse latest resume for personalized job search...")
        resume_data = get_latest_resume()
        if resume_data:
            resume_name, resume_text = resume_data
            logger.info(f"Using resume: {resume_name}. Extracting profile details...")
            profile = await extract_resume_profile(resume_text)
            logger.info(f"Extracted resume profile: {profile}")
        else:
            logger.warning("No resume found in uploads directory. Proceeding with keyword-only search.")
            
    # 1. Generate queries with search operators
    queries = generate_queries(request.keyword, profile)
    logger.info(f"Generated search operator queries: {queries}")
    
    # 2. Run searches concurrently
    import asyncio
    platforms = list(queries.keys())
    tasks = [search_yahoo(queries[p], p) for p in platforms]
    
    logger.info(f"Executing search tasks in parallel for platforms: {platforms}")
    results_list = await asyncio.gather(*tasks)
    raw_results = {p: jobs for p, jobs in zip(platforms, results_list)}
        
    # 3. Normalize results
    normalized_jobs = normalize_results(raw_results)
    logger.info(f"Normalized {len(normalized_jobs)} total listings.")
    
    # 4. Remove duplicates
    unique_jobs = remove_duplicates(normalized_jobs)
    logger.info(f"Deduplicated to {len(unique_jobs)} unique listings.")
    
    # 5. Rank the listings
    ranked_jobs = await rank_jobs(unique_jobs, request.keyword, profile)
    logger.info(f"Ranked and returning {len(ranked_jobs)} listings.")
    
    return JobSearchResponse(
        keyword=request.keyword,
        resume_used=resume_name,
        queries=queries,
        jobs=ranked_jobs
    )
