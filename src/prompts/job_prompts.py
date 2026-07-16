RESUME_EXTRACTION_PROMPT = """
Analyze the following resume and extract the candidate profile details in JSON format.
Return ONLY a raw JSON object with these keys:
- "skills": List of technical skills, frameworks, or programming languages (e.g. ["Python", "Django", "FastAPI"]).
- "experience_years": Float or integer representing total years of experience (e.g. 2.5).
- "salary_expectation": An estimated yearly salary expectation in INR or USD based on experience or profile (e.g. "8-12 LPA" or "800000").
- "location": Candidate's current/preferred location (e.g. "Chennai").
- "seniority": The candidate's seniority level (Junior, Mid, Senior, Lead).
- "notice_period_days": Integer notice period in days (e.g. 0 for immediate, 30 for 1 month).
- "role_keywords": Key search terms for roles matching their experience (e.g. ["Python Developer", "Django Backend Developer"]).

Resume:
{resume_text}
"""

JOB_EXTRACTION_PROMPT = """
Analyze the following job listing (Title & Snippet) and extract the requirements in JSON format.
Return ONLY a raw JSON object with these keys:
- "required_skills": List of programming languages/frameworks required (e.g. ["Python", "FastAPI"]).
- "experience_years_required": Integer or float representing minimum years of experience required. If not mentioned, set to 0.
- "salary_lpa": Est. salary in LPA if mentioned (e.g. 12.0), else null.
- "location": Job location (e.g. "Chennai", "Remote", "Bangalore").
- "seniority": Seniority required (Junior, Mid, Senior, Lead, or "Any").
- "notice_period_days": Notice period required if mentioned (e.g. 30, 0 for immediate), else null.

Job Title: {job_title}
Job Snippet: {job_snippet}
"""
