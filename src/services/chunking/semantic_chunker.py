import json
import re
from typing import Dict, List
from litellm import acompletion
from src.config.settings import settings
from core.exceptions import AIError
from config.logger import get_logger

logger = get_logger(__name__)

# Standard target sections
TARGET_SECTIONS = [
    "Summary",
    "Skills",
    "Experience",
    "Education",
    "Projects",
    "Certificates",
    "Achievements",
    "Languages",
    "Contact",
    "Links"
]

# Spoken/written languages list to help filter out programming languages in heuristic parser
COMMON_SPOKEN_LANGUAGES = {
    "english", "tamil", "hindi", "telugu", "kannada", "malayalam", "spanish", 
    "french", "german", "mandarin", "japanese", "korean", "russian"
}

# Keyword synonyms for rule-based matching fallback
SYNONYMS = {
    "Summary": [r"\bsummary\b", r"\bprofile\b", r"\bobjective\b", r"\bcareer objective\b", r"\babout me\b", r"\bprofessional summary\b"],
    "Skills": [r"\bskills\b", r"\btechnical skills\b", r"\bkey skills\b", r"\bcore competencies\b", r"\bexpertise\b", r"\btechnologies\b", r"\btools\b", r"\bprogramming languages\b"],
    "Experience": [r"\bexperience\b", r"\bwork experience\b", r"\bemployment\b", r"\bwork history\b", r"\bemployment history\b", r"\bprofessional experience\b", r"\binternship\b", r"\binternships\b"],
    "Education": [r"\beducation\b", r"\bacademic\b", r"\beducational qualification\b", r"\bacademics\b", r"\beducational background\b"],
    "Projects": [r"\bprojects\b", r"\bpersonal projects\b", r"\bacademic projects\b", r"\bkey projects\b", r"\bproduction projects\b", r"\bopen source\b"],
    "Certificates": [r"\bcertificates\b", r"\bcertifications\b", r"\blicenses\b", r"\bcredentials\b"],
    "Achievements": [r"\bachievements\b", r"\baccomplishments\b", r"\bawards\b", r"\bhonors\b", r"\bactivities\b"],
    "Languages": [r"\blanguages\b", r"\blanguage\b"],
    "Contact": [r"\bcontact\b", r"\bcontact info\b", r"\bpersonal info\b", r"\baddress\b", r"\bphone\b", r"\bemail\b"],
    "Links": [r"\blinks\b", r"\bwebsites\b", r"\bsocials\b", r"\bportfolio\b", r"\bgithub\b", r"\blinkedin\b"]
}

SYSTEM_PROMPT = """You are a professional resume parser.
Analyze the provided resume text and segment it into the following 10 standard categories:
1. Summary (include summary, profile, objective)
2. Skills (include technical skills, soft skills, core competencies, programming languages, databases, tools, frameworks)
3. Experience (include work experience, employment history, internships)
4. Education (include degrees, schools, coursework)
5. Projects (include personal projects, academic projects, open source contributions)
6. Certificates (include certifications, licenses)
7. Achievements (include awards, accomplishments, honors)
8. Languages (include ONLY spoken/written languages, e.g. English, Tamil, Spanish. Do NOT put programming languages like Python or SQL here; programming languages belong in Skills!)
9. Contact (include name, email, phone number, address)
10. Links (include GitHub, LinkedIn, portfolio website, personal links. Do NOT leave URLs in Contact; extract them and place them here!)

Rules for chunking:
- For Experience, Projects, and Education, each chunk MUST represent a single complete entry (e.g. one complete job role including title, company, dates and description; one complete project description; one degree with its university). Do NOT split a single job entry or sentence across multiple chunks.
- Clean up any orphan bullet points ("•", "-", "*"), stray symbols, or trailing fragments.
- Keep text flow cohesive. Ensure sentences are not broken or truncated.
- If a category is not present or has no information in the resume, return an empty list `[]` for it.

Output MUST be a valid JSON object matching the following structure and no other text:
{
  "Summary": ["chunk 1", ...],
  "Skills": ["chunk 1", ...],
  "Experience": ["chunk 1", "chunk 2", ...],
  "Education": ["chunk 1", "chunk 2", ...],
  "Projects": ["chunk 1", "chunk 2", ...],
  "Certificates": ["chunk 1", "chunk 2", ...],
  "Achievements": ["chunk 1", "chunk 2", ...],
  "Languages": ["chunk 1", "chunk 2", ...],
  "Contact": ["chunk 1", "chunk 2", ...],
  "Links": ["chunk 1", "chunk 2", ...]
}
"""

def clean_chunk(text: str) -> str:
    """Clean orphan bullet points and stray characters from the start/end of a chunk."""
    text = text.strip()
    # Remove leading/trailing bullet points or dashes
    text = re.sub(r'^[-*•◦▪]+\s*', '', text)
    text = re.sub(r'\s*[-*•◦▪]+$', '', text)
    return text.strip()

def is_link(text: str) -> bool:
    """Check if the text represents a URL or profile link (excluding email addresses)."""
    text_lower = text.lower()
    if "@" in text_lower:
        return False
    indicators = ["github.com", "linkedin.com", "http", "www.", ".co", ".io", ".in", ".com", ".org", "portfolio", "website"]
    for ind in indicators:
        if ind in text_lower:
            return True
    return False

def collapse_spaced_header(text: str) -> str:
    """Normalize headers with characters separated by spaces (e.g., 'E D U C A T I O N')."""
    words = text.split()
    if not words:
        return text
    
    # If the line consists mostly of single characters, collapse it
    single_char_words = [w for w in words if len(w) == 1]
    if len(single_char_words) / len(words) > 0.7:
        # Split by multiple spaces (which separate words in spaced-out headings)
        parts = re.split(r'\s{2,}', text)
        collapsed_parts = []
        for part in parts:
            collapsed_parts.append("".join(part.split()))
        collapsed = " ".join(collapsed_parts)
        return collapsed
    return text

def heuristic_chunker(text: str) -> Dict[str, List[str]]:
    """Rule-based fallback chunker using keyword synonym matching and line heuristics."""
    logger.info("Running heuristic chunker fallback")
    
    sections = {sec: [] for sec in TARGET_SECTIONS}
    lines = [line.strip() for line in text.split("\n")]
    
    current_section = "Contact"  # Start with Contact
    current_buffer = []

    def flush_buffer():
        if current_buffer and current_section:
            section_content = "\n".join(current_buffer).strip()
            if section_content:
                chunks = []
                if current_section in ["Experience", "Projects", "Education"]:
                    # Split by paragraph double-newlines
                    parts = re.split(r'\n{2,}', section_content)
                    for part in parts:
                        part_clean = clean_chunk(part.replace("\n", " "))
                        if part_clean and len(part_clean) > 5:
                            chunks.append(part_clean)
                elif current_section in ["Skills", "Languages"]:
                    # Split by commas, semicolons or newlines
                    parts = re.split(r'[;,\n]', section_content)
                    for part in parts:
                        part_clean = clean_chunk(part)
                        if part_clean:
                            if current_section == "Languages" and part_clean.lower() not in COMMON_SPOKEN_LANGUAGES:
                                sections["Skills"].append(part_clean)
                            else:
                                chunks.append(part_clean)
                elif current_section == "Contact":
                    # Split contact details by | or newline to separate name, email, phone, and links
                    parts = re.split(r'[|\n]', section_content)
                    for part in parts:
                        part_clean = clean_chunk(part)
                        if part_clean:
                            if is_link(part_clean):
                                sections["Links"].append(part_clean)
                            else:
                                chunks.append(part_clean)
                else:
                    # Summary, Links, etc.
                    part_clean = clean_chunk(section_content.replace("\n", " "))
                    if part_clean:
                        chunks.append(part_clean)
                        
                if chunks:
                    sections[current_section].extend(chunks)
            current_buffer.clear()

    for line in lines:
        if not line:
            current_buffer.append("")
            continue
            
        # Check if line matches a header pattern
        matched_section = None
        
        # Clean line for matching: lowercase, strip punctuation
        raw_clean = re.sub(r'[^\w\s]', '', line.lower()).strip()
        # Collapse spaced-out letters (e.g. "e d u c a t i o n" -> "education")
        clean_line = collapse_spaced_header(raw_clean)
        clean_line_no_spaces = clean_line.replace(" ", "")
        
        # Header detection: short lines matching synonyms
        if len(clean_line) < 40:
            for sec, patterns in SYNONYMS.items():
                for pat in patterns:
                    # Match standard pattern (with spaces)
                    if re.match(f"^{pat}$", clean_line) or (len(clean_line.split()) <= 3 and re.search(pat, clean_line)):
                        matched_section = sec
                        break
                    # Or match space-stripped pattern
                    pat_clean_str = pat.replace(r"\b", "").replace(" ", "")
                    pat_no_spaces = re.sub(r'[^\w]', '', pat_clean_str)
                    if pat_no_spaces and pat_no_spaces in clean_line_no_spaces:
                        if len(pat_no_spaces) >= 4:
                            matched_section = sec
                            break
                if matched_section:
                    break
                    
        if matched_section:
            flush_buffer()
            current_section = matched_section
        else:
            current_buffer.append(line)
            
    flush_buffer()
    
    # Ensure empty arrays instead of None/null and perform final cleanup
    cleaned_sections = {}
    for sec, chunks in sections.items():
        cleaned_chunks = []
        for c in chunks:
            c_cleaned = clean_chunk(c)
            if c_cleaned and c_cleaned not in ["•", "-", "*"]:
                cleaned_chunks.append(c_cleaned)
        cleaned_sections[sec] = cleaned_chunks
        
    return cleaned_sections

async def chunk_resume(text: str) -> Dict[str, List[str]]:
    """Segment resume text into standard sections using LiteLLM (with fallback)."""
    primary_model = settings.GROQ_MODEL or "groq/llama-3.1-70b-versatile"
    fallback_model = settings.GEMINI_MODEL or "gemini/gemini-1.5-flash"
    
    # Try primary model
    try:
        logger.info(f"Chunking resume using primary model: {primary_model}")
        chunks = await call_llm_chunker(primary_model, text)
        if chunks:
            return chunks
    except Exception as e:
        logger.warning(f"Primary model {primary_model} failed for chunking: {e}")
        
    # Try fallback model
    try:
        logger.info(f"Chunking resume using fallback model: {fallback_model}")
        chunks = await call_llm_chunker(fallback_model, text)
        if chunks:
            return chunks
    except Exception as e:
        logger.error(f"Fallback model {fallback_model} failed for chunking: {e}")
        
    # Run heuristic fallback
    return heuristic_chunker(text)

async def call_llm_chunker(model: str, text: str) -> Dict[str, List[str]]:
    """Helper to request LiteLLM for section-aware chunking."""
    response = await acompletion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Resume Text:\n\n{text}"}
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    
    content = response.choices[0].message.content
    if not content:
        raise AIError("Empty response from LLM")
    
    # Robustly strip markdown code blocks if LLM wraps JSON in ```json ... ```
    content_str = content.strip()
    if content_str.startswith("```"):
        # Strip backticks
        content_str = content_str.strip("`").strip()
        # Strip optional "json" language identifier
        if content_str.startswith("json"):
            content_str = content_str[4:].strip()
            
    parsed = json.loads(content_str)
    
    # Standardize result: ensure all 10 target sections are present and clean
    standardized = {}
    for sec in TARGET_SECTIONS:
        matched_val = []
        for k, v in parsed.items():
            if k.lower() == sec.lower() and isinstance(v, list):
                matched_val = [clean_chunk(str(item)) for item in v if clean_chunk(str(item)) not in ["•", "-", "*", ""]]
                break
        standardized[sec] = matched_val
        
    return standardized
