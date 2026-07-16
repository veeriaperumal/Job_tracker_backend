import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

def test_grounding():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("No GEMINI_API_KEY found")
        return
        
    genai.configure(api_key=api_key)
    
    # Enable Google Search grounding
    model = genai.GenerativeModel(
        model_name='gemini-1.5-flash',
        tools=[{"google_search": {}}]
    )
    
    prompt = "Find recent Python Developer jobs posted in the past 24 hours on linkedin.com/jobs"
    response = model.generate_content(prompt)
    
    print("Response text:", response.text)
    
    # Print grounding metadata if available
    metadata = getattr(response.candidates[0], "grounding_metadata", None)
    if metadata:
        print("\nGrounding Metadata:")
        # We can inspect the chunks
        chunks = getattr(metadata, "grounding_chunks", [])
        print(f"Found {len(chunks)} grounding chunks")
        for chunk in chunks:
            web = getattr(chunk, "web", None)
            if web:
                print(f"Title: {web.title}\nURI: {web.uri}\n")
    else:
        print("No grounding metadata found")

if __name__ == '__main__':
    test_grounding()
