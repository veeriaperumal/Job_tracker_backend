from dotenv import load_dotenv
import os


load_dotenv()

class Settings:
    def __init__(self):
        # API Keys
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
        

        # Models
        self.GROQ_MODEL = os.getenv("GROQ_MODEL")
        self.GEMINI_MODEL = os.getenv("GEMINI_MODEL")
        

        # Retrieval
        self.TOP_K = int(os.getenv("TOP_K", 4))
        self.HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://huggingface.co")

settings = Settings()