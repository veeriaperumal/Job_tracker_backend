import os
from src.config.settings import settings

# Bypass hf-mirror.com if it is configured but unreachable on user's system
hf_env = os.environ.get("HF_ENDPOINT", "")
if "hf-mirror.com" in hf_env:
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"

if settings.HF_ENDPOINT:
    if "hf-mirror.com" in settings.HF_ENDPOINT:
        os.environ["HF_ENDPOINT"] = "https://huggingface.co"
    else:
        os.environ["HF_ENDPOINT"] = settings.HF_ENDPOINT

from sentence_transformers import SentenceTransformer
from typing import List
from config.logger import get_logger

logger = get_logger(__name__)

class EmbeddingService:
    def __init__(self):
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading BAAI/bge-small-en-v1.5 embedding model...")
            self._model = SentenceTransformer('BAAI/bge-small-en-v1.5')
            logger.info("BAAI/bge-small-en-v1.5 embedding model loaded successfully.")
        return self._model

    def embed_text(self, text: str) -> List[float]:
        """
        Embeds a single text chunk into a list of floats (dense embedding).
        BAAI/bge-small-en-v1.5 default dimension is 384.
        """
        if not text.strip():
            # Return zero vector if text is empty
            return [0.0] * 384
        
        embedding = self.model.encode(text)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a batch of text chunks.
        """
        if not texts:
            return []
        
        # Filter empty/whitespace only strings to avoid issues
        cleaned_texts = [t if t.strip() else " " for t in texts]
        embeddings = self.model.encode(cleaned_texts)
        return [emb.tolist() for emb in embeddings]

embedding_service = EmbeddingService()
