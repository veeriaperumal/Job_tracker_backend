from pinecone import Pinecone, ServerlessSpec
from src.config.settings import settings
from config.logger import get_logger

logger = get_logger(__name__)

class PineconeService:
    def __init__(self):
        self._pc = None
        self._index = None

    @property
    def pc(self) -> Pinecone:
        if self._pc is None:
            api_key = settings.PINECONE_API_KEY
            if not api_key:
                raise ValueError("PINECONE_API_KEY environment variable is not set")
            self._pc = Pinecone(api_key=api_key)
        return self._pc

    @property
    def index(self):
        if self._index is None:
            index_name = settings.PINECONE_INDEX_NAME
            if not index_name:
                raise ValueError("PINECONE_INDEX_NAME environment variable is not set")
            self._index = self.pc.Index(index_name)
        return self._index

    def reset_database(self):
        """
        Resets the Pinecone index on application startup.
        If the index exists, deletes all vectors in it.
        If it does not exist, creates it with us-east-1 AWS serverless spec.
        """
        index_name = settings.PINECONE_INDEX_NAME
        if not index_name:
            logger.warning("PINECONE_INDEX_NAME is not configured. Skipping reset.")
            return

        api_key = settings.PINECONE_API_KEY
        if not api_key:
            logger.warning("PINECONE_API_KEY is not configured. Skipping reset.")
            return

        try:
            logger.info(f"Resetting Pinecone index '{index_name}'...")
            indexes = [idx.name for idx in self.pc.list_indexes()]
            
            if index_name in indexes:
                desc = self.pc.describe_index(index_name)
                if desc.dimension == 384:
                    logger.info(f"Index '{index_name}' exists with correct dimension (384). Deleting all vectors inside it...")
                    self.index.delete(delete_all=True)
                    logger.info(f"Successfully deleted all vectors in Pinecone index '{index_name}'.")
                else:
                    logger.warning(f"Index '{index_name}' exists but has dimension {desc.dimension} (expected 384). Deleting index to recreate...")
                    self.pc.delete_index(index_name)
                    # Force recreate
                    indexes.remove(index_name)
            
            if index_name not in indexes:
                logger.info(f"Creating a new Pinecone index '{index_name}' with 384 dimensions...")
                self.pc.create_index(
                    name=index_name,
                    dimension=384,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                logger.info(f"Successfully created Pinecone index '{index_name}'.")
        except Exception as e:
            logger.error(f"Error during Pinecone database reset: {e}")
            raise e

    def upsert_chunks(self, document_name: str, chunks: dict):
        """
        Embeds and upserts chunks to Pinecone.
        chunks: dict of format { section_name: [chunk1, chunk2, ...] }
        """
        from src.services.embeddings.embedding_service import embedding_service
        
        vectors = []
        for section, section_chunks in chunks.items():
            if not section_chunks:
                continue
            
            # Embed all chunks for the section in a batch
            logger.info(f"Embedding {len(section_chunks)} chunks for section: {section}")
            embeddings = embedding_service.embed_batch(section_chunks)
            
            for idx, (chunk, embedding) in enumerate(zip(section_chunks, embeddings)):
                # Sanitize document name and section name for ID
                sanitized_doc = "".join(c for c in document_name if c.isalnum() or c in "._-").strip()
                vector_id = f"{sanitized_doc}_{section}_{idx}"
                
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "document": document_name,
                        "section": section,
                        "chunk_index": idx,
                        "text": chunk
                    }
                })
        
        if vectors:
            logger.info(f"Upserting {len(vectors)} total vectors to Pinecone...")
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
            logger.info(f"Successfully upserted {len(vectors)} vectors to Pinecone.")

pinecone_service = PineconeService()
