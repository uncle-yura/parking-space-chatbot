import os

from langchain_ollama import OllamaEmbeddings


class OllamaEmbeddingsService:
    def __init__(self):
        self.embeddings = self.get_embeddings()

    def get_embeddings(self) -> OllamaEmbeddings:
        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.environ.get("OLLAMA_EMBEDDINGS_MODEL", "qwen3-embedding:8b")

        embeddings = OllamaEmbeddings(
            model=model,
            base_url=ollama_url,
        )
        return embeddings
