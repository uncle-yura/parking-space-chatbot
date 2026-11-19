import os

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings


class EmbeddingsService:
    def __init__(self):
        self.embeddings = self.get_embeddings()

    def get_embeddings(self) -> Embeddings:
        provider = os.environ.get("EMBEDDINGS_PROVIDER", "ollama").strip().lower()
        if provider == "openai":
            if not os.environ.get("OPENAI_API_KEY"):
                raise ValueError(
                    "OPENAI_API_KEY is required when EMBEDDINGS_PROVIDER=openai"
                )
            return OpenAIEmbeddings(
                model=os.environ.get(
                    "OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small"
                )
            )
        if provider != "ollama":
            raise ValueError(
                f"Unsupported EMBEDDINGS_PROVIDER: {provider!r}; use ollama or openai"
            )

        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.environ.get("OLLAMA_EMBEDDINGS_MODEL", "qwen3-embedding:8b")

        embeddings = OllamaEmbeddings(
            model=model,
            base_url=ollama_url,
        )
        return embeddings
