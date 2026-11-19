import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama.chat_models import ChatOllama
from langchain_openai import ChatOpenAI


class LLMService:
    def __init__(self):
        self.llm = self.get_llm()

    def get_llm(self) -> BaseChatModel:
        provider = os.environ.get("LLM_PROVIDER", "ollama").strip().lower()
        temperature = os.environ.get("LLM_TEMPERATURE", "").strip()
        options = {"temperature": float(temperature)} if temperature else {}
        if provider == "openai":
            if not os.environ.get("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
            return ChatOpenAI(
                model=os.environ.get("OPENAI_LLM_MODEL", "gpt-6-luna"),
                **options,  # type: ignore[arg-type]
            )
        if provider != "ollama":
            raise ValueError(
                f"Unsupported LLM_PROVIDER: {provider!r}; use ollama or openai"
            )

        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.environ.get("OLLAMA_LLM_MODEL", "qwen3.6:35b")

        return ChatOllama(model=model, base_url=ollama_url, **options)   # type: ignore[arg-type]
