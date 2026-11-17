import os

from langchain_ollama.chat_models import ChatOllama


class OllamaLLMService:
    def __init__(self):
        self.llm = self.get_llm()

    def get_llm(self):
        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.environ.get("OLLAMA_LLM_MODEL", "qwen3.6:35b")

        return ChatOllama(model=model, temperature=0, base_url=ollama_url)
