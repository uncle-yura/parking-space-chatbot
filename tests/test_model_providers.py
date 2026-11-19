from unittest.mock import Mock

import pytest

from app.services import embeddings, llm_chat


@pytest.fixture(params=["chat", "embeddings"])
def service(request, monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDINGS_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_TEMPERATURE", raising=False)
    if request.param == "chat":
        return (
            llm_chat,
            llm_chat.LLMService,
            "llm",
            "LLM_PROVIDER",
            "ChatOllama",
            "ChatOpenAI",
            "OLLAMA_LLM_MODEL",
            "OPENAI_LLM_MODEL",
            "qwen3.6:35b",
            "gpt-6-luna",
        )
    return (
        embeddings,
        embeddings.EmbeddingsService,
        "embeddings",
        "EMBEDDINGS_PROVIDER",
        "OllamaEmbeddings",
        "OpenAIEmbeddings",
        "OLLAMA_EMBEDDINGS_MODEL",
        "OPENAI_EMBEDDINGS_MODEL",
        "qwen3-embedding:8b",
        "text-embedding-3-small",
    )


@pytest.mark.parametrize("provider", [None, "ollama", " OpenAI "])
@pytest.mark.parametrize("custom_model", [False, True])
@pytest.mark.parametrize("temperature", [None, "", "0", "0.7"])
def test_provider_selection(service, monkeypatch, provider, custom_model, temperature):
    module, factory, attr, env, ollama, openai, ollama_env, openai_env, *defaults = (
        service
    )
    local, remote = Mock(), Mock()
    monkeypatch.setattr(module, ollama, local)
    monkeypatch.setattr(module, openai, remote)
    for name in (ollama_env, openai_env):
        monkeypatch.delenv(name, raising=False)
        if custom_model:
            monkeypatch.setenv(name, "custom-model")
    if provider:
        monkeypatch.setenv(env, provider)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama-test:11434")
    use_openai = provider == " OpenAI "
    if use_openai:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    selected, unused = (remote, local) if use_openai else (local, remote)
    expected = {"model": "custom-model" if custom_model else defaults[use_openai]}
    if not use_openai:
        expected["base_url"] = "http://ollama-test:11434"
    if temperature is not None:
        monkeypatch.setenv("LLM_TEMPERATURE", temperature)
    if attr == "llm" and temperature:
        expected["temperature"] = float(temperature)
    assert getattr(factory(), attr) is selected.return_value
    selected.assert_called_once_with(**expected)
    unused.assert_not_called()


def test_openai_requires_key(service, monkeypatch):
    module, factory, _, env, _, openai, *_ = service
    constructor = Mock()
    monkeypatch.setattr(module, openai, constructor)
    monkeypatch.setenv(env, "openai")
    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        factory()
    constructor.assert_not_called()


def test_unknown_provider_fails(service, monkeypatch):
    _, factory, _, env, *_ = service
    monkeypatch.setenv(env, "unsupported")
    with pytest.raises(ValueError, match=f"Unsupported {env}"):
        factory()
