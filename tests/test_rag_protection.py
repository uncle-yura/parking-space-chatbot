import asyncio
from collections.abc import Iterable, Iterator
from typing import Any, cast

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import InMemorySaver
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpArtifacts, NlpEngine
from presidio_analyzer.predefined_recognizers import (
    CreditCardRecognizer,
    EmailRecognizer,
)

from app.graphs import rag_graph
from app.graphs.utils import RagGraphContextSchema
from app.services.llm_chat import LLMService
from app.services.sql_service import SqlService


class RegexOnlyNlpEngine(NlpEngine):
    def __init__(self):
        import spacy

        self.nlp = spacy.blank("en")

    def load(self):
        return None

    def is_loaded(self):
        return True

    def process_text(self, text, language):
        doc = self.nlp(text)
        return NlpArtifacts(
            entities=[],
            tokens=doc,
            tokens_indices=[token.idx for token in doc],
            lemmas=[token.text.lower() for token in doc],
            nlp_engine=self,
            language=language,
        )

    def process_batch(
        self, texts: Iterable[str], language: str, batch_size=1, n_process=1, **kwargs
    ) -> Iterator[tuple[str, NlpArtifacts]]:
        for text in texts:
            yield text, self.process_text(text, language)

    def is_stopword(self, word, language):
        return False

    def is_punct(self, word, language):
        return False

    def get_supported_languages(self):
        return ["en"]

    def get_supported_entities(self):
        return []


class KeywordEmbeddings(Embeddings):
    terms = ("parking", "access", "garage", "charging", "price")

    def _embed(self, text):
        words = text.lower().split()
        return [float(words.count(term)) for term in self.terms]

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.bound_tools = []
        self.calls = 0

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        response = self.responses.pop(0)
        return response(messages) if callable(response) else response


class FakeSqlService(SqlService):
    def __init__(self):
        self.price_queries = []

    def get_prices(self, as_of=None):
        self.price_queries.append(as_of)
        return [
            {
                "price_per_hour": 8,
                "price_per_day": 40,
                "max_stay_hours": 24,
                "effective_from": "2030-01-01",
            }
        ]


def _presidio_analyzer():
    registry = RecognizerRegistry(supported_languages=["en"])
    email_recognizer = EmailRecognizer()
    import tldextract

    local_suffixes = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)
    email_recognizer.validate_result = lambda pattern_text: (
        local_suffixes(pattern_text).fqdn != ""
    )
    for recognizer in (email_recognizer, CreditCardRecognizer()):
        registry.add_recognizer(recognizer)
    return AnalyzerEngine(
        registry=registry,
        nlp_engine=RegexOnlyNlpEngine(),
        supported_languages=["en"],
    )


def _context(
    analyzer, *, llm=None, sql_service=None, vectorstore=None, embeddings=None
):
    embeddings = embeddings or KeywordEmbeddings()
    return RagGraphContextSchema.model_construct(
        llm=llm or LLMService().llm,
        sql_service=sql_service or FakeSqlService(),
        pii_analyzer=analyzer,
        embeddings=embeddings,
        vectorstore=vectorstore or InMemoryVectorStore(embeddings),
        thread_id="rag-protection-test",
    )


def _run(graph, graph_input, context):
    return asyncio.run(
        graph.ainvoke(
            graph_input,
            config=cast(Any, {"configurable": {"thread_id": context.thread_id}}),
            context=context,
        )
    )


@pytest.mark.parametrize(
    "text",
    [
        "Contact me at jordan.lee@example.com",
        "My card number is 4111 1111 1111 1111",
    ],
)
def test_presidio_recognizes_realistic_sensitive_values(text):
    analyzer = _presidio_analyzer()

    detections = rag_graph._analyze_text(text, analyzer)

    assert detections
    assert max(result.score for result in detections) >= rag_graph.PII_SCORE_THRESHOLD


def test_compiled_parent_graph_blocks_sensitive_model_output():
    analyzer = _presidio_analyzer()
    llm = ScriptedLLM([AIMessage(content="Contact Jordan at jordan.lee@example.com")])
    graph = rag_graph.build_rag_graph().compile(checkpointer=InMemorySaver())
    context = _context(analyzer, llm=llm)

    result = _run(
        graph,
        {"messages": [HumanMessage(content="What contact details do you have?")]},
        context,
    )

    assert "private information" in result["messages"][-1].content.lower()
    assert all(
        "jordan.lee@example.com" not in str(message.content)
        for message in result["messages"]
    )
    assert "vector_search" in [tool.name for tool in llm.bound_tools]


def test_compiled_parent_graph_rejects_realistic_sensitive_input():
    llm = ScriptedLLM([AIMessage(content="safe response")])
    graph = rag_graph.build_rag_graph().compile(checkpointer=InMemorySaver())
    context = _context(_presidio_analyzer(), llm=llm)

    result = _run(
        graph,
        {"messages": [HumanMessage(content="Email me at jordan.lee@example.com")]},
        context,
    )

    assert "private information" in result["messages"][-1].content.lower()
    assert all(
        "jordan.lee@example.com" not in str(message.content)
        for message in result["messages"]
    )
    assert llm.calls == 0


def test_compiled_graph_retrieves_parking_docs_and_filters_private_metadata():
    analyzer = _presidio_analyzer()
    embeddings = KeywordEmbeddings()
    vectorstore = InMemoryVectorStore(embeddings)
    vectorstore.add_documents(
        [
            Document(
                page_content="The garage provides parking access and EV charging.",
                metadata={"source": "public parking guide"},
            ),
            Document(
                page_content="Parking access is available for registered users.",
                metadata={"contact": "private.user@example.com"},
            ),
        ]
    )

    llm = ScriptedLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "vector_search",
                        "args": {"query": "parking access garage"},
                        "id": "search-parking-docs",
                        "type": "tool_call",
                    }
                ],
            ),
            lambda messages: AIMessage(content=str(messages[-1].content)),
        ]
    )
    context = _context(
        analyzer,
        llm=llm,
        vectorstore=vectorstore,
        embeddings=embeddings,
    )
    graph = rag_graph.build_rag_graph().compile(checkpointer=InMemorySaver())
    result = _run(
        graph,
        {"messages": [HumanMessage(content="How does parking access work?")]},
        context,
    )

    response = result["messages"][-1].content
    assert "garage provides parking access and EV charging" in response
    assert "private.user@example.com" not in response
    assert "vector_search" in [tool.name for tool in llm.bound_tools]


def test_compiled_graph_executes_selected_dynamic_sql_tool():
    sql_service = FakeSqlService()
    llm = ScriptedLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_prices",
                        "args": {},
                        "id": "lookup-price",
                        "type": "tool_call",
                    }
                ],
            ),
            lambda messages: AIMessage(content=str(messages[-1].content)),
        ]
    )
    context = _context(_presidio_analyzer(), llm=llm, sql_service=sql_service)
    graph = rag_graph.build_rag_graph().compile(checkpointer=InMemorySaver())
    result = _run(
        graph,
        {"messages": [HumanMessage(content="What are the current prices?")]},
        context,
    )

    assert sql_service.price_queries == [None]
    assert "$8/hr" in result["messages"][-1].content
    assert "get_prices" in [tool.name for tool in llm.bound_tools]
