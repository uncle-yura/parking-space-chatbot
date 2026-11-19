# from dataclasses import dataclass
from uuid import uuid4

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.vectorstores import VectorStore
from langgraph.graph import MessagesState
from presidio_analyzer.analyzer_engine import AnalyzerEngine
from pydantic import BaseModel, ConfigDict, model_validator

from app.services.embeddings import EmbeddingsService
from app.services.llm_chat import LLMService
from app.services.sql_service import SqlService
from app.services.weaviate_vectorstore import WeaviateService


# @dataclass
class RagGraphContextSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: BaseChatModel
    sql_service: SqlService
    pii_analyzer: AnalyzerEngine
    embeddings: Embeddings
    vectorstore: VectorStore
    thread_id: str

    # Initialization of the runtime context for local development in LangGraph Studio
    # LangGraph Studio operates as a separate web server,
    # so context data from `graph.invoke(context=...)` is not passed.
    # If the language studio is not needed, you can replace it with a @dataclass.
    @model_validator(mode="before")
    @classmethod
    def fill_missing(cls, data):
        if isinstance(data, dict):
            if not data.get("thread_id"):
                data["thread_id"] = str(uuid4())
            if not data.get("llm"):
                data["llm"] = LLMService().llm
            if not data.get("sql_service"):
                data["sql_service"] = SqlService()
            if not data.get("pii_analyzer"):
                data["pii_analyzer"] = AnalyzerEngine()
            if not data.get("embeddings"):
                data["embeddings"] = EmbeddingsService().embeddings
            if not data.get("vectorstore"):
                data["vectorstore"] = WeaviateService(
                    index_name="ParkingKnowledge",
                    embeddings=data["embeddings"],
                ).get_vectorstore()

        return data


class AdministratorGraphContextSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: BaseChatModel
    sql_service: SqlService

    @model_validator(mode="before")
    @classmethod
    def fill_missing(cls, data):
        if isinstance(data, dict):
            if not data.get("llm"):
                data["llm"] = LLMService().llm
            if not data.get("sql_service"):
                data["sql_service"] = SqlService()
        return data


def extract_last_message(
    state: MessagesState,
    messages_key: str = "messages",
) -> AnyMessage:

    if (isinstance(state, dict) and (messages := state.get(messages_key, []))) or (
        messages := getattr(state, messages_key, [])
    ):
        message = messages[-1]
    else:
        msg = f"No messages found in input state to input_check: {state}"
        raise ValueError(msg)

    return message
