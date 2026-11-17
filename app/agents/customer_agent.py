from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.graph.state import RunnableConfig
from langgraph.types import Command
from presidio_analyzer import AnalyzerEngine

from app.graphs.rag_graph import RagGraphContextSchema, get_rag_graph
from app.services.embeddings import OllamaEmbeddingsService
from app.services.llm_chat import OllamaLLMService
from app.services.sql_service import SqlService
from app.services.weaviate_vectorstore import WeaviateService


def main():
    sql_service = SqlService()
    thread_id = str(uuid4())

    config = RunnableConfig(configurable={"thread_id": thread_id})

    pii_analyzer = AnalyzerEngine()
    llm_service = OllamaLLMService()
    embeddings = OllamaEmbeddingsService().embeddings
    vectorstore = WeaviateService(
        index_name="ParkingKnowledge",
        embeddings=embeddings,
    ).get_vectorstore()
    
    context = RagGraphContextSchema(
        llm=llm_service.llm,
        pii_analyzer=pii_analyzer,
        sql_service=sql_service,
        embeddings=embeddings,
        vectorstore=vectorstore,
        thread_id=thread_id,
    )

    with sql_service.checkpointer() as cp:
        graph = get_rag_graph().compile(checkpointer=cp)

        print("Chat started. Type 'exit' to quit.\n")

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() == "exit":
                print("Chat ended.")
                break

            if not user_input:
                continue

            graph_input = {"messages": [HumanMessage(content=user_input)]}

            while True:
                state = graph.invoke(
                    graph_input,  # type: ignore[arg-type]
                    config=config,
                    context=context,
                )

                if not state.get("__interrupt__"):
                    break

                interrupts = state["__interrupt__"]

                for interrupt_item in interrupts:
                    interrupt_data = interrupt_item.value

                    field = interrupt_data.get("field", "")
                    message = interrupt_data.get("message", "")
                    error = interrupt_data.get("error_message")

                    print(f"\nAssistant: {message}")

                    if error:
                        print(f"Error: {error}")

                    if field:
                        print(f"Field: {field}")

                    response = input("You: ").strip()

                    if response.lower() == "cancel":
                        response = "[cancel]"

                    graph_input = Command(resume=response)

            messages = state.get("messages", [])

            if messages:
                last_message = messages[-1]
                if hasattr(last_message, "content"):
                    print(f"\nAssistant: {last_message.content}\n")


if __name__ == "__main__":
    main()
