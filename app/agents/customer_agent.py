import asyncio
from uuid import UUID, uuid4

from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage
from langgraph.graph.state import RunnableConfig
from langgraph.types import Command
from presidio_analyzer import AnalyzerEngine

from app.graphs.rag_graph import get_rag_graph
from app.graphs.utils import RagGraphContextSchema
from app.services.embeddings import OllamaEmbeddingsService
from app.services.llm_chat import OllamaLLMService
from app.services.sql_service import SqlService
from app.services.weaviate_vectorstore import WeaviateService
from app.tools.booking_tools import get_booking_status

POLL_INTERVAL_SECONDS = 5


async def _poll_booking_status(
    booking_id: UUID,
    context: RagGraphContextSchema,
    config: RunnableConfig,
    stop_event: asyncio.Event,
):
    runtime = ToolRuntime(
        state={"booking_id": booking_id},
        context=context,
        config=config,
        stream_writer=lambda _: None,
        tool_call_id=None,
        store=None,
    )

    while not stop_event.is_set():
        result = await get_booking_status.ainvoke({"runtime": runtime})
        if (
            "is still pending." not in result
            and "temporarily unavailable" not in result
        ):
            print(f"\nAssistant (background): {result}\n")
            return
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=POLL_INTERVAL_SECONDS)
        except TimeoutError:
            pass


async def main():
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

    async with sql_service.async_checkpointer() as cp:
        graph = get_rag_graph().compile(checkpointer=cp)
        print("Chat started. Type 'exit' to quit.\n")
        polling_jobs: dict[UUID, tuple[asyncio.Task, asyncio.Event]] = {}

        try:
            while True:
                user_input = (await asyncio.to_thread(input, "You: ")).strip()
                if user_input.lower() == "exit":
                    print("Chat ended.")
                    break
                if not user_input:
                    continue

                graph_input = {"messages": [HumanMessage(content=user_input)]}
                while True:
                    state = await graph.ainvoke(
                        graph_input,  # type: ignore[arg-type]
                        config=config,
                        context=context,
                    )
                    interrupts = state.get("__interrupt__")
                    if not interrupts:
                        break

                    for interrupt_item in interrupts:
                        interrupt_data = interrupt_item.value
                        message = interrupt_data.get("message", "")
                        error = interrupt_data.get("error_message")
                        field = interrupt_data.get("field", "")
                        print(f"\nAssistant: {message}")
                        if error:
                            print(f"Error: {error}")
                        if field:
                            print(f"Field: {field}")

                        details = interrupt_data.get("details")
                        if details:
                            print("Booking details:")
                            for key, value in details.items():
                                print(f"  {key}: {value}")

                        options = interrupt_data.get("options", [])
                        if options:
                            print(f"Options: {' | '.join(map(str, options))}")

                        response = (await asyncio.to_thread(input, "You: ")).strip()
                        if response.lower() == "cancel":
                            response = "[cancel]"
                        graph_input = Command(resume=response)

                booking_id = state.get("booking_id")
                if booking_id is not None and booking_id not in polling_jobs:
                    stop_event = asyncio.Event()
                    task = asyncio.create_task(
                        _poll_booking_status(booking_id, context, config, stop_event)
                    )
                    polling_jobs[booking_id] = (task, stop_event)

                messages = state.get("messages", [])
                if messages and hasattr(messages[-1], "content"):
                    print(f"\nAssistant: {messages[-1].content}\n")
        finally:
            for task, stop_event in polling_jobs.values():
                stop_event.set()
                task.cancel()
            await asyncio.gather(
                *(task for task, _ in polling_jobs.values()),
                return_exceptions=True,
            )


if __name__ == "__main__":
    asyncio.run(main())
