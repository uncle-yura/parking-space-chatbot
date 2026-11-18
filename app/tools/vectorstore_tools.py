from langchain.tools import ToolRuntime, tool

from app.graphs.utils import RagGraphContextSchema

PII_SCORE_THRESHOLD = 0.5
SAFE_ENTITIES = [
    "PHONE_NUMBER",
    "MEDICAL_LICENSE",
    "PERSON",
    "NRP",
    "MAC_ADDRESS",
    "IP_ADDRESS",
    "IBAN_CODE",
    "EMAIL_ADDRESS",
    "CRYPTO",
    "CREDIT_CARD",
]


@tool
def vector_search(query: str, runtime: ToolRuntime[RagGraphContextSchema]) -> str:
    """
    Search static parking knowledge for general descriptive information and policies.
    Do not use it to claim live pricing, working hours, availability, nearby facility
    search, address lookup, distance comparison, or directions; use the SQL tools for
    supported current data.

    Args:
        query: The static knowledge question to search for.

    Returns:
        Relevant safe knowledge entries.
    """

    results = runtime.context.vectorstore.similarity_search(query, k=3)
    safe_results = []
    for doc in results:
        content = f"{doc.metadata or {}}\n{doc.page_content}"
        detections = runtime.context.pii_analyzer.analyze(
            text=content, entities=SAFE_ENTITIES, language="en"
        )
        if any(result.score >= PII_SCORE_THRESHOLD for result in detections):
            continue
        safe_results.append(str(doc))

    if not safe_results:
        return "No safe, relevant parking information was found."
    return "\n".join(safe_results)
