"""Seed Weaviate vector store with parking knowledge data using local Ollama embeddings."""

import json
import sys
from pathlib import Path

from langchain_core.documents.base import Document

from app.services.embeddings import EmbeddingsService
from app.services.weaviate_vectorstore import WeaviateService

BASE_DIR = Path(__file__).resolve().parents[2]  # project root
DATA_FILE = BASE_DIR / "data" / "parking_knowledge.json"


def main():
    if not DATA_FILE.exists():
        print(f"ERROR: Knowledge data file not found: {DATA_FILE}")
        sys.exit(1)

    with open(DATA_FILE) as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} knowledge chunks from {DATA_FILE.name}")

    docs = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk["content"],
            metadata={
                "doc_id": chunk["id"],
                "title": chunk["title"],
                "category": chunk["category"],
                "tags": chunk.get("tags", []),
            },
        )
        docs.append(doc)

    index_name = "ParkingKnowledge"
    embeddings_service = EmbeddingsService()
    weaviate_service = WeaviateService(
        index_name, embeddings=embeddings_service.embeddings
    )
    weaviate_service.ensure_collection()

    vector_store = weaviate_service.get_vectorstore()

    vector_store.add_documents(documents=docs)
    print(
        f"Done! {len(docs)} documents seeded into Weaviate collection '{index_name}'."
    )

    print("\n--- Verification: sample query ---")
    results = vector_store.similarity_search("How do I book a parking space?", k=3)
    for i, doc in enumerate(results, 1):
        print(
            f" {dict(doc)}  [{i}] {doc.metadata['title']} ({doc.metadata['category']})"
        )


if __name__ == "__main__":
    main()
