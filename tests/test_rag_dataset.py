import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_retrieval_dataset_integrity():
    documents = json.loads((DATA_DIR / "parking_knowledge.json").read_text())
    questions = json.loads((DATA_DIR / "rag_questions.json").read_text())

    assert documents and questions
    by_id = {document["id"]: document for document in documents}
    assert len(by_id) == len(documents), "Duplicate knowledge document IDs"

    for document in documents:
        assert set(document) == {"id", "title", "category", "tags", "content"}
        for field in ("id", "title", "category", "content"):
            assert isinstance(document[field], str) and document[field].strip()
        assert isinstance(document["tags"], list) and document["tags"]
        assert all(isinstance(tag, str) and tag.strip() for tag in document["tags"])
        if "synthetic" in document["tags"]:
            assert document["content"].startswith(("SYNTHETIC", "TEST DATA"))

    seen_questions = set()
    covered = set()
    for item in questions:
        assert set(item) == {"question", "category", "relevant_doc_ids"}
        assert isinstance(item["question"], str) and item["question"].strip()
        normalized = " ".join(item["question"].casefold().split())
        assert normalized not in seen_questions, "Duplicate retrieval question"
        seen_questions.add(normalized)

        relevant = item["relevant_doc_ids"]
        assert isinstance(relevant, list) and relevant
        assert len(relevant) == len(set(relevant)), "Duplicate relevance labels"
        for doc_id in relevant:
            assert doc_id in by_id, f"Unknown document ID: {doc_id}"
            assert item["category"] == by_id[doc_id]["category"]
        covered.update(relevant)

    assert covered == set(by_id), "Some knowledge documents have no test questions"
