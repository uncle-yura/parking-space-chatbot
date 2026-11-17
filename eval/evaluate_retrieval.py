"""Evaluate the ParkingKnowledge vector store against labeled retrieval questions."""

import argparse
import csv
import json
import sys
from math import ceil
from pathlib import Path
from time import perf_counter

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = BASE_DIR / "data" / "rag_questions.json"


def evaluate(vectorstore, questions):
    results = []
    for item in questions:
        question = item["question"]
        relevant = list(dict.fromkeys(item["relevant_doc_ids"]))
        started_at = perf_counter()
        try:
            documents = vectorstore.similarity_search(question, k=3)
            retrieval_latency_ms = (perf_counter() - started_at) * 1000
            retrieved = list(
                dict.fromkeys(
                    doc_id
                    for document in documents
                    if (doc_id := (document.metadata or {}).get("doc_id")) is not None
                )
            )
            hits = [doc_id for doc_id in retrieved if doc_id in relevant]
            rank = next(
                (index for index, doc_id in enumerate(retrieved, 1) if doc_id in relevant),
                None,
            )
            error = ""
        except Exception as exc:
            retrieved, hits, rank = [], [], None
            retrieval_latency_ms = (perf_counter() - started_at) * 1000
            error = str(exc)

        results.append(
            {
                "question": question,
                "category": item.get("category", ""),
                "relevant_doc_ids": relevant,
                "retrieved_doc_ids": retrieved,
                "recall_at_3": len(hits) / len(relevant) if relevant else 0.0,
                "precision_at_3": len(hits) / 3,
                "mrr": 1 / rank if rank else 0.0,
                "retrieval_latency_ms": retrieval_latency_ms,
                "error": error,
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--json-out", type=Path, default=BASE_DIR / "eval" / "retrieval_results.json"
    )
    parser.add_argument(
        "--csv-out", type=Path, default=BASE_DIR / "eval" / "retrieval_results.csv"
    )
    args = parser.parse_args()

    try:
        with args.dataset.open() as dataset_file:
            questions = json.load(dataset_file)
        if not isinstance(questions, list):
            raise TypeError("Dataset must contain a JSON array")

        # Load the same services/configuration used by the chatbot; don't alter vector_search.
        from app.services.embeddings import OllamaEmbeddingsService
        from app.services.weaviate_vectorstore import WeaviateService

        weaviate_service = WeaviateService(
            "ParkingKnowledge", OllamaEmbeddingsService().embeddings
        )
        vectorstore = weaviate_service.get_vectorstore()
    except Exception as exc:
        print(f"Evaluation setup failed: {exc}", file=sys.stderr)
        return 1

    try:
        results = evaluate(vectorstore, questions)
    finally:
        weaviate_service.client.close()

    count = len(results)
    metrics = {
        key: sum(result[key] for result in results) / count if count else 0.0
        for key in ("recall_at_3", "precision_at_3", "mrr", "retrieval_latency_ms")
    }
    latencies = sorted(result["retrieval_latency_ms"] for result in results)
    metrics["retrieval_latency_p95_ms"] = (
        latencies[min(len(latencies) - 1, ceil(len(latencies) * 0.95) - 1)]
        if latencies
        else 0.0
    )
    for result in results:
        status = f"ERROR: {result['error']}" if result["error"] else ""
        print(
            f"[{result['category']}] {result['question']}\n"
            f"  retrieved={result['retrieved_doc_ids']} relevant={result['relevant_doc_ids']} "
            f"Recall@3={result['recall_at_3']:.3f} Precision@3={result['precision_at_3']:.3f} "
            f"MRR={result['mrr']:.3f} retrieval_ms={result['retrieval_latency_ms']:.2f} {status}"
        )
    print("\nAggregate metrics (macro average):")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps({"metrics": metrics, "results": results}, indent=2) + "\n")
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="") as csv_file:
        fields = [
            "question",
            "category",
            "relevant_doc_ids",
            "retrieved_doc_ids",
            "recall_at_3",
            "precision_at_3",
            "mrr",
            "retrieval_latency_ms",
            "error",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {**result, "relevant_doc_ids": json.dumps(result["relevant_doc_ids"]),
                 "retrieved_doc_ids": json.dumps(result["retrieved_doc_ids"])}
            )
    print(f"\nJSON: {args.json_out}\nCSV: {args.csv_out}")
    return 1 if any(result["error"] for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
