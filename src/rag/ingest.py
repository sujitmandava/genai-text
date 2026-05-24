"""CLI script to build FAISS index from passages.jsonl."""

import json
from pathlib import Path

from src.rag.config import (
    PASSAGES_PATH,
    FAISS_INDEX_PATH,
    PASSAGES_METADATA_PATH,
)
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore


def main():
    """Build FAISS index from passages.jsonl and save to disk."""
    print("=" * 60)
    print("NIETZSCHE RAG INGESTION")
    print("=" * 60)
    print()

    # Validate input file exists
    if not PASSAGES_PATH.exists():
        raise FileNotFoundError(
            f"Passages file not found: {PASSAGES_PATH}\n"
            "Run 'python -m src.data.prepare' first to generate passages."
        )

    # Load passages from JSONL
    print(f"[1/4] Loading passages from {PASSAGES_PATH}...")
    passages = []
    with open(PASSAGES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                passages.append(json.loads(line))
    print(f"Loaded {len(passages)} passages")
    print()

    # Extract texts for embedding
    print("[2/4] Encoding passages with embedding model...")
    texts = [p["text"] for p in passages]
    embedding_model = EmbeddingModel()
    embeddings = embedding_model.encode(texts)
    print(f"Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
    print()

    # Build vector store and add embeddings
    print("[3/4] Building FAISS index...")
    vector_store = VectorStore()
    vector_store.add(embeddings)
    print(f"Index contains {len(vector_store)} vectors")
    print()

    # Save index and metadata
    print("[4/4] Saving index and metadata to disk...")
    vector_store.save(FAISS_INDEX_PATH)
    print(f"Index saved to: {FAISS_INDEX_PATH}")

    # Save metadata (strip to only required fields)
    metadata = [
        {
            "text": p["text"],
            "source": p["source"],
            "section": p["section"],
        }
        for p in passages
    ]
    PASSAGES_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PASSAGES_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Metadata saved to: {PASSAGES_METADATA_PATH}")
    print()

    print("=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print()
    print(f"Total passages indexed: {len(passages)}")
    print(f"Index size: {len(vector_store)} vectors")
    print()
    print("You can now use Retriever to search passages:")
    print('  from src.rag.retrieval import Retriever')
    print('  retriever = Retriever()')
    print('  passages = retriever.retrieve("eternal recurrence", k=3)')


if __name__ == "__main__":
    main()
