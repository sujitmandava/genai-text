"""Retrieval layer combining embeddings and vector store for semantic search."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.rag.config import FAISS_INDEX_PATH, PASSAGES_METADATA_PATH
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore

if TYPE_CHECKING:
    from numpy.typing import NDArray
    import numpy as np


@dataclass
class Passage:
    """A retrieved text passage with metadata and relevance score."""

    text: str
    source: str  # book title
    section: str  # aphorism/chapter number
    score: float  # similarity score (lower distance = better)


class Retriever:
    """Combines embeddings and vector store for end-to-end retrieval."""

    def __init__(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        metadata_path: Path = PASSAGES_METADATA_PATH,
    ):
        """Initialize retriever by loading index and metadata from disk.

        Args:
            index_path: Path to FAISS index file.
            metadata_path: Path to passages metadata JSON file.

        Raises:
            FileNotFoundError: If index or metadata file does not exist.
        """
        self.vector_store = VectorStore.load(index_path)
        self.embedding_model = EmbeddingModel()

        # Load passages metadata
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.passages_metadata = json.load(f)

        # Validate metadata count matches index size
        if len(self.passages_metadata) != len(self.vector_store):
            raise ValueError(
                f"Metadata count ({len(self.passages_metadata)}) does not match "
                f"index size ({len(self.vector_store)})"
            )

    def retrieve(self, query: str, k: int = 3) -> list[Passage]:
        """Retrieve k most relevant passages for a query.

        Args:
            query: Text query to search for.
            k: Number of passages to retrieve (default: 3).

        Returns:
            List of Passage objects ranked by relevance (lowest score first = most similar).
        """
        # Encode query into embedding
        query_embedding = self.embedding_model.encode(query)

        # Search vector store
        distances, indices = self.vector_store.search(query_embedding, k=k)

        # Build list of Passage objects
        # distances and indices are 2D arrays of shape (1, k) since we have 1 query
        passages = []
        for score, idx in zip(distances[0], indices[0]):
            meta = self.passages_metadata[int(idx)]
            passages.append(
                Passage(
                    text=meta["text"],
                    source=meta["source"],
                    section=meta["section"],
                    score=float(score),
                )
            )

        return passages
