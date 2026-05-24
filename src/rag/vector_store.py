"""FAISS-based vector store for semantic search."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import faiss
import numpy as np

from src.rag.config import EMBEDDING_DIM, FAISS_INDEX_PATH

if TYPE_CHECKING:
    from numpy.typing import NDArray


class VectorStore:
    """Wrapper around FAISS IndexFlatL2 for storing and searching embeddings."""

    def __init__(self, dimension: int = EMBEDDING_DIM):
        """Initialize FAISS index with L2 distance metric.

        Args:
            dimension: Dimension of embedding vectors (default: 384).
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)

    def add(self, embeddings: NDArray[np.float32]) -> None:
        """Add vectors to the index.

        Args:
            embeddings: 2D array of shape (n_vectors, dimension).
        """
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {embeddings.shape}")
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match "
                f"index dimension {self.dimension}"
            )

        # FAISS requires contiguous float32 arrays
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add(embeddings)

    def search(
        self, query_embedding: NDArray[np.float32], k: int = 5
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        """Search for k nearest neighbors.

        Args:
            query_embedding: 1D or 2D query vector(s) of shape (dimension,) or (n_queries, dimension).
            k: Number of nearest neighbors to return.

        Returns:
            Tuple of (distances, indices) arrays:
            - distances: shape (n_queries, k) — L2 distances to neighbors
            - indices: shape (n_queries, k) — indices of neighbors in the index
        """
        # Reshape to 2D if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension {query_embedding.shape[1]} does not match "
                f"index dimension {self.dimension}"
            )

        query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)
        distances, indices = self.index.search(query_embedding, k)
        return distances, indices

    def save(self, path: Path = FAISS_INDEX_PATH) -> None:
        """Write index to disk.

        Args:
            path: Path to save the index file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path = FAISS_INDEX_PATH) -> VectorStore:
        """Load index from disk.

        Args:
            path: Path to the saved index file.

        Returns:
            VectorStore instance with loaded index.

        Raises:
            FileNotFoundError: If index file does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        index = faiss.read_index(str(path))
        store = cls(dimension=index.d)
        store.index = index
        return store

    def __len__(self) -> int:
        """Return number of vectors in the index."""
        return self.index.ntotal
