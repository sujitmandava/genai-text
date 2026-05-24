import numpy as np
from typing import Union
from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL, EMBEDDING_DIM


class EmbeddingModel:
    """Wrapper around sentence-transformers for generating L2-normalized embeddings."""

    def __init__(self):
        self._model = None

    def _load_model(self):
        """Lazy load the model on first use."""
        if self._model is None:
            self._model = SentenceTransformer(EMBEDDING_MODEL)

    def encode(self, texts: Union[str, list[str]]) -> np.ndarray:
        """
        Encode text(s) into L2-normalized embeddings.

        Args:
            texts: Single string or list of strings to encode

        Returns:
            np.ndarray of shape (n, EMBEDDING_DIM) with L2-normalized embeddings
        """
        self._load_model()

        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]

        # Generate embeddings
        embeddings = self._model.encode(texts, convert_to_numpy=True)

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        return embeddings
