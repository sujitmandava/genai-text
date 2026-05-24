"""Test script for RAG foundation layer."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag.embeddings import EmbeddingModel
from rag.config import (
    EMBEDDING_DIM,
    PROJECT_ROOT,
    DATA_DIR,
    PASSAGES_PATH,
    INDEX_DIR,
    FAISS_INDEX_PATH,
    PASSAGES_METADATA_PATH,
)


def test_embedding_model():
    """Test EmbeddingModel.encode with acceptance criteria."""
    print("Testing EmbeddingModel...")

    model = EmbeddingModel()

    # Test single text input
    embeddings = model.encode(["test"])

    print(f"  Shape: {embeddings.shape}")
    print(f"  Expected: (1, {EMBEDDING_DIM})")
    assert embeddings.shape == (1, EMBEDDING_DIM), f"Expected shape (1, {EMBEDDING_DIM}), got {embeddings.shape}"

    # Verify L2 normalization
    import numpy as np
    norm = np.linalg.norm(embeddings[0])
    print(f"  L2 norm: {norm:.6f}")
    assert abs(norm - 1.0) < 1e-5, f"Expected L2 norm ~1.0, got {norm}"

    # Test multiple texts
    embeddings_multi = model.encode(["test1", "test2", "test3"])
    print(f"  Multi-text shape: {embeddings_multi.shape}")
    assert embeddings_multi.shape == (3, EMBEDDING_DIM), f"Expected shape (3, {EMBEDDING_DIM})"

    # Test single string (not list)
    embeddings_single = model.encode("single test")
    print(f"  Single string shape: {embeddings_single.shape}")
    assert embeddings_single.shape == (1, EMBEDDING_DIM), f"Expected shape (1, {EMBEDDING_DIM})"

    print("  ✓ All embedding tests passed")


def test_config_paths():
    """Test that config paths resolve correctly."""
    print("\nTesting config paths...")

    print(f"  PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"  DATA_DIR: {DATA_DIR}")
    print(f"  PASSAGES_PATH: {PASSAGES_PATH}")
    print(f"  INDEX_DIR: {INDEX_DIR}")
    print(f"  FAISS_INDEX_PATH: {FAISS_INDEX_PATH}")
    print(f"  PASSAGES_METADATA_PATH: {PASSAGES_METADATA_PATH}")

    # Verify PROJECT_ROOT is absolute
    assert PROJECT_ROOT.is_absolute(), "PROJECT_ROOT must be absolute"

    # Verify all paths are absolute
    assert DATA_DIR.is_absolute(), "DATA_DIR must be absolute"
    assert PASSAGES_PATH.is_absolute(), "PASSAGES_PATH must be absolute"
    assert INDEX_DIR.is_absolute(), "INDEX_DIR must be absolute"

    # Verify paths are under PROJECT_ROOT
    assert str(DATA_DIR).startswith(str(PROJECT_ROOT)), "DATA_DIR must be under PROJECT_ROOT"
    assert str(INDEX_DIR).startswith(str(PROJECT_ROOT)), "INDEX_DIR must be under PROJECT_ROOT"

    print("  ✓ All config path tests passed")


if __name__ == "__main__":
    print("=" * 60)
    print("RAG Foundation Layer Tests")
    print("=" * 60)

    test_config_paths()
    test_embedding_model()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
