"""Test script for VectorStore wrapper."""
import sys
from pathlib import Path
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from rag.vector_store import VectorStore
from rag.config import EMBEDDING_DIM


def test_basic_operations():
    """Test basic VectorStore operations."""
    print("Testing VectorStore basic operations...")

    store = VectorStore(dimension=EMBEDDING_DIM)

    # Test initial length
    assert len(store) == 0, "New store should be empty"
    print(f"  Initial length: {len(store)} ✓")

    # Create 10 random vectors
    np.random.seed(42)
    embeddings = np.random.randn(10, EMBEDDING_DIM).astype(np.float32)

    # Add vectors
    store.add(embeddings)
    assert len(store) == 10, f"Expected 10 vectors, got {len(store)}"
    print(f"  Added 10 vectors, length: {len(store)} ✓")

    # Test search with single query
    query = embeddings[0]  # Use first vector as query
    distances, indices = store.search(query, k=5)

    print(f"  Search result shape - distances: {distances.shape}, indices: {indices.shape}")
    assert distances.shape == (1, 5), f"Expected distances shape (1, 5), got {distances.shape}"
    assert indices.shape == (1, 5), f"Expected indices shape (1, 5), got {indices.shape}"

    # First result should be the query itself (distance ~0)
    assert indices[0, 0] == 0, f"Expected first index to be 0, got {indices[0, 0]}"
    assert distances[0, 0] < 1e-5, f"Expected first distance ~0, got {distances[0, 0]}"
    print(f"  First result is query itself (distance={distances[0, 0]:.6f}) ✓")

    # Test search with 2D query
    query_2d = embeddings[:2]  # First two vectors
    distances_2d, indices_2d = store.search(query_2d, k=3)
    assert distances_2d.shape == (2, 3), f"Expected shape (2, 3), got {distances_2d.shape}"
    assert indices_2d.shape == (2, 3), f"Expected shape (2, 3), got {indices_2d.shape}"
    print(f"  2D query search works ✓")

    print("  ✓ All basic operation tests passed")
    return store, embeddings


def test_save_load():
    """Test save/load roundtrip."""
    print("\nTesting VectorStore save/load...")

    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as tmpdir:
        test_index_path = Path(tmpdir) / "test.index"

        # Create and populate store
        store = VectorStore(dimension=EMBEDDING_DIM)
        np.random.seed(42)
        embeddings = np.random.randn(10, EMBEDDING_DIM).astype(np.float32)
        store.add(embeddings)

        print(f"  Original store length: {len(store)}")

        # Save to disk
        store.save(test_index_path)
        assert test_index_path.exists(), f"Index file not created at {test_index_path}"
        print(f"  Saved to {test_index_path} ✓")

        # Load from disk
        loaded_store = VectorStore.load(test_index_path)
        assert len(loaded_store) == 10, f"Expected 10 vectors after load, got {len(loaded_store)}"
        print(f"  Loaded store length: {len(loaded_store)} ✓")

        # Verify search results are identical
        query = embeddings[0]
        orig_distances, orig_indices = store.search(query, k=5)
        loaded_distances, loaded_indices = loaded_store.search(query, k=5)

        assert np.allclose(orig_distances, loaded_distances), "Distances differ after load"
        assert np.array_equal(orig_indices, loaded_indices), "Indices differ after load"
        print(f"  Search results identical after load ✓")

    print("  ✓ All save/load tests passed")


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("\nTesting VectorStore error handling...")

    store = VectorStore(dimension=EMBEDDING_DIM)

    # Test wrong dimension
    try:
        wrong_dim = np.random.randn(5, 128).astype(np.float32)  # Wrong dimension
        store.add(wrong_dim)
        assert False, "Should raise ValueError for wrong dimension"
    except ValueError as e:
        assert "dimension" in str(e).lower()
        print(f"  Wrong dimension rejected ✓")

    # Test 1D array (should fail)
    try:
        wrong_shape = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        store.add(wrong_shape)
        assert False, "Should raise ValueError for 1D array"
    except ValueError as e:
        assert "2D" in str(e)
        print(f"  1D array rejected ✓")

    # Test load non-existent file
    try:
        VectorStore.load(Path("/nonexistent/path/index.faiss"))
        assert False, "Should raise FileNotFoundError"
    except FileNotFoundError:
        print(f"  Non-existent file rejected ✓")

    print("  ✓ All error handling tests passed")


def test_k_results():
    """Test that search returns correct number of results."""
    print("\nTesting k parameter in search...")

    store = VectorStore(dimension=EMBEDDING_DIM)
    np.random.seed(42)
    embeddings = np.random.randn(10, EMBEDDING_DIM).astype(np.float32)
    store.add(embeddings)

    query = embeddings[0]

    # Test different k values
    for k in [1, 3, 5, 10]:
        distances, indices = store.search(query, k=k)
        assert distances.shape == (1, k), f"Expected {k} results, got {distances.shape}"
        assert indices.shape == (1, k), f"Expected {k} results, got {indices.shape}"
        print(f"  k={k}: returned {distances.shape[1]} results ✓")

    print("  ✓ All k parameter tests passed")


if __name__ == "__main__":
    print("=" * 60)
    print("VectorStore Tests")
    print("=" * 60)

    test_basic_operations()
    test_save_load()
    test_error_handling()
    test_k_results()

    print("\n" + "=" * 60)
    print("All VectorStore tests passed!")
    print("=" * 60)
