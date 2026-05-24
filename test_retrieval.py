"""Test script for Retrieval layer."""
import sys
from pathlib import Path
import tempfile
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from rag.retrieval import Retriever, Passage
from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore
from rag.config import EMBEDDING_DIM


def test_passage_dataclass():
    """Test Passage dataclass creation."""
    print("Testing Passage dataclass...")

    passage = Passage(
        text="God is dead",
        source="Thus Spoke Zarathustra",
        section="Prologue:2",
        score=0.42,
    )

    assert passage.text == "God is dead"
    assert passage.source == "Thus Spoke Zarathustra"
    assert passage.section == "Prologue:2"
    assert passage.score == 0.42
    print("  Passage dataclass works ✓")


def test_retriever_with_mock_data():
    """Test Retriever with a small mock index and metadata."""
    print("\nTesting Retriever with mock data...")

    with tempfile.TemporaryDirectory() as tmpdir:
        test_index_path = Path(tmpdir) / "test.index"
        test_metadata_path = Path(tmpdir) / "test_meta.json"

        # Create mock passages
        mock_passages = [
            {
                "text": "What does not kill me makes me stronger.",
                "source": "Twilight of the Idols",
                "section": "Maxims:8",
            },
            {
                "text": "He who has a why to live can bear almost any how.",
                "source": "Twilight of the Idols",
                "section": "Maxims:12",
            },
            {
                "text": "That which is done out of love always occurs beyond good and evil.",
                "source": "Beyond Good and Evil",
                "section": "Aphorism:153",
            },
        ]

        # Generate embeddings for mock passages
        embedding_model = EmbeddingModel()
        texts = [p["text"] for p in mock_passages]
        embeddings = embedding_model.encode(texts)

        # Create and save vector store
        store = VectorStore(dimension=EMBEDDING_DIM)
        store.add(embeddings)
        store.save(test_index_path)
        print(f"  Created mock index with {len(store)} passages ✓")

        # Save metadata
        with open(test_metadata_path, "w", encoding="utf-8") as f:
            json.dump(mock_passages, f, indent=2)
        print(f"  Saved mock metadata ✓")

        # Initialize retriever
        retriever = Retriever(
            index_path=test_index_path, metadata_path=test_metadata_path
        )
        print(f"  Loaded retriever with {len(retriever.vector_store)} passages ✓")

        # Test retrieval with exact match query
        query = "strength"
        results = retriever.retrieve(query, k=2)

        # Verify results structure
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        assert all(isinstance(p, Passage) for p in results), "All results should be Passage objects"
        print(f"  Retrieved {len(results)} passages ✓")

        # Verify result structure
        first = results[0]
        assert isinstance(first.text, str) and len(first.text) > 0
        assert isinstance(first.source, str) and len(first.source) > 0
        assert isinstance(first.section, str) and len(first.section) > 0
        assert isinstance(first.score, float) and first.score >= 0
        print(f"  First result structure valid ✓")

        # Verify ranking (lower score = better)
        for i in range(len(results) - 1):
            assert results[i].score <= results[i + 1].score, "Results should be ranked by score (ascending)"
        print(f"  Results ranked by score (ascending) ✓")

        # Display results for manual inspection
        print("\n  Sample query results for 'strength':")
        for i, passage in enumerate(results, 1):
            print(f"    {i}. [{passage.source} - {passage.section}] (score={passage.score:.4f})")
            print(f"       {passage.text[:60]}...")


def test_retriever_k_parameter():
    """Test that Retriever respects k parameter."""
    print("\nTesting Retriever k parameter...")

    with tempfile.TemporaryDirectory() as tmpdir:
        test_index_path = Path(tmpdir) / "test.index"
        test_metadata_path = Path(tmpdir) / "test_meta.json"

        # Create 5 mock passages
        mock_passages = [
            {"text": f"This is passage number {i}.", "source": "Test Book", "section": f"{i}"}
            for i in range(5)
        ]

        # Generate embeddings
        embedding_model = EmbeddingModel()
        texts = [p["text"] for p in mock_passages]
        embeddings = embedding_model.encode(texts)

        # Save index and metadata
        store = VectorStore(dimension=EMBEDDING_DIM)
        store.add(embeddings)
        store.save(test_index_path)

        with open(test_metadata_path, "w", encoding="utf-8") as f:
            json.dump(mock_passages, f)

        # Test different k values
        retriever = Retriever(index_path=test_index_path, metadata_path=test_metadata_path)

        for k in [1, 3, 5]:
            results = retriever.retrieve("passage", k=k)
            assert len(results) == k, f"Expected {k} results, got {len(results)}"
            print(f"  k={k}: returned {len(results)} results ✓")


def test_retriever_error_handling():
    """Test Retriever error handling for missing files."""
    print("\nTesting Retriever error handling...")

    # Test missing index file
    with tempfile.TemporaryDirectory() as tmpdir:
        nonexistent_index = Path(tmpdir) / "missing.index"
        fake_metadata = Path(tmpdir) / "meta.json"

        # Create metadata but not index
        with open(fake_metadata, "w") as f:
            json.dump([], f)

        try:
            Retriever(index_path=nonexistent_index, metadata_path=fake_metadata)
            assert False, "Should raise FileNotFoundError for missing index"
        except FileNotFoundError:
            print("  Missing index rejected ✓")

    # Test missing metadata file
    with tempfile.TemporaryDirectory() as tmpdir:
        test_index_path = Path(tmpdir) / "test.index"
        nonexistent_meta = Path(tmpdir) / "missing_meta.json"

        # Create minimal index
        store = VectorStore(dimension=EMBEDDING_DIM)
        embedding_model = EmbeddingModel()
        embeddings = embedding_model.encode(["test"])
        store.add(embeddings)
        store.save(test_index_path)

        try:
            Retriever(index_path=test_index_path, metadata_path=nonexistent_meta)
            assert False, "Should raise FileNotFoundError for missing metadata"
        except FileNotFoundError:
            print("  Missing metadata rejected ✓")

    # Test mismatched index and metadata counts
    with tempfile.TemporaryDirectory() as tmpdir:
        test_index_path = Path(tmpdir) / "test.index"
        test_metadata_path = Path(tmpdir) / "meta.json"

        # Create index with 3 passages
        store = VectorStore(dimension=EMBEDDING_DIM)
        embedding_model = EmbeddingModel()
        embeddings = embedding_model.encode(["one", "two", "three"])
        store.add(embeddings)
        store.save(test_index_path)

        # Create metadata with only 2 passages
        metadata = [
            {"text": "one", "source": "Book", "section": "1"},
            {"text": "two", "source": "Book", "section": "2"},
        ]
        with open(test_metadata_path, "w") as f:
            json.dump(metadata, f)

        try:
            Retriever(index_path=test_index_path, metadata_path=test_metadata_path)
            assert False, "Should raise ValueError for mismatched counts"
        except ValueError as e:
            assert "does not match" in str(e)
            print("  Mismatched index/metadata counts rejected ✓")


if __name__ == "__main__":
    print("=" * 60)
    print("Retrieval Layer Tests")
    print("=" * 60)

    test_passage_dataclass()
    test_retriever_with_mock_data()
    test_retriever_k_parameter()
    test_retriever_error_handling()

    print("\n" + "=" * 60)
    print("All Retrieval tests passed!")
    print("=" * 60)
