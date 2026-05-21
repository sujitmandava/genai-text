#!/usr/bin/env python3
import json
from pathlib import Path

def test_pipeline():
    base_dir = Path(__file__).parent / "src" / "data"

    raw_dir = base_dir / "raw"
    processed_dir = base_dir / "processed"

    assert raw_dir.exists(), "Raw directory missing"
    assert processed_dir.exists(), "Processed directory missing"

    raw_files = list(raw_dir.glob("*.txt"))
    assert len(raw_files) == 4, f"Expected 4 raw files, found {len(raw_files)}"

    corpus_path = processed_dir / "training_corpus.txt"
    assert corpus_path.exists(), "Training corpus missing"

    corpus = corpus_path.read_text(encoding="utf-8")
    assert len(corpus) > 1_000_000, "Corpus too small"
    assert "Beyond Good and Evil" in corpus, "Missing book in corpus"
    assert "Thus Spoke Zarathustra" in corpus, "Missing book in corpus"

    passages_path = processed_dir / "passages.jsonl"
    assert passages_path.exists(), "Passages file missing"

    with passages_path.open("r", encoding="utf-8") as f:
        passages = [json.loads(line) for line in f]

    assert len(passages) >= 200, f"Expected 200+ passages, found {len(passages)}"

    for passage in passages[:10]:
        assert "text" in passage, "Missing 'text' field"
        assert "source" in passage, "Missing 'source' field"
        assert "section" in passage, "Missing 'section' field"
        assert 50 <= len(passage["text"]) <= 2000, "Passage length out of range"

    print("All tests passed!")
    print(f"  Raw files: {len(raw_files)}")
    print(f"  Corpus size: {len(corpus):,} chars")
    print(f"  Passages: {len(passages)}")

if __name__ == "__main__":
    test_pipeline()
