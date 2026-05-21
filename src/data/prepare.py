import argparse
from pathlib import Path

from .download import download_all_texts
from .clean_for_training import create_training_corpus
from .extract_passages import extract_all_passages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Nietzsche texts for training and RAG"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they exist",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("NIETZSCHE DATA PIPELINE")
    print("=" * 60)

    print("\n[1/3] Downloading texts from Project Gutenberg...")
    download_all_texts(force=args.force)

    print("\n[2/3] Creating training corpus...")
    create_training_corpus()

    print("\n[3/3] Extracting passages for RAG...")
    extract_all_passages()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE - SUMMARY")
    print("=" * 60)

    raw_dir = Path(__file__).parent / "raw"
    processed_dir = Path(__file__).parent / "processed"

    raw_files = list(raw_dir.glob("*.txt"))
    print(f"\nRaw texts: {len(raw_files)} books")
    for f in sorted(raw_files):
        size = f.stat().st_size
        print(f"  - {f.name}: {size:,} bytes")

    corpus_path = processed_dir / "training_corpus.txt"
    if corpus_path.exists():
        corpus_size = corpus_path.stat().st_size
        print(f"\nTraining corpus: {corpus_size:,} bytes")

    passages_path = processed_dir / "passages.jsonl"
    if passages_path.exists():
        with passages_path.open("r", encoding="utf-8") as f:
            num_passages = sum(1 for _ in f)
        passages_size = passages_path.stat().st_size
        print(f"RAG passages: {num_passages} passages ({passages_size:,} bytes)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
