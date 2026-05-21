import re
from pathlib import Path


def strip_gutenberg_boilerplate(text: str) -> str:
    start_marker = re.search(
        r"\*\*\* START OF (THIS|THE) PROJECT GUTENBERG EBOOK .+ \*\*\*",
        text,
        re.IGNORECASE,
    )
    end_marker = re.search(
        r"\*\*\* END OF (THIS|THE) PROJECT GUTENBERG EBOOK .+ \*\*\*",
        text,
        re.IGNORECASE,
    )

    if start_marker and end_marker:
        return text[start_marker.end() : end_marker.start()].strip()
    return text


def normalize_text(text: str) -> str:
    text = re.sub(r"\[Pg \d+\]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def create_training_corpus() -> None:
    raw_dir = Path(__file__).parent / "raw"
    processed_dir = Path(__file__).parent / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    corpus_parts = []

    for txt_file in sorted(raw_dir.glob("*.txt")):
        print(f"Processing {txt_file.name}...")
        raw_text = txt_file.read_text(encoding="utf-8")
        clean_text = strip_gutenberg_boilerplate(raw_text)
        clean_text = normalize_text(clean_text)

        title = txt_file.stem
        corpus_parts.append(f"--- [{title}] ---\n\n{clean_text}")

    output_path = processed_dir / "training_corpus.txt"
    full_corpus = "\n\n".join(corpus_parts)
    output_path.write_text(full_corpus, encoding="utf-8")

    print(f"Training corpus created: {len(full_corpus)} chars")
