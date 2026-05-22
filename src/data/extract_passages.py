import json
import re
import sys
from pathlib import Path
from typing import Dict, List


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _extract_from_zarathustra(text: str, title: str) -> List[Dict[str, str]]:
    passages = []
    sections = re.split(r"\n+(\d+\.)\n+", text)

    current_section = "Introduction"
    for i, part in enumerate(sections):
        if re.match(r"\d+\.", part):
            current_section = f"Section {part}"
        elif i > 0:
            paragraphs = part.split("\n\n")
            for para in paragraphs:
                para = para.strip()
                if 50 <= len(para) <= 2000:
                    passages.append({
                        "text": para,
                        "source": title,
                        "section": current_section,
                    })

    return passages


def _extract_aphorisms(text: str, title: str) -> List[Dict[str, str]]:
    passages = []
    sections = re.split(r"\n+(\d+\.)\s*", text)

    current_section_num = "0"
    for i, part in enumerate(sections):
        if re.match(r"\d+\.$", part):
            current_section_num = part.strip()
        elif i > 0:
            paragraphs = part.split("\n\n")
            for para in paragraphs:
                para = para.strip()
                if 50 <= len(para) <= 2000:
                    passages.append({
                        "text": para,
                        "source": title,
                        "section": f"Aphorism {current_section_num}",
                    })

    return passages


def _extract_paragraphs(text: str, title: str) -> List[Dict[str, str]]:
    passages = []
    paragraphs = text.split("\n\n")

    for idx, para in enumerate(paragraphs):
        para = para.strip()
        if 50 <= len(para) <= 2000:
            passages.append({
                "text": para,
                "source": title,
                "section": f"Paragraph {idx + 1}",
            })

    return passages


def extract_all_passages() -> None:
    raw_dir = Path(__file__).parent / "raw"
    processed_dir = Path(__file__).parent / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    all_passages = []

    for txt_file in sorted(raw_dir.glob("*.txt")):
        title = txt_file.stem
        print(f"Extracting passages from {title}...")

        from src.data.clean_for_training import strip_gutenberg_boilerplate, normalize_text

        raw_text = txt_file.read_text(encoding="utf-8")
        clean_text = strip_gutenberg_boilerplate(raw_text)
        clean_text = normalize_text(clean_text)

        if "Zarathustra" in title:
            passages = _extract_from_zarathustra(clean_text, title)
        elif title in ["Beyond Good and Evil", "The Gay Science"]:
            passages = _extract_aphorisms(clean_text, title)
        else:
            passages = _extract_paragraphs(clean_text, title)

        all_passages.extend(passages)
        print(f"  Extracted {len(passages)} passages")

    output_path = processed_dir / "passages.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for passage in all_passages:
            f.write(json.dumps(passage, ensure_ascii=False) + "\n")

    print(f"Total passages extracted: {len(all_passages)}")


def main() -> None:
    extract_all_passages()


if __name__ == "__main__":
    main()
