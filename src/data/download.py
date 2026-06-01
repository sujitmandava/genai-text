from pathlib import Path
from time import sleep
from typing import Dict
import argparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BOOKS: Dict[str, int] = {
    "Thus Spoke Zarathustra": 1998,
    "Beyond Good and Evil": 4363,
    "Ecce Homo": 7206,
    "The Antichrist": 19322,
    "The Genealogy of Morals": 52190,
    "Twilight of the Idols": 52263,
    "The Birth of Tragedy": 51356,
    "Human All Too Human": 38145,
    "The Dawn of Day": 39955,
    "The Joyful Wisdom": 52881,
    "The Will to Power Vol 1": 52915,
    "The Will to Power Vol 2": 52914,
    "Early Greek Philosophy": 51548,
    "Thoughts Out of Season Part 1": 37841,
    "Thoughts Out of Season Part 2": 38226,
}


def _get_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def download_all_texts(force: bool = False) -> None:
    raw_dir = Path(__file__).parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    session = _get_session()

    for title, book_id in BOOKS.items():
        filepath = raw_dir / f"{title}.txt"

        if filepath.exists() and not force:
            print(f"Skipping {title} (already exists)")
            continue

        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        print(f"Downloading {title} from {url}...")

        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
            filepath.write_text(response.text, encoding="utf-8")
            print(f"Saved {title} ({len(response.text)} chars)")
            sleep(2)
        except Exception as e:
            print(f"Error downloading {title}: {e}")
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Nietzsche texts")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist",
    )
    args = parser.parse_args()
    download_all_texts(force=args.force)


if __name__ == "__main__":
    main()
