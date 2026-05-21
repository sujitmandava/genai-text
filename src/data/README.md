# Nietzsche Data Pipeline

Automated pipeline to download, clean, and extract passages from Nietzsche's works for training and RAG.

## Books Included

1. Thus Spoke Zarathustra (Project Gutenberg #1998)
2. Beyond Good and Evil (Project Gutenberg #4363)
3. Ecce Homo (Project Gutenberg #7206)
4. The Antichrist (Project Gutenberg #19322)

## Usage

```bash
# Run full pipeline
python -m src.data.prepare

# Force re-download
python -m src.data.prepare --force
```

## Output

- `raw/` - Original texts from Project Gutenberg
- `processed/training_corpus.txt` - Concatenated clean text for LLM training (~1.5MB)
- `processed/passages.jsonl` - Discrete passages for RAG (3900+ passages)

## Passage Format

Each line in `passages.jsonl`:
```json
{
  "text": "passage content",
  "source": "Book Title",
  "section": "Aphorism X" or "Section Y"
}
```

Passage length: 50-2000 characters each.
