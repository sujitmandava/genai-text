# NietzscheBot - Project Report

A conversational AI that channels Friedrich Nietzsche, combining a
fine-tuned GPT-2 language model with Retrieval-Augmented Generation
(RAG) over his philosophical corpus. The system features a Gradio web
interface with two modes: conversational Q&A (RAG-grounded) and open
prose continuation. This report covers the architecture, results,
difficulties, and future directions.

For setup and CLI flags see the codebase. This report focuses on
findings and analysis.

## 1. Dataset

**10 Nietzsche works** sourced from Project Gutenberg, totaling
**~29,000 lines** of processed training text:

| Work | Gutenberg ID |
|---|---:|
| Thus Spoke Zarathustra | 1998 |
| Beyond Good and Evil | 4363 |
| Ecce Homo | 7206 |
| The Antichrist | 19322 |
| The Genealogy of Morals | 52190 |
| Twilight of the Idols | 52263 |
| The Birth of Tragedy | 51356 |
| Human, All Too Human | 38145 |
| The Dawn of Day | 39955 |
| The Joyful Wisdom | 52881 |

**RAG Index**: 3,974 passages extracted from 4 core texts (Beyond Good
and Evil, Thus Spoke Zarathustra, Ecce Homo, The Antichrist), embedded
with sentence-transformers and indexed in FAISS for semantic retrieval.

**Train/Val Split**: 90/10 split for language model training (seed 42).

## 2. Architecture

### Language Model

**GPT-2 (124M parameters)** fine-tuned with HuggingFace Transformers:

- Causal language modeling objective
- 512 token context window
- Dynamic padding via `DataCollatorForLanguageModeling`
- AdamW optimizer with weight decay and gradient clipping

### RAG Pipeline

```
Query → EmbeddingModel → VectorStore (FAISS) → Top-k Passages → PromptBuilder → GPT-2 → Response
```

Components:
- **EmbeddingModel**: Sentence-transformers encoder for query/passage embeddings
- **VectorStore**: FAISS index wrapper with L2 similarity search
- **Retriever**: Combines embedding + search, returns `Passage` dataclass with metadata
- **NietzschePromptBuilder**: Constructs few-shot prompts with retrieved context

### Scripts

| Script | Purpose |
|---|---|
| `src/data/download.py` | Fetch texts from Project Gutenberg with retry logic |
| `src/data/preprocess.py` | Clean Gutenberg boilerplate, normalize text, create corpus |
| `src/data/extract_passages.py` | Split into ~200-word passages with source/section metadata |
| `src/rag/ingest.py` | Embed passages and build FAISS index |
| `src/training/train.py` | GPT-2 fine-tuning with Trainer API, checkpointing, TensorBoard |
| `src/training/evaluate.py` | Compute perplexity, generate samples |
| `app.py` | Gradio GUI with chat and prose generation tabs |

## 3. Training Configuration

| Hyperparameter | Value |
|---|---|
| Base model | gpt2 |
| Batch size | 4 |
| Gradient accumulation | 4 (effective batch 16) |
| Learning rate | 5e-5 |
| Epochs | 3 |
| Max sequence length | 512 |
| Warmup steps | 500 |
| Weight decay | 0.01 |
| Max grad norm | 1.0 |

Training uses mixed precision (fp16) when CUDA is available, with
checkpoints saved every 500 steps and best-model selection by
validation loss.

## 4. Results

### Quantitative Metrics

| Metric | Value |
|---|---|
| Validation perplexity | 44.1 |
| Validation set size | 85 samples |
| RAG index size | 3,974 passages |
| Retrieval latency | <100ms (CPU) |

Perplexity of 44 on a small validation set is reasonable for a
domain-specific fine-tune but leaves room for improvement.

### Qualitative Observations

**Generation samples** from `evaluation_report.json`:

| Prompt | Observation |
|---|---|
| "God is" | Heavy repetition ("who is like God" loops) |
| "The will to power" | Coherent philosophical tone, stays on theme |
| "What does not kill me" | Biblical/aphoristic cadence, some drift |
| "Morality is" | Repetitive structure, circular phrasing |
| "There are no facts," | Severe repetition collapse |

**Pattern**: The model captures Nietzsche's rhetorical style and
vocabulary but suffers from **repetition collapse** on many prompts.
The "will to power" and "overman" prompts fare best, likely due to
higher training frequency.

**RAG-grounded chat** (via `app.py`) produces more coherent responses
by conditioning on retrieved passages, but the generation still
occasionally loops or drifts from the grounded context.

## 5. Caveats

1. **No held-out test set evaluation.** Perplexity is computed on the
   validation split only; no blind test set exists.
2. **Limited RAG coverage.** Only 4 of 10 downloaded texts are
   indexed. Queries about The Birth of Tragedy or Human, All Too Human
   will not retrieve relevant passages.
3. **No human evaluation.** "Nietzsche-likeness" is judged informally;
   no systematic annotation study.
4. **Repetition not penalized.** Generation uses temperature/top-p
   sampling but no repetition penalty, leading to loops.
5. **Small base model.** GPT-2-small (124M) has limited capacity;
   larger variants may help.

## 6. Future Work

### A. Fix repetition collapse

The dominant failure mode is looping. Immediate fixes:

1. **Add repetition penalty** to generation config (e.g., `repetition_penalty=1.2`).
2. **Increase temperature** on short prompts where the model is most
   confident and most prone to loops.
3. **Try nucleus sampling with lower top-p** (e.g., 0.8 instead of 0.9).

### B. Improve generation quality

4. **Fine-tune GPT-2-medium or GPT-2-large.** More parameters = better
   long-range coherence.
5. **Increase training epochs.** Current 3 epochs may underfit;
   perplexity curve suggests continued descent.
6. **Add LPIPS or style-consistency loss.** Not directly applicable to
   text, but a style classifier head could encourage Nietzschean tone.

### C. Expand RAG coverage

7. **Index all 10 texts.** Current pipeline only covers 4; extending
   to full corpus improves retrieval breadth.
8. **Chunk overlap.** Current ~200-word passages have no overlap;
   sliding window would improve retrieval recall.
9. **Hybrid retrieval.** Add BM25 alongside dense retrieval for
   keyword-heavy queries.

### D. Evaluation

10. **Add a held-out test split** and report final perplexity.
11. **Human evaluation.** A-B preference study for Nietzsche-likeness.
12. **Retrieval quality metrics.** Measure MRR/Recall@k on manually
    labeled query-passage pairs.

## 7. Difficulties Faced

1. **Gutenberg boilerplate.** Raw texts include lengthy legal headers
   and footers that required custom stripping logic. Some editions had
   inconsistent formatting.

2. **Repetition collapse.** The fine-tuned model quickly falls into
   loops, especially on short or generic prompts. This is a known GPT-2
   failure mode exacerbated by small data and low temperature.

3. **Embedding model selection.** Initial experiments with smaller
   embedding models produced poor retrieval quality; switching to
   sentence-transformers improved semantic matching.

4. **Prompt engineering for RAG.** The prompt format significantly
   affects response quality. Early templates produced answers that
   ignored retrieved context; the current `NietzschePromptBuilder`
   explicitly marks context and instructs the model to respond "as
   Nietzsche."

5. **Limited compute.** Training on a MacBook Air limits batch size
   and epoch count. HPC/Colab runs helped but introduced quota and
   session timeout issues.

## 8. Conclusion

NietzscheBot demonstrates a working pipeline for domain-specific
conversational AI: data acquisition, preprocessing, fine-tuning, RAG
indexing, and a polished Gradio interface. The core system works —
the model speaks in Nietzsche's voice and the retriever grounds
responses in his actual writings.

The main gap is **generation quality**: repetition collapse undermines
coherence, and perplexity has room to improve. The recommended next
step is to add a repetition penalty and increase training epochs, then
expand RAG coverage to all 10 texts. With those changes, the system
should produce fluent, grounded Nietzsche-style responses.
