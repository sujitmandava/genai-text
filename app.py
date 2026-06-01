"""Gradio GUI for NietzscheBot: conversational RAG + prose generation."""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.rag.retrieval import Retriever
from src.rag.prompt_builder import NietzschePromptBuilder
from src.app_config import (
    MODEL_PATH,
    CHAT_NUM_PASSAGES,
    CHAT_MAX_NEW_TOKENS,
    CHAT_TEMPERATURE,
    CHAT_TOP_P,
    CHAT_TOP_K,
    PROSE_MAX_LENGTH,
    PROSE_TEMPERATURE,
    PROSE_TOP_P,
    PROSE_TOP_K,
)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
print(f"Model loaded on {device}")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading RAG retriever...")
try:
    retriever = Retriever()
    prompt_builder = NietzschePromptBuilder()
    rag_available = True
    print(f"RAG loaded: {len(retriever.passages_metadata)} passages indexed")
except FileNotFoundError as e:
    print(f"RAG not available: {e}")
    retriever = None
    prompt_builder = None
    rag_available = False


def chat_with_nietzsche(message: str, history: list) -> str:
    """Conversational Q&A with Nietzsche using RAG.

    Args:
        message: User's current message.
        history: Chat history (list of [user_msg, assistant_msg] pairs).

    Returns:
        Nietzsche's response string.
    """
    if not message.strip():
        return "Please ask a question."

    if not rag_available:
        return "RAG index not available. Run `python -m src.rag.ingest` first."

    passages = retriever.retrieve(message, k=CHAT_NUM_PASSAGES)
    prompt = prompt_builder.build(message, passages)

    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=CHAT_MAX_NEW_TOKENS,
            temperature=CHAT_TEMPERATURE,
            top_p=CHAT_TOP_P,
            top_k=CHAT_TOP_K,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    response_marker = "Nietzsche:"
    if response_marker in full_output:
        response = full_output.split(response_marker)[-1].strip()
    else:
        response = full_output[len(prompt):].strip()

    return response


def generate_prose(prompt: str) -> str:
    """Generate prose continuation in Nietzsche's style."""
    if not prompt.strip():
        return "Please enter a prompt."

    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=PROSE_MAX_LENGTH,
            temperature=PROSE_TEMPERATURE,
            top_p=PROSE_TOP_P,
            top_k=PROSE_TOP_K,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated


custom_css = """
/* Nietzsche aesthetic: elegant serif, moody palette */
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&display=swap');

:root {
    --abyss-dark: #1a1512;
    --parchment: #f4e8d0;
    --gold-accent: #d4a574;
    --deep-shadow: #2d2419;
}

body, .gradio-container {
    font-family: 'Cormorant Garamond', serif !important;
    background: linear-gradient(135deg, var(--abyss-dark) 0%, var(--deep-shadow) 100%);
    color: var(--parchment) !important;
}

h1, h2, h3, .markdown h1, .markdown h2, .markdown h3 {
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 600;
    color: var(--gold-accent) !important;
    letter-spacing: 0.05em;
}

.markdown p, .markdown em {
    color: var(--parchment) !important;
    font-style: italic;
    font-size: 1.1em;
}

button {
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    background: var(--gold-accent) !important;
    color: var(--abyss-dark) !important;
    border: none !important;
}

button:hover {
    background: var(--parchment) !important;
    color: var(--abyss-dark) !important;
}

.textbox, textarea, input {
    background: var(--deep-shadow) !important;
    color: var(--parchment) !important;
    border: 1px solid var(--gold-accent) !important;
    font-family: 'Cormorant Garamond', serif !important;
}

.tab-nav button {
    color: var(--gold-accent) !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
}

.tab-nav button.selected {
    border-bottom: 2px solid var(--gold-accent) !important;
}
"""

with gr.Blocks(title="Speak with Nietzsche", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# Thus Spoke the Philosopher")
    gr.Markdown("*And if you gaze long into an abyss, the abyss also gazes into you.*")

    with gr.Tabs():
        with gr.Tab("Dialogue"):
            gr.Markdown("Pose your question. The philosopher draws from his corpus to respond.")

            chat_interface = gr.ChatInterface(
                fn=chat_with_nietzsche,
                textbox=gr.Textbox(
                    placeholder="Speak your question into the abyss...",
                    container=False,
                    scale=7,
                    submit_btn="Summon Response"
                ),
                examples=[
                    "What is the will to power?",
                    "How should one live?",
                    "What do you think of Christianity?",
                    "Is God dead?",
                    "What is the Ubermensch?",
                ],
            )

        with gr.Tab("Continue Writing"):
            gr.Markdown("Enter a fragment and let Nietzsche continue the thought in his philosophical voice.")

            prose_input = gr.Textbox(
                label="Opening Fragment",
                placeholder="The will to power...",
                lines=3
            )

            prose_btn = gr.Button("Continue the Thought", variant="primary")
            prose_output = gr.Textbox(label="Generated Prose", lines=10)

            prose_btn.click(
                generate_prose,
                inputs=prose_input,
                outputs=prose_output
            )

            gr.Examples(
                examples=[
                    "The will to power",
                    "Man is something that shall be overcome",
                    "God is dead",
                    "What does not kill me",
                ],
                inputs=prose_input,
            )

if __name__ == "__main__":
    demo.launch()
