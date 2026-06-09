"""Gradio GUI for NietzscheBot: conversational RAG with Gemma-LoRA."""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.rag.retrieval import Retriever
from src.rag.prompt_builder import NietzschePromptBuilder
from src.app_config import (
    BASE_MODEL,
    LORA_PATH,
    CHAT_NUM_PASSAGES,
    CHAT_MAX_NEW_TOKENS,
)

print("Loading Gemma base model with LoRA adapter...")
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if device == "cuda":
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        device_map="auto",
    )
else:
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if device == "mps" else torch.float32,
    )
    base_model.to(device)

model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()
print(f"Gemma-LoRA loaded on {device}")

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


def chat_with_nietzsche(
    message: str,
    history: list,
    style: str,
    temperature: float,
    top_p: float,
    top_k: int,
) -> str:
    """Conversational Q&A with Nietzsche using RAG."""
    if not message.strip():
        return "Please ask a question."

    if not rag_available:
        return "RAG index not available. Run `python -m src.rag.ingest` first."

    passages = retriever.retrieve(message, k=CHAT_NUM_PASSAGES)
    prompt = prompt_builder.build(message, passages, style=style)

    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=CHAT_MAX_NEW_TOKENS,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
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

    gr.Markdown("Pose your question. The philosopher draws from his corpus to respond.")

    with gr.Accordion("Advanced Settings", open=False):
        style_dropdown = gr.Dropdown(
            choices=["Provocative", "Philosophical"],
            value="Provocative",
            label="Style",
        )
        temperature_slider = gr.Slider(
            minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="Temperature"
        )
        top_p_slider = gr.Slider(
            minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-P"
        )
        top_k_slider = gr.Slider(
            minimum=1, maximum=100, value=50, step=1, label="Top-K"
        )

    chat_interface = gr.ChatInterface(
        fn=chat_with_nietzsche,
        additional_inputs=[style_dropdown, temperature_slider, top_p_slider, top_k_slider],
        textbox=gr.Textbox(
            placeholder="Speak your question into the abyss...",
            container=False,
            scale=7,
        ),
        examples=[
            ["What is the will to power?"],
            ["How should one live?"],
            ["What do you think of Christianity?"],
            ["Is God dead?"],
            ["What is the Ubermensch?"],
        ],
    )

if __name__ == "__main__":
    demo.launch()
