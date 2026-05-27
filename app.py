"""Gradio GUI for NietzscheBot: conversational RAG + prose generation."""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.rag.retrieval import Retriever
from src.rag.prompt_builder import NietzschePromptBuilder

MODEL_PATH = "models/nietzsche-bot/final"

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


def chat_with_nietzsche(question: str, num_passages: int, max_length: int, temperature: float) -> tuple[str, str]:
    """Conversational Q&A with Nietzsche using RAG."""
    if not question.strip():
        return "Please ask a question.", ""

    if not rag_available:
        return "RAG index not available. Run `python -m src.rag.ingest` first.", ""

    passages = retriever.retrieve(question, k=num_passages)
    prompt = prompt_builder.build(question, passages)

    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=0.9,
            top_k=50,
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

    context_display = "\n\n".join(
        f"**{p.source}** ({p.section}):\n> {p.text[:200]}..." if len(p.text) > 200 else f"**{p.source}** ({p.section}):\n> {p.text}"
        for p in passages
    )

    return response, context_display


def generate_prose(prompt: str, max_length: int, temperature: float, top_p: float, top_k: int) -> str:
    """Generate prose continuation in Nietzsche's style."""
    if not prompt.strip():
        return "Please enter a prompt."

    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=int(top_k),
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated


with gr.Blocks(title="NietzscheBot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# NietzscheBot")
    gr.Markdown("GPT-2 fine-tuned on Nietzsche's works. Ask questions or continue his prose.")

    with gr.Tabs():
        with gr.Tab("Chat with Nietzsche"):
            gr.Markdown("Ask Nietzsche a question. The model retrieves relevant passages from his works to inform the response.")

            with gr.Row():
                with gr.Column(scale=2):
                    chat_input = gr.Textbox(
                        label="Your Question",
                        placeholder="What is the will to power?",
                        lines=2
                    )
                    with gr.Row():
                        chat_passages = gr.Slider(1, 5, value=3, step=1, label="Retrieved Passages")
                        chat_length = gr.Slider(50, 300, value=150, step=10, label="Max Response Length")
                        chat_temp = gr.Slider(0.3, 1.5, value=0.8, step=0.1, label="Temperature")
                    chat_btn = gr.Button("Ask Nietzsche", variant="primary")

                with gr.Column(scale=2):
                    chat_output = gr.Textbox(label="Nietzsche's Response", lines=8)

            with gr.Accordion("Retrieved Context", open=False):
                context_output = gr.Markdown(label="Passages used for context")

            chat_btn.click(
                chat_with_nietzsche,
                inputs=[chat_input, chat_passages, chat_length, chat_temp],
                outputs=[chat_output, context_output]
            )

            gr.Examples(
                examples=[
                    ["What is the will to power?", 3, 150, 0.8],
                    ["How should one live?", 3, 200, 0.7],
                    ["What do you think of Christianity?", 3, 150, 0.8],
                    ["Is God dead?", 3, 150, 0.9],
                ],
                inputs=[chat_input, chat_passages, chat_length, chat_temp],
            )

        with gr.Tab("Continue Writing"):
            gr.Markdown("Enter a fragment and let Nietzsche continue writing in his philosophical style.")

            prose_input = gr.Textbox(
                label="Prompt",
                placeholder="The will to power...",
                lines=3
            )
            with gr.Row():
                prose_length = gr.Slider(50, 500, value=150, step=10, label="Max Length")
                prose_temp = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature")
                prose_top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top P")
                prose_top_k = gr.Slider(1, 100, value=50, step=1, label="Top K")

            prose_btn = gr.Button("Generate", variant="primary")
            prose_output = gr.Textbox(label="Generated Text", lines=10)

            prose_btn.click(
                generate_prose,
                inputs=[prose_input, prose_length, prose_temp, prose_top_p, prose_top_k],
                outputs=prose_output
            )

            gr.Examples(
                examples=[
                    ["The will to power", 150, 0.8, 0.9, 50],
                    ["Man is something that shall be overcome", 200, 0.7, 0.9, 50],
                    ["God is dead", 150, 0.8, 0.9, 50],
                    ["What does not kill me", 100, 0.9, 0.9, 50],
                ],
                inputs=[prose_input, prose_length, prose_temp, prose_top_p, prose_top_k],
            )

if __name__ == "__main__":
    demo.launch()
