"""Configuration for Gradio app: model path and generation hyperparameters."""

# Model path
MODEL_PATH = "models/nietzsche-bot/final"

# Chat (conversational RAG) parameters
CHAT_NUM_PASSAGES = 3
CHAT_MAX_NEW_TOKENS = 150
CHAT_TEMPERATURE = 0.8
CHAT_TOP_P = 0.9
CHAT_TOP_K = 50

# Prose generation parameters
PROSE_MAX_LENGTH = 150
PROSE_TEMPERATURE = 0.8
PROSE_TOP_P = 0.9
PROSE_TOP_K = 50
