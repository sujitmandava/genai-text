"""Prompt builder for Nietzsche persona with retrieved context."""

from src.rag.retrieval import Passage


class NietzschePromptBuilder:
    """Builds prompts for Nietzsche persona with retrieved passages as context."""

    STYLE_PROMPTS = {
        "Provocative": (
            "You are Friedrich Nietzsche, the philosopher. Respond in first person. "
            "Be confrontational, challenging, and rhetorically fierce. Question the "
            "questioner's assumptions. Use sharp aphorisms and bold declarations. "
            "Do not soften your words — speak with the hammer."
        ),
        "Philosophical": (
            "You are Friedrich Nietzsche, the philosopher. Respond in first person. "
            "Be exploratory, nuanced, and deeply reflective. Draw connections across "
            "your works. Invite the reader to think alongside you, unfolding ideas "
            "layer by layer with careful reasoning."
        ),
    }

    DEFAULT_STYLE = "Provocative"

    def build(self, query: str, passages: list[Passage], style: str | None = None) -> str:
        """Build a complete prompt with system message, retrieved context, and query.

        Args:
            query: User's question or topic.
            passages: List of retrieved Passage objects to use as context.
            style: Voice style - "Provocative" or "Philosophical".

        Returns:
            Formatted prompt string ready for LLM input.
        """
        style = style if style in self.STYLE_PROMPTS else self.DEFAULT_STYLE
        system_message = self.STYLE_PROMPTS[style]

        prompt_parts = [system_message, ""]

        # Add retrieved passages if any exist
        if passages:
            prompt_parts.append("From your works:")
            for i, passage in enumerate(passages, start=1):
                prompt_parts.append(f'{i}. "{passage.text}" — {passage.source}')
            prompt_parts.append("")

        # Add the question
        prompt_parts.append(f"Question: {query}")
        prompt_parts.append("")

        # Add the response prefix
        prompt_parts.append("Nietzsche:")

        return "\n".join(prompt_parts)
