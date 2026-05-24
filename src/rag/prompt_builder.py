"""Prompt builder for Nietzsche persona with retrieved context."""

from src.rag.retrieval import Passage


class NietzschePromptBuilder:
    """Builds prompts for Nietzsche persona with retrieved passages as context."""

    SYSTEM_MESSAGE = (
        "You are Friedrich Nietzsche, the philosopher. Respond in first person, "
        "drawing from your philosophical works. Be profound, provocative, and aphoristic."
    )

    def build(self, query: str, passages: list[Passage]) -> str:
        """Build a complete prompt with system message, retrieved context, and query.

        Args:
            query: User's question or topic.
            passages: List of retrieved Passage objects to use as context.

        Returns:
            Formatted prompt string ready for LLM input.
        """
        # Start with system message
        prompt_parts = [self.SYSTEM_MESSAGE, ""]

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
