"""Text generation using Together API endpoint."""

from together import Together
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    model: str = "sviteri/Qwen/Qwen3-30B-A3B-Base-3737eb6e"
    max_tokens: int = 100
    temperature: float = 0.8
    top_p: float = 0.9
    num_branches: int = 3


class Generator:
    """Generate text continuations using Together API."""

    def __init__(self, config: GenerationConfig | None = None):
        self.client = Together()
        self.config = config or GenerationConfig()

    def generate_continuations(
        self,
        prompt: str,
        num_branches: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[str]:
        """Generate multiple possible continuations for a prompt."""
        n = num_branches or self.config.num_branches
        tokens = max_tokens or self.config.max_tokens
        temp = temperature or self.config.temperature

        continuations = []
        for _ in range(n):
            response = self.client.completions.create(
                model=self.config.model,
                prompt=prompt,
                max_tokens=tokens,
                temperature=temp,
                top_p=self.config.top_p,
            )
            text = response.choices[0].text
            continuations.append(text)

        return continuations

    def generate_single(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate a single continuation."""
        tokens = max_tokens or self.config.max_tokens
        temp = temperature or self.config.temperature

        response = self.client.completions.create(
            model=self.config.model,
            prompt=prompt,
            max_tokens=tokens,
            temperature=temp,
            top_p=self.config.top_p,
        )
        return response.choices[0].text
