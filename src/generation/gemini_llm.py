"""
src/generation/gemini_llm.py
────────────────────────────────────────────────────────────────
Gemini LLM adapter for Tamil cultural story generation.
Uses google-genai 2.8.0 SDK.

Implements three generation methods:
  - generate_story(): proposed evidence-grounded method
  - generate_baseline_a(): direct generation (no retrieval)
  - generate_baseline_b(): fixed-example prompting
"""

from google import genai

from src.generation.schemas import GeneratedStory
from src.reasoning.schemas import ReasoningTrace
from src.retrieval.schemas import RetrievalResult
from src.utils.errors import LLMAPIError, LLMRateLimitError
from src.utils.logger import get_logger
from src.utils.text_utils import count_words

log = get_logger(__name__)

# Fixed cultural examples for Baseline B (never change during experiments)
FIXED_EXAMPLES = """
Example 1 (Pongal):
In the Tamil month of Thai, when the sugarcane stands tall and the rice harvest gleams golden,
families gather at dawn to cook the sacred pongal in clay pots. The moment the rice overflows
with the cry of 'Pongal-o-Pongal!' marks the sun's northward journey and abundance for the year.

Example 2 (Bharatanatyam):
The dancer's anklet bells rang out across the sabha hall as she struck the aramandi — the
characteristic bent-knee stance of Bharatanatyam. Her hands formed the Alapadma mudra,
petals opening towards the deity above the stage, her eyes alive with abhinaya.

Example 3 (Temple Festival):
At the Meenakshi Temple, the gopuram's thousand painted figures watched over the city as
the chariot procession wound through Madurai's ancient streets, the air thick with jasmine
and incense, devotees pressing forward for a glimpse of the goddess.
"""

STORY_SYSTEM_PROMPT = """You are a culturally sensitive Tamil cultural storyteller.

Generate a coherent, engaging, respectful short story (approximately {target_words} words)
grounded in Tamil cultural traditions.

Rules:
1. Only make cultural claims supported by the provided evidence.
2. Clearly separate fictional narrative from cultural facts.
3. Never stereotype or overgeneralise Tamil culture.
4. If evidence is insufficient, write a culturally humble story acknowledging uncertainty.
5. Use specific Tamil cultural details (festival names, foods, rituals) from the evidence.
6. The story should be inclusive and respectful.
7. Reading level: {reading_level}

Return ONLY the story text. No headings, no meta-commentary.
"""

BASELINE_A_PROMPT = """Based on this visual observation of an image, write a short story
(approximately {target_words} words) that could be told about the scene.

Visual Observation:
{observation}

Write a culturally aware story. Return only the story text."""

BASELINE_B_PROMPT = """Based on this visual observation, write a short Tamil-cultural story
(approximately {target_words} words).

Visual Observation:
{observation}

Use inspiration from these Tamil cultural examples:
{examples}

Return only the story text."""


class GeminiLLM:
    """Gemini LLM adapter for story generation (google-genai 2.8.0 SDK)."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        log.info(f"GeminiLLM initialised: {model_name}")

    def generate_story(
        self,
        reasoning_trace: ReasoningTrace,
        retrieval_result: RetrievalResult,
        target_words: int = 400,
        reading_level: str = "general",
    ) -> GeneratedStory:
        """Proposed method: evidence-grounded story generation."""
        evidence_block = "\n\n".join(
            e.to_evidence_block()
            for e in retrieval_result.retrieved_records
            if e.record_id in reasoning_trace.evidence_used_ids
        ) or "No relevant cultural evidence retrieved."

        outline = reasoning_trace.story_outline
        outline_block = ""
        if outline:
            outline_block = (
                f"\nStory outline to follow:\n"
                f"- Setting: {outline.setting}\n"
                f"- Character: {outline.main_character}\n"
                f"- Cultural elements to include: {', '.join(outline.cultural_elements_to_use)}\n"
                f"- Narrative arc: {outline.narrative_arc}\n"
                f"- Avoid: {', '.join(outline.cultural_claims_to_avoid)}\n"
            )

        prompt = (
            STORY_SYSTEM_PROMPT.format(target_words=target_words, reading_level=reading_level)
            + f"\n\nImage observations:\n{reasoning_trace.observation_summary}"
            + f"\n\nRetrieved cultural evidence:\n{evidence_block}"
            + outline_block
        )

        raw_text = self._call_gemini(prompt)
        return GeneratedStory(
            story_text=raw_text,
            word_count=count_words(raw_text),
            evidence_ids_used=reasoning_trace.evidence_used_ids,
            generation_method="proposed",
            model_used=self.model_name,
            reading_level=reading_level,
            raw_llm_response=raw_text,
        )

    def generate_baseline_a(self, observation_text: str, target_words: int = 400) -> GeneratedStory:
        """Baseline A: direct generation from image observation only."""
        prompt = BASELINE_A_PROMPT.format(observation=observation_text, target_words=target_words)
        raw_text = self._call_gemini(prompt)
        return GeneratedStory(
            story_text=raw_text,
            word_count=count_words(raw_text),
            evidence_ids_used=[],
            generation_method="baseline_a",
            model_used=self.model_name,
        )

    def generate_baseline_b(
        self,
        observation_text: str,
        target_words: int = 400,
        fixed_examples: str = FIXED_EXAMPLES,
    ) -> GeneratedStory:
        """Baseline B: fixed-example prompting."""
        prompt = BASELINE_B_PROMPT.format(
            observation=observation_text,
            examples=fixed_examples,
            target_words=target_words,
        )
        raw_text = self._call_gemini(prompt)
        return GeneratedStory(
            story_text=raw_text,
            word_count=count_words(raw_text),
            evidence_ids_used=[],
            generation_method="baseline_b",
            model_used=self.model_name,
        )

    def _call_gemini(self, prompt: str) -> str:
        """Make a Gemini API call and return the text response."""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            err_str = str(e).lower()
            if "quota" in err_str or "rate" in err_str or "429" in err_str:
                raise LLMRateLimitError(f"Gemini LLM rate limit: {e}") from e
            raise LLMAPIError(f"Gemini LLM API call failed: {e}") from e
