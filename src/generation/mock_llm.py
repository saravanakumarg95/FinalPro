"""
src/generation/mock_llm.py — Deterministic mock LLM for testing.
"""
from src.generation.schemas import GeneratedStory
from src.reasoning.schemas import ReasoningTrace
from src.utils.logger import get_logger

log = get_logger(__name__)

MOCK_STORY = """The morning sun had barely kissed the terracotta rooftops of the old Tamil village when Meenakshi stepped onto the threshold of her ancestral home. In her hands, she carried a freshly drawn pot of water from the well — the same well her grandmother had drawn from decades ago.

The kolam she had drawn at dawn still gleamed on the swept courtyard floor — a perfect pulli kolam, sixty-four dots connected in loops of pure white rice flour. The pattern was more than decoration; it was a daily conversation with prosperity, a welcome to Lakshmi herself.

Today was Thai Pongal. The clay pot — the new one her mother had bought at the Tuesday market — sat on three bricks over the wood fire, filled with freshly harvested rice and thick buffalo milk. Around it, she had tied sugarcane stalks and turmeric leaves. When the milk began to foam and rise, Meenakshi would call out "Pongal-o-Pongal!" and the village would echo her joy.

Her grandmother sat on the wooden swing in the verandah, dressed in her finest Kanchipuram silk — deep crimson with a peacock-bordered gold zari, the one reserved for sacred days. The golden jhumkas at her ears caught the morning light as she hummed a Carnatic melody, a Tyagaraja krithi in Bhairavi raga.

At the cattle shed, her younger brother Murugan was tying fresh marigolds to the horns of their farm bull. Tomorrow would be Mattu Pongal — the day the patient, faithful animals who ploughed the earth would receive their share of gratitude.

This was not merely a festival. It was memory made visible — the unbroken thread connecting this courtyard to every courtyard before it, in every Tamil village, across every rice harvest season that had ever been."""

class MockLLM:
    """Deterministic mock LLM for tests — returns a fixed Tamil cultural story."""

    def generate_story(
        self,
        reasoning_trace: ReasoningTrace,
        target_words: int = 400,
        reading_level: str = "general",
    ) -> GeneratedStory:
        log.debug("MockLLM.generate_story called.")
        return GeneratedStory(
            story_text=MOCK_STORY,
            word_count=len(MOCK_STORY.split()),
            evidence_ids_used=["ta-001", "ta-002", "ta-004"],
            unsupported_claims=[],
            cultural_elements_included=["Pongal", "kolam", "Kanchipuram saree", "Carnatic music", "Mattu Pongal"],
            generation_method="proposed",
            model_used="mock",
            reading_level=reading_level,
        )

    def generate_baseline_a(self, observation_text: str, target_words: int = 400) -> GeneratedStory:
        """Baseline A: direct generation without retrieval."""
        return GeneratedStory(
            story_text="[Baseline A Mock] " + MOCK_STORY[:300] + "...",
            evidence_ids_used=[],
            generation_method="baseline_a",
            model_used="mock",
        )

    def generate_baseline_b(self, observation_text: str, fixed_examples: str, target_words: int = 400) -> GeneratedStory:
        """Baseline B: fixed-example prompting."""
        return GeneratedStory(
            story_text="[Baseline B Mock] " + MOCK_STORY[:400] + "...",
            evidence_ids_used=[],
            generation_method="baseline_b",
            model_used="mock",
        )
