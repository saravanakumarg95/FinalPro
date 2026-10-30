"""
src/vision/mock_vlm.py
────────────────────────────────────────────────────────────────
Deterministic mock VLM for testing and development.

Returns a realistic VisualObservation for ANY image without
making any API calls. The response is tailored for Tamil cultural
context so tests are meaningful.
"""

from PIL.Image import Image
from src.vision.schemas import ObservationConfidence, VisualObservation
from src.utils.logger import get_logger

log = get_logger(__name__)


class MockVLM:
    """
    Deterministic mock VLM adapter for tests and offline development.

    Returns consistent, realistic Tamil-cultural observations without
    any API calls. Safe to use in CI/CD pipelines.
    """

    def analyze_image(self, image: Image, prompt_hint: str = "") -> VisualObservation:
        """Return a deterministic VisualObservation for any input image."""
        log.debug(f"MockVLM.analyze_image called. Image size: {image.size}. hint='{prompt_hint}'")

        return VisualObservation(
            objects_present=[
                "clay pot", "oil lamp (vilakku)", "flowers (marigold)",
                "banana leaves", "kolam pattern on floor", "wooden swing"
            ],
            people_description=[
                "A woman wearing a bright silk saree with gold border",
                "An elderly man wearing a white veshti (dhoti) and angavastram",
                "Children in traditional dress carrying lamps"
            ],
            setting=(
                "A courtyard of a traditional South Indian house with "
                "a tiled roof, pillared verandah, and decorated entrance."
            ),
            activities=[
                "Lighting oil lamps at the threshold",
                "Preparing rice in a clay pot over an open fire",
                "Drawing kolam patterns on the floor",
                "Family gathering for a celebration"
            ],
            clothing_and_objects=[
                "Bright silk saree with contrasting zari border",
                "White veshti (dhoti) with gold border",
                "Gold jewellery (jhumka earrings, thali, bangles)",
                "Clay oil lamps (vilakkus)",
                "New clay pot with turmeric and sugarcane decoration"
            ],
            colours_prominent=["saffron/orange", "green", "gold", "white", "red"],
            possible_activities=[
                "Celebrating a Tamil festival such as Pongal or Karthigai Deepam",
                "Performing a household ritual",
                "Welcoming guests at a festive occasion"
            ],
            architectural_features=[
                "Tiled Chettinad-style roof",
                "Wooden pillared verandah",
                "Intricate kolam design at threshold"
            ],
            cultural_clues=[
                "Tamil-style kolam pattern at entrance",
                "Clay pot decoration with turmeric and sugarcane (Pongal context)",
                "Kanchipuram-style silk saree",
                "Vilakku oil lamp",
                "Banana leaf arrangement"
            ],
            potential_cultural_context="Tamil Nadu, South India — likely a Tamil festival celebration",
            uncertainty_notes=[
                "Cannot confirm which specific festival from image alone",
                "Saree style may be Kanchipuram but cannot confirm without closer detail",
                "The specific occasion (Pongal vs Karthigai) requires additional context"
            ],
            claims_requiring_evidence=[
                "Whether the clay pot overflow ritual is Pongal-specific",
                "Specific significance of the kolam design shown",
                "The community or caste-specific customs visible"
            ],
            image_quality="acceptable",
            raw_vlm_response="[MockVLM] Deterministic Tamil cultural scene response."
        )
