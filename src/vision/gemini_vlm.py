"""
src/vision/gemini_vlm.py
────────────────────────────────────────────────────────────────
Google Gemini Flash VLM adapter — uses google-genai 2.8.0 SDK.

The prompt instructs Gemini to:
  1. Separate visible facts from interpretations
  2. Never assert cultural identity from appearance alone
  3. Use Tamil cultural context as a hypothesis, not a conclusion
  4. Flag all uncertainty explicitly
"""

import json
from PIL.Image import Image as PILImage

from google import genai
from google.genai import types

from src.vision.schemas import VisualObservation
from src.utils.errors import VLMAPIError, VLMRateLimitError, VLMResponseParseError
from src.utils.logger import get_logger

log = get_logger(__name__)

VISION_PROMPT = """You are a careful, culturally-sensitive image analyst.

Analyze this image and produce a structured JSON response with the following fields.
Be precise and honest. Never assume a person's religion, caste, ethnicity, or cultural
identity solely from their appearance.

Respond ONLY with valid JSON. No markdown, no code fences, just the JSON object.

Required JSON structure:
{
  "objects_present": ["list of clearly visible objects"],
  "people_description": ["physical descriptions of people — clothing, posture, activity only"],
  "setting": "description of the physical environment",
  "activities": ["directly observable activities"],
  "clothing_and_objects": ["specific clothing items and significant objects"],
  "colours_prominent": ["dominant colours in the image"],
  "possible_activities": ["inferred activities — mark as inferred"],
  "architectural_features": ["any visible architectural elements"],
  "cultural_clues": ["observable features that MAY suggest cultural context — scripts, specific garments, decorations — never conclusive"],
  "potential_cultural_context": "cautious cultural context suggestion or null if unclear",
  "uncertainty_notes": ["what is ambiguous or could be misidentified"],
  "claims_requiring_evidence": ["statements that would need external cultural knowledge to verify"],
  "image_quality": "good | acceptable | poor"
}

Rules:
- Distinguish between SEEN (directly visible) and INFERRED (your interpretation).
- Never state cultural identity as a fact; use phrases like "may suggest", "appears to be".
- If you cannot determine something, say so in uncertainty_notes.
- Tamil cultural context: look for kolam patterns, gopurams, traditional dress, festival items, etc.
"""


class GeminiVLM:
    """Gemini Flash vision-language model adapter (google-genai 2.8.0 SDK)."""

    def __init__(self, api_key: str, model_name: str = "gemini-3.8-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        log.info(f"GeminiVLM initialised with model: {model_name}")

    def analyze_image(self, image: PILImage, prompt_hint: str = "") -> VisualObservation:
        """Analyze image using Gemini and return VisualObservation."""
        import io

        additional = f"\nAdditional context: {prompt_hint}" if prompt_hint else ""
        full_prompt = VISION_PROMPT + additional

        # Convert PIL Image to bytes for the API
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        image_bytes = buf.getvalue()

        try:
            log.info(f"Sending image to Gemini VLM ({self.model_name})...")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    full_prompt,
                ],
            )
            raw_text = response.text
            log.debug(f"Gemini VLM raw response (first 500 chars): {raw_text[:500]}")
        except Exception as e:
            err_str = str(e).lower()
            if "quota" in err_str or "rate" in err_str or "429" in err_str:
                raise VLMRateLimitError(f"Gemini API rate limit exceeded: {e}") from e
            raise VLMAPIError(f"Gemini API call failed: {e}") from e

        return self._parse_response(raw_text)

    def _parse_response(self, raw_text: str) -> VisualObservation:
        """Parse Gemini's JSON response into VisualObservation."""
        text = raw_text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise VLMResponseParseError(
                f"Gemini returned non-JSON response: {e}\nResponse: {raw_text[:300]}"
            ) from e

        try:
            obs = VisualObservation(
                objects_present=data.get("objects_present", []),
                people_description=data.get("people_description", []),
                setting=data.get("setting", ""),
                activities=data.get("activities", []),
                clothing_and_objects=data.get("clothing_and_objects", []),
                colours_prominent=data.get("colours_prominent", []),
                possible_activities=data.get("possible_activities", []),
                architectural_features=data.get("architectural_features", []),
                cultural_clues=data.get("cultural_clues", []),
                potential_cultural_context=data.get("potential_cultural_context"),
                uncertainty_notes=data.get("uncertainty_notes", []),
                claims_requiring_evidence=data.get("claims_requiring_evidence", []),
                image_quality=data.get("image_quality", "acceptable"),
                raw_vlm_response=raw_text,
            )
        except Exception as e:
            raise VLMResponseParseError(
                f"Could not construct VisualObservation from parsed data: {e}"
            ) from e

        return obs
