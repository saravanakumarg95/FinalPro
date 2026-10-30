"""
src/vision/adapter.py
────────────────────────────────────────────────────────────────
VLMAdapter Protocol — defines the interface every VLM backend must implement.

Using Protocol (structural subtyping) means adapters do not need to inherit
from a base class — they just need to implement the interface.
"""

from typing import Protocol, runtime_checkable
from PIL.Image import Image
from src.vision.schemas import VisualObservation


@runtime_checkable
class VLMAdapter(Protocol):
    """
    Interface for Vision-Language Model adapters.

    Any class implementing analyze_image() satisfies this protocol
    and can be used as a drop-in VLM backend.
    """

    def analyze_image(self, image: Image, prompt_hint: str = "") -> VisualObservation:
        """
        Analyze an image and return structured visual observations.

        Args:
            image: A PIL RGB Image (already validated).
            prompt_hint: Optional cultural context hint to guide the VLM.

        Returns:
            VisualObservation with structured observations.

        Raises:
            VLMAPIError: If the API call fails.
            VLMResponseParseError: If the response cannot be parsed.
        """
        ...
