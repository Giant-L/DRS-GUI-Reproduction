"""DeepSeek Vision API adapter for pipeline smoke testing."""

from __future__ import annotations

from drsgui.evaluator import validate_pixel_point
from drsgui.models.base import (
    GroundingModel,
    GroundingPrediction,
    ImageInput,
    image_to_data_url,
    load_image,
    parse_coordinate,
    post_chat_completion,
)


class DeepSeekVisionModel(GroundingModel):
    """Call DeepSeek's vision model and require original-pixel JSON output."""

    backend_name = "deepseek"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-v4-flash-vision-exp",
        timeout_seconds: float = 120.0,
    ) -> None:
        if not api_key:
            raise ValueError("DeepSeek API key must not be empty")
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model
        self.timeout_seconds = timeout_seconds

    def predict(self, image: ImageInput, instruction: str) -> GroundingPrediction:
        screenshot = load_image(image)
        width, height = screenshot.size
        prompt = (
            "Locate the target GUI element described by the instruction in the "
            "provided screenshot image. Return one representative point inside the "
            f"target. The provided image size is {width}x{height} pixels. Return JSON "
            'only as {"x": <input-image-pixel-x>, "y": <input-image-pixel-y>}. '
            f"Instruction: {instruction}"
        )
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_to_data_url(screenshot),
                                "detail": "original",
                            },
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 64,
            "response_format": {"type": "json_object"},
        }
        raw_output, metadata = post_chat_completion(
            base_url=self.base_url,
            api_key=self.api_key,
            payload=payload,
            timeout_seconds=self.timeout_seconds,
        )
        point = parse_coordinate(raw_output)
        validate_pixel_point(point, width, height)
        return GroundingPrediction(
            point=point,
            backend=self.backend_name,
            model=self.model_name,
            raw_output=raw_output,
            metadata=metadata,
        )
