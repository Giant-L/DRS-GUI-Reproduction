"""UGround-V1 adapter for an OpenAI-compatible inference server."""

from __future__ import annotations

from drsgui.evaluator import normalized_to_pixel
from drsgui.models.base import (
    GroundingModel,
    GroundingPrediction,
    ImageInput,
    image_to_data_url,
    load_image,
    parse_coordinate,
    post_chat_completion,
)


class UGroundModel(GroundingModel):
    """Call official UGround-V1 through vLLM/SGLang and convert 0-1000 output."""

    backend_name = "uground"

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model: str = "osunlp/UGround-V1-2B",
        timeout_seconds: float = 120.0,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model
        self.timeout_seconds = timeout_seconds

    def predict(self, image: ImageInput, instruction: str) -> GroundingPrediction:
        screenshot = load_image(image)
        width, height = screenshot.size
        prompt = (
            "Your task is to help the user identify the precise coordinates (x, y) "
            "of a specific area/element/object on the screen based on a description.\n"
            "- Point to the center or a representative point within the described "
            "area/element/object as accurately as possible.\n"
            "- If the description is unclear or ambiguous, infer the most relevant "
            "area or element from its likely context or purpose.\n"
            "- Return a single string (x, y).\n\n"
            f"Description: {instruction}\n\nAnswer:"
        )
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_to_data_url(screenshot)},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 64,
        }
        raw_output, metadata = post_chat_completion(
            base_url=self.base_url,
            api_key=self.api_key,
            payload=payload,
            timeout_seconds=self.timeout_seconds,
        )
        normalized_point = parse_coordinate(raw_output)
        pixel_point = normalized_to_pixel(
            normalized_point,
            width=width,
            height=height,
            scale=1000.0,
        )
        return GroundingPrediction(
            point=pixel_point,
            backend=self.backend_name,
            model=self.model_name,
            raw_output=raw_output,
            metadata={
                **metadata,
                "source_coordinate": normalized_point.to_list(),
                "source_coordinate_space": "normalized_0_1000",
            },
        )
