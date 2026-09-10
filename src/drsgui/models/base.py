"""Backend-neutral grounding model contract and response utilities."""

from __future__ import annotations

import base64
import io
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, TypeAlias

import requests
from PIL import Image

from drsgui.evaluator import PixelPoint

ImageInput: TypeAlias = str | Path | Image.Image


class ModelError(RuntimeError):
    """Base exception for grounding backend failures."""


class ModelConfigurationError(ModelError):
    """Raised when a backend is not configured for inference."""


class ModelRequestError(ModelError):
    """Raised when a model endpoint request fails."""


class ModelResponseError(ModelError):
    """Raised when a model response cannot be converted to one coordinate."""


@dataclass(frozen=True, slots=True)
class GroundingPrediction:
    """A backend prediction with an explicit coordinate-space declaration."""

    point: PixelPoint
    backend: str
    model: str
    raw_output: str
    coordinate_space: str = "original_image_pixels"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "point": self.point.to_list(),
            "coordinate_space": self.coordinate_space,
            "backend": self.backend,
            "model": self.model,
            "raw_output": self.raw_output,
            "metadata": dict(self.metadata),
        }


class GroundingModel(ABC):
    """Backends return pixels for their exact input image.

    Stage 1 passes the original screenshot. DRS passes a crop and relabels the
    backend output as crop-local before restoring it to original pixels.
    """

    backend_name: str
    model_name: str

    @abstractmethod
    def predict(self, image: ImageInput, instruction: str) -> GroundingPrediction:
        """Ground an instruction in a full screenshot."""


def load_image(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        loaded = image.copy()
    else:
        path = Path(image).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"image not found: {path}")
        with Image.open(path) as opened:
            opened.load()
            loaded = opened.copy()
    if loaded.mode not in {"RGB", "RGBA"}:
        loaded = loaded.convert("RGB")
    return loaded


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    # GUI grounding depends on tiny text and icons, so avoid lossy JPEG artifacts.
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def parse_coordinate(raw_output: str) -> PixelPoint:
    """Parse a single JSON/list/tuple coordinate and reject ambiguous output."""

    text = raw_output.strip()
    if not text:
        raise ModelResponseError("model returned an empty response")

    candidates: list[PixelPoint] = []
    json_texts = [text]
    json_texts.extend(re.findall(r"\{[^{}]*\}|\[[^\[\]]*\]", text))
    for candidate in json_texts:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        parsed = _point_from_json(value)
        if parsed is not None:
            candidates.append(parsed)

    number = r"(-?(?:\d+(?:\.\d*)?|\.\d+))"
    bbox_pattern = re.compile(
        rf"[\(\[]\s*{number}\s*,\s*{number}\s*,\s*"
        rf"{number}\s*,\s*{number}\s*[\)\]]"
    )
    candidates.extend(
        PixelPoint((float(x1) + float(x2)) / 2, (float(y1) + float(y2)) / 2)
        for x1, y1, x2, y2 in bbox_pattern.findall(text)
    )

    pair_pattern = re.compile(
        r"[\(\[]\s*(-?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
        r"(-?(?:\d+(?:\.\d*)?|\.\d+))\s*[\)\]]"
    )
    candidates.extend(
        PixelPoint(float(x), float(y)) for x, y in pair_pattern.findall(text)
    )

    unique = {(point.x, point.y): point for point in candidates}
    if len(unique) != 1:
        if not unique:
            raise ModelResponseError(
                f"could not parse one coordinate from response: {text[:300]!r}"
            )
        raise ModelResponseError(
            f"ambiguous response contains multiple coordinates: {text[:300]!r}"
        )
    return next(iter(unique.values()))


def post_chat_completion(
    *,
    base_url: str,
    api_key: str,
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> tuple[str, dict[str, Any]]:
    if not base_url.strip():
        raise ModelConfigurationError("model API base URL must not be empty")
    endpoint = _chat_completion_endpoint(base_url)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=dict(payload),
            timeout=timeout_seconds,
        )
    except requests.RequestException as exc:
        raise ModelRequestError(f"request to {endpoint} failed: {exc}") from exc

    if not response.ok:
        body = response.text[:1000]
        raise ModelRequestError(
            f"model endpoint returned HTTP {response.status_code}: {body}"
        )
    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ModelResponseError(
            f"unexpected chat-completion response: {response.text[:1000]}"
        ) from exc

    if isinstance(content, list):
        text_parts = [
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in {"text", "output_text"}
        ]
        content = "".join(text_parts)
    if not isinstance(content, str):
        raise ModelResponseError(f"response content is not text: {content!r}")

    metadata = {
        "request_id": response.headers.get("x-request-id"),
        "usage": data.get("usage"),
        "finish_reason": data["choices"][0].get("finish_reason"),
    }
    return content, metadata


def _point_from_json(value: Any) -> PixelPoint | None:
    if isinstance(value, dict):
        if "x" in value and "y" in value:
            try:
                return PixelPoint(float(value["x"]), float(value["y"]))
            except (TypeError, ValueError):
                return None
        if "point" in value:
            return _point_from_json(value["point"])
    if isinstance(value, list) and len(value) in {2, 4}:
        try:
            numbers = [float(item) for item in value]
        except (TypeError, ValueError):
            return None
        if len(numbers) == 2:
            return PixelPoint(*numbers)
        return PixelPoint(
            (numbers[0] + numbers[2]) / 2,
            (numbers[1] + numbers[3]) / 2,
        )
    return None


def _chat_completion_endpoint(base_url: str) -> str:
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/chat/completions"):
        return cleaned
    return f"{cleaned}/chat/completions"
