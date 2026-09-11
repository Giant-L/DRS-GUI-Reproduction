"""GPU service exposing scored OmniParser V2 elements over a private tunnel."""

from __future__ import annotations

import base64
import hmac
import io
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

from drsgui.dataset import BBox
from drsgui.drs.perception import (
    InstructorLargeEmbeddingBackend,
    SemanticRelevanceScorer,
)
from drsgui.drs.types import UIElement


class PerceiveRequest(BaseModel):
    sample_id: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    application: str | None = None
    platform: str | None = None
    image_data_url: str = Field(min_length=1)


class PerceptionRuntime:
    """Load official models lazily and serialize GPU inference calls."""

    def __init__(self) -> None:
        self._omniparser: Any = None
        self._scorer: SemanticRelevanceScorer | None = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._omniparser is not None and self._scorer is not None

    def _load(self) -> None:
        if self.loaded:
            return
        root = Path(os.environ["OMNIPARSER_ROOT"]).expanduser().resolve()
        if not (root / "util" / "omniparser.py").is_file():
            raise RuntimeError(f"invalid OMNIPARSER_ROOT: {root}")
        sys.path.insert(0, str(root))
        from util.omniparser import Omniparser

        detector = os.getenv(
            "OMNIPARSER_DETECTOR_PATH",
            str(root / "weights" / "icon_detect_v3" / "model.pt"),
        )
        caption = os.getenv(
            "OMNIPARSER_CAPTION_PATH",
            str(root / "weights" / "icon_caption_florence"),
        )
        self._omniparser = Omniparser(
            {
                "som_model_path": detector,
                "caption_model_name": "florence2",
                "caption_model_path": caption,
                "BOX_TRESHOLD": float(os.getenv("OMNIPARSER_BOX_THRESHOLD", "0.05")),
            }
        )
        embedding = InstructorLargeEmbeddingBackend(
            device=os.getenv("INSTRUCTOR_DEVICE", "cuda"),
            batch_size=int(os.getenv("INSTRUCTOR_BATCH_SIZE", "32")),
            max_length=int(os.getenv("INSTRUCTOR_MAX_LENGTH", "512")),
        )
        self._scorer = SemanticRelevanceScorer(embedding)

    def perceive(self, request: PerceiveRequest) -> dict[str, Any]:
        with self._lock:
            self._load()
            image_bytes = _decode_data_url(request.image_data_url)
            with Image.open(io.BytesIO(image_bytes)) as opened:
                opened.load()
                image = opened.convert("RGB")
            width, height = image.size
            raw_base64 = base64.b64encode(image_bytes).decode("ascii")
            parse_started = time.perf_counter()
            _, parsed = self._omniparser.parse(raw_base64)
            parse_latency = time.perf_counter() - parse_started
            elements = _convert_elements(parsed, width=width, height=height)
            score_started = time.perf_counter()
            assert self._scorer is not None
            scored = self._scorer.score(
                request.instruction,
                elements,
                application=request.application,
                platform=request.platform,
            )
            score_latency = time.perf_counter() - score_started
            return {
                "coordinate_space": "original_image_pixels",
                "sample_id": request.sample_id,
                "image_size": [width, height],
                "source": "microsoft/OmniParser-v2.0",
                "relevance_source": "hkunlp/instructor-large",
                "elements": [element.to_dict() for element in scored],
                "metadata": {
                    "omniparser_box_threshold": float(
                        os.getenv("OMNIPARSER_BOX_THRESHOLD", "0.05")
                    ),
                    "parse_latency_seconds": parse_latency,
                    "score_latency_seconds": score_latency,
                    "element_count": len(scored),
                    "ground_truth_used": False,
                },
            }


def _decode_data_url(value: str) -> bytes:
    prefix = "data:image/"
    if not value.startswith(prefix) or ";base64," not in value:
        raise ValueError("image_data_url must be a base64 image data URL")
    encoded = value.split(",", 1)[1]
    try:
        return base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ValueError("image_data_url contains invalid base64") from exc


def _convert_elements(
    parsed: list[dict[str, Any]], *, width: int, height: int
) -> tuple[UIElement, ...]:
    elements: list[UIElement] = []
    for index, raw in enumerate(parsed):
        normalized = raw.get("bbox")
        if not isinstance(normalized, list) or len(normalized) != 4:
            raise ValueError(f"OmniParser element {index} has invalid bbox")
        x1, y1, x2, y2 = [float(value) for value in normalized]
        if not all(0 <= value <= 1 for value in (x1, y1, x2, y2)):
            raise ValueError(f"OmniParser element {index} bbox is not normalized")
        description = str(raw.get("content") or "").strip()
        element_type = str(raw.get("type") or "unknown")
        used_fallback = not description
        if used_fallback:
            description = f"unlabeled {element_type}"
        elements.append(
            UIElement(
                element_id=f"omni-{index}",
                bbox=BBox(x1 * width, y1 * height, x2 * width, y2 * height),
                description=description,
                interactive=bool(raw.get("interactivity", False)),
                metadata={
                    "omniparser_type": element_type,
                    "omniparser_source": raw.get("source"),
                    "normalized_bbox": normalized,
                    "description_fallback": used_fallback,
                },
            )
        )
    return tuple(elements)


runtime = PerceptionRuntime()
app = FastAPI(title="DRS-GUI Perception Service", version="1.0")


def _authorize(provided: str | None) -> None:
    expected = os.getenv("PERCEPTION_API_KEY", "")
    if expected and (provided is None or not hmac.compare_digest(provided, expected)):
        raise HTTPException(status_code=401, detail="invalid API key")


@app.get("/health")
def health(x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    _authorize(x_api_key)
    return {"status": "ok", "models_loaded": runtime.loaded}


@app.post("/v1/perceive")
def perceive(
    request: PerceiveRequest,
    x_api_key: str | None = Header(default=None),
) -> dict[str, Any]:
    _authorize(x_api_key)
    try:
        return runtime.perceive(request)
    except (KeyError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
