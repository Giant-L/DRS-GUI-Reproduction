"""UI parsing and instruction-element semantic relevance abstractions."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from drsgui.dataset import BBox
from drsgui.drs.geometry import visible_elements
from drsgui.drs.types import UIElement, require_scored
from drsgui.models.base import ImageInput


class UIElementPerceptor(Protocol):
    """OmniParser-compatible contract returning global-pixel UI elements."""

    source_name: str

    def parse(self, image: ImageInput, region: BBox) -> tuple[UIElement, ...]: ...


class SemanticScorer(Protocol):
    source_name: str

    def score(
        self,
        instruction: str,
        elements: Sequence[UIElement],
        *,
        application: str | None,
        platform: str | None,
    ) -> tuple[UIElement, ...]: ...


class EmbeddingBackend(Protocol):
    model_name: str

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class CachedUIElementPerceptor:
    """Load cached OmniParser-like output for deterministic core testing."""

    source_name = "cached_ui_elements"

    def __init__(self, elements: Sequence[UIElement]) -> None:
        self._elements = tuple(elements)
        ids = [element.element_id for element in self._elements]
        if len(ids) != len(set(ids)):
            raise ValueError("cached UI element ids must be unique")

    @classmethod
    def from_json(cls, path: str | Path) -> "CachedUIElementPerceptor":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            coordinate_space = payload.get("coordinate_space")
            if coordinate_space != "original_image_pixels":
                raise ValueError(
                    "cached elements must declare coordinate_space="
                    "'original_image_pixels'"
                )
            values = payload.get("elements")
        else:
            values = payload
        if not isinstance(values, list):
            raise ValueError("cached element JSON must contain an elements list")
        return cls([UIElement.from_dict(value) for value in values])

    def parse(self, image: ImageInput, region: BBox) -> tuple[UIElement, ...]:
        del image
        return visible_elements(self._elements, region)


class PrecomputedSemanticScorer:
    """Use cached cosine scores; intended for isolated core tests and demos."""

    source_name = "precomputed_relevance"

    def score(
        self,
        instruction: str,
        elements: Sequence[UIElement],
        *,
        application: str | None,
        platform: str | None,
    ) -> tuple[UIElement, ...]:
        del instruction, application, platform
        require_scored(elements)
        return tuple(elements)


class SemanticRelevanceScorer:
    """Cosine similarity after shared domain-prefix encoding (paper Eq. 3)."""

    source_name = "domain_prefixed_cosine_similarity"

    def __init__(self, backend: EmbeddingBackend) -> None:
        self.backend = backend

    def score(
        self,
        instruction: str,
        elements: Sequence[UIElement],
        *,
        application: str | None,
        platform: str | None,
    ) -> tuple[UIElement, ...]:
        if not instruction.strip():
            raise ValueError("instruction must not be empty")
        prefix = domain_prefix(application=application, platform=platform)
        texts = [f"{prefix}{instruction}"] + [
            f"{prefix}{element.description}" for element in elements
        ]
        vectors = self.backend.encode(texts)
        if len(vectors) != len(texts):
            raise ValueError(
                f"embedding backend returned {len(vectors)} vectors for {len(texts)} texts"
            )
        instruction_vector = vectors[0]
        return tuple(
            element.with_relevance(cosine_similarity(instruction_vector, vector))
            for element, vector in zip(elements, vectors[1:], strict=True)
        )


class InstructorLargeEmbeddingBackend:
    """Lazy optional backend implementing final-hidden-state masked mean pooling."""

    model_name = "hkunlp/instructor-large"

    def __init__(
        self,
        *,
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("batch_size and max_length must be positive")
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._tokenizer: Any = None
        self._model: Any = None

    def _load(self) -> tuple[Any, Any, Any]:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Instructor-large requires optional dependencies torch and transformers"
            ) from exc
        if self._tokenizer is None or self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()
        return torch, self._tokenizer, self._model

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        torch, tokenizer, model = self._load()
        encoded_vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokens = {name: value.to(self.device) for name, value in tokens.items()}
            with torch.inference_mode():
                hidden = model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            encoded_vectors.extend(pooled.detach().cpu().tolist())
        return encoded_vectors


def domain_prefix(*, application: str | None, platform: str | None) -> str:
    """Template inferred from the Word/macOS example in paper Figure 2."""

    app = application.strip() if application else "current application"
    system = platform.strip() if platform else "current system"
    return f"Represent the {app} {system} UI element: "


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("embedding vectors must be non-empty and have equal dimensions")
    if not all(math.isfinite(value) for value in (*left, *right)):
        raise ValueError("embedding vectors must contain only finite values")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("cosine similarity is undefined for zero vectors")
    return sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )
