from __future__ import annotations

import json
from pathlib import Path

import pytest

from drsgui.dataset import BBox
from drsgui.drs.perception import (
    CachedUIElementPerceptor,
    InstructorLargeEmbeddingBackend,
    PrecomputedSemanticScorer,
    SemanticRelevanceScorer,
    cosine_similarity,
    domain_prefix,
)
from drsgui.drs.types import UIElement


def test_instructor_backend_accepts_local_model_path() -> None:
    backend = InstructorLargeEmbeddingBackend(model_name="/models/instructor-large")

    assert backend.model_name == "/models/instructor-large"


def test_instructor_backend_rejects_empty_model_path() -> None:
    with pytest.raises(ValueError, match="model_name cannot be empty"):
        InstructorLargeEmbeddingBackend(model_name="  ")


class FixedEmbeddingBackend:
    model_name = "fixed-test-embedder"

    def encode(self, texts: list[str]) -> list[list[float]]:
        assert texts[0].startswith("Represent the Word macOS UI element: ")
        return [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]


def test_cached_perceptor_loads_declared_global_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "elements.json"
    path.write_text(
        json.dumps(
            {
                "coordinate_space": "original_image_pixels",
                "source": "test_parser_cache",
                "relevance_source": "test_scores",
                "elements": [
                    {
                        "element_id": "save",
                        "bbox": [10, 10, 20, 20],
                        "description": "Save",
                        "interactive": True,
                        "relevance": 0.8,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    perceptor = CachedUIElementPerceptor.from_json(path)
    parsed = perceptor.parse("unused.png", BBox(0, 0, 100, 100))
    assert parsed[0].element_id == "save"
    assert perceptor.source_name == "test_parser_cache"
    assert perceptor.relevance_source == "test_scores"


def test_cached_perceptor_rejects_unspecified_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "elements.json"
    path.write_text(json.dumps({"elements": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="coordinate_space"):
        CachedUIElementPerceptor.from_json(path)


def test_semantic_scorer_applies_domain_prefix_and_cosine() -> None:
    elements = (
        UIElement("target", BBox(0, 0, 10, 10), "Wrap text", True),
        UIElement("other", BBox(20, 0, 30, 10), "Insert shape", True),
    )
    scored = SemanticRelevanceScorer(FixedEmbeddingBackend()).score(
        "Set wrap text", elements, application="Word", platform="macOS"
    )
    assert scored[0].relevance == pytest.approx(1.0)
    assert scored[1].relevance == pytest.approx(0.0)


def test_precomputed_scorer_requires_every_score() -> None:
    element = UIElement("missing", BBox(0, 0, 1, 1), "Missing", False)
    with pytest.raises(ValueError, match="missing"):
        PrecomputedSemanticScorer().score(
            "instruction", [element], application=None, platform=None
        )


def test_prefix_and_cosine_validation() -> None:
    assert domain_prefix(application="Word", platform="macOS") == (
        "Represent the Word macOS UI element: "
    )
    assert cosine_similarity([1, 1], [1, 1]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="zero"):
        cosine_similarity([0, 0], [1, 0])


def test_ui_element_requires_boolean_interaction_flag() -> None:
    with pytest.raises(ValueError, match="JSON boolean"):
        UIElement.from_dict(
            {
                "element_id": "bad",
                "bbox": [0, 0, 1, 1],
                "description": "Bad",
                "interactive": "false",
            }
        )
