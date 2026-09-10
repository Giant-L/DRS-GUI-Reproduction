"""Core data structures for DRS-GUI region search."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping, Sequence

from drsgui.dataset import BBox


class PerceptualAction(str, Enum):
    FOCUS = "focus"
    SHIFT = "shift"
    SCATTER = "scatter"


@dataclass(frozen=True, slots=True)
class UIElement:
    """One OmniParser-style element in original-screenshot coordinates."""

    element_id: str
    bbox: BBox
    description: str
    interactive: bool
    relevance: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.element_id:
            raise ValueError("element_id must not be empty")
        if not self.description.strip():
            raise ValueError(f"description is empty for element {self.element_id}")
        if self.relevance is not None and not math.isfinite(self.relevance):
            raise ValueError(
                f"relevance must be finite for {self.element_id}, got {self.relevance}"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "UIElement":
        required = {"element_id", "bbox", "description", "interactive"}
        missing = sorted(required.difference(value))
        if missing:
            raise ValueError(f"UI element is missing fields: {missing}")
        if not isinstance(value["interactive"], bool):
            raise ValueError("UI element interactive must be a JSON boolean")
        relevance = value.get("relevance")
        known = required | {"relevance", "metadata"}
        metadata = dict(value.get("metadata") or {})
        metadata.update({key: item for key, item in value.items() if key not in known})
        return cls(
            element_id=str(value["element_id"]),
            bbox=BBox.from_sequence(value["bbox"]),
            description=str(value["description"]),
            interactive=value["interactive"],
            relevance=None if relevance is None else float(relevance),
            metadata=metadata,
        )

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.bbox.x1 + self.bbox.x2) / 2,
            (self.bbox.y1 + self.bbox.y2) / 2,
        )

    def with_relevance(self, relevance: float) -> "UIElement":
        return replace(self, relevance=float(relevance))

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "bbox": self.bbox.to_list(),
            "description": self.description,
            "interactive": self.interactive,
            "relevance": self.relevance,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    interaction_weighted_relevance: float
    ui_coverage_consistency: float
    semantic_concentration: float
    total: float
    element_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "interaction_weighted_relevance": self.interaction_weighted_relevance,
            "ui_coverage_consistency": self.ui_coverage_consistency,
            "semantic_concentration": self.semantic_concentration,
            "total": self.total,
            "element_count": self.element_count,
        }


@dataclass(frozen=True, slots=True)
class SearchTraceNode:
    node_id: int
    parent_id: int | None
    depth: int
    action: PerceptualAction | None
    region: BBox
    reward: RewardBreakdown
    visits: int
    mean_value: float
    action_path: tuple[PerceptualAction, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "action": None if self.action is None else self.action.value,
            "region": self.region.to_list(),
            "reward": self.reward.to_dict(),
            "visits": self.visits,
            "mean_value": self.mean_value,
            "action_path": [action.value for action in self.action_path],
        }


@dataclass(frozen=True, slots=True)
class SearchResult:
    full_region: BBox
    initial_region: BBox
    best_region: BBox
    initial_reward: RewardBreakdown
    best_reward: RewardBreakdown
    action_path: tuple[PerceptualAction, ...]
    rollout_budget: int
    iterations: int
    nodes: tuple[SearchTraceNode, ...]
    assumptions: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "full_region": self.full_region.to_list(),
            "initial_region": self.initial_region.to_list(),
            "best_region": self.best_region.to_list(),
            "initial_reward": self.initial_reward.to_dict(),
            "best_reward": self.best_reward.to_dict(),
            "action_path": [action.value for action in self.action_path],
            "rollout_budget": self.rollout_budget,
            "iterations": self.iterations,
            "nodes": [node.to_dict() for node in self.nodes],
            "assumptions": dict(self.assumptions),
        }


def require_scored(elements: Sequence[UIElement]) -> None:
    missing = [element.element_id for element in elements if element.relevance is None]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"semantic relevance is missing for UI elements: {preview}")
