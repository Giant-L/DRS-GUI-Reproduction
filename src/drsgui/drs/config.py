"""Paper constants and explicit paper-reimplementation assumptions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DRSConfig:
    # Values specified in Sec. 4.1 of the paper.
    focus_top_fraction: float = 0.15
    scatter_external_fraction: float = 0.10
    scatter_max_area_scale: float = 1.5
    shift_external_fraction: float = 0.15
    shift_max_iou: float = 0.3
    rollout_budget: int = 8
    max_depth: int = 3
    uct_exploration_constant: float = 1.0
    reward_alpha: float = 0.4
    reward_beta: float = 0.4
    reward_gamma: float = 0.2

    # Configurable reproduction assumptions absent from the main paper.
    # Calibrated conservatively against the paper's reported ~64% mean area
    # reduction. These remain assumptions because the paper omits both values.
    focus_outlier_distance_fraction: float = 0.55
    focus_target_area_ratio: float = 0.80
    non_interactive_weight: float = 0.50
    semantic_temperature: float = 0.10
    numerical_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        fractions = {
            "focus_top_fraction": self.focus_top_fraction,
            "scatter_external_fraction": self.scatter_external_fraction,
            "shift_external_fraction": self.shift_external_fraction,
            "focus_outlier_distance_fraction": self.focus_outlier_distance_fraction,
            "focus_target_area_ratio": self.focus_target_area_ratio,
            "non_interactive_weight": self.non_interactive_weight,
        }
        for name, value in fractions.items():
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1], got {value}")
        if self.scatter_max_area_scale < 1:
            raise ValueError("scatter_max_area_scale must be at least 1")
        if not 0 <= self.shift_max_iou <= 1:
            raise ValueError("shift_max_iou must be in [0, 1]")
        if self.rollout_budget <= 0 or self.max_depth <= 0:
            raise ValueError("rollout_budget and max_depth must be positive")
        if self.uct_exploration_constant < 0:
            raise ValueError("uct_exploration_constant must be non-negative")
        if self.semantic_temperature <= 0 or self.numerical_epsilon <= 0:
            raise ValueError("semantic_temperature and numerical_epsilon must be positive")
        weights = (self.reward_alpha, self.reward_beta, self.reward_gamma)
        if any(weight < 0 for weight in weights):
            raise ValueError("reward weights must be non-negative")
        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("reward weights must sum to one")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def assumptions(self) -> dict[str, Any]:
        return {
            "focus_outlier_rule": (
                "discard selected centers farther than "
                f"{self.focus_outlier_distance_fraction} of the current-region "
                "diagonal from their centroid"
            ),
            "focus_target_area_ratio": self.focus_target_area_ratio,
            "shift_direction_groups": "dominant-axis left/right/up/down",
            "shift_region_size": "preserve current width and height",
            "scatter_overflow": "skip external anchors that violate the 1.5x cap",
            "non_interactive_weight": self.non_interactive_weight,
            "semantic_temperature": self.semantic_temperature,
            "numerical_epsilon": self.numerical_epsilon,
            "cached_element_visibility": "element center lies inside region",
            "parser_reuse": "parse full screenshot once and filter global elements",
        }
