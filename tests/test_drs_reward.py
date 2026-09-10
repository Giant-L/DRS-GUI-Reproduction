from __future__ import annotations

import math

import pytest

from drsgui.dataset import BBox
from drsgui.drs.config import DRSConfig
from drsgui.drs.reward import RegionQualityReward
from drsgui.drs.types import UIElement


def test_reward_matches_paper_components() -> None:
    config = DRSConfig(non_interactive_weight=0.5, semantic_temperature=1.0)
    elements = [
        UIElement("a", BBox(0, 0, 10, 10), "A", True, 1.0),
        UIElement("b", BBox(10, 0, 20, 10), "B", False, 0.0),
    ]
    result = RegionQualityReward(config).evaluate(BBox(0, 0, 100, 100), elements)
    expected_rel = 1.0 / (1.5 + config.numerical_epsilon)
    probabilities = [math.e / (math.e + 1), 1 / (math.e + 1)]
    entropy = -sum(value * math.log(value) for value in probabilities)
    expected_con = 1 - entropy / math.log(2 + config.numerical_epsilon)
    assert result.interaction_weighted_relevance == pytest.approx(expected_rel)
    assert result.ui_coverage_consistency == pytest.approx(0.02)
    assert result.semantic_concentration == pytest.approx(expected_con)
    assert result.total == pytest.approx(
        0.4 * expected_rel + 0.4 * 0.02 + 0.2 * expected_con
    )


def test_reward_uniform_scores_have_near_zero_concentration() -> None:
    elements = [
        UIElement(str(i), BBox(i * 10, 0, i * 10 + 5, 5), str(i), True, 0.5)
        for i in range(4)
    ]
    result = RegionQualityReward().evaluate(BBox(0, 0, 100, 100), elements)
    assert result.semantic_concentration == pytest.approx(0.0, abs=1e-7)


def test_single_element_has_maximum_concentration() -> None:
    element = UIElement("a", BBox(0, 0, 10, 10), "A", True, 0.5)
    result = RegionQualityReward().evaluate(BBox(0, 0, 20, 20), [element])
    assert result.semantic_concentration == pytest.approx(1.0)


def test_empty_region_reward_is_explicitly_zero() -> None:
    element = UIElement("a", BBox(50, 50, 60, 60), "A", True, 0.5)
    result = RegionQualityReward().evaluate(BBox(0, 0, 20, 20), [element])
    assert result.total == 0.0
    assert result.element_count == 0
