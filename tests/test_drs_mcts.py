from __future__ import annotations

from drsgui.dataset import BBox
from drsgui.drs.config import DRSConfig
from drsgui.drs.mcts import MCTSActionPlanner
from drsgui.drs.types import PerceptualAction, UIElement


def make_elements() -> list[UIElement]:
    result = []
    for index in range(20):
        x = 100 + index * 8
        result.append(
            UIElement(
                f"target-{index:02d}",
                BBox(x, 100, x + 6, 110),
                f"Target {index}",
                True,
                1.0 - index * 0.02,
            )
        )
    for index in range(20):
        x = 700 + index * 5
        result.append(
            UIElement(
                f"external-{index:02d}",
                BBox(x, 400, x + 4, 408),
                f"External {index}",
                False,
                0.3 - index * 0.005,
            )
        )
    return result


def test_mcts_uses_paper_budget_depth_and_initial_focus() -> None:
    config = DRSConfig()
    full = BBox(0, 0, 1000, 600)
    result = MCTSActionPlanner(config).search(full_region=full, elements=make_elements())
    assert result.rollout_budget == 8
    assert result.iterations == 8
    assert result.initial_region != full
    assert result.action_path[0] is PerceptualAction.FOCUS
    assert len(result.nodes) <= config.rollout_budget + 1
    assert max(node.depth for node in result.nodes) <= config.max_depth
    assert result.best_reward.total == max(node.reward.total for node in result.nodes)


def test_mcts_is_deterministic_for_equal_input() -> None:
    planner = MCTSActionPlanner()
    first = planner.search(full_region=BBox(0, 0, 1000, 600), elements=make_elements())
    second = planner.search(full_region=BBox(0, 0, 1000, 600), elements=make_elements())
    assert first.to_dict() == second.to_dict()


def test_mcts_falls_back_to_full_region_when_no_elements_exist() -> None:
    full = BBox(0, 0, 100, 100)
    result = MCTSActionPlanner().search(full_region=full, elements=[])
    assert result.initial_region == full
    assert result.best_region == full
    assert result.best_reward.total == 0.0
    assert result.action_path == ()
