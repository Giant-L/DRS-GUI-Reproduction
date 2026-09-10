"""MCTS action planner following paper Sec. 3.3."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

from drsgui.dataset import BBox
from drsgui.drs.actions import PerceptualActions
from drsgui.drs.config import DRSConfig
from drsgui.drs.geometry import same_bbox
from drsgui.drs.reward import RegionQualityReward
from drsgui.drs.types import (
    PerceptualAction,
    RegionProposal,
    RewardBreakdown,
    SearchResult,
    SearchTraceNode,
    UIElement,
    require_scored,
)


ACTION_ORDER = (
    PerceptualAction.FOCUS,
    PerceptualAction.SHIFT,
    PerceptualAction.SCATTER,
)


@dataclass(slots=True)
class _Node:
    node_id: int
    parent: _Node | None
    action: PerceptualAction | None
    proposal: RegionProposal | None
    region: BBox
    reward: RewardBreakdown
    depth: int
    action_path: tuple[PerceptualAction, ...]
    children: dict[PerceptualAction, _Node] = field(default_factory=dict)
    remaining_actions: list[PerceptualAction] = field(
        default_factory=lambda: list(ACTION_ORDER)
    )
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class MCTSActionPlanner:
    """Plan over region states with one reward evaluation per expansion."""

    def __init__(
        self,
        config: DRSConfig | None = None,
        *,
        actions: PerceptualActions | None = None,
        reward: RegionQualityReward | None = None,
    ) -> None:
        self.config = config or DRSConfig()
        self.actions = actions or PerceptualActions(self.config)
        self.reward = reward or RegionQualityReward(self.config)

    def search(
        self,
        *,
        full_region: BBox,
        elements: Iterable[UIElement],
    ) -> SearchResult:
        all_elements = tuple(elements)
        require_scored(all_elements)
        initial = self.actions.focus(
            current_region=full_region,
            all_elements=all_elements,
            full_region=full_region,
        )
        initial_region = initial.region if initial is not None else full_region
        initial_path = (
            (PerceptualAction.FOCUS,) if initial is not None else tuple()
        )
        root = _Node(
            node_id=0,
            parent=None,
            action=None,
            proposal=initial,
            region=initial_region,
            reward=self.reward.evaluate(initial_region, all_elements),
            depth=0,
            action_path=initial_path,
        )
        nodes = [root]

        iterations = 0
        for _ in range(self.config.rollout_budget):
            path = self._select(root)
            selected = path[-1]
            expanded = self._expand(selected, all_elements, full_region, len(nodes))
            if expanded is not None:
                nodes.append(expanded)
                path.append(expanded)
            value = path[-1].reward.total
            self._backpropagate(path, value)
            iterations += 1

        best = max(
            nodes,
            key=lambda node: (node.reward.total, -node.region.area, -node.node_id),
        )
        traces = tuple(
            SearchTraceNode(
                node_id=node.node_id,
                parent_id=None if node.parent is None else node.parent.node_id,
                depth=node.depth,
                action=node.action,
                region=node.region,
                reward=node.reward,
                visits=node.visits,
                mean_value=node.mean_value,
                action_path=node.action_path,
            )
            for node in nodes
        )
        assumptions = {
            **self.config.assumptions(),
            "initial_focus_fallback": (
                "use full screenshot when Focus cannot propose a smaller region"
            ),
            "mcts_root_depth": "post-initial-Focus root is depth zero",
            "mcts_action_order": [action.value for action in ACTION_ORDER],
            "mcts_backup": "arithmetic mean of immediate candidate rewards",
            "mcts_invalid_action": "mark attempted and try the next action in rollout",
            "best_region_tie_break": "higher reward, then smaller area, then older node",
        }
        return SearchResult(
            full_region=full_region,
            initial_region=initial_region,
            best_region=best.region,
            initial_reward=root.reward,
            best_reward=best.reward,
            action_path=best.action_path,
            rollout_budget=self.config.rollout_budget,
            iterations=iterations,
            nodes=traces,
            assumptions=assumptions,
        )

    def _select(self, root: _Node) -> list[_Node]:
        node = root
        path = [node]
        while node.depth < self.config.max_depth:
            if node.remaining_actions or not node.children:
                break
            node = max(
                node.children.values(),
                key=lambda child: (
                    self._uct(node, child),
                    -ACTION_ORDER.index(child.action),  # type: ignore[arg-type]
                ),
            )
            path.append(node)
        return path

    def _expand(
        self,
        node: _Node,
        all_elements: tuple[UIElement, ...],
        full_region: BBox,
        next_node_id: int,
    ) -> _Node | None:
        if node.depth >= self.config.max_depth:
            return None
        while node.remaining_actions:
            action = node.remaining_actions.pop(0)
            proposal = self.actions.propose(
                action,
                current_region=node.region,
                all_elements=all_elements,
                full_region=full_region,
            )
            if proposal is None or same_bbox(proposal.region, node.region):
                continue
            child = _Node(
                node_id=next_node_id,
                parent=node,
                action=action,
                proposal=proposal,
                region=proposal.region,
                reward=self.reward.evaluate(proposal.region, all_elements),
                depth=node.depth + 1,
                action_path=(*node.action_path, action),
            )
            node.children[action] = child
            return child
        return None

    def _uct(self, parent: _Node, child: _Node) -> float:
        if child.visits == 0:
            return math.inf
        exploration = self.config.uct_exploration_constant * math.sqrt(
            math.log(max(1, parent.visits)) / child.visits
        )
        return child.mean_value + exploration

    @staticmethod
    def _backpropagate(path: list[_Node], value: float) -> None:
        for node in path:
            node.visits += 1
            node.value_sum += value
