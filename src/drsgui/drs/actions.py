"""Focus, Shift, and Scatter region proposals from paper Sec. 3.2."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from drsgui.dataset import BBox
from drsgui.drs.config import DRSConfig
from drsgui.drs.geometry import (
    bbox_center,
    bbox_iou,
    center_inside,
    clip_bbox,
    enclosing_bbox,
    recentered_region,
    region_diagonal,
    same_bbox,
    visible_elements,
)
from drsgui.drs.types import (
    PerceptualAction,
    RegionProposal,
    UIElement,
    require_scored,
)


class PerceptualActions:
    def __init__(self, config: DRSConfig | None = None) -> None:
        self.config = config or DRSConfig()

    def propose(
        self,
        action: PerceptualAction,
        *,
        current_region: BBox,
        all_elements: Sequence[UIElement],
        full_region: BBox,
    ) -> RegionProposal | None:
        require_scored(all_elements)
        if action is PerceptualAction.FOCUS:
            return self.focus(
                current_region=current_region,
                all_elements=all_elements,
                full_region=full_region,
            )
        if action is PerceptualAction.SHIFT:
            return self.shift(
                current_region=current_region,
                all_elements=all_elements,
                full_region=full_region,
            )
        if action is PerceptualAction.SCATTER:
            return self.scatter(
                current_region=current_region,
                all_elements=all_elements,
                full_region=full_region,
            )
        raise ValueError(f"unsupported action: {action}")

    def focus(
        self,
        *,
        current_region: BBox,
        all_elements: Sequence[UIElement],
        full_region: BBox,
    ) -> RegionProposal | None:
        require_scored(all_elements)
        visible = _ranked(visible_elements(all_elements, current_region))
        if not visible:
            return None
        selected = list(visible[: _top_count(len(visible), self.config.focus_top_fraction)])
        selected = self._discard_focus_outliers(selected, current_region)
        selected = self._prune_to_target_area(selected, current_region)
        candidate = clip_bbox(enclosing_bbox(element.bbox for element in selected), full_region)
        if same_bbox(candidate, current_region):
            return None
        return RegionProposal(
            action=PerceptualAction.FOCUS,
            source_region=current_region,
            region=candidate,
            selected_element_ids=tuple(element.element_id for element in selected),
            details={
                "visible_count": len(visible),
                "paper_top_fraction": self.config.focus_top_fraction,
                "assumed_target_area_ratio": self.config.focus_target_area_ratio,
            },
        )

    def _discard_focus_outliers(
        self, selected: list[UIElement], current_region: BBox
    ) -> list[UIElement]:
        if len(selected) <= 2:
            return selected
        centroid = _centroid(selected)
        threshold = (
            self.config.focus_outlier_distance_fraction
            * region_diagonal(current_region)
        )
        retained = [
            element
            for element in selected
            if math.dist(element.center, centroid) <= threshold
        ]
        return retained or [selected[0]]

    def _prune_to_target_area(
        self, selected: list[UIElement], current_region: BBox
    ) -> list[UIElement]:
        target_area = current_region.area * self.config.focus_target_area_ratio
        retained = list(selected)
        while len(retained) > 1:
            crop = enclosing_bbox(element.bbox for element in retained)
            if crop.area <= target_area:
                break
            centroid = _centroid(retained)
            remove_index = max(
                range(len(retained)),
                key=lambda index: (
                    math.dist(retained[index].center, centroid),
                    -_score(retained[index]),
                    retained[index].element_id,
                ),
            )
            retained.pop(remove_index)
        return retained

    def shift(
        self,
        *,
        current_region: BBox,
        all_elements: Sequence[UIElement],
        full_region: BBox,
    ) -> RegionProposal | None:
        require_scored(all_elements)
        external = _ranked(
            element for element in all_elements if not center_inside(element, current_region)
        )
        if not external:
            return None
        anchors = list(
            external[: _top_count(len(external), self.config.shift_external_fraction)]
        )
        current_center = bbox_center(current_region)
        direction = _direction(current_center, anchors[0].center)
        grouped = [
            element
            for element in anchors
            if _direction(current_center, element.center) == direction
        ]
        target_center = _centroid(grouped)
        constrained_center = _enforce_shift_distance(
            target_center=target_center,
            current_center=current_center,
            direction=direction,
            current_region=current_region,
            maximum_iou=self.config.shift_max_iou,
        )
        candidate = recentered_region(
            center=constrained_center,
            width=current_region.width,
            height=current_region.height,
            bounds=full_region,
        )
        overlap = bbox_iou(current_region, candidate)
        if same_bbox(candidate, current_region) or overlap > self.config.shift_max_iou + 1e-9:
            return None
        return RegionProposal(
            action=PerceptualAction.SHIFT,
            source_region=current_region,
            region=candidate,
            selected_element_ids=tuple(element.element_id for element in grouped),
            details={
                "external_count": len(external),
                "paper_top_fraction": self.config.shift_external_fraction,
                "direction": direction,
                "iou_with_previous": overlap,
                "paper_max_iou": self.config.shift_max_iou,
            },
        )

    def scatter(
        self,
        *,
        current_region: BBox,
        all_elements: Sequence[UIElement],
        full_region: BBox,
    ) -> RegionProposal | None:
        require_scored(all_elements)
        external = _ranked(
            element for element in all_elements if not center_inside(element, current_region)
        )
        if not external:
            return None
        selected = external[
            : _top_count(len(external), self.config.scatter_external_fraction)
        ]
        maximum_area = current_region.area * self.config.scatter_max_area_scale
        candidate = current_region
        admitted: list[UIElement] = []
        for element in selected:
            expanded = clip_bbox(
                enclosing_bbox((candidate, element.bbox)),
                full_region,
            )
            if expanded.area <= maximum_area + 1e-9:
                candidate = expanded
                admitted.append(element)
        if not admitted or same_bbox(candidate, current_region):
            return None
        return RegionProposal(
            action=PerceptualAction.SCATTER,
            source_region=current_region,
            region=candidate,
            selected_element_ids=tuple(element.element_id for element in admitted),
            details={
                "external_count": len(external),
                "paper_top_fraction": self.config.scatter_external_fraction,
                "area_scale": candidate.area / current_region.area,
                "paper_max_area_scale": self.config.scatter_max_area_scale,
            },
        )


def _score(element: UIElement) -> float:
    if element.relevance is None:
        raise ValueError(f"element {element.element_id} has no relevance score")
    return element.relevance


def _ranked(elements: Iterable[UIElement]) -> tuple[UIElement, ...]:
    materialized = tuple(elements)
    return tuple(
        sorted(materialized, key=lambda element: (-_score(element), element.element_id))
    )


def _top_count(size: int, fraction: float) -> int:
    return min(size, max(1, math.ceil(size * fraction)))


def _centroid(elements: Sequence[UIElement]) -> tuple[float, float]:
    return (
        sum(element.center[0] for element in elements) / len(elements),
        sum(element.center[1] for element in elements) / len(elements),
    )


def _direction(
    origin: tuple[float, float], target: tuple[float, float]
) -> str:
    dx = target[0] - origin[0]
    dy = target[1] - origin[1]
    if abs(dx) >= abs(dy):
        return "right" if dx >= 0 else "left"
    return "down" if dy >= 0 else "up"


def _enforce_shift_distance(
    *,
    target_center: tuple[float, float],
    current_center: tuple[float, float],
    direction: str,
    current_region: BBox,
    maximum_iou: float,
) -> tuple[float, float]:
    # For equal-size boxes displaced along one axis, IoU <= t when the center
    # displacement is at least size * (1 - t) / (1 + t).
    if maximum_iou >= 1:
        return target_center
    if direction in {"left", "right"}:
        minimum = current_region.width * (1 - maximum_iou) / (1 + maximum_iou)
        sign = -1 if direction == "left" else 1
        x = target_center[0]
        if sign * (x - current_center[0]) < minimum:
            x = current_center[0] + sign * minimum
        return x, target_center[1]
    minimum = current_region.height * (1 - maximum_iou) / (1 + maximum_iou)
    sign = -1 if direction == "up" else 1
    y = target_center[1]
    if sign * (y - current_center[1]) < minimum:
        y = current_center[1] + sign * minimum
    return target_center[0], y
