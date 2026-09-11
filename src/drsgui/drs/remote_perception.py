"""Client for a remote OmniParser V2 + Instructor-large perception service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import requests

from drsgui.drs.types import UIElement
from drsgui.models.base import ImageInput, image_to_data_url, load_image


class RemotePerceptionError(RuntimeError):
    """Raised when remote perception fails or violates the cache contract."""


@dataclass(frozen=True, slots=True)
class RemotePerceptionClient:
    """Fetch paper-aligned, scored UI elements without sending ground truth."""

    base_url: str
    api_key: str = ""
    timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("remote perception base URL must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("remote perception timeout must be positive")

    def perceive(
        self,
        image: ImageInput,
        instruction: str,
        *,
        sample_id: str,
        application: str | None,
        platform: str | None,
    ) -> dict[str, Any]:
        if not instruction.strip():
            raise ValueError("instruction must not be empty")
        screenshot = load_image(image)
        payload = {
            "sample_id": sample_id,
            "instruction": instruction,
            "application": application,
            "platform": platform,
            "image_data_url": image_to_data_url(screenshot),
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        endpoint = f"{self.base_url.rstrip('/')}/v1/perceive"
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise RemotePerceptionError(
                f"request to perception service failed: {exc}"
            ) from exc
        if not response.ok:
            raise RemotePerceptionError(
                f"perception service returned HTTP {response.status_code}: "
                f"{response.text[:1000]}"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise RemotePerceptionError("perception service returned invalid JSON") from exc
        return _validated_cache_payload(
            data,
            expected_sample_id=sample_id,
            expected_size=screenshot.size,
        )


def _validated_cache_payload(
    data: Mapping[str, Any],
    *,
    expected_sample_id: str,
    expected_size: tuple[int, int],
) -> dict[str, Any]:
    if data.get("coordinate_space") != "original_image_pixels":
        raise RemotePerceptionError(
            "perception response must declare original_image_pixels"
        )
    if data.get("sample_id") != expected_sample_id:
        raise RemotePerceptionError("perception response sample_id mismatch")
    if list(data.get("image_size") or []) != list(expected_size):
        raise RemotePerceptionError("perception response image_size mismatch")
    source = str(data.get("source") or "")
    relevance_source = str(data.get("relevance_source") or "")
    if not source or not relevance_source:
        raise RemotePerceptionError("perception response is missing provenance")
    raw_elements = data.get("elements")
    if not isinstance(raw_elements, list):
        raise RemotePerceptionError("perception response elements must be a list")
    try:
        elements = [UIElement.from_dict(value) for value in raw_elements]
    except (TypeError, ValueError) as exc:
        raise RemotePerceptionError(f"invalid UI element response: {exc}") from exc
    ids = [element.element_id for element in elements]
    if len(ids) != len(set(ids)):
        raise RemotePerceptionError("perception response element ids are not unique")
    if any(element.relevance is None for element in elements):
        raise RemotePerceptionError("perception response contains unscored elements")
    return {
        "coordinate_space": "original_image_pixels",
        "sample_id": expected_sample_id,
        "image_size": list(expected_size),
        "source": source,
        "relevance_source": relevance_source,
        "elements": [element.to_dict() for element in elements],
        "metadata": dict(data.get("metadata") or {}),
    }
