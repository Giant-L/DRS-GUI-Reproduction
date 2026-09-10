"""Grounding model interfaces and backend factory."""

from __future__ import annotations

from drsgui.config import Settings
from drsgui.models.base import GroundingModel, GroundingPrediction
from drsgui.models.deepseek import DeepSeekVisionModel
from drsgui.models.uground import UGroundModel

__all__ = [
    "DeepSeekVisionModel",
    "GroundingModel",
    "GroundingPrediction",
    "UGroundModel",
    "build_model",
]


def build_model(backend: str, settings: Settings) -> GroundingModel:
    if backend == "deepseek":
        if not settings.deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for the DeepSeek backend")
        return DeepSeekVisionModel(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_api_base,
            model=settings.deepseek_model,
            timeout_seconds=settings.deepseek_timeout_seconds,
        )
    if backend == "uground":
        return UGroundModel(
            api_key=settings.uground_api_key,
            base_url=settings.uground_api_base,
            model=settings.uground_model,
            timeout_seconds=settings.uground_timeout_seconds,
        )
    raise ValueError(f"unknown grounding backend: {backend!r}")
