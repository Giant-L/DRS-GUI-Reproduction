"""Environment-backed configuration without secret persistence."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings loaded from environment variables and an optional .env."""

    dataset_dir: Path
    output_dir: Path
    deepseek_api_key: str | None
    deepseek_api_base: str
    deepseek_model: str
    deepseek_timeout_seconds: float
    uground_api_base: str
    uground_api_key: str
    uground_model: str
    uground_timeout_seconds: float
    perception_api_base: str
    perception_api_key: str
    perception_timeout_seconds: float
    perception_cache_dir: Path

    @classmethod
    def from_env(cls, env_file: str | Path | None = ".env") -> "Settings":
        if env_file is not None:
            load_dotenv(dotenv_path=env_file, override=False)

        return cls(
            dataset_dir=Path(
                os.getenv("SCREENSPOT_PRO_DIR", "data/ScreenSpot-Pro")
            ).expanduser(),
            output_dir=Path(os.getenv("OUTPUT_DIR", "outputs")).expanduser(),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY") or None,
            deepseek_api_base=os.getenv(
                "DEEPSEEK_API_BASE", "https://api.deepseek.com"
            ),
            deepseek_model=os.getenv(
                "DEEPSEEK_MODEL", "deepseek-v4-flash-vision-exp"
            ),
            deepseek_timeout_seconds=_positive_float_env(
                "DEEPSEEK_TIMEOUT_SECONDS", 120.0
            ),
            uground_api_base=os.getenv(
                "UGROUND_API_BASE", "http://localhost:8000/v1"
            ),
            uground_api_key=os.getenv("UGROUND_API_KEY", "EMPTY"),
            uground_model=os.getenv("UGROUND_MODEL", "osunlp/UGround-V1-2B"),
            uground_timeout_seconds=_positive_float_env(
                "UGROUND_TIMEOUT_SECONDS", 120.0
            ),
            perception_api_base=os.getenv(
                "PERCEPTION_API_BASE", "http://127.0.0.1:8010"
            ),
            perception_api_key=os.getenv("PERCEPTION_API_KEY", ""),
            perception_timeout_seconds=_positive_float_env(
                "PERCEPTION_TIMEOUT_SECONDS", 300.0
            ),
            perception_cache_dir=Path(
                os.getenv("PERCEPTION_CACHE_DIR", "tasks/perception_cache")
            ).expanduser(),
        )


def _positive_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value
