from __future__ import annotations

from typing import Any

import pytest
from PIL import Image

from drsgui.evaluator import PixelPoint
from drsgui.models.base import ModelResponseError, parse_coordinate
from drsgui.models.deepseek import DeepSeekVisionModel
from drsgui.models.uground import UGroundModel


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.ok = True
        self.status_code = 200
        self.text = ""
        self.headers = {"x-request-id": "request-test"}
        self._content = content

    def json(self) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {"content": self._content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"x": 12, "y": 34}', PixelPoint(12, 34)),
        ("(12.5, 34.25)", PixelPoint(12.5, 34.25)),
        ("[12, 34]", PixelPoint(12, 34)),
        ('```json\n{"x": 12, "y": 34}\n```', PixelPoint(12, 34)),
    ],
)
def test_parse_coordinate(raw: str, expected: PixelPoint) -> None:
    assert parse_coordinate(raw) == expected


@pytest.mark.parametrize(
    "raw", ["", "no coordinate", "(1, 2) or (3, 4)", '{"x": "bad", "y": 2}']
)
def test_parse_coordinate_rejects_invalid_or_ambiguous_output(raw: str) -> None:
    with pytest.raises(ModelResponseError):
        parse_coordinate(raw)


def test_deepseek_adapter_returns_original_pixel_coordinate(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        captured["url"] = url
        captured["payload"] = kwargs["json"]
        return FakeResponse('{"x": 40, "y": 20}')

    monkeypatch.setattr("drsgui.models.base.requests.post", fake_post)
    model = DeepSeekVisionModel(api_key="test-only", base_url="https://example.test")
    prediction = model.predict(Image.new("RGB", (100, 50)), "Click Save")
    assert prediction.point == PixelPoint(40, 20)
    assert captured["url"] == "https://example.test/chat/completions"
    assert captured["payload"]["temperature"] == 0
    image_url = captured["payload"]["messages"][0]["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")


def test_uground_adapter_converts_official_0_1000_output(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        return FakeResponse("(250, 500)")

    monkeypatch.setattr("drsgui.models.base.requests.post", fake_post)
    model = UGroundModel(base_url="http://localhost:8000/v1")
    prediction = model.predict(Image.new("RGB", (3840, 1080)), "Click Save")
    assert prediction.point == PixelPoint(960, 540)
    assert prediction.metadata["source_coordinate_space"] == "normalized_0_1000"
