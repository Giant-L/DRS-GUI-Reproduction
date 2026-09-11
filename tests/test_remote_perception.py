from __future__ import annotations

from typing import Any

import pytest
from PIL import Image

from drsgui.drs.remote_perception import (
    RemotePerceptionClient,
    RemotePerceptionError,
)


class FakeResponse:
    ok = True
    status_code = 200
    text = ""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


def valid_response() -> dict[str, Any]:
    return {
        "coordinate_space": "original_image_pixels",
        "sample_id": "sample-1",
        "image_size": [100, 50],
        "source": "microsoft/OmniParser-v2.0",
        "relevance_source": "hkunlp/instructor-large",
        "elements": [
            {
                "element_id": "omni-0",
                "bbox": [10, 5, 30, 15],
                "description": "Save",
                "interactive": True,
                "relevance": 0.9,
            }
        ],
        "metadata": {"ground_truth_used": False},
    }


def test_remote_perception_builds_gt_free_request_and_validates_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(session: Any, url: str, **kwargs: Any) -> FakeResponse:
        captured["trust_env"] = session.trust_env
        captured["url"] = url
        captured["headers"] = kwargs["headers"]
        captured["payload"] = kwargs["json"]
        return FakeResponse(valid_response())

    monkeypatch.setattr("requests.Session.post", fake_post)
    client = RemotePerceptionClient("https://perception.test/", api_key="test-token")
    cache = client.perceive(
        Image.new("RGB", (100, 50)),
        "Click Save",
        sample_id="sample-1",
        application="word",
        platform="windows",
    )

    assert captured["url"] == "https://perception.test/v1/perceive"
    assert captured["trust_env"] is False
    assert captured["headers"]["X-API-Key"] == "test-token"
    assert captured["payload"]["image_data_url"].startswith("data:image/png;base64,")
    assert "gt_bbox" not in captured["payload"]
    assert cache["elements"][0]["relevance"] == 0.9
    assert cache["metadata"]["ground_truth_used"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("coordinate_space", "normalized_0_1"),
        ("sample_id", "wrong-sample"),
        ("image_size", [50, 100]),
        ("source", ""),
    ],
)
def test_remote_perception_rejects_contract_mismatch(
    monkeypatch: pytest.MonkeyPatch, field: str, value: Any
) -> None:
    payload = valid_response()
    payload[field] = value
    monkeypatch.setattr(
        "requests.Session.post",
        lambda self, *args, **kwargs: FakeResponse(payload),
    )
    client = RemotePerceptionClient("https://perception.test")
    with pytest.raises(RemotePerceptionError):
        client.perceive(
            Image.new("RGB", (100, 50)),
            "Click Save",
            sample_id="sample-1",
            application=None,
            platform=None,
        )
