from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from drsgui.dataset import DatasetFormatError, ScreenSpotProDataset


def _make_dataset(root: Path, *, actual_size: tuple[int, int] = (100, 50)) -> Path:
    annotations = root / "annotations"
    image_dir = root / "images" / "demo_windows"
    annotations.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    Image.new("RGB", actual_size).save(image_dir / "shot.png")
    row = {
        "img_filename": "demo_windows/shot.png",
        "bbox": [10, 10, 30, 30],
        "instruction": "Click Save",
        "instruction_cn": "点击保存",
        "id": "demo_0",
        "application": "demo",
        "platform": "windows",
        "img_size": [100, 50],
        "ui_type": "text",
        "group": "Office",
    }
    (annotations / "demo_windows.json").write_text(
        json.dumps([row]), encoding="utf-8"
    )
    return root


def test_load_screenspot_pro_sample(tmp_path: Path) -> None:
    dataset = ScreenSpotProDataset(_make_dataset(tmp_path))
    sample = dataset[0]
    assert len(dataset) == 1
    assert sample.sample_id == "demo_0"
    assert sample.image_width == 100
    assert sample.image_height == 50
    assert sample.gt_bbox.to_list() == [10.0, 10.0, 30.0, 30.0]
    assert sample.image_path.is_file()
    assert dataset.validate_images(limit=1)[0]["image_size"] == [100, 50]


def test_load_chinese_instruction(tmp_path: Path) -> None:
    dataset = ScreenSpotProDataset(_make_dataset(tmp_path), language="cn")
    assert dataset[0].instruction == "点击保存"


def test_validate_image_detects_annotation_size_mismatch(tmp_path: Path) -> None:
    dataset = ScreenSpotProDataset(_make_dataset(tmp_path, actual_size=(80, 50)))
    with pytest.raises(DatasetFormatError, match="image size mismatch"):
        dataset.validate_images(limit=1)


def test_loader_rejects_path_traversal(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path)
    annotation = root / "annotations" / "demo_windows.json"
    rows = json.loads(annotation.read_text(encoding="utf-8"))
    rows[0]["img_filename"] = "../../outside.png"
    annotation.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(DatasetFormatError, match="escapes"):
        ScreenSpotProDataset(root, require_images=False)


def test_loader_preserves_partially_out_of_bounds_official_bbox(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path)
    annotation = root / "annotations" / "demo_windows.json"
    rows = json.loads(annotation.read_text(encoding="utf-8"))
    rows[0]["bbox"] = [10, -1, 30, 20]
    annotation.write_text(json.dumps(rows), encoding="utf-8")
    dataset = ScreenSpotProDataset(root)
    assert dataset[0].gt_bbox.to_list() == [10.0, -1.0, 30.0, 20.0]


def test_loader_rejects_bbox_fully_outside_image(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path)
    annotation = root / "annotations" / "demo_windows.json"
    rows = json.loads(annotation.read_text(encoding="utf-8"))
    rows[0]["bbox"] = [10, -30, 30, -1]
    annotation.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(DatasetFormatError, match="does not intersect"):
        ScreenSpotProDataset(root)
