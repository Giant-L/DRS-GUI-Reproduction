"""ScreenSpot-Pro dataset abstractions and loader."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from PIL import Image


class DatasetFormatError(ValueError):
    """Raised when a dataset annotation violates the expected schema."""


@dataclass(frozen=True, slots=True)
class BBox:
    """Bounding box in original-image pixel coordinates: [x1, y1, x2, y2]."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"bbox values must be finite, got {values}")
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"invalid bbox ordering: {values}")

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> "BBox":
        if len(values) != 4:
            raise ValueError(f"bbox must contain four values, got {values!r}")
        return cls(*(float(value) for value in values))

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(frozen=True, slots=True)
class GroundingSample:
    """One GUI-grounding task with an original screenshot and target bbox."""

    sample_id: str
    image_path: Path
    image_filename: str
    instruction: str
    gt_bbox: BBox
    image_width: int
    image_height: int
    ui_type: str | None = None
    application: str | None = None
    platform: str | None = None
    group: str | None = None
    annotation_file: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if not self.instruction.strip():
            raise ValueError(f"instruction is empty for sample {self.sample_id}")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError(
                f"invalid image size for {self.sample_id}: "
                f"{self.image_width}x{self.image_height}"
            )
        if (
            self.gt_bbox.x1 < 0
            or self.gt_bbox.y1 < 0
            or self.gt_bbox.x2 > self.image_width
            or self.gt_bbox.y2 > self.image_height
        ):
            raise ValueError(
                f"bbox {self.gt_bbox.to_list()} lies outside "
                f"{self.image_width}x{self.image_height} for {self.sample_id}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "image_path": str(self.image_path),
            "image_filename": self.image_filename,
            "instruction": self.instruction,
            "gt_bbox": self.gt_bbox.to_list(),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "ui_type": self.ui_type,
            "application": self.application,
            "platform": self.platform,
            "group": self.group,
            "annotation_file": self.annotation_file,
            "metadata": dict(self.metadata),
        }


class ScreenSpotProDataset(Sequence[GroundingSample]):
    """Load official ScreenSpot-Pro JSON annotations from a local snapshot."""

    def __init__(
        self,
        root: str | Path,
        *,
        language: str = "en",
        require_images: bool = True,
    ) -> None:
        self.root = Path(root).expanduser()
        self.annotation_root = self.root / "annotations"
        self.image_root = self.root / "images"
        self.language = language
        self.require_images = require_images

        if language not in {"en", "cn"}:
            raise ValueError("language must be 'en' or 'cn'")
        if not self.annotation_root.is_dir():
            raise FileNotFoundError(
                f"ScreenSpot-Pro annotations not found: {self.annotation_root}"
            )
        if require_images and not self.image_root.is_dir():
            raise FileNotFoundError(
                f"ScreenSpot-Pro images not found: {self.image_root}"
            )

        self._samples = self._load_samples()
        self._by_id = {sample.sample_id: sample for sample in self._samples}
        if len(self._by_id) != len(self._samples):
            raise DatasetFormatError("duplicate sample ids found in annotations")

    def _load_samples(self) -> list[GroundingSample]:
        annotation_paths = sorted(self.annotation_root.glob("*.json"))
        if not annotation_paths:
            raise FileNotFoundError(
                f"no JSON annotations found in {self.annotation_root}"
            )

        samples: list[GroundingSample] = []
        for annotation_path in annotation_paths:
            try:
                rows = json.loads(annotation_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise DatasetFormatError(
                    f"failed to read annotation {annotation_path}: {exc}"
                ) from exc
            if not isinstance(rows, list):
                raise DatasetFormatError(
                    f"annotation must be a list: {annotation_path}"
                )
            for row_index, row in enumerate(rows):
                samples.append(
                    self._parse_row(row, annotation_path=annotation_path, row_index=row_index)
                )
        return samples

    def _parse_row(
        self,
        row: Any,
        *,
        annotation_path: Path,
        row_index: int,
    ) -> GroundingSample:
        context = f"{annotation_path.name}[{row_index}]"
        if not isinstance(row, dict):
            raise DatasetFormatError(f"{context} must be a JSON object")

        required = {"id", "img_filename", "bbox", "img_size"}
        missing = sorted(required.difference(row))
        instruction_key = "instruction" if self.language == "en" else "instruction_cn"
        if instruction_key not in row:
            missing.append(instruction_key)
        if missing:
            raise DatasetFormatError(f"{context} missing fields: {missing}")

        img_size = row["img_size"]
        if not isinstance(img_size, list) or len(img_size) != 2:
            raise DatasetFormatError(f"{context} has invalid img_size: {img_size!r}")
        try:
            image_width, image_height = (int(value) for value in img_size)
            bbox = BBox.from_sequence(row["bbox"])
        except (TypeError, ValueError) as exc:
            raise DatasetFormatError(f"{context} has invalid geometry: {exc}") from exc

        image_filename = str(row["img_filename"])
        image_path = self._safe_image_path(image_filename, context)
        if self.require_images and not image_path.is_file():
            raise FileNotFoundError(f"image referenced by {context} is missing: {image_path}")

        known = {
            "id",
            "img_filename",
            "bbox",
            "instruction",
            "instruction_cn",
            "img_size",
            "ui_type",
            "application",
            "platform",
            "group",
        }
        try:
            return GroundingSample(
                sample_id=str(row["id"]),
                image_path=image_path,
                image_filename=image_filename,
                instruction=str(row[instruction_key]),
                gt_bbox=bbox,
                image_width=image_width,
                image_height=image_height,
                ui_type=_optional_string(row.get("ui_type")),
                application=_optional_string(row.get("application")),
                platform=_optional_string(row.get("platform")),
                group=_optional_string(row.get("group")),
                annotation_file=annotation_path.name,
                metadata={key: value for key, value in row.items() if key not in known},
            )
        except ValueError as exc:
            raise DatasetFormatError(f"{context} is invalid: {exc}") from exc

    def _safe_image_path(self, image_filename: str, context: str) -> Path:
        image_root = self.image_root.resolve()
        image_path = (image_root / image_filename).resolve()
        if image_path != image_root and image_root not in image_path.parents:
            raise DatasetFormatError(
                f"{context} image path escapes the images directory: {image_filename!r}"
            )
        return image_path

    def validate_images(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Open images and verify actual dimensions against annotations."""

        selected = self._samples if limit is None else self._samples[:limit]
        results: list[dict[str, Any]] = []
        for sample in selected:
            with Image.open(sample.image_path) as image:
                image.load()
                actual_size = image.size
            expected_size = (sample.image_width, sample.image_height)
            if actual_size != expected_size:
                raise DatasetFormatError(
                    f"image size mismatch for {sample.sample_id}: "
                    f"annotation={expected_size}, actual={actual_size}"
                )
            results.append(
                {
                    "sample_id": sample.sample_id,
                    "image_filename": sample.image_filename,
                    "image_size": list(actual_size),
                }
            )
        return results

    def get(self, sample_id: str) -> GroundingSample:
        try:
            return self._by_id[sample_id]
        except KeyError as exc:
            raise KeyError(f"unknown ScreenSpot-Pro sample id: {sample_id}") from exc

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int | slice) -> GroundingSample | list[GroundingSample]:
        return self._samples[index]

    def __iter__(self) -> Iterator[GroundingSample]:
        return iter(self._samples)


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)
