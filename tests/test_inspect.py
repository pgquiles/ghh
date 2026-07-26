"""Tests for ghh.inspect diagnostic overlays."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from ghh.inspect import (
    InspectError,
    inspect_image,
    parse_color,
    resolve_checkpoint_name,
    resolve_image_path,
    suggest_hue_params,
)


def make_synthetic_staff_image(path: Path, w: int = 800, h: int = 600) -> None:
    """Create a synthetic page with red staff lines for testing."""
    img = np.full((h, w, 3), 230, dtype=np.uint8)
    for y in range(100, 500, 24):
        cv2.line(img, (50, y), (w - 50, y), (0, 0, 200), 3)
    cv2.imwrite(str(path), img)


class TestParseColor:
    def test_hex(self):
        assert parse_color("#ff0000") == (0, 0, 255)

    def test_rgb_string(self):
        assert parse_color("255,0,0") == (0, 0, 255)

    def test_rgb_tuple(self):
        assert parse_color((0, 255, 0)) == (0, 255, 0)

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid color"):
            parse_color("red")


class TestResolveCheckpoint:
    def test_by_number(self):
        assert resolve_checkpoint_name("8") == "08_deskewed"

    def test_by_name(self):
        assert resolve_checkpoint_name("deskew") == "08_deskewed"

    def test_dewarp_alias(self):
        assert resolve_checkpoint_name("dewarp") == "08_deskewed"

    def test_checkpoint_dirname(self):
        assert resolve_checkpoint_name("08_deskewed") == "08_deskewed"

    def test_unknown_raises(self):
        with pytest.raises(InspectError, match="Unknown stage"):
            resolve_checkpoint_name("not_a_stage")


class TestResolveImagePath:
    def test_existing_file(self, tmp_path: Path):
        img = tmp_path / "page.png"
        make_synthetic_staff_image(img)
        resolved, label = resolve_image_path(img, stage="deskew")
        assert resolved == img.resolve()
        assert label == "deskew"

    def test_stem_from_checkpoint(self, tmp_path: Path):
        out = tmp_path / "book_output"
        ckpt = out / "08_deskewed"
        ckpt.mkdir(parents=True)
        img = ckpt / "IMG_0045.png"
        make_synthetic_staff_image(img)

        resolved, label = resolve_image_path(
            "IMG_0045", stage="deskew", output_dir=out,
        )
        assert resolved == img.resolve()
        assert label == "08_deskewed"

    def test_missing_raises(self, tmp_path: Path):
        with pytest.raises(InspectError, match="not found"):
            resolve_image_path("nope.png", output_dir=tmp_path)


class TestSuggestHue:
    def test_peak_near_red(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Pure red in BGR
        img[:, :] = (0, 0, 200)
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        sug = suggest_hue_params(img, mask)
        assert "staff_color_hue" in sug
        # OpenCV red hue is near 0
        assert sug["staff_color_hue"] <= 10 or sug["staff_color_hue"] >= 170
        assert sug["staff_color_range"] >= 5


class TestInspectImage:
    def test_creates_outputs(self, tmp_path: Path):
        img_path = tmp_path / "sample.png"
        make_synthetic_staff_image(img_path)
        out_path = tmp_path / "out_inspect.png"

        res = inspect_image(
            image_path=img_path,
            max_width=600,
            output_path=out_path,
            open_in_viewer=False,
        )

        assert res.exists()
        sidecar = res.with_suffix(".json")
        assert sidecar.exists()
        data = json.loads(sidecar.read_text())
        assert data["staff_lines"] >= 1
        assert "suggestions" in data
        assert "current" in data

    def test_jpeg_overlay_only(self, tmp_path: Path):
        img_path = tmp_path / "sample2.png"
        make_synthetic_staff_image(img_path)

        res = inspect_image(
            image_path=img_path,
            max_width=600,
            output_path=tmp_path / "overlay_out",
            open_in_viewer=False,
            output_format="jpg",
            overlay_only=True,
        )
        assert res.exists()
        assert res.suffix.lower() in (".jpg", ".jpeg")

    def test_bad_color_raises(self, tmp_path: Path):
        img_path = tmp_path / "sample.png"
        make_synthetic_staff_image(img_path)
        with pytest.raises(ValueError, match="Invalid color"):
            inspect_image(img_path, ink_color="not-a-color")

    def test_bad_alpha_raises(self, tmp_path: Path):
        img_path = tmp_path / "sample.png"
        make_synthetic_staff_image(img_path)
        with pytest.raises(InspectError, match="alpha"):
            inspect_image(img_path, alpha=1.5)

    def test_stem_with_stage(self, tmp_path: Path):
        out = tmp_path / "book_output"
        ckpt = out / "07_staff_extract"
        ckpt.mkdir(parents=True)
        img = ckpt / "IMG_0001.png"
        make_synthetic_staff_image(img)

        res = inspect_image(
            "IMG_0001",
            stage="7",
            output_dir=out,
            max_width=600,
            output_path=tmp_path / "from_stage.png",
        )
        assert res.exists()
        data = json.loads(res.with_suffix(".json").read_text())
        assert data["stage"] == "07_staff_extract"
        assert "IMG_0001.png" in data["source"]
