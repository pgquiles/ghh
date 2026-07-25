"""Tests for camera chessboard calibration (ghh calibrate)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from ghh.cli import main

# Known distortion used when rendering synthetic calibration photos
_TRUE_K1 = -0.15
_TRUE_K2 = 0.05
_SQUARE_MM = 25.0


def _render_board_views(
    out_dir: Path,
    n: int = 15,
    squares: tuple[int, int] = (9, 6),
    square_size_mm: float = _SQUARE_MM,
    k1: float = _TRUE_K1,
    k2: float = _TRUE_K2,
) -> Path:
    """Project a chessboard with known camera + distortion into synthetic photos."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cols_sq, rows_sq = squares
    w, h = 640, 480
    fx = fy = 500.0
    cx, cy = w / 2.0, h / 2.0
    camera = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)

    rng = np.random.default_rng(1)
    written = 0
    attempt = 0
    while written < n and attempt < n * 3:
        attempt += 1
        rvec = rng.normal(0, 0.25, size=3).astype(np.float64).reshape(3, 1)
        tvec = np.array(
            [
                rng.uniform(-40, 40),
                rng.uniform(-30, 30),
                rng.uniform(350, 550),
            ],
            dtype=np.float64,
        ).reshape(3, 1)

        img = np.full((h, w, 3), 200, dtype=np.uint8)
        grid = np.zeros(((cols_sq + 1) * (rows_sq + 1), 3), np.float32)
        grid[:, :2] = np.mgrid[0 : cols_sq + 1, 0 : rows_sq + 1].T.reshape(-1, 2)
        grid *= float(square_size_mm)
        pts, _ = cv2.projectPoints(grid, rvec, tvec, camera, dist)
        pts = pts.reshape(rows_sq + 1, cols_sq + 1, 2)

        # Skip poses where the board leaves the frame
        if (
            pts[:, :, 0].min() < 5
            or pts[:, :, 1].min() < 5
            or pts[:, :, 0].max() > w - 5
            or pts[:, :, 1].max() > h - 5
        ):
            continue

        for r in range(rows_sq):
            for c in range(cols_sq):
                if (r + c) % 2 == 0:
                    quad = np.array(
                        [pts[r, c], pts[r, c + 1], pts[r + 1, c + 1], pts[r + 1, c]],
                        dtype=np.int32,
                    )
                    cv2.fillConvexPoly(img, quad, (0, 0, 0))

        cv2.imwrite(
            str(out_dir / f"calib_{written:02d}.jpg"),
            img,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        written += 1

    if written < 6:
        raise RuntimeError(f"Only rendered {written} calibration views")
    return out_dir


class TestParsePattern:
    def test_parse_pattern_basic(self):
        from ghh.calibrate import parse_pattern

        assert parse_pattern("8x5") == (8, 5)
        assert parse_pattern("7X10") == (7, 10)
        assert parse_pattern(" 9×6 ") == (9, 6)

    def test_parse_pattern_invalid(self):
        import click

        from ghh.calibrate import parse_pattern

        with pytest.raises(click.BadParameter):
            parse_pattern("abc")
        with pytest.raises(click.BadParameter):
            parse_pattern("0x5")


class TestCalibrateCamera:
    def test_calibrate_synthetic_board(self, tmp_path):
        from ghh.calibrate import calibrate_camera

        img_dir = _render_board_views(tmp_path / "calib")
        result = calibrate_camera(
            img_dir,
            pattern=(8, 5),
            square_size=_SQUARE_MM,
        )

        assert result.images_used >= 6
        assert result.images_total >= 6
        assert result.rms < 1.5
        # Free calibration should recover the planted k1 closely
        assert result.k1_free == pytest.approx(_TRUE_K1, abs=0.05)
        # Stage-3 coeffs must be finite and same sign as true distortion
        assert np.isfinite(result.k1) and np.isfinite(result.k2)
        assert result.k1 < 0

    def test_calibrate_wrong_pattern_raises(self, tmp_path):
        from ghh.calibrate import CalibrationError, calibrate_camera

        img_dir = _render_board_views(tmp_path / "calib", n=8)
        with pytest.raises(CalibrationError, match="[Cc]hessboard"):
            calibrate_camera(img_dir, pattern=(3, 3), square_size=_SQUARE_MM)

    def test_calibrate_empty_dir_raises(self, tmp_path):
        from ghh.calibrate import CalibrationError, calibrate_camera

        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(CalibrationError, match="No images"):
            calibrate_camera(empty, pattern=(8, 5), square_size=25.0)

    def test_toml_snippet(self, tmp_path):
        from ghh.calibrate import calibrate_camera, format_toml_snippet

        img_dir = _render_board_views(tmp_path / "calib", n=8)
        result = calibrate_camera(img_dir, pattern=(8, 5), square_size=_SQUARE_MM)
        snippet = format_toml_snippet(result)
        assert "lens_distortion_k1" in snippet
        assert "lens_distortion_k2" in snippet
        assert "[photography]" in snippet


class TestCalibrateCLI:
    def test_calibrate_command_exists(self, tmp_path):
        img_dir = _render_board_views(tmp_path / "calib", n=8)
        runner = CliRunner()
        result = runner.invoke(main, [
            "calibrate", str(img_dir),
            "--pattern", "8x5",
            "--square-size", str(_SQUARE_MM),
        ])
        assert result.exit_code == 0, result.output
        assert "lens_distortion_k1" in result.output
        assert "lens_distortion_k2" in result.output
        assert "RMS" in result.output or "rms" in result.output.lower()

    def test_calibrate_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["calibrate", "--help"])
        assert result.exit_code == 0
        assert "--pattern" in result.output
        assert "--square-size" in result.output
