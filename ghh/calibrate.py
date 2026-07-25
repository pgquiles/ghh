"""Chessboard camera calibration for lens distortion coefficients.

Produces OpenCV radial distortion coefficients suitable for Stage 3
(``lens_correct``), which applies ``cv2.undistort`` with a simplified
camera matrix ``fx = fy = max(w, h)``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import click
import cv2
import numpy as np

from ghh.utils.image_io import load_image

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
_PATTERN_RE = re.compile(r"^\s*(\d+)\s*[x×X]\s*(\d+)\s*$")

class CalibrationError(Exception):
    """Raised when calibration cannot proceed or finds no usable boards."""


@dataclass(frozen=True)
class CalibrationResult:
    """Result of chessboard camera calibration."""

    k1: float
    k2: float
    rms: float
    images_used: int
    images_total: int
    pattern: tuple[int, int]
    square_size: float
    image_size: tuple[int, int]  # (width, height)
    # Free (unconstrained) calibration extras for lensfun / diagnostics
    fx: float
    fy: float
    cx: float
    cy: float
    k1_free: float
    k2_free: float
    k3_free: float
    p1_free: float
    p2_free: float
    rms_free: float


def parse_pattern(value: str) -> tuple[int, int]:
    """Parse ``COLSxROWS`` inner-corner pattern size for OpenCV."""
    match = _PATTERN_RE.match(value or "")
    if not match:
        raise click.BadParameter(
            f"Invalid pattern '{value}'. Expected COLSxROWS inner corners "
            f"(e.g. 8x5 for a BoofCV 9x6-square board)."
        )
    cols, rows = int(match.group(1)), int(match.group(2))
    if cols < 2 or rows < 2:
        raise click.BadParameter(
            f"Pattern must be at least 2x2 inner corners, got {cols}x{rows}."
        )
    return cols, rows


def calibrate_camera(
    input_dir: Path,
    pattern: tuple[int, int],
    square_size: float,
) -> CalibrationResult:
    """Calibrate lens distortion from chessboard photos in *input_dir*.

    Parameters
    ----------
    input_dir:
        Directory of calibration photos.
    pattern:
        OpenCV inner-corner grid size ``(cols, rows)``.
    square_size:
        Physical square size in millimetres (scale-invariant for k1/k2,
        but reported for completeness).
    """
    input_dir = Path(input_dir)
    paths = _list_images(input_dir)
    if not paths:
        raise CalibrationError(f"No images found in {input_dir}")

    if square_size <= 0:
        raise CalibrationError("square_size must be positive")

    patterns_to_try = [pattern]
    swapped = (pattern[1], pattern[0])
    if swapped != pattern:
        patterns_to_try.append(swapped)

    best: tuple[tuple[int, int], list, list, tuple[int, int]] | None = None
    for pat in patterns_to_try:
        objpoints, imgpoints, size = _detect_boards(paths, pat, square_size)
        logger.info(
            "Pattern %dx%d: detected board in %d/%d images",
            pat[0], pat[1], len(objpoints), len(paths),
        )
        if best is None or len(objpoints) > len(best[1]):
            best = (pat, objpoints, imgpoints, size)

    assert best is not None
    used_pattern, objpoints, imgpoints, (w, h) = best

    if len(objpoints) < 3:
        raise CalibrationError(
            f"Chessboard not found in enough images "
            f"({len(objpoints)}/{len(paths)} with pattern "
            f"{used_pattern[0]}x{used_pattern[1]}). "
            f"Hint: --pattern is OpenCV *inner corners*. "
            f"BoofCV A4 9x6 squares → --pattern 8x5."
        )

    # Free calibration (for lensfun / diagnostics)
    rms_free, mtx, dist, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None,
    )
    dist = dist.ravel()

    # Refit k1/k2 for Stage 3's simplified camera matrix
    # (fx = fy = max(w, h), principal point at image centre).
    k1_s3, k2_s3 = _fit_stage3_k1_k2(mtx, dist, w, h)

    return CalibrationResult(
        k1=k1_s3,
        k2=k2_s3,
        rms=float(rms_free),
        images_used=len(objpoints),
        images_total=len(paths),
        pattern=used_pattern,
        square_size=float(square_size),
        image_size=(w, h),
        fx=float(mtx[0, 0]),
        fy=float(mtx[1, 1]),
        cx=float(mtx[0, 2]),
        cy=float(mtx[1, 2]),
        k1_free=float(dist[0]),
        k2_free=float(dist[1]),
        p1_free=float(dist[2]),
        p2_free=float(dist[3]),
        k3_free=float(dist[4]) if len(dist) > 4 else 0.0,
        rms_free=float(rms_free),
    )


def format_toml_snippet(result: CalibrationResult) -> str:
    """Format Stage-3-compatible coefficients for book.toml."""
    return (
        "[photography]\n"
        f"lens_distortion_k1 = {result.k1}\n"
        f"lens_distortion_k2 = {result.k2}\n"
    )


def format_report(result: CalibrationResult) -> str:
    """Human-readable calibration report."""
    lines = [
        f"Detected chessboard in {result.images_used}/{result.images_total} "
        f"images (pattern {result.pattern[0]}x{result.pattern[1]} inner corners)",
        f"Image size: {result.image_size[0]}x{result.image_size[1]}",
        f"Square size: {result.square_size:g} mm",
        "",
        f"RMS reprojection error: {result.rms:.4f} px",
        "",
        "Stage 3 coefficients (for book.toml):",
        f"  k1 = {result.k1}",
        f"  k2 = {result.k2}",
        "",
        format_toml_snippet(result).rstrip(),
        "",
        "Free calibration (for lensfun / diagnostics):",
        f"  fx={result.fx:.1f} fy={result.fy:.1f} "
        f"cx={result.cx:.1f} cy={result.cy:.1f}",
        f"  k1={result.k1_free:.6f} k2={result.k2_free:.6f} "
        f"k3={result.k3_free:.6f}",
        f"  p1={result.p1_free:.6f} p2={result.p2_free:.6f}",
    ]
    if result.rms > 1.5:
        lines.extend([
            "",
            "Note: RMS > 1.5 px — target may not be flat, or coverage is "
            "limited. Coefficients can still be used; retake on a rigid "
            "board for a lensfun submission.",
        ])
    return "\n".join(lines) + "\n"


def _fit_stage3_k1_k2(
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    w: int,
    h: int,
) -> tuple[float, float]:
    """Convert free-calibration k1/k2 to Stage 3's simplified camera model.

    Stage 3 calls ``cv2.undistort`` with ``fx = fy = max(w, h)``.  OpenCV
    radial coefficients are defined in focal-length-normalized coordinates,
    so they rescale as::

        k1_s3 = k1 * (fx_s3 / fx) ** 2
        k2_s3 = k2 * (fx_s3 / fx) ** 4
    """
    fx = float(camera_matrix[0, 0])
    if fx <= 0:
        return 0.0, 0.0
    fx_s3 = float(max(w, h))
    scale = fx_s3 / fx
    k1 = float(dist_coeffs.ravel()[0]) * scale**2
    k2 = float(dist_coeffs.ravel()[1]) * scale**4
    return round(k1, 8), round(k2, 8)


def _list_images(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )


def _detect_boards(
    paths: list[Path],
    pattern: tuple[int, int],
    square_size: float,
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int]]:
    """Detect chessboards; return object points, image points, and size.

    Images that do not match the majority (width, height) are skipped so
    landscape and portrait frames are never mixed in one calibration.
    """
    cols, rows = pattern
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size)

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    term = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    # First pass: load greyscale images and pick the dominant size
    loaded: list[tuple[Path, np.ndarray]] = []
    size_counts: dict[tuple[int, int], int] = {}
    for path in paths:
        img, _meta = load_image(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        size_counts[(w, h)] = size_counts.get((w, h), 0) + 1
        loaded.append((path, gray))

    if not loaded:
        return [], [], (0, 0)

    target_size = max(size_counts, key=size_counts.get)  # type: ignore[arg-type]
    if len(size_counts) > 1:
        skipped = sum(c for s, c in size_counts.items() if s != target_size)
        logger.info(
            "Using image size %dx%d; skipping %d image(s) with other sizes",
            target_size[0], target_size[1], skipped,
        )

    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    w, h = target_size

    for path, gray in loaded:
        if gray.shape[1] != w or gray.shape[0] != h:
            logger.debug("Skipping %s (size mismatch)", path.name)
            continue

        # Downscale for faster detection, refine on full-resolution
        scale = 0.5 if max(w, h) > 1200 else 1.0
        if scale != 1.0:
            small = cv2.resize(
                gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA,
            )
        else:
            small = gray

        ok, corners = cv2.findChessboardCorners(small, pattern, flags)
        if not ok:
            logger.debug("No board in %s for pattern %s", path.name, pattern)
            continue

        if scale != 1.0:
            corners = (corners / scale).astype(np.float32)

        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
        objpoints.append(objp.copy())
        imgpoints.append(refined)

    return objpoints, imgpoints, target_size
