"""Inspect a single image and render diagnostic overlays.

Produces an annotated image showing ink mask, staff lines, polynomial
fits, page quad, and foxing spots. Also emits parameter suggestions for
``staff_color_hue`` / ``staff_color_range`` from the ink hue histogram.

Invoked from the CLI as ``ghh inspect IMAGE [--stage STAGE]``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import cv2
import numpy as np

from ghh.config import Config
from ghh.stages import STAGE_BY_NAME, STAGE_BY_NUMBER, STAGE_CLASSES
from ghh.utils.image_io import load_image
from ghh.utils.line_detect import (
    detect_ink_mask,
    detect_ink_mask_geometric,
    detect_staff_lines,
)
from ghh.utils.page_find import find_page_quad

logger = logging.getLogger(__name__)

# PLAN.md and older docs still say "--stage dewarp"
_STAGE_ALIASES = {
    "dewarp": "deskew",
}

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


class InspectError(Exception):
    """Raised when inspect cannot resolve inputs or write output."""


def parse_color(value: str | tuple | list) -> tuple[int, int, int]:
    """Parse a color specifier into an OpenCV BGR tuple.

    Accepts ``"#RRGGBB"``, ``"R,G,B"`` (0-255), or a 3-tuple/list in RGB order.
    """
    if isinstance(value, (tuple, list)):
        if len(value) != 3:
            raise ValueError(f"Color tuple must have 3 channels, got {value!r}")
        r, g, b = (int(v) for v in value)
        return b, g, r

    if not isinstance(value, str):
        raise ValueError(f"Invalid color specification: {value!r}")

    s = value.strip()
    if s.startswith("#") and len(s) == 7:
        r = int(s[1:3], 16)
        g = int(s[3:5], 16)
        b = int(s[5:7], 16)
        return b, g, r

    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Expected R,G,B got {value!r}")
        r, g, b = (int(p) for p in parts)
        for c in (r, g, b):
            if not 0 <= c <= 255:
                raise ValueError(f"Color channel out of range 0-255: {value!r}")
        return b, g, r

    raise ValueError(
        f"Invalid color {value!r}. Use #RRGGBB or R,G,B (0-255)."
    )


def resolve_checkpoint_name(stage: str) -> str:
    """Map a stage number/name/alias to a checkpoint directory name."""
    key = stage.strip().lower()
    key = _STAGE_ALIASES.get(key, key)

    if key.isdigit():
        num = int(key)
        if num not in STAGE_BY_NUMBER:
            raise InspectError(
                f"Unknown stage number {num}. "
                f"Available: {sorted(STAGE_BY_NUMBER)}"
            )
        return STAGE_BY_NUMBER[num].checkpoint_name

    if key in STAGE_BY_NAME:
        return STAGE_BY_NAME[key].checkpoint_name

    # Accept checkpoint directory names directly (e.g. 08_deskewed)
    for cls in STAGE_CLASSES:
        if cls.checkpoint_name.lower() == key:
            return cls.checkpoint_name
        # Allow bare suffix match: "deskewed" → 08_deskewed
        suffix = cls.checkpoint_name.lstrip("0123456789").lstrip("_").lower()
        if suffix == key:
            return cls.checkpoint_name

    known = sorted({c.name for c in STAGE_CLASSES} | {c.checkpoint_name for c in STAGE_CLASSES})
    raise InspectError(
        f"Unknown stage {stage!r}. Use a number, name, or checkpoint "
        f"dir (e.g. deskew, 8, 08_deskewed). Known: {', '.join(known)}"
    )


def resolve_image_path(
    image_ref: str | Path,
    stage: str | None = None,
    output_dir: str | Path | None = None,
) -> tuple[Path, str | None]:
    """Resolve *image_ref* to an on-disk image path.

    * If *image_ref* is an existing file, return it (stage is informational).
    * Otherwise treat it as a stem (``IMG_0045`` / ``IMG_0045.jpg``) and look
      under ``output_dir`` / ``<checkpoint>`` when ``stage`` is set, or search
      all checkpoints.
    """
    ref = Path(image_ref)
    if ref.is_file():
        return ref.resolve(), stage

    stem = ref.stem if ref.suffix else ref.name
    out = Path(output_dir).resolve() if output_dir else _guess_output_dir(ref)

    search_dirs: list[Path] = []
    stage_label = stage
    if stage and out is not None:
        ckpt = resolve_checkpoint_name(stage)
        stage_label = ckpt
        search_dirs.append(out / ckpt)
        # Branch subdirs (book/ / score/)
        for branch in ("book", "score"):
            search_dirs.append(out / branch / ckpt)
    elif out is not None:
        search_dirs.extend(sorted(p for p in out.iterdir() if p.is_dir()))
        for branch in ("book", "score"):
            bdir = out / branch
            if bdir.is_dir():
                search_dirs.extend(sorted(p for p in bdir.iterdir() if p.is_dir()))

    for directory in search_dirs:
        for ext in _IMAGE_EXTENSIONS:
            candidate = directory / f"{stem}{ext}"
            if candidate.is_file():
                return candidate.resolve(), stage_label or directory.name

    # Also try ref as a path relative to cwd that almost exists
    if out is None and stage:
        raise InspectError(
            f"Cannot resolve {image_ref!r} with --stage {stage}: "
            f"pass an existing image path or --output-dir pointing at a "
            f"pipeline output directory."
        )

    raise InspectError(
        f"Image not found: {image_ref}. Pass a file path, or a stem "
        f"with --stage / --output-dir to load a checkpoint."
    )


def suggest_hue_params(img: np.ndarray, mask: np.ndarray) -> dict:
    """Suggest ``staff_color_hue`` and ``staff_color_range`` from ink hues."""
    if img.ndim != 3:
        return {}

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    vals = hsv[:, :, 0][mask.astype(bool)]
    if vals.size == 0:
        return {}

    hist = cv2.calcHist([vals.astype(np.uint8)], [0], None, [180], [0, 180]).ravel()
    peak = int(np.argmax(hist))

    # Width around the peak that covers ~80% of mass near the peak
    # (circular on OpenCV's 0-179 hue wheel).
    total = float(hist.sum())
    if total <= 0:
        return {"staff_color_hue": peak, "staff_color_range": 15}

    target = 0.8 * total
    acc = float(hist[peak])
    half = 0
    while acc < target and half < 90:
        half += 1
        acc += float(hist[(peak - half) % 180] + hist[(peak + half) % 180])

    return {
        "staff_color_hue": peak,
        "staff_color_range": max(5, half),
    }


def inspect_image(
    image_path: str | Path,
    stage: str | None = None,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    max_width: int = 1600,
    output_path: str | Path | None = None,
    open_in_viewer: bool = False,
    output_format: str = "png",
    overlay_only: bool = False,
    alpha: float = 0.45,
    ink_color: str | tuple = "#00ff00",
    geom_color: str | tuple = "#ffff00",
    foxing_color: str | tuple = "#0000ff",
    staff_seg_color: str | tuple = "#ff0000",
    poly_color: str | tuple = "#00ffff",
    quad_color: str | tuple = "#ff00ff",
    show_coeffs: bool = True,
    rms_only: bool = False,
    font_scale: float = 0.45,
    show_line_text: bool = True,
) -> Path:
    """Generate a diagnostic overlay for a single image.

    Returns the path to the annotated image written to disk.
    """
    if not 0.0 <= alpha <= 1.0:
        raise InspectError(f"--alpha must be between 0 and 1, got {alpha}")

    resolved, stage_label = resolve_image_path(
        image_path, stage=stage, output_dir=output_dir,
    )

    ink_bgr = parse_color(ink_color)
    geom_bgr = parse_color(geom_color)
    fox_bgr = parse_color(foxing_color)
    staff_bgr = parse_color(staff_seg_color)
    poly_bgr = parse_color(poly_color)
    quad_bgr = parse_color(quad_color)

    img, _meta = load_image(resolved)
    cfg = _load_config(resolved, config_path, output_dir)

    h0, w0 = img.shape[:2]
    if w0 > max_width:
        scale = max_width / float(w0)
        img_small = cv2.resize(
            img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA,
        )
    else:
        img_small = img

    color_mask = detect_ink_mask(img_small, cfg)
    geom_mask = detect_ink_mask_geometric(img_small, cfg)
    foxing = cv2.bitwise_and(color_mask, cv2.bitwise_not(geom_mask))

    # Build color overlays on a blank layer, then blend once onto the base
    color_layer = np.zeros_like(img_small)
    color_layer[color_mask.astype(bool)] = ink_bgr
    color_layer[geom_mask.astype(bool)] = geom_bgr
    mask_any = (color_mask.astype(bool) | geom_mask.astype(bool))
    annotated = img_small.copy()
    if np.any(mask_any):
        blended = cv2.addWeighted(img_small, 1.0 - alpha, color_layer, alpha, 0)
        annotated[mask_any] = blended[mask_any]

    _draw_foxing(annotated, foxing, fox_bgr)

    staff_lines = detect_staff_lines(img_small, cfg)
    for sl in staff_lines:
        _draw_staff_line(
            annotated, sl,
            staff_bgr=staff_bgr,
            poly_bgr=poly_bgr,
            show_line_text=show_line_text,
            show_coeffs=show_coeffs,
            rms_only=rms_only,
            font_scale=font_scale,
        )

    quad = find_page_quad(img_small)
    if quad is not None:
        quad_int = np.round(quad).astype(np.int32)
        for i in range(4):
            a = tuple(int(v) for v in quad_int[i])
            b = tuple(int(v) for v in quad_int[(i + 1) % 4])
            cv2.line(annotated, a, b, quad_bgr, 2)

    suggestions = suggest_hue_params(img_small, color_mask)

    if overlay_only:
        out_img = annotated
    else:
        out_img = _compose_panel(
            annotated,
            image_name=resolved.name,
            stage_label=stage_label or stage or "(none)",
            ink_pixels=int(np.count_nonzero(color_mask)),
            staff_count=len(staff_lines),
            suggestions=suggestions,
            cfg=cfg,
        )

    out_path = _output_path(resolved, output_path, output_format)
    _write_image(out_path, out_img, output_format)

    sidecar = out_path.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "source": str(resolved),
        "stage": stage_label or stage,
        "suggestions": suggestions,
        "staff_lines": len(staff_lines),
        "ink_pixels": int(np.count_nonzero(color_mask)),
        "current": {
            "staff_color_hue": cfg.staff_color_hue,
            "staff_color_range": cfg.staff_color_range,
        },
    }, indent=2))

    if open_in_viewer:
        try:
            import webbrowser
            webbrowser.open(out_path.resolve().as_uri())
        except Exception:
            logger.exception("Failed to open image in viewer")

    return out_path


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _guess_output_dir(ref: Path) -> Path | None:
    """Best-effort discovery of a pipeline ``*_output`` directory."""
    # CWD / <stem>_output, parent_output, or literal nearby *_output
    cwd = Path.cwd()
    candidates = [
        cwd / f"{ref.stem}_output",
        cwd / "output",
    ]
    # If ref looks like photos/IMG_x, try photos_output
    if ref.parent != Path("."):
        candidates.insert(0, ref.parent.parent / f"{ref.parent.name}_output")
        candidates.insert(0, ref.parent / f"{ref.parent.name}_output")

    for c in candidates:
        if c.is_dir():
            return c.resolve()

    # Any *_output in cwd
    outs = sorted(cwd.glob("*_output"))
    if len(outs) == 1 and outs[0].is_dir():
        return outs[0].resolve()
    return None


def _load_config(
    image_path: Path,
    config_path: str | Path | None,
    output_dir: str | Path | None,
) -> Config:
    if config_path:
        return Config.from_toml(image_path.parent, toml_path=Path(config_path))

    candidates = [
        image_path.parent / "book.toml",
        image_path.parent.parent / "book.toml",
    ]
    if output_dir:
        candidates.insert(0, Path(output_dir) / "book.toml")
    guessed = _guess_output_dir(image_path)
    if guessed is not None:
        candidates.append(guessed / "book.toml")

    for c in candidates:
        if c.is_file():
            return Config.from_toml(image_path.parent, toml_path=c)
    return Config(input_dir=image_path.parent)


def _draw_poly(
    img: np.ndarray,
    coeffs: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    h, w = img.shape[:2]
    xs = np.arange(0, w, dtype=np.float64)
    ys = np.polyval(np.atleast_1d(coeffs), xs)
    pts = np.column_stack([xs, ys])
    pts = pts[(pts[:, 1] >= 0) & (pts[:, 1] < h)]
    if len(pts) < 2:
        return
    cv2.polylines(
        img,
        [np.round(pts).astype(np.int32)],
        isClosed=False,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def _draw_foxing(
    img: np.ndarray,
    foxing: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    """Circle compact non-linear ink blobs (likely foxing), not every leftover pixel."""
    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        foxing, connectivity=8,
    )
    h, w = img.shape[:2]
    min_area = max(20, int(0.00002 * h * w))
    max_area = int(0.01 * h * w)
    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        if area < min_area or area > max_area:
            continue
        aspect = bw / max(bh, 1)
        if aspect > 4.0 or aspect < 0.25:
            # Elongated → probably residual line fragments, not foxing spots
            continue
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        radius = max(6, int(max(bw, bh) / 2))
        cv2.circle(img, (cx, cy), radius, color, 2, lineType=cv2.LINE_AA)


def _draw_staff_line(
    img: np.ndarray,
    sl,
    *,
    staff_bgr: tuple[int, int, int],
    poly_bgr: tuple[int, int, int],
    show_line_text: bool,
    show_coeffs: bool,
    rms_only: bool,
    font_scale: float,
) -> None:
    pts = np.asarray(sl.points)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return

    # points are stored as endpoint pairs from Hough segments
    for i in range(0, len(pts) - 1, 2):
        a = (int(pts[i, 0]), int(pts[i, 1]))
        b = (int(pts[i + 1, 0]), int(pts[i + 1, 1]))
        cv2.line(img, a, b, staff_bgr, 2, lineType=cv2.LINE_AA)

    coeffs = np.atleast_1d(sl.polynomial_coeffs)
    try:
        _draw_poly(img, coeffs, color=poly_bgr, thickness=2)
    except Exception:
        logger.debug("Could not draw polynomial for staff line", exc_info=True)
        return

    if not show_line_text:
        return

    xs = pts[:, 0].astype(np.float64)
    ys = pts[:, 1].astype(np.float64)
    pred = np.polyval(coeffs, xs)
    residuals = ys - pred
    rms = float(np.sqrt(np.mean(residuals ** 2))) if residuals.size else 0.0
    deg = len(coeffs) - 1

    if rms_only:
        txt = f"rms={rms:.2f} deg={deg}"
    elif show_coeffs:
        coeffs_fmt = ", ".join(f"{c:.3g}" for c in coeffs)
        txt = f"deg={deg} rms={rms:.2f} coeffs=[{coeffs_fmt}]"
    else:
        txt = f"deg={deg} rms={rms:.2f}"

    min_x = int(np.min(xs))
    y_pos = int(sl.y_center)
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    box_tl = (max(0, min_x), max(0, y_pos - th - 6))
    box_br = (
        min(img.shape[1] - 1, min_x + tw + 8),
        min(img.shape[0] - 1, y_pos + 4),
    )
    cv2.rectangle(img, box_tl, box_br, (240, 240, 240), -1)
    cv2.putText(
        img, txt, (box_tl[0] + 4, box_br[1] - 4),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (10, 10, 10), 1, cv2.LINE_AA,
    )


def _compose_panel(
    annotated: np.ndarray,
    *,
    image_name: str,
    stage_label: str,
    ink_pixels: int,
    staff_count: int,
    suggestions: dict,
    cfg: Config,
) -> np.ndarray:
    panel_w = 340
    h_ann = annotated.shape[0]
    panel = np.full((h_ann, panel_w, 3), 240, dtype=np.uint8)
    lines = [
        f"Image: {image_name}",
        f"Stage: {stage_label}",
        f"Ink pixels: {ink_pixels}",
        f"Staff lines: {staff_count}",
        f"Config hue={cfg.staff_color_hue} range={cfg.staff_color_range}",
    ]
    if suggestions:
        lines.append(
            f"Suggest hue={suggestions['staff_color_hue']} "
            f"range={suggestions['staff_color_range']}"
        )
    y = 30
    for ln in lines:
        # Soft-wrap long lines
        for chunk in _wrap(ln, 36):
            cv2.putText(
                panel, chunk, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA,
            )
            y += 26
    return np.hstack((annotated, panel))


def _wrap(text: str, width: int) -> list[str]:
    if len(text) <= width:
        return [text]
    parts = re.findall(rf".{{1,{width}}}", text)
    return parts or [text]


def _output_path(
    source: Path,
    output_path: str | Path | None,
    output_format: str,
) -> Path:
    fmt = (output_format or "png").lower()
    if fmt == "jpeg":
        fmt = "jpg"
    if output_path:
        out = Path(output_path)
    else:
        out = source.with_name(f"{source.stem}_inspect.{fmt}")
    if out.suffix.lower() not in {f".{fmt}", ".jpeg"}:
        out = out.with_suffix(f".{fmt}")
    return out


def _write_image(path: Path, img: np.ndarray, output_format: str) -> None:
    fmt = (output_format or "png").lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt in ("jpg", "jpeg"):
        ok = cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    else:
        ok = cv2.imwrite(str(path), img)
    if not ok:
        raise InspectError(f"Failed to write {path}")
