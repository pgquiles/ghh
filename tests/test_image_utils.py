"""Tests for utils/image_utils.py: estimate_background and trim_to_content."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from ghh.utils.image_utils import estimate_background, trim_to_content

# ---------------------------------------------------------------------------
# TestEstimateBackground
# ---------------------------------------------------------------------------

class TestEstimateBackground:

    def test_uniform_image_returns_that_color(self):
        img = np.full((100, 100, 3), (200, 180, 160), dtype=np.uint8)
        bg = estimate_background(img)
        assert bg == (200, 180, 160)

    def test_ignores_center_content(self):
        img = np.full((200, 200, 3), (230, 230, 230), dtype=np.uint8)
        img[40:160, 40:160] = (10, 10, 10)
        bg = estimate_background(img)
        assert all(c > 200 for c in bg)

    def test_returns_tuple_of_ints(self):
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        bg = estimate_background(img)
        assert isinstance(bg, tuple)
        assert all(isinstance(c, int) for c in bg)
        assert len(bg) == 3

    def test_grayscale_returns_single_value(self):
        img = np.full((100, 100), 128, dtype=np.uint8)
        bg = estimate_background(img)
        assert bg == (128,)
        assert len(bg) == 1

    def test_custom_border_frac(self):
        img = np.full((100, 100, 3), (100, 100, 100), dtype=np.uint8)
        img[:20, :] = (200, 200, 200)
        img[-20:, :] = (200, 200, 200)
        img[:, :20] = (200, 200, 200)
        img[:, -20:] = (200, 200, 200)
        bg = estimate_background(img, border_frac=0.20)
        assert all(c > 150 for c in bg)

    def test_small_image(self):
        img = np.full((5, 5, 3), (42, 42, 42), dtype=np.uint8)
        bg = estimate_background(img)
        assert bg == (42, 42, 42)


# ---------------------------------------------------------------------------
# TestTrimToContent
# ---------------------------------------------------------------------------

class TestTrimToContent:

    def test_trims_background_border(self):
        content = np.full((50, 50, 3), (0, 0, 0), dtype=np.uint8)
        img = np.full((100, 100, 3), (240, 240, 240), dtype=np.uint8)
        img[25:75, 25:75] = content

        result, x_off, y_off = trim_to_content(img, bg_color=(240, 240, 240))
        assert result.shape[0] < img.shape[0]
        assert result.shape[1] < img.shape[1]
        assert x_off == 25
        assert y_off == 25

    def test_adds_margin(self):
        content = np.full((50, 50, 3), (0, 0, 0), dtype=np.uint8)
        img = np.full((200, 200, 3), (240, 240, 240), dtype=np.uint8)
        img[75:125, 75:125] = content

        result, x_off, y_off = trim_to_content(img, bg_color=(240, 240, 240), margin_frac=0.05)
        assert result.shape[0] > 50
        assert result.shape[1] > 50
        margin = max(1, int(50 * 0.05))
        assert x_off == 75 - margin
        assert y_off == 75 - margin

    def test_uniform_image_returns_unchanged(self):
        img = np.full((100, 100, 3), (200, 200, 200), dtype=np.uint8)
        result, x_off, y_off = trim_to_content(img, bg_color=(200, 200, 200))
        assert result.shape == img.shape
        assert x_off == 0
        assert y_off == 0

    def test_auto_estimates_background(self):
        img = np.full((100, 100, 3), (230, 230, 230), dtype=np.uint8)
        img[30:70, 30:70] = (10, 10, 10)
        result, x_off, y_off = trim_to_content(img)
        assert result.shape[0] < img.shape[0]
        assert x_off == 30
        assert y_off == 30

    def test_preserves_content(self):
        bg = (230, 230, 230)
        img = np.full((100, 100, 3), bg, dtype=np.uint8)
        img[40:60, 40:60] = (50, 50, 50)
        result, x_off, y_off = trim_to_content(img, bg_color=bg, margin_frac=0.0, threshold=30)
        center = result[result.shape[0] // 2, result.shape[1] // 2]
        assert all(c < 100 for c in center)
        assert x_off == 40
        assert y_off == 40

    def test_grayscale(self):
        img = np.full((100, 100), 200, dtype=np.uint8)
        img[30:70, 30:70] = 20
        result, x_off, y_off = trim_to_content(img, bg_color=(200,))
        assert result.shape[0] < 100
        assert x_off == 30
        assert y_off == 30

    def test_custom_threshold(self):
        img = np.full((100, 100, 3), (200, 200, 200), dtype=np.uint8)
        img[30:70, 30:70] = (180, 180, 180)
        result_strict, _, _ = trim_to_content(img, bg_color=(200, 200, 200), threshold=10)
        result_loose, x_off, y_off = trim_to_content(img, bg_color=(200, 200, 200), threshold=100)
        assert result_strict.shape[0] < img.shape[0]
        assert result_loose.shape == img.shape
        assert x_off == 0
        assert y_off == 0

    def test_zero_margin_produces_no_padding(self):
        """margin_frac=0 should produce no padding at all (0px, not 1px)."""
        img = np.full((100, 100, 3), (240, 240, 240), dtype=np.uint8)
        img[25:75, 25:75] = (0, 0, 0)
        result, x_off, y_off = trim_to_content(img, bg_color=(240, 240, 240), margin_frac=0.0)
        assert result.shape == (50, 50, 3)
        assert x_off == 25
        assert y_off == 25


def _gradient_page(h=400, w=400):
    """Synthetic page: diagonal light gradient + dark text bars everywhere."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg = 40 + 170 * (xx / w + yy / h) / 2  # 40 (dark corner) .. 210 (bright)
    gray = bg.astype(np.uint8)
    truth = np.zeros((h, w), dtype=bool)
    for y in range(30, h - 20, 40):  # text bars span the full width
        gray[y:y + 8, 10:w - 10] = 20
        truth[y:y + 8, 10:w - 10] = True
    # A text-free strip in the dark corner to measure false ink.
    dark_strip = np.zeros((h, w), dtype=bool)
    dark_strip[:, :8] = True
    return gray, truth, dark_strip


class TestFlattenIllumination:
    """flatten_illumination / threshold_otsu_flattened (ghh#89)."""

    def test_recovers_text_under_gradient(self):
        from ghh.utils.image_utils import threshold_otsu_flattened

        gray, truth, dark_strip = _gradient_page()
        _, raw = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        flat = threshold_otsu_flattened(gray)
        raw_ink = raw == 0
        flat_ink = flat == 0

        # Dark-zone background must not read as ink ...
        assert float(raw_ink[dark_strip].mean()) > float(flat_ink[dark_strip].mean())
        # ... and real text must survive everywhere, dark corner included.
        assert float(flat_ink[truth].mean()) > 0.95
        # Overall: raw Otsu buys its recall by calling dark background ink;
        # flattening must produce strictly fewer false positives.
        raw_fp = float((raw_ink & ~truth).mean())
        flat_fp = float((flat_ink & ~truth).mean())
        assert flat_fp < raw_fp

    def test_output_is_binary_ink_black(self):
        from ghh.utils.image_utils import threshold_otsu_flattened

        gray, _, _ = _gradient_page()
        flat = threshold_otsu_flattened(gray)
        assert flat.dtype == np.uint8
        assert set(np.unique(flat).tolist()) <= {0, 255}

    def test_clean_page_mostly_unchanged(self):
        """Do-no-harm: on evenly lit pages flattening is ~a no-op."""
        from ghh.utils.image_utils import threshold_otsu_flattened

        gray = np.full((200, 200), 200, dtype=np.uint8)
        gray[40:48, 20:180] = 20
        gray[100:108, 20:180] = 20
        _, raw = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        flat = threshold_otsu_flattened(gray)
        agreement = float((raw == flat).mean())
        assert agreement > 0.98

    def test_blur_frac_validated(self):
        from ghh.utils.image_utils import flatten_illumination

        gray, _, _ = _gradient_page()
        for bad in (0.0, -0.1, 1.0, 2.0):
            with pytest.raises(ValueError, match="blur_frac"):
                flatten_illumination(gray, blur_frac=bad)
