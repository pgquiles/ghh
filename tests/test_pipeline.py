"""TDD tests for ghh.pipeline -- BaseStage contract and PipelineState."""

from __future__ import annotations

import json

import cv2
import numpy as np

from ghh.config import Config
from ghh.pipeline import BaseStage, PipelineState, StageResult

# ---------------------------------------------------------------------------
# Concrete test stage (minimal implementation for testing the contract)
# ---------------------------------------------------------------------------

class InvertStage(BaseStage):
    """Test stage that inverts image colors."""

    name = "invert"
    number = 99
    checkpoint_name = "99_inverted"
    error_class = "skippable"
    config_keys = ("staff_color_hue", "staff_color_range")

    def process_image(self, img, metadata, cfg):
        return cv2.bitwise_not(img), {**metadata, "inverted": True}


class FailingStage(BaseStage):
    """Test stage that always raises an error."""

    name = "failing"
    number = 98
    checkpoint_name = "98_failing"
    error_class = "skippable"

    def process_image(self, img, metadata, cfg):
        raise RuntimeError("Intentional failure for testing")


class CriticalFailStage(BaseStage):
    """Test stage with critical error class that always fails."""

    name = "critical_fail"
    number = 97
    checkpoint_name = "97_critical"
    error_class = "critical"

    def process_image(self, img, metadata, cfg):
        raise RuntimeError("Critical failure")


# ---------------------------------------------------------------------------
# TestPipelineState
# ---------------------------------------------------------------------------

class TestPipelineState:
    """Tests for PipelineState: tracking stage completion and config hashes."""

    def test_creates_empty_state(self, tmp_path):
        state = PipelineState(tmp_path)
        assert state.output_dir == tmp_path

    def test_save_and_load(self, tmp_path):
        state = PipelineState(tmp_path)
        state.set_stage_hash("stage_2", "abc123")
        state.mark_image_done("stage_2", "IMG_0001")
        state.save()

        loaded = PipelineState.load(tmp_path)
        assert loaded.get_stage_hash("stage_2") == "abc123"
        assert loaded.is_image_done("stage_2", "IMG_0001")

    def test_pipeline_json_location(self, tmp_path):
        state = PipelineState(tmp_path)
        state.save()
        assert (tmp_path / "pipeline.json").exists()

    def test_is_image_done_returns_false_for_unknown(self, tmp_path):
        state = PipelineState(tmp_path)
        assert state.is_image_done("stage_2", "IMG_0099") is False

    def test_config_hash_invalidation(self, tmp_path):
        state = PipelineState(tmp_path)
        state.set_stage_hash("stage_7", "old_hash")
        assert state.is_stage_invalidated("stage_7", "new_hash") is True
        assert state.is_stage_invalidated("stage_7", "old_hash") is False

    def test_invalidate_clears_images(self, tmp_path):
        state = PipelineState(tmp_path)
        state.mark_image_done("stage_7", "IMG_0001")
        state.mark_image_done("stage_7", "IMG_0002")
        state.invalidate_stage("stage_7")
        assert state.is_image_done("stage_7", "IMG_0001") is False
        assert state.is_image_done("stage_7", "IMG_0002") is False

    def test_config_source_tracking(self, tmp_path):
        state = PipelineState(tmp_path)
        state.config_source = "analyzed"
        state.save()

        loaded = PipelineState.load(tmp_path)
        assert loaded.config_source == "analyzed"

    def test_record_stage_result(self, tmp_path):
        state = PipelineState(tmp_path)
        result = StageResult(
            stage_name="orientation",
            processed=10,
            skipped=2,
            failed=1,
            excluded=0,
        )
        state.record_result(result)
        assert state.get_result("orientation").processed == 10

    def test_load_from_nonexistent_returns_empty(self, tmp_path):
        state = PipelineState.load(tmp_path / "nonexistent")
        assert state is not None


# ---------------------------------------------------------------------------
# TestBaseStage
# ---------------------------------------------------------------------------

class TestBaseStage:
    """Tests for BaseStage.run(): orchestration loop with checkpointing."""

    def _setup_input(self, tmp_path, num_images=3):
        """Create a fake input directory with synthetic PNGs."""
        input_dir = tmp_path / "02_oriented"
        input_dir.mkdir(parents=True)
        cfg = Config(input_dir=tmp_path, output_dir=tmp_path)

        for i in range(num_images):
            img = np.full((100, 100, 3), (100 + i * 30, 80, 60), dtype=np.uint8)
            cv2.imwrite(str(input_dir / f"IMG_{i:04d}.png"), img)

        return input_dir, cfg

    def test_produces_output_files(self, tmp_path):
        input_dir, cfg = self._setup_input(tmp_path)
        stage = InvertStage()
        state = PipelineState(tmp_path)

        result = stage.run(input_dir, tmp_path, cfg, state)

        out_dir = tmp_path / "99_inverted"
        assert out_dir.exists()
        assert len(list(out_dir.glob("*.png"))) == 3

    def test_returns_stage_result(self, tmp_path):
        input_dir, cfg = self._setup_input(tmp_path)
        stage = InvertStage()
        state = PipelineState(tmp_path)

        result = stage.run(input_dir, tmp_path, cfg, state)
        assert isinstance(result, StageResult)
        assert result.processed == 3
        assert result.failed == 0

    def test_skippable_stage_passes_through_on_failure(self, tmp_path):
        input_dir, cfg = self._setup_input(tmp_path)
        stage = FailingStage()
        state = PipelineState(tmp_path)

        result = stage.run(input_dir, tmp_path, cfg, state)

        # Skippable: all images should pass through (copied from input)
        out_dir = tmp_path / "98_failing"
        assert out_dir.exists()
        assert len(list(out_dir.glob("*.png"))) == 3
        assert result.failed == 3
        assert result.processed == 0

    def test_critical_stage_excludes_on_failure(self, tmp_path):
        input_dir, cfg = self._setup_input(tmp_path)
        stage = CriticalFailStage()
        state = PipelineState(tmp_path)

        result = stage.run(input_dir, tmp_path, cfg, state)

        # Critical: failed images should be excluded (not written)
        out_dir = tmp_path / "97_critical"
        assert out_dir.exists()
        assert len(list(out_dir.glob("*.png"))) == 0
        assert result.excluded == 3

    def test_resumes_completed_images(self, tmp_path):
        input_dir, cfg = self._setup_input(tmp_path)
        stage = InvertStage()
        state = PipelineState(tmp_path)

        # Pre-create one output (simulating a previous run)
        out_dir = tmp_path / "99_inverted"
        out_dir.mkdir(parents=True)
        first_input = cv2.imread(str(input_dir / "IMG_0000.png"))
        cv2.imwrite(str(out_dir / "IMG_0000.png"), cv2.bitwise_not(first_input))
        state.mark_image_done("99_inverted", "IMG_0000")

        result = stage.run(input_dir, tmp_path, cfg, state)
        # Should process only 2 new images, skip the pre-existing one
        assert result.processed == 2
        assert result.skipped == 1

    def test_writes_metadata_sidecars(self, tmp_path):
        input_dir, cfg = self._setup_input(tmp_path, num_images=1)
        stage = InvertStage()
        state = PipelineState(tmp_path)

        stage.run(input_dir, tmp_path, cfg, state)

        sidecar = tmp_path / "99_inverted" / "IMG_0000.json"
        assert sidecar.exists()
        meta = json.loads(sidecar.read_text())
        assert meta["inverted"] is True

    def test_should_skip_respects_config(self, tmp_path):
        cfg = Config(input_dir=tmp_path, skip_enhance=True)

        class EnhanceStage(BaseStage):
            name = "enhance"
            number = 9
            checkpoint_name = "09_enhanced"
            error_class = "skippable"
            def process_image(self, img, metadata, cfg):
                return img, metadata

        stage = EnhanceStage()
        assert stage.should_skip(cfg) is True


# ---------------------------------------------------------------------------
# TestConfigHash
# ---------------------------------------------------------------------------

class TestConfigHash:
    """Tests for BaseStage.config_hash() -- deterministic hashing of
    config fields for cache invalidation."""

    def test_deterministic(self, tmp_path):
        cfg = Config(input_dir=tmp_path, staff_color_hue=5, staff_color_range=15)
        stage = InvertStage()
        assert stage.config_hash(cfg) == stage.config_hash(cfg)

    def test_changes_with_config_value(self, tmp_path):
        cfg_a = Config(input_dir=tmp_path, staff_color_hue=5)
        cfg_b = Config(input_dir=tmp_path, staff_color_hue=10)
        stage = InvertStage()
        assert stage.config_hash(cfg_a) != stage.config_hash(cfg_b)

    def test_empty_config_keys_gives_stable_hash(self, tmp_path):
        """Stages with no config_keys always produce the same hash."""
        cfg_a = Config(input_dir=tmp_path, staff_color_hue=5)
        cfg_b = Config(input_dir=tmp_path, staff_color_hue=99)

        class NoKeysStage(BaseStage):
            name = "nokeys"
            number = 90
            checkpoint_name = "90_nokeys"
            error_class = "skippable"
            config_keys = ()
            def process_image(self, img, metadata, cfg):
                return img, metadata

        stage = NoKeysStage()
        assert stage.config_hash(cfg_a) == stage.config_hash(cfg_b)

    def test_hash_length(self, tmp_path):
        cfg = Config(input_dir=tmp_path)
        stage = InvertStage()
        h = stage.config_hash(cfg)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_all_implemented_stages_have_config_keys(self):
        """Every stage class must declare config_keys (even if empty)."""
        from ghh.stages import STAGE_CLASSES
        for cls in STAGE_CLASSES:
            assert hasattr(cls, "config_keys"), (
                f"{cls.__name__} missing config_keys"
            )
            assert isinstance(cls.config_keys, tuple), (
                f"{cls.__name__}.config_keys must be a tuple"
            )


# ---------------------------------------------------------------------------
# TestInvalidateFrom
# ---------------------------------------------------------------------------

class TestInvalidateFrom:
    """Tests for _invalidate_from() -- downstream cascade invalidation."""

    def test_clears_state_and_checkpoint_dir(self, tmp_path):
        from ghh.cli import _invalidate_from

        state = PipelineState(tmp_path)
        state.set_stage_hash("invert", "old_hash")
        state.mark_image_done("99_inverted", "IMG_0001")

        ckpt = tmp_path / "99_inverted"
        ckpt.mkdir()
        (ckpt / "IMG_0001.png").write_bytes(b"fake")

        stage = InvertStage()
        _invalidate_from([stage], state, tmp_path)

        assert state.get_stage_hash("invert") is None
        assert not state.is_image_done("99_inverted", "IMG_0001")
        assert not ckpt.exists()

    def test_cascades_to_multiple_stages(self, tmp_path):
        from ghh.cli import _invalidate_from

        state = PipelineState(tmp_path)

        class StageA(BaseStage):
            name = "a"
            number = 80
            checkpoint_name = "80_a"
            error_class = "skippable"
            def process_image(self, img, metadata, cfg):
                return img, metadata

        class StageB(BaseStage):
            name = "b"
            number = 81
            checkpoint_name = "81_b"
            error_class = "skippable"
            def process_image(self, img, metadata, cfg):
                return img, metadata

        state.set_stage_hash("a", "ha")
        state.set_stage_hash("b", "hb")
        (tmp_path / "80_a").mkdir()
        (tmp_path / "81_b").mkdir()

        _invalidate_from([StageA(), StageB()], state, tmp_path)

        assert state.get_stage_hash("a") is None
        assert state.get_stage_hash("b") is None
        assert not (tmp_path / "80_a").exists()
        assert not (tmp_path / "81_b").exists()

    def test_no_error_when_no_checkpoint_dir(self, tmp_path):
        """Invalidating a stage with no checkpoint dir should not raise."""
        from ghh.cli import _invalidate_from

        state = PipelineState(tmp_path)
        stage = InvertStage()
        _invalidate_from([stage], state, tmp_path)  # should not raise


class TestStaleUpstreamInvalidation:
    """_prepend_stale_upstream(): --stages subsets must notice upstream
    config changes (e.g. new lens coefficients with --stages 4-6)."""

    def _lens_cfg(self, tmp_path, k1):
        return Config(
            input_dir=tmp_path,
            lens_distortion_k1=k1,
            lens_distortion_k2=0.0,
        )

    def _setup_selection(self, tmp_path):
        from ghh.stages import get_stages
        from ghh.stages.page_detect import PageDetectStage

        cfg = self._lens_cfg(tmp_path, -0.14)
        state = PipelineState(tmp_path)
        # Realistic resume: stages 0-2 ran with the current config.
        for upstream in get_stages([0, 1, 2]):
            d = tmp_path / upstream.checkpoint_name
            d.mkdir()
            (d / "IMG_0001.png").write_bytes(b"fake")
            state.set_stage_hash(upstream.name, upstream.config_hash(cfg))
        upstream_dir = tmp_path / "03_lens_corrected"
        upstream_dir.mkdir()
        (upstream_dir / "IMG_0001.png").write_bytes(b"fake")
        downstream_dir = tmp_path / "04_page_detected"
        downstream_dir.mkdir()

        state.set_stage_hash("lens_correct", "old_hash")
        state.mark_image_done("page_detect", "IMG_0001")
        return PageDetectStage(), state, downstream_dir

    def _names(self, stages):
        return [s.name for s in stages]

    def test_upstream_change_prepends_upstream(self, tmp_path):
        from ghh.cli import _prepend_stale_upstream

        page, state, _ = self._setup_selection(tmp_path)
        # Stored hash differs from hash of the new coefficients.
        extended = _prepend_stale_upstream(
            [page], self._lens_cfg(tmp_path, -0.14), state, tmp_path,
        )
        assert self._names(extended) == ["lens_correct", "page_detect"]

    def test_upstream_unchanged_leaves_selection_alone(self, tmp_path):
        from ghh.cli import _prepend_stale_upstream
        from ghh.stages.lens_correct import LensCorrectStage

        page, state, _ = self._setup_selection(tmp_path)
        cfg = self._lens_cfg(tmp_path, -0.14)
        state.set_stage_hash("lens_correct", LensCorrectStage().config_hash(cfg))
        assert _prepend_stale_upstream([page], cfg, state, tmp_path) == [page]

    def test_missing_never_ran_upstream_is_prepended(self, tmp_path):
        """Newly activated stage (no checkpoint, no stored hash) is included."""
        import shutil

        from ghh.cli import _prepend_stale_upstream
        page, state, _ = self._setup_selection(tmp_path)
        shutil.rmtree(tmp_path / "03_lens_corrected")
        state.invalidate_stage("lens_correct")
        extended = _prepend_stale_upstream(
            [page], self._lens_cfg(tmp_path, -0.14), state, tmp_path,
        )
        assert self._names(extended) == ["lens_correct", "page_detect"]

    def test_fresh_selection_is_untouched(self, tmp_path):
        """First runs never prepend: nothing can be stale yet."""
        from ghh.cli import _prepend_stale_upstream

        page, state, _ = self._setup_selection(tmp_path)
        import shutil
        # Selection has no results yet; upstream changed meanwhile.
        shutil.rmtree(tmp_path / "04_page_detected")
        extended = _prepend_stale_upstream(
            [page], self._lens_cfg(tmp_path, -0.14), state, tmp_path,
        )
        assert extended == [page]

    def test_skipped_upstream_is_ignored(self, tmp_path):
        from ghh.cli import _prepend_stale_upstream

        page, state, _ = self._setup_selection(tmp_path)
        # k1=k2=0 → lens_correct should_skip → no invalidation.
        extended = _prepend_stale_upstream(
            [page], self._lens_cfg(tmp_path, 0.0), state, tmp_path,
        )
        assert extended == [page]


class TestStagesSubsetUpstreamE2E:
    """End-to-end: a --stages subset re-runs after an upstream config change.

    Regression test for silently reusing stale inputs: run stages 0-4 on
    a synthetic image, change an upstream (stage 3) coefficient, then run
    only stage 4. Stage 3 must be pulled back in and both must re-process.
    """

    def test_subset_reruns_after_upstream_config_change(self, tmp_path):
        import cv2
        import numpy as np

        from ghh.cli import _run_stage_list
        from ghh.config import Config
        from ghh.stages import get_stages

        input_dir = tmp_path / "in"
        input_dir.mkdir()
        img = np.full((300, 400, 3), 200, dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (350, 250), (30, 30, 30), 3)
        cv2.imwrite(str(input_dir / "IMG_0001.png"), img)

        out = tmp_path / "out"
        cfg = Config(input_dir=input_dir, output_dir=out)
        state = PipelineState(out)
        total, failed = _run_stage_list(
            get_stages([0, 1, 2, 3, 4]), input_dir, out, cfg, state, 1, True,
        )
        assert failed == 0
        assert total > 0

        cfg2 = Config(
            input_dir=input_dir, output_dir=out, lens_distortion_k1=-0.5,
        )
        total2, failed2 = _run_stage_list(
            get_stages([4]), input_dir, out, cfg2, state, 1, True,
        )
        assert failed2 == 0
        # Stage 3 (prepended) and stage 4 both re-processed the image.
        assert total2 == 2


# ---------------------------------------------------------------------------
# TestCheckDiskSpace
# ---------------------------------------------------------------------------

class TestCheckDiskSpace:
    """Tests for _check_disk_space() -- warns when disk is tight."""

    def test_no_crash_on_empty_input(self, tmp_path):
        from ghh.cli import _check_disk_space

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        _check_disk_space(output_dir, input_dir, quiet=True)

    def test_no_crash_on_normal_input(self, tmp_path):
        from ghh.cli import _check_disk_space

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(input_dir / "test.png"), img)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        _check_disk_space(output_dir, input_dir, quiet=False)


# ---------------------------------------------------------------------------
# TestPrintEndReport
# ---------------------------------------------------------------------------

class TestPrintEndReport:
    """Tests for _print_end_report() -- per-stage summary."""

    def test_prints_without_error(self, tmp_path, capsys):
        from ghh.cli import _print_end_report

        state = PipelineState(tmp_path)
        state.record_result(StageResult("orientation", processed=10, skipped=0, failed=0))
        state.record_result(StageResult("deskew", processed=8, skipped=2, failed=0))

        _print_end_report(state, total_p=18, total_f=0)

        captured = capsys.readouterr()
        assert "Pipeline Summary" in captured.out
        assert "orientation" in captured.out
        assert "deskew" in captured.out
        assert "completed" in captured.out

    def test_shows_errors_status(self, tmp_path, capsys):
        from ghh.cli import _print_end_report

        state = PipelineState(tmp_path)
        state.record_result(StageResult("preprocess", processed=5, failed=2))

        _print_end_report(state, total_p=5, total_f=2)

        captured = capsys.readouterr()
        assert "completed with errors" in captured.out
